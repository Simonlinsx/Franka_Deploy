"""Injected, discrete Inspire RH56 driver for staged grasp execution.

Importing this module performs no serial discovery and does not import anything
from ``examples``.  A caller may inject an already-open ``RH56Hand`` plus the
module-like register API, or explicitly call :meth:`RH56SequenceDriver.connect`
to perform the lazy import and open a serial context.

The driver intentionally does not impose a host-side aggregate-current fault
during active motion.  Device CURRENT_LIMIT remains authoritative per axis;
commissioning paths may retain a stricter reviewed per-axis cap, and the same
effective caps are checked during their read-only bounded hold.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import inspect
import sys
import threading
import time
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .rh56_hand_path import (
    Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS,
    RH56HandExecutionPath,
    build_rh56_no_contact_execution_path,
    require_rh56_feedback_in_interval,
    rh56_feedback_envelope_policy,
    validate_rh56_hand_execution_path,
    validate_rh56_feedback_envelope_policy,
)


AXIS_NAMES: Tuple[str, ...] = (
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotate",
)
OPEN_TARGETS: Tuple[int, ...] = (1000, 1000, 1000, 1000, 1000, 1000)
DISABLED_TARGETS: Tuple[int, ...] = (-1, -1, -1, -1, -1, -1)
COMMISSIONING_SPEED = 40
COMMISSIONING_FORCE_G = 80
DEVICE_MAX_AXIS_CURRENT_MA = 1400


class RH56SequenceDriverError(RuntimeError):
    """Invalid state, communication, or actuator feedback during a sequence."""


class RH56MotionStopped(RH56SequenceDriverError):
    """A stop request interrupted active monitoring."""


class RH56StopUnconfirmed(RH56SequenceDriverError):
    """The driver could not prove that all six outputs and feedback are idle."""


class RH56ValidatedFeedbackObserverError(RH56SequenceDriverError):
    """A validated-feedback observer failed and latched the current session."""


@dataclass(frozen=True)
class RH56Telemetry:
    phase: str
    elapsed_s: float
    angle_targets: Tuple[int, ...]
    angles: Tuple[int, ...]
    positions: Optional[Tuple[int, ...]]
    forces: Optional[Tuple[int, ...]]
    currents: Tuple[int, ...]
    errors: Tuple[int, ...]
    statuses: Tuple[int, ...]
    temperatures: Tuple[int, ...]


@dataclass(frozen=True)
class RH56StateSnapshot:
    """One read-only six-axis sample captured at a sequence boundary."""

    angle_targets: Tuple[int, ...]
    angles: Tuple[int, ...]
    positions: Optional[Tuple[int, ...]]
    forces: Optional[Tuple[int, ...]]
    currents: Tuple[int, ...]
    errors: Tuple[int, ...]
    statuses: Tuple[int, ...]
    temperatures: Tuple[int, ...]
    contact_axes: Tuple[str, ...]


@dataclass(frozen=True)
class _Feedback:
    # Capture/read-complete time for the register values below.  A zero pair
    # means no validated-feedback observer was installed, so the driver
    # intentionally avoided two otherwise-unused clock calls.
    timestamp_unix_ns: int
    timestamp_monotonic_ns: int
    angle_targets: Tuple[int, ...]
    angles: Tuple[int, ...]
    positions: Optional[Tuple[int, ...]]
    forces: Optional[Tuple[int, ...]]
    currents: Tuple[int, ...]
    errors: Tuple[int, ...]
    statuses: Tuple[int, ...]
    temperatures: Tuple[int, ...]


@dataclass(frozen=True)
class RH56ValidatedFeedbackObservation:
    """One fully checked RH56 feedback sample delivered to a passive observer.

    ``timestamp_unix_ns`` and ``timestamp_monotonic_ns`` describe when the
    final register belonging to this sample finished reading.  They are
    captured before subsequent envelope/external-Franka checks; the observer
    receives the sample only after every check applicable to ``phase`` has
    accepted those same values.  Installing an observer never adds a register
    read.
    ``observer_identity`` binds the event to the exact installation and lets a
    downstream publisher reject cross-run feedback.
    """

    observer_identity: str
    phase: str
    timestamp_unix_ns: int
    timestamp_monotonic_ns: int
    angle_targets: Tuple[int, ...]
    angles: Tuple[int, ...]
    positions: Optional[Tuple[int, ...]]
    forces: Optional[Tuple[int, ...]]
    currents: Tuple[int, ...]
    errors: Tuple[int, ...]
    statuses: Tuple[int, ...]
    temperatures: Tuple[int, ...]


def _six_integral(values: Sequence[Any], name: str, *, allow_disabled: bool) -> Tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a six-value sequence") from exc
    if len(raw) != 6:
        raise ValueError(f"{name} must contain exactly six values")
    result = []
    minimum = -1 if allow_disabled else 0
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name}[{index}] must be an integer")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be an integer") from exc
        if not np.isfinite(numeric) or numeric != round(numeric):
            raise ValueError(f"{name}[{index}] must be a finite integer")
        integer = int(numeric)
        if not minimum <= integer <= 1000:
            raise ValueError(f"{name}[{index}] must be in {minimum}..1000")
        result.append(integer)
    return tuple(result)


class RH56SequenceDriver:
    """Discrete six-axis RH56 driver implementing the staged hand protocol.

    The conservative default thumb range is the already commissioned realtime
    range 900..1000.  Expanding it is an explicit construction-time decision and
    should only follow a supervised, staged axis-six acceptance procedure.
    """

    def __init__(
        self,
        hand: Any,
        api: Any,
        *,
        thumb_rotate_range: Tuple[int, int] = (900, 1000),
        motion_timeout_s: float = 20.0,
        poll_interval_s: float = 0.08,
        angle_tolerance: int = 20,
        open_min_angle: int = 980,
        thumb_preshape_step_units: int = 25,
        stop_max_axis_current_ma: int = 100,
        stop_verify_samples: int = 3,
        stop_verify_interval_s: float = 0.10,
        stop_verify_timeout_s: float = 1.50,
        external_safety_check: Optional[Callable[[], None]] = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        time_ns: Callable[[], int] = time.time_ns,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.hand = hand
        self.api = api
        if hand is None or api is None:
            raise ValueError("hand and api are required")
        if len(tuple(thumb_rotate_range)) != 2:
            raise ValueError("thumb_rotate_range must contain min and max")
        thumb_min, thumb_max = (
            int(thumb_rotate_range[0]),
            int(thumb_rotate_range[1]),
        )
        if not 0 <= thumb_min <= thumb_max <= 1000:
            raise ValueError("thumb_rotate_range must lie within 0..1000")
        if not 5.0 <= float(motion_timeout_s) <= 30.0:
            raise ValueError("motion_timeout_s must be in 5..30")
        if float(poll_interval_s) <= 0.0:
            raise ValueError("poll_interval_s must be positive")
        if not 0 <= int(angle_tolerance) <= 100:
            raise ValueError("angle_tolerance must be in 0..100")
        if not 900 <= int(open_min_angle) <= 1000:
            raise ValueError("open_min_angle must be in 900..1000")
        if (
            isinstance(thumb_preshape_step_units, (bool, np.bool_))
            or int(thumb_preshape_step_units) != float(thumb_preshape_step_units)
            or not 10 <= int(thumb_preshape_step_units) <= 50
        ):
            raise ValueError("thumb_preshape_step_units must be an integer in 10..50")
        if (
            isinstance(stop_max_axis_current_ma, (bool, np.bool_))
            or int(stop_max_axis_current_ma) != float(stop_max_axis_current_ma)
            or not 1 <= int(stop_max_axis_current_ma) <= 400
        ):
            raise ValueError("stop_max_axis_current_ma must be an integer in 1..400")
        if not 2 <= int(stop_verify_samples) <= 10:
            raise ValueError("stop_verify_samples must be in 2..10")
        if float(stop_verify_interval_s) < 0.0:
            raise ValueError("stop_verify_interval_s cannot be negative")
        if not 0.10 <= float(stop_verify_timeout_s) <= 5.0:
            raise ValueError("stop_verify_timeout_s must be in 0.10..5.0")
        minimum_stop_window_s = (
            (int(stop_verify_samples) - 1) * float(stop_verify_interval_s)
        )
        if float(stop_verify_timeout_s) < minimum_stop_window_s:
            raise ValueError(
                "stop_verify_timeout_s is too short for stop_verify_samples "
                "and stop_verify_interval_s"
            )
        if external_safety_check is not None and not callable(external_safety_check):
            raise ValueError("external_safety_check must be callable or None")
        if not callable(time_ns) or not callable(monotonic_ns):
            raise ValueError("time_ns and monotonic_ns must be callable")
        required_api = (
            "REG_ANGLE_SET",
            "REG_ANGLE_ACT",
            "REG_FORCE_SET",
            "REG_SPEED_SET",
            "REG_CURRENT",
            "REG_ERROR",
            "REG_STATUS",
            "REG_TEMP",
            "open_hand",
        )
        missing = [name for name in required_api if not hasattr(api, name)]
        if missing:
            raise ValueError("api is missing RH56 members: " + ", ".join(missing))

        self.thumb_rotate_range = (thumb_min, thumb_max)
        self.motion_timeout_s = float(motion_timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self.angle_tolerance = int(angle_tolerance)
        self.open_min_angle = int(open_min_angle)
        self.thumb_preshape_step_units = int(thumb_preshape_step_units)
        self.stop_max_axis_current_ma = int(stop_max_axis_current_ma)
        self.stop_verify_samples = int(stop_verify_samples)
        self.stop_verify_interval_s = float(stop_verify_interval_s)
        self.stop_verify_timeout_s = float(stop_verify_timeout_s)
        self._external_safety_check = external_safety_check
        self._monotonic = monotonic
        self._sleep = sleep
        self._time_ns = time_ns
        self._monotonic_ns = monotonic_ns
        self._operation_lock = threading.RLock()
        self._operation_depth = 0
        self._motion_write_lock = threading.RLock()
        self._io_lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._preshaped_q6: Optional[int] = None
        self._commissioned_q6_waypoints: Optional[Tuple[int, ...]] = None
        self._commissioned_bend_waypoints: Optional[Tuple[Tuple[int, ...], ...]] = None
        self._numeric_hold_targets: Optional[Tuple[int, ...]] = None
        self._numeric_hold_current_caps: Optional[Tuple[int, ...]] = None
        self._audited_no_contact_path: Optional[RH56HandExecutionPath] = None
        self._audited_feedback_envelope_policy: Optional[dict] = None
        self._disabled_verified = False
        self._closed = False
        self._serial_context: Optional[Any] = None
        self._original_speeds: Optional[Tuple[int, ...]] = None
        self._original_forces: Optional[Tuple[int, ...]] = None
        self._validated_feedback_observer: Optional[
            Callable[[RH56ValidatedFeedbackObservation], None]
        ] = None
        self._validated_feedback_observer_identity: Optional[str] = None
        self._validated_feedback_observer_active = False
        self.telemetry: List[RH56Telemetry] = []
        self.last_contact_axes: Tuple[str, ...] = ()
        self.last_emergency_disable_errors: Tuple[str, ...] = ()

    @contextmanager
    def _operation_scope(self):
        """Hold unique-owner state and make observer replacement impossible."""

        with self._operation_lock:
            if self._validated_feedback_observer_active:
                raise RH56SequenceDriverError(
                    "validated feedback observer cannot enter a driver operation"
                )
            self._operation_depth += 1
            try:
                yield
            finally:
                self._operation_depth -= 1

    def install_validated_feedback_observer(
        self,
        observer: Callable[[RH56ValidatedFeedbackObservation], None],
        *,
        identity: str,
    ) -> None:
        """Install one passive observer outside an active driver operation.

        The non-empty identity is copied into every observation and must match
        exactly when the observer is removed.  An existing observer can never
        be replaced in place; callers must remove it between operations first.
        """

        if not callable(observer):
            raise ValueError("validated feedback observer must be callable")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("validated feedback observer identity must be non-empty text")
        bound_identity = identity.strip()
        with self._operation_lock:
            if self._operation_depth:
                raise RH56SequenceDriverError(
                    "validated feedback observer cannot be installed during an operation"
                )
            if self._validated_feedback_observer is not None:
                raise RH56SequenceDriverError(
                    "validated feedback observer is already installed"
                )
            self._validated_feedback_observer = observer
            self._validated_feedback_observer_identity = bound_identity

    def remove_validated_feedback_observer(self, *, identity: str) -> None:
        """Remove the observer only for the exact identity that installed it."""

        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("validated feedback observer identity must be non-empty text")
        bound_identity = identity.strip()
        with self._operation_lock:
            if self._operation_depth:
                raise RH56SequenceDriverError(
                    "validated feedback observer cannot be removed during an operation"
                )
            if self._validated_feedback_observer is None:
                raise RH56SequenceDriverError(
                    "validated feedback observer is not installed"
                )
            if self._validated_feedback_observer_identity != bound_identity:
                raise RH56SequenceDriverError(
                    "validated feedback observer identity does not match"
                )
            self._validated_feedback_observer = None
            self._validated_feedback_observer_identity = None

    def _observe_validated_feedback(self, feedback: _Feedback, phase: str) -> None:
        """Publish an already-validated sample without performing any RH56 IO."""

        observer = self._validated_feedback_observer
        if observer is None:
            return
        identity = self._validated_feedback_observer_identity
        if identity is None:
            raise AssertionError("validated feedback observer has no bound identity")
        observation = RH56ValidatedFeedbackObservation(
            observer_identity=identity,
            phase=str(phase),
            timestamp_unix_ns=feedback.timestamp_unix_ns,
            timestamp_monotonic_ns=feedback.timestamp_monotonic_ns,
            angle_targets=feedback.angle_targets,
            angles=feedback.angles,
            positions=feedback.positions,
            forces=feedback.forces,
            currents=feedback.currents,
            errors=feedback.errors,
            statuses=feedback.statuses,
            temperatures=feedback.temperatures,
        )
        try:
            self._validated_feedback_observer_active = True
            observer(observation)
        except BaseException as exc:
            # Latch immediately.  Unloaded motion methods subsequently enter
            # their normal fail-and-disable handler.  verify_loaded_hold is the
            # deliberate exception: it propagates this latched failure while
            # preserving the numeric grasp target so telemetry failure cannot
            # drop a suspended object.
            self._stop_requested.set()
            raise RH56ValidatedFeedbackObserverError(
                f"{phase}: validated feedback observer failed: {exc}"
            ) from exc
        finally:
            self._validated_feedback_observer_active = False

    def install_external_safety_check(self, callback: Callable[[], None]) -> None:
        """Install the one-shot arm watchdog while RH56 is proven disabled.

        Formal joint execution connects and disables the hand before it can
        safely construct/read the Franka client.  This setter closes that
        ordering gap without permitting a watchdog swap during numeric hold.
        """

        if not callable(callback):
            raise ValueError("external safety check must be callable")
        with self._operation_lock:
            if not self._disabled_verified or self._numeric_hold_targets is not None:
                raise RH56SequenceDriverError(
                    "external safety check can only be installed while disabled"
                )
            if self._external_safety_check is not None:
                raise RH56SequenceDriverError(
                    "external safety check is already installed"
                )
            self._external_safety_check = callback

    def bind_audited_no_contact_execution_path(
        self,
        payload: Any,
        *,
        feedback_envelope_policy: Any,
    ) -> None:
        """Bind the exact path and runtime feedback envelope while disabled."""

        path = validate_rh56_hand_execution_path(payload)
        bound_policy = dict(
            validate_rh56_feedback_envelope_policy(
                feedback_envelope_policy, self.angle_tolerance
            )
        )
        lower, upper = self.thumb_rotate_range
        if not lower <= path.target[5] <= upper:
            raise RH56SequenceDriverError(
                "audited RH56 q6 target is outside the configured realtime range"
            )
        with self._operation_lock:
            if not self._disabled_verified or self._numeric_hold_targets is not None:
                raise RH56SequenceDriverError(
                    "RH56 hand path can only be bound while all outputs are disabled"
                )
            if path.step_units != self.thumb_preshape_step_units:
                raise RH56SequenceDriverError(
                    "audited RH56 step size differs from configured driver step"
                )
            self._audited_no_contact_path = path
            self._audited_feedback_envelope_policy = bound_policy

    @property
    def feedback_envelope_policy(self) -> dict:
        """Return the immutable contract enforced for every numeric interval."""

        return dict(rh56_feedback_envelope_policy(self.angle_tolerance))

    @classmethod
    def connect(
        cls,
        *,
        port: Optional[str] = None,
        baud: int = 115200,
        hand_id: int = 1,
        serial_timeout_s: float = 0.5,
        debug: bool = False,
        api: Any = None,
        **driver_kwargs: Any,
    ) -> "RH56SequenceDriver":
        """Explicitly open one RH56 connection; imports the API lazily."""

        active_api = (
            importlib.import_module("examples.inspire_rh56_test")
            if api is None
            else api
        )
        resolved_port = port or active_api.find_serial_port()
        serial_context = active_api.LinuxSerial(
            resolved_port,
            int(baud),
            float(serial_timeout_s),
            bool(debug),
        )
        entered = None
        enter_completed = False
        try:
            entered = serial_context.__enter__()
            enter_completed = True
            serial_port = serial_context if entered is None else entered
            hand = active_api.RH56Hand(serial_port, int(hand_id))
            driver = cls(hand, active_api, **driver_kwargs)
            driver._serial_context = serial_context
            return driver
        except BaseException:
            if enter_completed:
                serial_context.__exit__(*sys.exc_info())
            raise

    def __enter__(self) -> "RH56SequenceDriver":
        if self._closed:
            raise RuntimeError("RH56 sequence driver is already closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    @property
    def numeric_hold_targets(self) -> Optional[Tuple[int, ...]]:
        return self._numeric_hold_targets

    @property
    def numeric_hold_current_caps(self) -> Optional[Tuple[int, ...]]:
        """Effective per-axis caps retained from the accepted numeric close."""

        return self._numeric_hold_current_caps

    def _constant(self, name: str) -> int:
        return int(getattr(self.api, name))

    def _ensure_motion_allowed(self) -> None:
        if self._closed:
            raise RH56MotionStopped("driver is closed")
        if self._stop_requested.is_set():
            raise RH56MotionStopped("RH56 stop has been requested")

    def request_stop(self) -> None:
        """Latch stop, then wait out any numeric write already in progress."""

        self._stop_requested.set()
        with self._motion_write_lock:
            pass

    def _read_six(self, address: int) -> Tuple[int, ...]:
        if self._validated_feedback_observer_active:
            raise RH56SequenceDriverError(
                "validated feedback observer cannot perform RH56 register IO"
            )
        with self._io_lock:
            values = self.hand.read_six_shorts(address, retries=0)
        result = tuple(int(value) for value in values)
        if len(result) != 6:
            raise RH56SequenceDriverError(
                f"register {address} returned {len(result)} values, expected six"
            )
        return result

    def _read_bytes(self, address: int) -> Tuple[int, ...]:
        if self._validated_feedback_observer_active:
            raise RH56SequenceDriverError(
                "validated feedback observer cannot perform RH56 register IO"
            )
        with self._io_lock:
            values = self.hand.read(address, 6, retries=0)
        result = tuple(int(value) for value in values)
        if len(result) != 6:
            raise RH56SequenceDriverError(
                f"register {address} returned {len(result)} values, expected six"
            )
        return result

    def _write_six_verified(
        self,
        address: int,
        values: Sequence[int],
        *,
        numeric_motion: bool,
    ) -> Tuple[int, ...]:
        if self._validated_feedback_observer_active:
            raise RH56SequenceDriverError(
                "validated feedback observer cannot perform RH56 register IO"
            )
        expected = _six_integral(
            values,
            "register values",
            allow_disabled=(address == self._constant("REG_ANGLE_SET")),
        )
        with self._motion_write_lock:
            if numeric_motion:
                self._ensure_motion_allowed()
            with self._io_lock:
                self.hand.write_six_shorts(address, expected, retries=1)
                readback = tuple(
                    int(value)
                    for value in self.hand.read_six_shorts(address, retries=1)
                )
        if readback != expected:
            raise RH56SequenceDriverError(
                f"register {address} readback mismatch: "
                f"expected={expected}, actual={readback}"
            )
        return readback

    def _configure_commissioning_settings(self) -> None:
        if self._original_speeds is None:
            self._original_speeds = self._read_six(
                self._constant("REG_SPEED_SET")
            )
            self._original_forces = self._read_six(
                self._constant("REG_FORCE_SET")
            )
        self._write_six_verified(
            self._constant("REG_SPEED_SET"),
            (COMMISSIONING_SPEED,) * 6,
            numeric_motion=True,
        )
        self._write_six_verified(
            self._constant("REG_FORCE_SET"),
            (COMMISSIONING_FORCE_G,) * 6,
            numeric_motion=True,
        )

    def _restore_original_settings(self) -> None:
        if self._original_speeds is None or self._original_forces is None:
            return
        self._write_six_verified(
            self._constant("REG_SPEED_SET"),
            self._original_speeds,
            numeric_motion=False,
        )
        self._write_six_verified(
            self._constant("REG_FORCE_SET"),
            self._original_forces,
            numeric_motion=False,
        )

    def _feedback_supported(self) -> bool:
        required = (
            "REG_ANGLE_SET",
            "REG_ANGLE_ACT",
            "REG_CURRENT",
            "REG_ERROR",
            "REG_STATUS",
            "REG_TEMP",
        )
        return (
            all(hasattr(self.api, name) for name in required)
            and hasattr(self.hand, "read_six_shorts")
            and hasattr(self.hand, "read")
        )

    def _run_external_safety_check(self, phase: str, boundary: str) -> None:
        callback = self._external_safety_check
        if callback is None:
            return
        try:
            callback()
        except BaseException as exc:
            raise RH56SequenceDriverError(
                f"{phase}: external safety check failed {boundary} feedback: {exc}"
            ) from exc

    def _run_reviewed_open_helper(
        self,
        *,
        include_thumb_rotate: bool,
        simultaneous_bend_open: bool = False,
        max_axis_current_ma: Optional[int] = None,
        endpoint_stable_samples: Optional[int] = None,
        max_inactive_drift_units: Optional[int] = None,
    ) -> None:
        """Run the reviewed helper only if its motion loop accepts our watchdog."""

        self._run_external_safety_check("open_helper", "before")
        open_kwargs = dict(
            speed=COMMISSIONING_SPEED,
            force_limit=COMMISSIONING_FORCE_G,
            motion_timeout=self.motion_timeout_s,
            include_thumb_rotate=bool(include_thumb_rotate),
        )
        strict_helper_gate = any(
            value is not None
            for value in (
                max_axis_current_ma,
                endpoint_stable_samples,
                max_inactive_drift_units,
            )
        )
        helper_started = self._monotonic()

        def record_helper_feedback(payload: Mapping[str, Any]) -> None:
            if not isinstance(payload, Mapping):
                raise RH56SequenceDriverError(
                    "open helper returned malformed validated feedback"
                )

            def six(name: str) -> Tuple[int, ...]:
                try:
                    values = tuple(int(value) for value in payload[name])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RH56SequenceDriverError(
                        f"open helper feedback lacks valid {name}"
                    ) from exc
                if len(values) != 6:
                    raise RH56SequenceDriverError(
                        f"open helper feedback {name} must contain six values"
                    )
                return values

            phase = str(payload.get("phase", ""))
            if not (
                phase.startswith("reset_open_bend_m")
                or phase == "reset_open_bends_all_1000"
            ):
                raise RH56SequenceDriverError(
                    f"open helper feedback phase is invalid: {phase!r}"
                )
            self.telemetry.append(
                RH56Telemetry(
                    phase=phase,
                    elapsed_s=max(
                        0.0, float(self._monotonic()) - float(helper_started)
                    ),
                    angle_targets=six("angle_targets"),
                    angles=six("angles"),
                    positions=six("positions"),
                    forces=None,
                    currents=six("currents"),
                    errors=six("errors"),
                    statuses=six("statuses"),
                    temperatures=six("temperatures"),
                )
            )

        if (
            self._external_safety_check is not None
            or strict_helper_gate
            or simultaneous_bend_open
        ):
            try:
                helper_parameters = inspect.signature(self.api.open_hand).parameters
            except (TypeError, ValueError) as exc:
                raise RH56SequenceDriverError(
                    "cannot prove that the open helper supports the continuous "
                    "external safety gate"
                ) from exc
            if "safety_check" not in helper_parameters:
                if self._external_safety_check is not None:
                    raise RH56SequenceDriverError(
                        "open helper lacks the required continuous external safety gate"
                    )
            elif self._external_safety_check is not None:
                open_kwargs["safety_check"] = self._external_safety_check
            if simultaneous_bend_open:
                if "simultaneous_bend_open" not in helper_parameters:
                    raise RH56SequenceDriverError(
                        "open helper lacks simultaneous bend-open support"
                    )
                open_kwargs["simultaneous_bend_open"] = True
            if max_axis_current_ma is not None:
                if "max_axis_current_ma" not in helper_parameters:
                    raise RH56SequenceDriverError(
                        "open helper lacks the required host current-cap gate"
                    )
                open_kwargs["max_axis_current_ma"] = int(max_axis_current_ma)
            if endpoint_stable_samples is not None:
                if "endpoint_stable_samples" not in helper_parameters:
                    raise RH56SequenceDriverError(
                        "open helper lacks the required stable-endpoint gate"
                    )
                open_kwargs["endpoint_stable_samples"] = int(
                    endpoint_stable_samples
                )
            if max_inactive_drift_units is not None:
                if "max_inactive_drift_units" not in helper_parameters:
                    raise RH56SequenceDriverError(
                        "open helper lacks the required inactive-axis drift gate"
                    )
                open_kwargs["max_inactive_drift_units"] = int(
                    max_inactive_drift_units
                )
            if strict_helper_gate:
                if "feedback_callback" not in helper_parameters:
                    raise RH56SequenceDriverError(
                        "open helper lacks the required validated-feedback transcript"
                    )
                open_kwargs["feedback_callback"] = record_helper_feedback
        with self._motion_write_lock:
            self._ensure_motion_allowed()
            with self._io_lock:
                self.api.open_hand(self.hand, **open_kwargs)
        self._run_external_safety_check("open_helper", "after")
        self._ensure_motion_allowed()

    def _read_feedback(
        self,
        phase: str,
        started_at: float,
        *,
        allow_external_gate_bypass_after_disabled_readback: bool = False,
    ) -> _Feedback:
        cleanup_bypass = bool(
            allow_external_gate_bypass_after_disabled_readback
        )
        if cleanup_bypass:
            # Cleanup must remain possible when Franka is in Move/Reflex and
            # its normal Idle watchdog therefore raises.  The one read allowed
            # before establishing this exception is ANGLE_SET itself.  No
            # other RH56 feedback bypasses the watchdog until all six outputs
            # have been proven disabled, and the targets are checked again at
            # the end of the sample.
            targets = self._read_six(self._constant("REG_ANGLE_SET"))
            if targets != DISABLED_TARGETS:
                raise RH56SequenceDriverError(
                    f"{phase}: external safety gate bypass requires all-six "
                    f"ANGLE_SET=-1; actual={targets}"
                )
        else:
            self._run_external_safety_check(str(phase), "before")
            targets = self._read_six(self._constant("REG_ANGLE_SET"))
        angles = self._read_six(self._constant("REG_ANGLE_ACT"))
        positions = (
            self._read_six(self._constant("REG_POS_ACT"))
            if hasattr(self.api, "REG_POS_ACT")
            else None
        )
        forces = (
            self._read_six(self._constant("REG_FORCE_ACT"))
            if hasattr(self.api, "REG_FORCE_ACT")
            else None
        )
        currents = self._read_six(self._constant("REG_CURRENT"))
        errors = self._read_bytes(self._constant("REG_ERROR"))
        statuses = self._read_bytes(self._constant("REG_STATUS"))
        temperatures = self._read_bytes(self._constant("REG_TEMP"))
        # These timestamps belong to the register payload above, not to the
        # later point at which phase-specific validation happens.  Avoid both
        # clock calls entirely when no observer can consume the provenance.
        if self._validated_feedback_observer is None:
            timestamp_unix_ns = 0
            timestamp_monotonic_ns = 0
        else:
            timestamp_unix_ns = int(self._time_ns())
            timestamp_monotonic_ns = int(self._monotonic_ns())
        feedback = _Feedback(
            timestamp_unix_ns=timestamp_unix_ns,
            timestamp_monotonic_ns=timestamp_monotonic_ns,
            angle_targets=targets,
            angles=angles,
            positions=positions,
            forces=forces,
            currents=currents,
            errors=errors,
            statuses=statuses,
            temperatures=temperatures,
        )
        self.telemetry.append(
            RH56Telemetry(
                phase=str(phase),
                elapsed_s=max(0.0, float(self._monotonic()) - float(started_at)),
                angle_targets=targets,
                angles=angles,
                positions=positions,
                forces=forces,
                currents=currents,
                errors=errors,
                statuses=statuses,
                temperatures=temperatures,
            )
        )
        if cleanup_bypass:
            final_targets = self._read_six(
                self._constant("REG_ANGLE_SET")
            )
            if final_targets != DISABLED_TARGETS:
                raise RH56SequenceDriverError(
                    f"{phase}: ANGLE_SET changed during cleanup feedback: "
                    f"{final_targets}"
                )
        else:
            self._run_external_safety_check(str(phase), "after")
        return feedback

    @staticmethod
    def _check_fault_feedback(feedback: _Feedback, phase: str) -> None:
        if any(feedback.errors):
            raise RH56SequenceDriverError(
                f"{phase}: actuator ERROR is nonzero: {feedback.errors}"
            )
        if max(feedback.temperatures) >= 60:
            raise RH56SequenceDriverError(
                f"{phase}: actuator temperature reached 60 C: "
                f"{feedback.temperatures}"
            )
        for index, status in enumerate(feedback.statuses):
            if status in (5, 6, 7):
                raise RH56SequenceDriverError(
                    f"{phase}: {AXIS_NAMES[index]} returned fault status {status}"
                )

    def _check_feedback_envelope(
        self,
        feedback: _Feedback,
        previous_configuration: Sequence[int],
        current_configuration: Sequence[int],
        phase: str,
        *,
        previous_phase: str = "",
        current_phase: str = "",
    ) -> None:
        """Enforce the audited adjacent-endpoint hull on every live sample.

        This runs after the new ANGLE_SET readback is verified.  A sample may
        still equal the preceding endpoint (normal actuator lag), may be in
        transit, or may lie inside the accepted new endpoint band.  It may not
        escape past either endpoint band on any axis.  The surrounding
        fail-and-latch handler requests all-six disable on any violation.
        """

        try:
            require_rh56_feedback_in_interval(
                feedback.angles,
                previous_configuration,
                current_configuration,
                self.angle_tolerance,
                previous_phase=previous_phase,
                current_phase=current_phase,
                name="{} ANGLE_ACT".format(phase),
            )
        except ValueError as exc:
            raise RH56SequenceDriverError(str(exc)) from exc

    def _commissioning_current_caps(
        self, max_axis_current_ma: Any
    ) -> Tuple[int, ...]:
        """Return strict per-axis caps bounded by the device registers.

        The installed-hand commissioning path intentionally has no arbitrary
        aggregate-current trip.  Each actuator is instead checked on every
        feedback sample against both its device CURRENT_LIMIT register and the
        explicitly requested, lower host-side cap.
        """

        if isinstance(max_axis_current_ma, (bool, np.bool_)):
            raise ValueError("max_axis_current_ma must be an integer in 1..1400")
        try:
            numeric = float(max_axis_current_ma)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "max_axis_current_ma must be an integer in 1..1400"
            ) from exc
        if (
            not np.isfinite(numeric)
            or numeric != round(numeric)
            or not 1 <= int(numeric) <= 1400
        ):
            raise ValueError("max_axis_current_ma must be an integer in 1..1400")
        if not hasattr(self.api, "REG_CURRENT_LIMIT"):
            raise RH56SequenceDriverError(
                "strict commissioning requires REG_CURRENT_LIMIT feedback"
            )
        device_limits = self._read_six(self._constant("REG_CURRENT_LIMIT"))
        if any(value <= 0 or value > 5000 for value in device_limits):
            raise RH56SequenceDriverError(
                "device CURRENT_LIMIT registers are invalid: "
                f"{device_limits}"
            )
        requested = int(numeric)
        return tuple(min(requested, value) for value in device_limits)

    @staticmethod
    def _check_commissioning_currents(
        feedback: _Feedback,
        caps_ma: Sequence[int],
        phase: str,
    ) -> None:
        caps = tuple(int(value) for value in caps_ma)
        if len(caps) != 6:
            raise ValueError("commissioning current caps must contain six values")
        for index, (current, cap) in enumerate(zip(feedback.currents, caps)):
            if abs(int(current)) > cap:
                raise RH56SequenceDriverError(
                    f"{phase}: {AXIS_NAMES[index]} current {current}mA "
                    f"exceeded strict cap {cap}mA"
                )

    def _fail_and_latch(self, error: BaseException) -> None:
        self.request_stop()
        failures = []
        # Do not leave a numeric target active while the exception unwinds to a
        # higher-level finally block.  The caller still performs the full
        # multi-sample stop verification; these are immediate best-effort passes.
        for pass_index in (1, 2):
            try:
                self._write_disable_pass()
            except BaseException as stop_exc:
                failures.append(f"emergency disable pass {pass_index}: {stop_exc}")
        self.last_emergency_disable_errors = tuple(failures)

    def adopt_disabled_state_and_verify(self) -> None:
        """Disable an already-connected hand before reading/changing settings.

        This is the first hardware action for an installed-hand session.  It
        prevents stale numeric targets from a prior process from continuing
        while speed/force registers are inspected or changed.
        """

        with self._operation_scope():
            try:
                self._ensure_motion_allowed()
                for _pass_index in (1, 2):
                    self._write_disable_pass()
                self._verify_disabled_feedback(phase="adopt_disable_verify")
                self._preshaped_q6 = None
                self._commissioned_q6_waypoints = None
                self._commissioned_bend_waypoints = None
                self._numeric_hold_targets = None
                self._numeric_hold_current_caps = None
                self._disabled_verified = True
                self.last_contact_axes = ()
            except BaseException as exc:
                self._fail_and_latch(exc)
                raise RH56StopUnconfirmed(
                    f"initial all-six disable/idle adoption failed: {exc}"
                ) from exc

    def open_and_verify(self, targets: Sequence[int]) -> None:
        requested = _six_integral(targets, "open targets", allow_disabled=False)
        if requested != OPEN_TARGETS:
            raise ValueError("open_and_verify requires [1000, 1000, 1000, 1000, 1000, 1000]")
        with self._operation_scope():
            try:
                self._ensure_motion_allowed()
                if not self._disabled_verified:
                    self.adopt_disabled_state_and_verify()
                self._configure_commissioning_settings()
                # The existing reviewed helper owns all serial IO during this
                # blocking operation; keep numeric-write exclusion until it exits.
                self._run_reviewed_open_helper(include_thumb_rotate=True)
                # open_hand restores per-axis defaults as it progresses.  Put all
                # axes back at the conservative settings before later staging.
                self._configure_commissioning_settings()
                started = self._monotonic()
                feedback = self._read_feedback("open_verify", started)
                self._check_fault_feedback(feedback, "open verification")
                if feedback.angle_targets != DISABLED_TARGETS:
                    raise RH56SequenceDriverError(
                        "open verification requires all-six ANGLE_SET=-1; "
                        f"actual={feedback.angle_targets}"
                    )
                if any(angle < self.open_min_angle for angle in feedback.angles):
                    raise RH56SequenceDriverError(
                        "open verification requires every ANGLE_ACT >= "
                        f"{self.open_min_angle}; actual={feedback.angles}"
                    )
                if not all(status in (2, 0xFF) for status in feedback.statuses):
                    raise RH56SequenceDriverError(
                        "open verification requires idle STATUS in {2,255}; "
                        f"actual={feedback.statuses}"
                    )
                self._observe_validated_feedback(feedback, "open_verify")
                self._preshaped_q6 = None
                self._commissioned_q6_waypoints = None
                self._commissioned_bend_waypoints = None
                self._numeric_hold_targets = None
                self._numeric_hold_current_caps = None
                self._disabled_verified = True
                self.last_contact_axes = ()
            except BaseException as exc:
                self._fail_and_latch(exc)
                raise

    def _validate_thumb_target(self, target_q6: Any) -> int:
        value = _six_integral(
            (0, 0, 0, 0, 0, target_q6),
            "thumb target",
            allow_disabled=False,
        )[5]
        lower, upper = self.thumb_rotate_range
        if not lower <= value <= upper:
            raise ValueError(
                f"thumb target {value} is outside configured q6 range {lower}..{upper}"
            )
        return value

    def preshape_thumb(self, target_q6: int) -> None:
        target = self._validate_thumb_target(target_q6)
        step = self.thumb_preshape_step_units
        waypoints = list(range(1000 - step, target, -step))
        if not waypoints or waypoints[-1] != target:
            waypoints.append(target)

        # In an audited no-contact run, reject any plan/driver disagreement
        # before the first feedback read, setting write, or numeric q6 command.
        # Waiting until bend closure would already have moved q6 over a path
        # that was not the one certified by the installed-tool sidecar.
        audited_path = self._audited_no_contact_path
        if audited_path is not None:
            audited_forward = tuple(
                item.command_targets[5]
                for item in audited_path.waypoints
                if item.phase.startswith("q6_forward_")
            )
            if target != audited_path.target[5] or tuple(waypoints) != audited_forward:
                raise RH56SequenceDriverError(
                    "thumb preshape target/path differs from bound installed audit"
                )
        with self._operation_scope():
            try:
                self._ensure_motion_allowed()
                started = self._monotonic()
                preflight = self._read_feedback("thumb_preflight", started)
                self._check_fault_feedback(preflight, "thumb preshape preflight")
                if preflight.angle_targets != DISABLED_TARGETS:
                    raise RH56SequenceDriverError(
                        "thumb preshape requires all-six ANGLE_SET=-1"
                    )
                # Waypoints below are generated from the proven 1000-side open
                # endpoint.  Requiring q6 open here prevents a stale low q6 from
                # turning the first nominal "descending" step into a large
                # unmonitored ascending jump.
                if any(angle < self.open_min_angle for angle in preflight.angles):
                    raise RH56SequenceDriverError(
                        "thumb preshape requires all six axes fully open"
                    )
                if not all(
                    status in (2, 0xFF) for status in preflight.statuses
                ):
                    raise RH56SequenceDriverError(
                        "thumb preshape requires all six axes idle"
                    )
                self._observe_validated_feedback(preflight, "thumb_preflight")

                self._configure_commissioning_settings()
                previous_endpoint = int(preflight.angles[5])
                inactive_reference = preflight.angles[:5]
                opposite_slack = max(4, self.angle_tolerance // 2)
                final_targets = None
                previous_configuration = OPEN_TARGETS
                previous_phase = "start_open"
                for waypoint_index, waypoint in enumerate(waypoints):
                    phase = f"thumb_preshape_{waypoint:04d}"
                    expected_targets = (-1, -1, -1, -1, -1, waypoint)
                    current_configuration = (1000, 1000, 1000, 1000, 1000, waypoint)
                    current_phase = "q6_forward_{:04d}".format(waypoint_index)
                    self._write_six_verified(
                        self._constant("REG_ANGLE_SET"),
                        expected_targets,
                        numeric_motion=True,
                    )
                    deadline = self._monotonic() + self.motion_timeout_s
                    stable = 0
                    while True:
                        if self._stop_requested.is_set():
                            raise RH56MotionStopped(
                                "thumb preshape interrupted by stop request"
                            )
                        feedback = self._read_feedback(phase, started)
                        self._check_fault_feedback(feedback, phase)
                        if feedback.angle_targets != expected_targets:
                            raise RH56SequenceDriverError(
                                f"{phase}: ANGLE_SET changed unexpectedly: "
                                f"{feedback.angle_targets}"
                            )
                        self._check_feedback_envelope(
                            feedback,
                            previous_configuration,
                            current_configuration,
                            phase,
                            previous_phase=previous_phase,
                            current_phase=current_phase,
                        )
                        if any(
                            angle < self.open_min_angle
                            or abs(angle - reference) > 8
                            for angle, reference in zip(
                                feedback.angles[:5], inactive_reference
                            )
                        ) or not all(
                            status in (2, 0xFF)
                            for status in feedback.statuses[:5]
                        ):
                            raise RH56SequenceDriverError(
                                f"{phase}: a bend axis moved during thumb preshape"
                            )
                        q6_status = feedback.statuses[5]
                        if q6_status == 3:
                            raise RH56SequenceDriverError(
                                f"{phase}: q6 stopped on force contact in free air"
                            )
                        if q6_status not in (0, 1, 2):
                            raise RH56SequenceDriverError(
                                f"{phase}: unsupported q6 status {q6_status}"
                            )
                        if feedback.angles[5] > previous_endpoint + opposite_slack:
                            raise RH56SequenceDriverError(
                                f"{phase}: q6 moved opposite the descending command"
                            )
                        self._observe_validated_feedback(feedback, phase)
                        reached = (
                            q6_status == 2
                            and abs(feedback.angles[5] - waypoint)
                            <= self.angle_tolerance
                        )
                        stable = stable + 1 if reached else 0
                        if stable >= 2:
                            previous_endpoint = int(feedback.angles[5])
                            previous_configuration = current_configuration
                            previous_phase = current_phase
                            final_targets = expected_targets
                            break
                        if self._monotonic() >= deadline:
                            raise RH56SequenceDriverError(
                                f"{phase}: timed out at ANGLE_ACT={feedback.angles[5]}, "
                                f"target={waypoint}, status={q6_status}"
                            )
                        self._sleep(self.poll_interval_s)
                if final_targets is None:
                    raise RH56SequenceDriverError("thumb preshape produced no waypoint")
                self._preshaped_q6 = target
                # Retain the exact monitored path so a successful no-contact
                # run can reopen over its reverse.  Fault paths never consume
                # this path automatically; they disable immediately instead.
                self._commissioned_q6_waypoints = tuple(waypoints)
                self._commissioned_bend_waypoints = None
                self._numeric_hold_targets = final_targets
                self._numeric_hold_current_caps = None
                self._disabled_verified = False
                return
            except BaseException as exc:
                self._fail_and_latch(exc)
                raise

    def commission_thumb_sweep(
        self,
        target_q6: int,
        *,
        step_units: int = 25,
        max_axis_current_ma: int = 400,
        endpoint_stable_samples: int = 3,
        max_inactive_drift_units: int = 8,
    ) -> Tuple[int, ...]:
        """Commission q6 from the verified open state using descending steps.

        The first five axes remain disabled and fully open.  Every waypoint is
        written as one all-six batch, read back exactly, and monitored until a
        stable, idle endpoint is observed.  The caller must still invoke
        :meth:`open_and_verify` on success and :meth:`disable_and_verify` on all
        exits; :meth:`close` provides the latter guarantee as a final fallback.
        """

        target = self._validate_thumb_target(target_q6)
        # q6=1000 is a legitimate official candidate value.  Its single
        # numeric 1000 waypoint proves the exact endpoint while producing no
        # requested rotation; coupled bend-axis commissioning may then proceed
        # without inventing 999 as a substitute target.
        if isinstance(step_units, (bool, np.bool_)) or not 10 <= int(step_units) <= 50:
            raise ValueError("q6 step_units must be an integer in 10..50")
        if int(step_units) != float(step_units):
            raise ValueError("q6 step_units must be an integer in 10..50")
        if (
            isinstance(endpoint_stable_samples, (bool, np.bool_))
            or not 2 <= int(endpoint_stable_samples) <= 10
        ):
            raise ValueError("endpoint_stable_samples must be an integer in 2..10")
        if (
            isinstance(max_inactive_drift_units, (bool, np.bool_))
            or not 1 <= int(max_inactive_drift_units) <= 20
        ):
            raise ValueError("max_inactive_drift_units must be an integer in 1..20")

        step = int(step_units)
        stable_required = int(endpoint_stable_samples)
        inactive_drift = int(max_inactive_drift_units)
        waypoints = list(range(1000 - step, target, -step))
        if not waypoints or waypoints[-1] != target:
            waypoints.append(target)

        with self._operation_scope():
            try:
                self._ensure_motion_allowed()
                if not self._disabled_verified:
                    raise RH56SequenceDriverError(
                        "q6 sweep requires a successful open_and_verify first"
                    )
                started = self._monotonic()
                preflight = self._read_feedback("q6_sweep_preflight", started)
                self._check_fault_feedback(preflight, "q6 sweep preflight")
                if preflight.angle_targets != DISABLED_TARGETS:
                    raise RH56SequenceDriverError(
                        "q6 sweep requires all-six ANGLE_SET=-1"
                    )
                if any(angle < self.open_min_angle for angle in preflight.angles):
                    raise RH56SequenceDriverError(
                        "q6 sweep requires all six axes fully open"
                    )
                if not all(status in (2, 0xFF) for status in preflight.statuses):
                    raise RH56SequenceDriverError(
                        "q6 sweep requires every actuator idle before its first step"
                    )

                caps = self._commissioning_current_caps(max_axis_current_ma)
                self._check_commissioning_currents(
                    preflight, caps, "q6 sweep preflight"
                )
                self._observe_validated_feedback(preflight, "q6_sweep_preflight")
                self._configure_commissioning_settings()
                inactive_reference = preflight.angles[:5]
                previous_endpoint = preflight.angles[5]
                previous_command = 1000
                previous_configuration = OPEN_TARGETS
                previous_phase = "start_open"

                for waypoint_index, waypoint in enumerate(waypoints):
                    phase = f"q6_step_{waypoint:04d}"
                    expected_targets = (-1, -1, -1, -1, -1, waypoint)
                    current_configuration = (1000, 1000, 1000, 1000, 1000, waypoint)
                    current_phase = "q6_forward_{:04d}".format(waypoint_index)
                    self._write_six_verified(
                        self._constant("REG_ANGLE_SET"),
                        expected_targets,
                        numeric_motion=True,
                    )
                    step_started = self._monotonic()
                    deadline = step_started + self.motion_timeout_s
                    stable = 0
                    commanded_delta = previous_command - waypoint
                    commissioning_tolerance = min(self.angle_tolerance, 20)
                    first_already_in_band = (
                        waypoint_index == 0
                        and len(waypoints) > 1
                        and abs(previous_endpoint - waypoint)
                        <= commissioning_tolerance
                    )
                    required_progress = (
                        0
                        if first_already_in_band or commanded_delta < 10
                        else max(3, commanded_delta // 2)
                    )
                    while True:
                        if self._stop_requested.is_set():
                            raise RH56MotionStopped(
                                f"q6 sweep interrupted at target {waypoint}"
                            )
                        feedback = self._read_feedback(phase, step_started)
                        self._check_fault_feedback(feedback, phase)
                        self._check_commissioning_currents(feedback, caps, phase)
                        if feedback.angle_targets != expected_targets:
                            raise RH56SequenceDriverError(
                                f"{phase}: ANGLE_SET changed unexpectedly: "
                                f"{feedback.angle_targets}"
                            )
                        self._check_feedback_envelope(
                            feedback,
                            previous_configuration,
                            current_configuration,
                            phase,
                            previous_phase=previous_phase,
                            current_phase=current_phase,
                        )
                        for index, (reference, actual) in enumerate(
                            zip(inactive_reference, feedback.angles[:5])
                        ):
                            if (
                                actual < self.open_min_angle
                                or abs(actual - reference) > inactive_drift
                                or feedback.statuses[index] not in (2, 0xFF)
                            ):
                                raise RH56SequenceDriverError(
                                    f"{phase}: inactive {AXIS_NAMES[index]} moved "
                                    "or left idle state"
                                )
                        status = feedback.statuses[5]
                        if status == 3:
                            raise RH56SequenceDriverError(
                                f"{phase}: q6 reported force contact in free air"
                            )
                        if status not in (0, 1, 2):
                            raise RH56SequenceDriverError(
                                f"{phase}: q6 returned unsupported status {status}"
                            )
                        opposite_slack = max(4, self.angle_tolerance // 2)
                        if feedback.angles[5] > previous_endpoint + opposite_slack:
                            raise RH56SequenceDriverError(
                                f"{phase}: q6 moved opposite the descending command; "
                                f"start={previous_endpoint}, actual={feedback.angles[5]}"
                            )
                        self._observe_validated_feedback(feedback, phase)
                        reached = (
                            status == 2
                            and abs(feedback.angles[5] - waypoint)
                            <= commissioning_tolerance
                            and previous_endpoint - feedback.angles[5]
                            >= required_progress
                        )
                        stable = stable + 1 if reached else 0
                        if stable >= stable_required:
                            previous_endpoint = feedback.angles[5]
                            previous_command = waypoint
                            previous_configuration = current_configuration
                            previous_phase = current_phase
                            break
                        if self._monotonic() >= deadline:
                            raise RH56SequenceDriverError(
                                f"{phase}: timed out; ANGLE_ACT={feedback.angles[5]}, "
                                f"STATUS={status}, target={waypoint}"
                            )
                        self._sleep(self.poll_interval_s)

                self._preshaped_q6 = target
                self._commissioned_q6_waypoints = tuple(waypoints)
                self._numeric_hold_targets = (-1, -1, -1, -1, -1, target)
                self._numeric_hold_current_caps = None
                self._disabled_verified = False
                return tuple(waypoints)
            except BaseException as exc:
                self._fail_and_latch(exc)
                raise

    def commission_coupled_air_close(
        self,
        bend_targets: Sequence[int],
        *,
        max_axis_current_ma: int = 400,
        endpoint_stable_samples: int = 3,
        minimum_bend_target: int = 800,
        maximum_bend_target: int = 950,
        step_units: int = 25,
    ) -> Tuple[int, ...]:
        """Partially close all five bend axes at the commissioned q6 in air.

        Contact status 3 is an error for this no-contact test.  All six axes
        must reach an idle status-2 endpoint inside the angle tolerance.
        """

        if self._preshaped_q6 is None:
            raise RH56SequenceDriverError(
                "coupled air close requires a completed q6 commissioning sweep"
            )
        try:
            bend_values = tuple(bend_targets)
        except TypeError as exc:
            raise ValueError("bend_targets must contain exactly five values") from exc
        if len(bend_values) != 5:
            raise ValueError("bend_targets must contain exactly five values")
        bends = _six_integral(
            bend_values + (self._preshaped_q6,),
            "coupled air-close targets",
            allow_disabled=False,
        )
        if not 0 <= int(minimum_bend_target) < int(maximum_bend_target) <= 1000:
            raise ValueError("commissioning bend bounds must satisfy 0<=min<max<=1000")
        if any(
            value < int(minimum_bend_target)
            or value > int(maximum_bend_target)
            for value in bends[:5]
        ):
            raise ValueError(
                "every bend target must stay inside the conservative air-close "
                f"range {int(minimum_bend_target)}..{int(maximum_bend_target)}"
            )
        if (
            isinstance(endpoint_stable_samples, (bool, np.bool_))
            or not 2 <= int(endpoint_stable_samples) <= 10
        ):
            raise ValueError("endpoint_stable_samples must be an integer in 2..10")
        if (
            isinstance(step_units, (bool, np.bool_))
            or int(step_units) != float(step_units)
            or not 10 <= int(step_units) <= 50
        ):
            raise ValueError("coupled step_units must be an integer in 10..50")
        step = int(step_units)
        bend_waypoints = []
        current = [1000] * 5
        # Deterministic one-axis-at-a-time path.  It is slower than a batch
        # interpolation, but removes multi-axis asynchronous motion from both
        # commissioning and formal execution and gives collision replay an
        # exact six-register waypoint sequence.
        for axis, target_value in enumerate(bends[:5]):
            while current[axis] != target_value:
                current[axis] = max(target_value, current[axis] - step)
                bend_waypoints.append(
                    tuple(current) + (int(self._preshaped_q6),)
                )

        with self._operation_scope():
            try:
                self._ensure_motion_allowed()
                started = self._monotonic()
                preflight = self._read_feedback(
                    "coupled_air_close_preflight", started
                )
                self._check_fault_feedback(preflight, "coupled air-close preflight")
                expected_preshape = (
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    self._preshaped_q6,
                )
                if preflight.angle_targets != expected_preshape:
                    raise RH56SequenceDriverError(
                        "coupled air close requires the verified q6 target active"
                    )
                if any(angle < self.open_min_angle for angle in preflight.angles[:5]):
                    raise RH56SequenceDriverError(
                        "coupled air close requires all five bend axes fully open"
                    )
                if (
                    abs(preflight.angles[5] - self._preshaped_q6)
                    > self.angle_tolerance
                ):
                    raise RH56SequenceDriverError(
                        "coupled air close requires q6 at its commissioned target"
                    )
                caps = self._commissioning_current_caps(max_axis_current_ma)
                self._check_commissioning_currents(
                    preflight, caps, "coupled air-close preflight"
                )
                self._observe_validated_feedback(
                    preflight, "coupled_air_close_preflight"
                )
                self._configure_commissioning_settings()
                previous_actual = tuple(int(value) for value in preflight.angles[:5])
                previous_configuration = (
                    1000,
                    1000,
                    1000,
                    1000,
                    1000,
                    int(self._preshaped_q6),
                )
                previous_phase = "q6_forward_complete"
                for waypoint_index, waypoint in enumerate(bend_waypoints):
                    phase = f"coupled_air_close_step_{waypoint_index:04d}"
                    current_phase = "bend_forward_{:04d}".format(waypoint_index)
                    self._write_six_verified(
                        self._constant("REG_ANGLE_SET"), waypoint, numeric_motion=True
                    )
                    deadline = self._monotonic() + self.motion_timeout_s
                    stable = 0
                    while True:
                        if self._stop_requested.is_set():
                            raise RH56MotionStopped(
                                "coupled air close interrupted by stop request"
                            )
                        feedback = self._read_feedback(phase, started)
                        self._check_fault_feedback(feedback, phase)
                        self._check_commissioning_currents(feedback, caps, phase)
                        if feedback.angle_targets != waypoint:
                            raise RH56SequenceDriverError(
                                f"{phase}: ANGLE_SET changed unexpectedly: {feedback.angle_targets}"
                            )
                        self._check_feedback_envelope(
                            feedback,
                            previous_configuration,
                            waypoint,
                            phase,
                            previous_phase=previous_phase,
                            current_phase=current_phase,
                        )
                        contact_axes = tuple(
                            AXIS_NAMES[index]
                            for index, status in enumerate(feedback.statuses)
                            if status == 3
                        )
                        if contact_axes:
                            raise RH56SequenceDriverError(
                                f"{phase}: no-contact free-air motion reported force "
                                f"contact on {', '.join(contact_axes)}"
                            )
                        if any(status not in (0, 1, 2) for status in feedback.statuses):
                            raise RH56SequenceDriverError(
                                f"{phase}: unsupported status {feedback.statuses}"
                            )
                        if any(
                            actual > previous + max(4, self.angle_tolerance // 2)
                            for actual, previous in zip(
                                feedback.angles[:5], previous_actual
                            )
                        ):
                            raise RH56SequenceDriverError(
                                f"{phase}: a bend axis moved opposite the closing command"
                            )
                        if abs(feedback.angles[5] - self._preshaped_q6) > self.angle_tolerance:
                            raise RH56SequenceDriverError(f"{phase}: q6 drifted during bend closure")
                        self._observe_validated_feedback(feedback, phase)
                        reached = all(
                            status == 2 and abs(actual - target) <= self.angle_tolerance
                            for actual, target, status in zip(
                                feedback.angles, waypoint, feedback.statuses
                            )
                        )
                        stable = stable + 1 if reached else 0
                        if stable >= int(endpoint_stable_samples):
                            previous_actual = tuple(int(value) for value in feedback.angles[:5])
                            previous_configuration = tuple(int(value) for value in waypoint)
                            previous_phase = current_phase
                            break
                        if self._monotonic() >= deadline:
                            raise RH56SequenceDriverError(
                                f"{phase}: timed out at ANGLE_ACT={feedback.angles}, target={waypoint}"
                            )
                        self._sleep(self.poll_interval_s)
                self._commissioned_bend_waypoints = tuple(bend_waypoints)
                self._numeric_hold_targets = bends
                self._numeric_hold_current_caps = tuple(caps)
                self._disabled_verified = False
                return bends
            except BaseException as exc:
                self._fail_and_latch(exc)
                raise

    def return_commissioned_thumb_to_open(
        self,
        *,
        max_axis_current_ma: int = 400,
        endpoint_stable_samples: int = 3,
        max_inactive_drift_units: int = 8,
        direct_q6_return_to_open: bool = False,
    ) -> Tuple[int, ...]:
        """Reopen bends, then return q6 over the audited bootstrap path.

        The reviewed generic q6-open helper intentionally refuses a starting
        angle below 885.  A wide-range commissioning run therefore returns over
        the already-observed forward waypoints instead of weakening that gate or
        issuing one direct command from the commissioned target to 1000.
        """

        forward = self._commissioned_q6_waypoints
        if self._preshaped_q6 is None or not forward:
            raise RH56SequenceDriverError(
                "q6 return requires a completed commissioning sweep"
            )
        if forward[-1] != self._preshaped_q6:
            raise RH56SequenceDriverError("stored q6 sweep endpoint is inconsistent")
        if (
            isinstance(endpoint_stable_samples, (bool, np.bool_))
            or int(endpoint_stable_samples) != float(endpoint_stable_samples)
            or not 2 <= int(endpoint_stable_samples) <= 10
        ):
            raise ValueError("endpoint_stable_samples must be an integer in 2..10")
        if (
            isinstance(max_inactive_drift_units, (bool, np.bool_))
            or int(max_inactive_drift_units) != float(max_inactive_drift_units)
            or not 1 <= int(max_inactive_drift_units) <= 20
        ):
            raise ValueError("max_inactive_drift_units must be an integer in 1..20")
        target = int(self._preshaped_q6)
        numeric_targets = self._numeric_hold_targets
        if numeric_targets is None or any(value < 0 for value in numeric_targets[:5]):
            numeric_targets = (1000, 1000, 1000, 1000, 1000, target)
        canonical_return = build_rh56_no_contact_execution_path(
            numeric_targets, step_units=self.thumb_preshape_step_units
        )
        return_waypoints = tuple(
            item.command_targets[5]
            for item in canonical_return.waypoints
            if item.phase.startswith("q6_reverse_")
        )
        if direct_q6_return_to_open:
            return_waypoints = (1000,)
        stable_required = int(endpoint_stable_samples)
        inactive_drift = int(max_inactive_drift_units)

        with self._operation_scope():
            try:
                self._ensure_motion_allowed()
                # Remove every numeric hold before changing any speed/force
                # setting or asking the reviewed helper to reopen bend axes.
                for _pass_index in (1, 2):
                    self._write_disable_pass()
                self._verify_disabled_feedback(phase="q6_return_adopt_verify")
                self._configure_commissioning_settings()
                if self._commissioned_bend_waypoints:
                    caps = self._commissioning_current_caps(max_axis_current_ma)
                    reverse_bends = tuple(
                        item.command_targets
                        for item in canonical_return.waypoints
                        if item.phase.startswith("bend_reverse_")
                    )
                    previous_actual = None
                    previous_configuration = tuple(
                        int(value) for value in numeric_targets
                    )
                    previous_phase = "bend_forward_complete"
                    for waypoint_index, waypoint in enumerate(reverse_bends):
                        phase = f"coupled_air_return_step_{waypoint_index:04d}"
                        current_phase = "bend_reverse_{:04d}".format(
                            waypoint_index
                        )
                        self._write_six_verified(
                            self._constant("REG_ANGLE_SET"), waypoint, numeric_motion=True
                        )
                        deadline = self._monotonic() + self.motion_timeout_s
                        stable = 0
                        while True:
                            feedback = self._read_feedback(phase, self._monotonic())
                            self._check_fault_feedback(feedback, phase)
                            self._check_commissioning_currents(feedback, caps, phase)
                            if feedback.angle_targets != waypoint:
                                raise RH56SequenceDriverError(f"{phase}: target readback changed")
                            self._check_feedback_envelope(
                                feedback,
                                previous_configuration,
                                waypoint,
                                phase,
                                previous_phase=previous_phase,
                                current_phase=current_phase,
                            )
                            if any(status == 3 for status in feedback.statuses):
                                raise RH56SequenceDriverError(f"{phase}: contact during free-air reopen")
                            if any(status not in (0, 1, 2) for status in feedback.statuses):
                                raise RH56SequenceDriverError(f"{phase}: unsupported status")
                            if abs(feedback.angles[5] - target) > self.angle_tolerance:
                                raise RH56SequenceDriverError(f"{phase}: q6 drifted during bend reopen")
                            if previous_actual is not None and any(
                                actual < previous - max(4, self.angle_tolerance // 2)
                                for actual, previous in zip(feedback.angles[:5], previous_actual)
                            ):
                                raise RH56SequenceDriverError(f"{phase}: bend moved opposite reopen")
                            self._observe_validated_feedback(feedback, phase)
                            reached = all(
                                status == 2 and abs(actual - wanted) <= self.angle_tolerance
                                for actual, wanted, status in zip(
                                    feedback.angles, waypoint, feedback.statuses
                                )
                            )
                            stable = stable + 1 if reached else 0
                            if stable >= int(endpoint_stable_samples):
                                previous_actual = tuple(int(value) for value in feedback.angles[:5])
                                previous_configuration = tuple(
                                    int(value) for value in waypoint
                                )
                                previous_phase = current_phase
                                break
                            if self._monotonic() >= deadline:
                                raise RH56SequenceDriverError(f"{phase}: timed out")
                            self._sleep(self.poll_interval_s)
                    for _pass_index in (1, 2):
                        self._write_disable_pass()
                    self._verify_disabled_feedback(phase="bend_return_disable_verify")
                else:
                    self._run_reviewed_open_helper(include_thumb_rotate=False)
                    self._configure_commissioning_settings()

                started = self._monotonic()
                preflight = self._read_feedback("q6_return_preflight", started)
                self._check_fault_feedback(preflight, "q6 return preflight")
                if preflight.angle_targets != DISABLED_TARGETS:
                    raise RH56SequenceDriverError(
                        "q6 return requires all-six ANGLE_SET=-1 after bend reopen"
                    )
                if any(angle < self.open_min_angle for angle in preflight.angles[:5]):
                    raise RH56SequenceDriverError(
                        "q6 return requires all five bend axes fully open"
                    )
                if not all(status in (2, 0xFF) for status in preflight.statuses):
                    raise RH56SequenceDriverError(
                        "q6 return requires every actuator idle before its first step"
                    )
                commissioning_tolerance = min(self.angle_tolerance, 20)
                if abs(int(preflight.angles[5]) - target) > commissioning_tolerance:
                    raise RH56SequenceDriverError(
                        "q6 drifted away from its commissioned endpoint while bends opened"
                    )
                caps = self._commissioning_current_caps(max_axis_current_ma)
                self._check_commissioning_currents(
                    preflight, caps, "q6 return preflight"
                )
                self._observe_validated_feedback(preflight, "q6_return_preflight")
                inactive_reference = preflight.angles[:5]
                previous_endpoint = int(preflight.angles[5])
                previous_command = target
                previous_configuration = (
                    1000,
                    1000,
                    1000,
                    1000,
                    1000,
                    target,
                )
                previous_phase = "bend_reverse_complete"

                for waypoint_index, waypoint in enumerate(return_waypoints):
                    phase = f"q6_return_{waypoint:04d}"
                    expected_targets = (-1, -1, -1, -1, -1, waypoint)
                    current_configuration = (
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        waypoint,
                    )
                    current_phase = "q6_reverse_{:04d}".format(waypoint_index)
                    self._write_six_verified(
                        self._constant("REG_ANGLE_SET"),
                        expected_targets,
                        numeric_motion=True,
                    )
                    step_started = self._monotonic()
                    deadline = step_started + self.motion_timeout_s
                    stable = 0
                    commanded_delta = waypoint - previous_command
                    required_progress = (
                        0
                        if waypoint == 1000 or commanded_delta < 10
                        else max(
                            3,
                            commanded_delta
                            - Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS,
                        )
                    )
                    opposite_slack = max(4, self.angle_tolerance // 2)
                    while True:
                        if self._stop_requested.is_set():
                            raise RH56MotionStopped(
                                f"q6 return interrupted at target {waypoint}"
                            )
                        feedback = self._read_feedback(phase, step_started)
                        self._check_fault_feedback(feedback, phase)
                        self._check_commissioning_currents(feedback, caps, phase)
                        if feedback.angle_targets != expected_targets:
                            raise RH56SequenceDriverError(
                                f"{phase}: ANGLE_SET changed unexpectedly: "
                                f"{feedback.angle_targets}"
                            )
                        self._check_feedback_envelope(
                            feedback,
                            previous_configuration,
                            current_configuration,
                            phase,
                            previous_phase=previous_phase,
                            current_phase=current_phase,
                        )
                        for index, (reference, actual) in enumerate(
                            zip(inactive_reference, feedback.angles[:5])
                        ):
                            if (
                                actual < self.open_min_angle
                                or abs(actual - reference) > inactive_drift
                                or feedback.statuses[index] not in (2, 0xFF)
                            ):
                                raise RH56SequenceDriverError(
                                    f"{phase}: inactive {AXIS_NAMES[index]} moved "
                                    "or left idle state"
                                )
                        status = feedback.statuses[5]
                        if status == 3:
                            raise RH56SequenceDriverError(
                                f"{phase}: q6 reported force contact in free air"
                            )
                        if status not in (0, 1, 2):
                            raise RH56SequenceDriverError(
                                f"{phase}: q6 returned unsupported status {status}"
                            )
                        if feedback.angles[5] < previous_endpoint - opposite_slack:
                            raise RH56SequenceDriverError(
                                f"{phase}: q6 moved opposite the ascending command"
                            )
                        endpoint_error_ok = (
                            feedback.angles[5] >= self.open_min_angle
                            if waypoint == 1000
                            else abs(feedback.angles[5] - waypoint)
                            <= Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS
                        )
                        stopped_current_ok = all(
                            abs(int(current)) <= self.stop_max_axis_current_ma
                            for current in feedback.currents
                        )
                        self._observe_validated_feedback(feedback, phase)
                        reached = (
                            status == 2
                            and endpoint_error_ok
                            and stopped_current_ok
                            and feedback.angles[5] - previous_endpoint
                            >= required_progress
                        )
                        stable = stable + 1 if reached else 0
                        if stable >= stable_required:
                            previous_endpoint = int(feedback.angles[5])
                            previous_command = waypoint
                            previous_configuration = current_configuration
                            previous_phase = current_phase
                            break
                        if self._monotonic() >= deadline:
                            raise RH56SequenceDriverError(
                                f"{phase}: timed out; ANGLE_ACT={feedback.angles[5]}, "
                                f"STATUS={status}, target={waypoint}"
                            )
                        self._sleep(self.poll_interval_s)

                for _pass_index in (1, 2):
                    self._write_disable_pass()
                self._verify_disabled_feedback(phase="q6_return_disable_verify")
                final_feedback = self._read_feedback(
                    "q6_return_open_verify", started
                )
                self._check_fault_feedback(final_feedback, "q6 return open verification")
                if final_feedback.angle_targets != DISABLED_TARGETS or any(
                    angle < self.open_min_angle for angle in final_feedback.angles
                ):
                    raise RH56SequenceDriverError(
                        "q6 reverse sweep did not finish six-axis open and disabled"
                    )
                if not all(
                    status in (2, 0xFF) for status in final_feedback.statuses
                ):
                    raise RH56SequenceDriverError(
                        "q6 reverse sweep final feedback is not idle"
                    )
                if any(
                    abs(int(current)) > self.stop_max_axis_current_ma
                    for current in final_feedback.currents
                ):
                    raise RH56SequenceDriverError(
                        "q6 reverse sweep final current did not return to idle"
                    )
                self._observe_validated_feedback(
                    final_feedback, "q6_return_open_verify"
                )
                self._preshaped_q6 = None
                self._commissioned_q6_waypoints = None
                self._commissioned_bend_waypoints = None
                self._numeric_hold_targets = None
                self._numeric_hold_current_caps = None
                self._disabled_verified = True
                self.last_contact_axes = ()
                return return_waypoints
            except BaseException as exc:
                self._fail_and_latch(exc)
                raise

    def close_bends_and_hold(self, targets: Sequence[int]) -> None:
        """Close a loaded grasp, allowing device force-contact completion."""

        self._close_bends_and_hold(targets, allow_contact=True)

    def verify_bounded_hold(
        self, targets: Sequence[int], *, no_contact: bool
    ) -> None:
        """Read-only proof for an ordinary or no-contact bounded hold.

        The unique RH56 owner performs exactly one normal feedback snapshot;
        no register is written and no secondary client/thread is created.  The
        sample passes the same external Franka gate, actuator fault checks,
        effective per-axis current caps, and command-path envelope used during
        the accepted close.  Ordinary grasps may remain settled (STATUS=2) or
        stopped on force contact (STATUS=3); air grasps must be settled on all
        six axes with no contact.

        Unlike ``verify_loaded_hold``, failure is allowed to propagate to the
        ordinary sequencer's coordinated stop/disable path.  Loaded-hold
        containment semantics are intentionally unchanged.
        """

        requested = _six_integral(
            targets, "bounded hold targets", allow_disabled=False
        )
        if not isinstance(no_contact, (bool, np.bool_)):
            raise ValueError("no_contact must be boolean")
        require_no_contact = bool(no_contact)
        with self._operation_scope():
            self._ensure_motion_allowed()
            if self._numeric_hold_targets != requested:
                raise RH56SequenceDriverError(
                    "bounded hold target does not match the active numeric hold: "
                    f"expected={requested}, active={self._numeric_hold_targets}"
                )
            caps = self._numeric_hold_current_caps
            if caps is None:
                raise RH56SequenceDriverError(
                    "bounded hold has no current-cap contract from its accepted close"
                )
            started = self._monotonic()
            feedback = self._read_feedback("bounded_hold_verify", started)
            self._check_fault_feedback(feedback, "bounded hold verification")
            self._check_commissioning_currents(
                feedback, caps, "bounded hold verification"
            )
            if feedback.angle_targets != requested:
                raise RH56SequenceDriverError(
                    "bounded hold ANGLE_SET changed unexpectedly: "
                    f"expected={requested}, actual={feedback.angle_targets}"
                )
            self._check_feedback_envelope(
                feedback,
                OPEN_TARGETS,
                requested,
                "bounded hold verification",
                previous_phase="start_open",
                current_phase="hold_target",
            )
            contacts = tuple(
                AXIS_NAMES[index]
                for index, status in enumerate(feedback.statuses[:5])
                if status == 3
            )
            if require_no_contact:
                if feedback.statuses != (2, 2, 2, 2, 2, 2) or contacts:
                    raise RH56SequenceDriverError(
                        "air bounded hold requires all-six STATUS=2 and zero contact: "
                        f"statuses={feedback.statuses}, contacts={contacts}"
                    )
            else:
                unsupported = tuple(
                    (AXIS_NAMES[index], status)
                    for index, status in enumerate(feedback.statuses)
                    if status not in (2, 3)
                )
                if unsupported:
                    raise RH56SequenceDriverError(
                        "bounded hold axis is neither settled nor in contact: "
                        f"{unsupported}"
                    )
            settled_drift = tuple(
                (
                    AXIS_NAMES[index],
                    int(actual),
                    int(target),
                )
                for index, (actual, target, status) in enumerate(
                    zip(feedback.angles, requested, feedback.statuses)
                )
                if status == 2 and abs(int(actual) - int(target)) > self.angle_tolerance
            )
            if settled_drift:
                raise RH56SequenceDriverError(
                    "bounded hold settled ANGLE_ACT is outside tolerance: "
                    f"axes={settled_drift}, tolerance={self.angle_tolerance}"
                )
            if abs(feedback.angles[5] - requested[5]) > self.angle_tolerance:
                raise RH56SequenceDriverError(
                    "bounded hold thumb rotation is outside target tolerance: "
                    f"actual={feedback.angles[5]}, target={requested[5]}, "
                    f"tolerance={self.angle_tolerance}"
                )
            self._observe_validated_feedback(feedback, "bounded_hold_verify")
            self.last_contact_axes = contacts

    def verify_loaded_hold(
        self, targets: Sequence[int], minimum_contact_axes: int
    ) -> None:
        """Read-only proof that the audited loaded numeric hold is still active.

        This method is intended for synchronization points between Franka
        control handles.  It never changes an RH56 register and deliberately
        does not call the normal fail-and-disable path: after lift begins, a
        failed proof must be contained by stopping Franka while leaving the
        last numeric hand target active for manual loaded recovery.
        """

        requested = _six_integral(targets, "loaded hold targets", allow_disabled=False)
        if (
            isinstance(minimum_contact_axes, (bool, np.bool_))
            or not isinstance(minimum_contact_axes, (int, np.integer))
            or not 1 <= int(minimum_contact_axes) <= 5
        ):
            raise ValueError("minimum_contact_axes must be an integer in 1..5")
        minimum_contacts = int(minimum_contact_axes)
        with self._operation_scope():
            self._ensure_motion_allowed()
            if self._numeric_hold_targets != requested:
                raise RH56SequenceDriverError(
                    "loaded hold target does not match the active numeric hold: "
                    f"expected={requested}, active={self._numeric_hold_targets}"
                )
            started = self._monotonic()
            feedback = self._read_feedback("loaded_hold_verify", started)
            self._check_fault_feedback(feedback, "loaded hold verification")
            if self._numeric_hold_current_caps is not None:
                self._check_commissioning_currents(
                    feedback,
                    self._numeric_hold_current_caps,
                    "loaded hold verification",
                )
            if feedback.angle_targets != requested:
                raise RH56SequenceDriverError(
                    "loaded hold ANGLE_SET changed unexpectedly: "
                    f"expected={requested}, actual={feedback.angle_targets}"
                )
            self._check_feedback_envelope(
                feedback,
                (1000, 1000, 1000, 1000, 1000, requested[5]),
                requested,
                "loaded hold verification",
                previous_phase="start_open",
                current_phase="loaded_hold_target",
            )
            unsupported = [
                (AXIS_NAMES[index], status)
                for index, status in enumerate(feedback.statuses)
                if status not in (2, 3)
            ]
            if unsupported:
                raise RH56SequenceDriverError(
                    "loaded hold is not settled; expected STATUS 2/3: "
                    f"{unsupported}"
                )
            contacts = tuple(
                AXIS_NAMES[index]
                for index, status in enumerate(feedback.statuses[:5])
                if status == 3
            )
            if len(contacts) < minimum_contacts:
                raise RH56SequenceDriverError(
                    "loaded hold contact count is below audited minimum: "
                    f"actual={len(contacts)}, required={minimum_contacts}, "
                    f"axes={contacts}"
                )
            for index, (actual, target, status) in enumerate(
                zip(feedback.angles[:5], requested[:5], feedback.statuses[:5])
            ):
                if status == 2 and abs(actual - target) > self.angle_tolerance:
                    raise RH56SequenceDriverError(
                        "loaded hold idle axis is outside target tolerance: "
                        f"axis={AXIS_NAMES[index]}, actual={actual}, target={target}"
                    )
            if abs(feedback.angles[5] - requested[5]) > self.angle_tolerance:
                raise RH56SequenceDriverError(
                    "loaded hold thumb rotation is outside target tolerance: "
                    f"actual={feedback.angles[5]}, target={requested[5]}"
                )
            self._observe_validated_feedback(feedback, "loaded_hold_verify")
            self.last_contact_axes = contacts

    def read_state_snapshot(self) -> RH56StateSnapshot:
        """Capture boundary-rate telemetry without changing any register.

        This is intentionally usable after a verified all-six disable, when
        ``stop_requested`` is latched.  It performs the same fresh external
        Franka safety checks and actuator fault checks as normal feedback, but
        it neither clears that latch nor grants permission for another motion.
        """

        with self._operation_scope():
            if self._closed:
                raise RH56MotionStopped("driver is closed")
            started = self._monotonic()
            feedback = self._read_feedback("boundary_state_snapshot", started)
            self._check_fault_feedback(feedback, "boundary state snapshot")
            contacts = tuple(
                AXIS_NAMES[index]
                for index, status in enumerate(feedback.statuses[:5])
                if status == 3
            )
            self._observe_validated_feedback(feedback, "boundary_state_snapshot")
            return RH56StateSnapshot(
                angle_targets=feedback.angle_targets,
                angles=feedback.angles,
                positions=feedback.positions,
                forces=feedback.forces,
                currents=feedback.currents,
                errors=feedback.errors,
                statuses=feedback.statuses,
                temperatures=feedback.temperatures,
                contact_axes=contacts,
            )

    def close_bends_no_contact_and_hold(self, targets: Sequence[int]) -> None:
        """Close an air grasp and reject any actuator force-contact status."""
        requested = _six_integral(targets, "close targets", allow_disabled=False)
        if self._audited_no_contact_path is None:
            raise RH56SequenceDriverError(
                "no-contact execution requires a bound audited RH56 waypoint path"
            )
        actual_path = build_rh56_no_contact_execution_path(
            requested, step_units=self.thumb_preshape_step_units
        )
        if actual_path.sha256 != self._audited_no_contact_path.sha256:
            raise RH56SequenceDriverError(
                "runtime RH56 waypoint path hash differs from installed audit"
            )
        if self._preshaped_q6 is None or requested[5] != self._preshaped_q6:
            raise RH56SequenceDriverError(
                "close q6 target does not match the verified thumb preshape"
            )
        self.commission_coupled_air_close(
            requested[:5],
            minimum_bend_target=0,
            maximum_bend_target=1000,
            step_units=self.thumb_preshape_step_units,
        )

    def return_no_contact_hand_to_open(self) -> None:
        """Monitored success-only return over the exact reverse small-step path."""

        self.return_commissioned_thumb_to_open()

    def _close_bends_and_hold(
        self, targets: Sequence[int], *, allow_contact: bool
    ) -> None:
        requested = _six_integral(targets, "close targets", allow_disabled=False)
        if self._preshaped_q6 is None:
            raise RH56SequenceDriverError(
                "close refused: thumb rotation has not been successfully preshaped"
            )
        if requested[5] != self._preshaped_q6:
            raise RH56SequenceDriverError(
                "close q6 target does not match the verified thumb preshape"
            )
        with self._operation_scope():
            try:
                self._ensure_motion_allowed()
                started = self._monotonic()
                preflight = self._read_feedback("close_preflight", started)
                self._check_fault_feedback(preflight, "close preflight")
                expected_preshape = (-1, -1, -1, -1, -1, self._preshaped_q6)
                if preflight.angle_targets != expected_preshape:
                    raise RH56SequenceDriverError(
                        "close requires the verified q6 preshape target to remain active"
                    )
                if any(angle < self.open_min_angle for angle in preflight.angles[:5]):
                    raise RH56SequenceDriverError(
                        "close requires all five bend axes to start fully open"
                    )
                if (
                    abs(preflight.angles[5] - requested[5])
                    > self.angle_tolerance
                ):
                    raise RH56SequenceDriverError(
                        "close requires q6 actual feedback at its preshaped target"
                    )
                # Loaded/contact closure uses the device's real per-axis
                # CURRENT_LIMIT policy (bounded by the official 1400 mA
                # register range), never an invented aggregate-current trip.
                # Retain the exact effective caps for subsequent read-only
                # bounded-hold verification.
                caps = self._commissioning_current_caps(
                    DEVICE_MAX_AXIS_CURRENT_MA
                )
                self._check_commissioning_currents(
                    preflight, caps, "close preflight"
                )
                self._observe_validated_feedback(preflight, "close_preflight")

                self._configure_commissioning_settings()
                previous_configuration = (
                    1000,
                    1000,
                    1000,
                    1000,
                    1000,
                    int(self._preshaped_q6),
                )
                self._write_six_verified(
                    self._constant("REG_ANGLE_SET"),
                    requested,
                    numeric_motion=True,
                )
                deadline = self._monotonic() + self.motion_timeout_s
                while True:
                    if self._stop_requested.is_set():
                        raise RH56MotionStopped("bend closure interrupted by stop request")
                    feedback = self._read_feedback("close", started)
                    self._check_fault_feedback(feedback, "bend closure")
                    self._check_commissioning_currents(
                        feedback, caps, "bend closure"
                    )
                    if feedback.angle_targets != requested:
                        raise RH56SequenceDriverError(
                            "close ANGLE_SET changed unexpectedly: "
                            f"{feedback.angle_targets}"
                        )
                    self._check_feedback_envelope(
                        feedback,
                        previous_configuration,
                        requested,
                        "bend closure",
                        previous_phase="start_open",
                        current_phase="bend_close_target",
                    )
                    unsupported = [
                        (AXIS_NAMES[index], status)
                        for index, status in enumerate(feedback.statuses)
                        if status not in (0, 1, 2, 3)
                    ]
                    if unsupported:
                        raise RH56SequenceDriverError(
                            f"bend closure returned unsupported status: {unsupported}"
                        )
                    contact_axes = tuple(
                        AXIS_NAMES[index]
                        for index, status in enumerate(feedback.statuses[:5])
                        if status == 3
                    )
                    if contact_axes and not allow_contact:
                        raise RH56SequenceDriverError(
                            "no-contact air closure reported force contact on "
                            + ", ".join(contact_axes)
                        )
                    self._observe_validated_feedback(feedback, "close")
                    bend_done = tuple(
                        (allow_contact and status == 3)
                        or (
                            status == 2
                            and abs(actual - target) <= self.angle_tolerance
                        )
                        for actual, target, status in zip(
                            feedback.angles[:5],
                            requested[:5],
                            feedback.statuses[:5],
                        )
                    )
                    q6_done = (
                        abs(feedback.angles[5] - requested[5])
                        <= self.angle_tolerance
                    )
                    if all(bend_done) and q6_done:
                        self.last_contact_axes = contact_axes
                        self._numeric_hold_targets = requested
                        self._numeric_hold_current_caps = tuple(caps)
                        self._disabled_verified = False
                        return
                    if self._monotonic() >= deadline:
                        raise RH56SequenceDriverError(
                            "bend closure timed out; "
                            f"ANGLE_ACT={feedback.angles}, STATUS={feedback.statuses}, "
                            f"target={requested}"
                        )
                    self._sleep(self.poll_interval_s)
            except BaseException as exc:
                self._fail_and_latch(exc)
                raise

    def _write_disable_pass(self) -> None:
        if self._validated_feedback_observer_active:
            raise RH56SequenceDriverError(
                "validated feedback observer cannot perform RH56 register IO"
            )
        with self._motion_write_lock:
            with self._io_lock:
                self.hand.write_six_shorts(
                    self._constant("REG_ANGLE_SET"),
                    DISABLED_TARGETS,
                    retries=1,
                )
                readback = tuple(
                    int(value)
                    for value in self.hand.read_six_shorts(
                        self._constant("REG_ANGLE_SET"), retries=1
                    )
                )
        if readback != DISABLED_TARGETS:
            raise RH56SequenceDriverError(
                f"ANGLE_SET disable readback is {readback}, expected {DISABLED_TARGETS}"
            )

    def _verify_disabled_feedback(
        self,
        *,
        phase: str = "disable_verify",
        allow_external_gate_bypass_after_disabled_readback: bool = False,
    ) -> None:
        if not self._feedback_supported():
            return
        started = self._monotonic()
        deadline = started + self.stop_verify_timeout_s
        stable_samples: List[_Feedback] = []
        observed_samples = 0
        last_detail = "no post-disable feedback was captured"
        idle_statuses = (2, 0xFF)
        transient_statuses = (0, 1)

        while True:
            feedback = self._read_feedback(
                str(phase),
                started,
                allow_external_gate_bypass_after_disabled_readback=(
                    allow_external_gate_bypass_after_disabled_readback
                ),
            )
            observed_samples += 1
            if feedback.angle_targets != DISABLED_TARGETS:
                raise RH56SequenceDriverError(
                    f"post-disable ANGLE_SET changed: {feedback.angle_targets}"
                )
            if any(feedback.errors):
                raise RH56SequenceDriverError(
                    f"post-disable actuator ERROR remains: {feedback.errors}"
                )
            if max(feedback.temperatures) >= 60:
                raise RH56SequenceDriverError(
                    f"post-disable temperature is unsafe: {feedback.temperatures}"
                )
            if any(
                abs(int(current)) > self.stop_max_axis_current_ma
                for current in feedback.currents
            ):
                raise RH56SequenceDriverError(
                    "post-disable actuator current did not return to the idle "
                    f"per-axis bound {self.stop_max_axis_current_ma}mA: "
                    f"{feedback.currents}"
                )
            fault_statuses = tuple(
                (AXIS_NAMES[index], int(status))
                for index, status in enumerate(feedback.statuses)
                if status in (5, 6, 7)
            )
            if fault_statuses:
                raise RH56SequenceDriverError(
                    "post-disable actuator fault status remains: "
                    f"{fault_statuses}"
                )
            unexpected_statuses = tuple(
                (AXIS_NAMES[index], int(status))
                for index, status in enumerate(feedback.statuses)
                if status not in idle_statuses + transient_statuses
            )
            if unexpected_statuses:
                raise RH56SequenceDriverError(
                    "post-disable actuator returned a non-idle, non-transient "
                    f"status: {unexpected_statuses}"
                )

            if all(status in idle_statuses for status in feedback.statuses):
                stable_samples.append(feedback)
                if len(stable_samples) > self.stop_verify_samples:
                    del stable_samples[0]
                if len(stable_samples) == self.stop_verify_samples:
                    motion_problems = []
                    for axis_index, name in enumerate(AXIS_NAMES):
                        angles = [
                            sample.angles[axis_index]
                            for sample in stable_samples
                        ]
                        if max(angles) - min(angles) > 2:
                            motion_problems.append(
                                f"post-disable {name} ANGLE_ACT is still "
                                f"moving: {angles}"
                            )
                        if all(
                            sample.positions is not None
                            for sample in stable_samples
                        ):
                            positions = [
                                int(sample.positions[axis_index])  # type: ignore[index]
                                for sample in stable_samples
                            ]
                            if max(positions) - min(positions) > 3:
                                motion_problems.append(
                                    f"post-disable {name} POS_ACT is still "
                                    f"moving: {positions}"
                                )
                    if not motion_problems:
                        if self._monotonic() > deadline:
                            last_detail = (
                                "stable idle tail completed after the deadline"
                            )
                            break
                        # Do not expose any member of the stop sample set until
                        # the complete multi-sample idle/stability proof has
                        # passed for all six axes.
                        for sample in stable_samples:
                            self._observe_validated_feedback(sample, str(phase))
                        return
                    last_detail = "; ".join(motion_problems)
            else:
                # STATUS 0/1 means the released actuator is still reporting an
                # opening/grasping transition.  It is never accepted as idle;
                # discard the candidate tail and wait only within the bounded
                # stop-verification deadline for a fresh all-idle tail.
                stable_samples.clear()
                last_detail = (
                    "latest status remains transitional (0/1): "
                    f"{feedback.statuses}"
                )

            now = self._monotonic()
            if now >= deadline:
                break
            remaining = deadline - now
            retry_interval_s = self.stop_verify_interval_s
            if retry_interval_s == 0.0:
                # A positive floor keeps an injected monotonic/sleep pair from
                # turning a persistent transition into an unbounded busy loop.
                retry_interval_s = 0.01
            self._sleep(min(retry_interval_s, remaining))

        raise RH56SequenceDriverError(
            "post-disable feedback did not reach a stable idle tail before "
            f"the {self.stop_verify_timeout_s:.3f}s deadline: "
            f"observed={observed_samples}, "
            f"consecutive_idle={len(stable_samples)}/"
            f"{self.stop_verify_samples}, detail={last_detail}"
        )

    def disable_and_verify(self) -> None:
        """Latch stop, disable twice with readback, then verify physical idle."""

        self.request_stop()
        with self._operation_scope():
            failures = []
            for pass_index in (1, 2):
                try:
                    self._write_disable_pass()
                except BaseException as exc:
                    failures.append(f"disable pass {pass_index}: {exc}")
            try:
                self._verify_disabled_feedback(
                    allow_external_gate_bypass_after_disabled_readback=True
                )
            except BaseException as exc:
                failures.append(f"physical stop verification: {exc}")
            if not failures:
                try:
                    self._restore_original_settings()
                except BaseException as exc:
                    failures.append(f"temporary setting restore: {exc}")
            if failures:
                self._disabled_verified = False
                raise RH56StopUnconfirmed(
                    "STOP UNCONFIRMED: " + "; ".join(failures)
                )
            self._preshaped_q6 = None
            self._commissioned_q6_waypoints = None
            self._commissioned_bend_waypoints = None
            self._numeric_hold_targets = None
            self._numeric_hold_current_caps = None
            self._disabled_verified = True
            self.last_contact_axes = ()

    def close(self) -> None:
        with self._operation_scope():
            if self._closed:
                return
            failures = []
            try:
                self.disable_and_verify()
            except BaseException as exc:
                failures.append(str(exc))
            serial_context = self._serial_context
            try:
                if serial_context is not None:
                    serial_context.__exit__(None, None, None)
            except BaseException as exc:
                failures.append(f"serial close failed: {exc}")
            finally:
                self._closed = True
            if failures:
                raise RH56StopUnconfirmed(
                    "STOP UNCONFIRMED during close: " + "; ".join(failures)
                )


__all__ = [
    "AXIS_NAMES",
    "COMMISSIONING_FORCE_G",
    "COMMISSIONING_SPEED",
    "DISABLED_TARGETS",
    "OPEN_TARGETS",
    "RH56MotionStopped",
    "RH56SequenceDriver",
    "RH56SequenceDriverError",
    "RH56StopUnconfirmed",
    "RH56StateSnapshot",
    "RH56Telemetry",
    "RH56ValidatedFeedbackObservation",
    "RH56ValidatedFeedbackObserverError",
]
