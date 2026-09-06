"""Lazy pylibfranka adapter for :mod:`robot_control.franka.session`.

Importing this module imports no Franka binding, opens no socket, constructs no
``Robot`` and creates no control handle.  The only production construction path
is :class:`PylibfrankaBackendFactory.__call__`; callers pass that *callable*, not
an already-open robot, to ``FrankaPersistentSession``.  The session invokes it
only after its authorization, preflight, interlock and fresh-target boundary
has passed.

The adapter is deliberately thin.  One session thread owns one persistent
``ActiveControlBase`` and uses pylibfranka's synchronous cadence::

    ActiveControlBase.readOnce() -> validate/sample-hold -> writeOnce(...)

The policy may publish a new target at 60 Hz, while the same target sequence is
held for multiple nominal 1 kHz FCI cycles.  This module never starts the
low-rate asynchronous position API and never performs an additional Robot
state read while the active handle is running.

All motion envelopes remain owned by the exact commissioning profile.  This
module contains no velocity, acceleration, jerk, tracking, collision, payload
or joint-limit defaults and does not change robot configuration.
"""

from __future__ import annotations

import gc
import importlib
import math
import threading
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

# The persistent-session validator consumes these pylibfranka RobotState names
# directly.  Values are intentionally not copied or renamed in the 1 kHz
# adapter; this table is the reviewed boundary contract and is also useful to
# offline compatibility tests.
PYLIBFRANKA_STATE_FIELDS: Mapping[str, str] = MappingProxyType(
    {
        "q_rad": "q",
        "dq_rad_s": "dq",
        "T_base_eef_column_major": "O_T_EE",
        "T_flange_eef_column_major": "F_T_EE",
        "end_effector_mass_kg": "m_ee",
        "end_effector_com_m": "F_x_Cee",
        "end_effector_inertia_column_major_kg_m2": "I_ee",
        "external_load_mass_kg": "m_load",
        "external_load_com_m": "F_x_Cload",
        "external_load_inertia_column_major_kg_m2": "I_load",
        "total_mass_kg": "m_total",
        "robot_mode": "robot_mode",
        "current_errors": "current_errors",
        "joint_contact": "joint_contact",
        "joint_collision": "joint_collision",
        "cartesian_contact": "cartesian_contact",
        "cartesian_collision": "cartesian_collision",
        "control_command_success_rate": "control_command_success_rate",
        "robot_time": "time",
    }
)

AUDITED_PYLIBFRANKA_VERSION = "0.21.2"


class PylibfrankaBackendError(RuntimeError):
    """One fail-closed pylibfranka boundary error.

    ``operation`` is stable machine-readable context.  The original exception
    is retained as ``__cause__``; the adapter never reconnects, calls automatic
    recovery or retries a command after this exception.
    """

    def __init__(self, operation: str, detail: str) -> None:
        self.operation = str(operation)
        self.detail = str(detail)
        super().__init__(f"pylibfranka {self.operation} failed: {self.detail}")


_BACKEND_SEAL = object()


def _backend_error(operation: str, error: Exception) -> PylibfrankaBackendError:
    if isinstance(error, PylibfrankaBackendError):
        return error
    detail = str(error).strip()
    if detail:
        detail = f"{type(error).__name__}: {detail}"
    else:
        detail = type(error).__name__
    return PylibfrankaBackendError(operation, detail)


def _positive_sequence(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("command sequence must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("command sequence must be a positive integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0.0:
        raise ValueError("command sequence must be a positive integer")
    return int(numeric)


def _fill_joint_values(value: Sequence[float], output: list[float]) -> None:
    try:
        if len(value) != 7:
            raise ValueError("joint command must contain seven values")
    except TypeError as exc:
        raise ValueError("joint command must contain seven values") from exc
    for index in range(7):
        try:
            numeric = float(value[index])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("joint command must contain seven finite values") from exc
        if not math.isfinite(numeric):
            raise ValueError("joint command must contain seven finite values")
        output[index] = numeric


def _duration_seconds(period: object) -> float:
    converter = getattr(period, "to_sec", None)
    try:
        value = converter() if callable(converter) else float(period)
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("active-control period is not numeric") from exc
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("active-control period must be finite and non-negative")
    return seconds


def _is_native_joint_command_type(command_type: object) -> bool:
    """Reuse the command object only for the reviewed native pybind type."""

    return getattr(command_type, "__name__", "") == "JointPositions" and str(
        getattr(command_type, "__module__", "")
    ).startswith("pylibfranka")


class _PylibfrankaJointPositionControl:
    """Single-thread wrapper around one pylibfranka active control handle."""

    def __init__(
        self,
        handle: Any,
        pylibfranka: Any,
        owner_ident: int,
        observe_robot_time: Callable[[Any], None],
    ) -> None:
        if handle is None:
            raise ValueError("active control handle is required")
        self._handle = handle
        self._pylibfranka = pylibfranka
        self._owner_ident = int(owner_ident)
        self._observe_robot_time = observe_robot_time
        self._awaiting_write = False
        self._released = False
        self._finish_succeeded = False
        self._last_sequence: Optional[int] = None
        self._command_values = [0.0] * 7
        self._reusable_command: Optional[Any] = None

    @property
    def finish_succeeded(self) -> bool:
        return self._finish_succeeded

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_ident:
            raise PylibfrankaBackendError(
                "thread_ownership",
                "the active control handle was accessed by a non-owner thread",
            )

    def _require_open(self) -> None:
        self._require_owner()
        if self._released or self._handle is None:
            raise PylibfrankaBackendError("control_lifecycle", "control is released")
        if self._finish_succeeded:
            raise PylibfrankaBackendError(
                "control_lifecycle", "motion-finished command was already accepted"
            )

    def read_once(self) -> Tuple[Any, float]:
        """Perform the unique active-handle read for one FCI cycle."""

        self._require_open()
        if self._awaiting_write:
            raise PylibfrankaBackendError(
                "active_read",
                "a second read was attempted before responding to the prior read",
            )
        try:
            result = self._handle.readOnce()
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("readOnce() must return (RobotState, Duration)")
            state, period = result
            if state is None:
                raise TypeError("readOnce() returned no RobotState")
            seconds = _duration_seconds(period)
            self._observe_robot_time(state)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise _backend_error("active_read", exc) from exc
        self._awaiting_write = True
        return state, seconds

    def _make_command(self, *, motion_finished: bool) -> Any:
        command_type = getattr(self._pylibfranka, "JointPositions", None)
        if not callable(command_type):
            raise TypeError("pylibfranka.JointPositions is unavailable")
        if self._reusable_command is None:
            command = command_type(self._command_values)
            if _is_native_joint_command_type(command_type):
                self._reusable_command = command
        else:
            command = self._reusable_command
            command.q = self._command_values
        command.motion_finished = bool(motion_finished)
        return command

    def write_once(self, q_rad: Sequence[float], *, sequence: int) -> None:
        """Respond once to the preceding read with the held joint target."""

        self._require_open()
        if not self._awaiting_write:
            raise PylibfrankaBackendError(
                "active_write", "writeOnce() has no preceding active-handle read"
            )
        try:
            active_sequence = _positive_sequence(sequence)
            if self._last_sequence is not None:
                if active_sequence < self._last_sequence:
                    raise ValueError("command sequence regressed")
                if active_sequence > self._last_sequence + 1:
                    raise ValueError("command sequence skipped")
            _fill_joint_values(q_rad, self._command_values)
            command = self._make_command(motion_finished=False)
            self._handle.writeOnce(command)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise _backend_error("active_write", exc) from exc
        self._last_sequence = active_sequence
        self._awaiting_write = False

    def write_bootstrap_hold(self, q_rad: Sequence[float]) -> None:
        """Respond once with the first measured-q hold, before sequence one.

        This is deliberately a separate operation from :meth:`write_once`:
        the command has the exact same non-terminal ``JointPositions`` wire
        representation, but it does not consume or advance a policy command
        sequence.  The persistent session exposes this operation only at its
        explicitly authorized C2 bootstrap boundary.
        """

        self._require_open()
        if not self._awaiting_write:
            raise PylibfrankaBackendError(
                "active_write",
                "bootstrap writeOnce() has no preceding active-handle read",
            )
        if self._last_sequence is not None:
            raise PylibfrankaBackendError(
                "active_write",
                "measured-q bootstrap is forbidden after policy sequence one",
            )
        try:
            _fill_joint_values(q_rad, self._command_values)
            command = self._make_command(motion_finished=False)
            self._handle.writeOnce(command)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise _backend_error("active_write", exc) from exc
        self._awaiting_write = False

    def finish(self, q_rad: Sequence[float]) -> None:
        """Send exactly one final hold with ``motion_finished=True``.

        libfranka permits this terminal write after the last ordinary write;
        it also serves as the response if a safety fault happened after an
        active read but before that cycle's ordinary write.  ``Robot.stop()``
        is still requested unconditionally by the backend/session cleanup.
        """

        self._require_open()
        try:
            _fill_joint_values(q_rad, self._command_values)
            command = self._make_command(motion_finished=True)
            self._handle.writeOnce(command)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise _backend_error("motion_finish", exc) from exc
        self._finish_succeeded = True
        self._awaiting_write = False

    def release(self) -> None:
        """Drop Python references after stop verification; send no command."""

        self._require_owner()
        self._released = True
        self._handle = None
        self._pylibfranka = None
        self._reusable_command = None
        self._observe_robot_time = None


class PylibfrankaPersistentBackend:
    """Concrete backend made only by :class:`PylibfrankaBackendFactory`."""

    def __init__(
        self,
        *,
        _seal: object,
        robot: Any,
        pylibfranka: Any,
        owner_ident: int,
    ) -> None:
        if _seal is not _BACKEND_SEAL:
            raise TypeError(
                "PylibfrankaPersistentBackend must be made by its delayed factory"
            )
        if robot is None or pylibfranka is None:
            raise ValueError("robot and pylibfranka module are required")
        self._robot = robot
        self._pylibfranka = pylibfranka
        self._owner_ident = int(owner_ident)
        self._control: Optional[_PylibfrankaJointPositionControl] = None
        self._start_attempted = False
        self._stop_attempted = False
        self._closed = False
        self._last_robot_time_s: Optional[float] = None
        self._gc_was_enabled: Optional[bool] = None

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_ident:
            raise PylibfrankaBackendError(
                "thread_ownership", "backend was accessed by a non-owner thread"
            )

    def _require_open(self) -> None:
        self._require_owner()
        if self._closed or self._robot is None:
            raise PylibfrankaBackendError("backend_lifecycle", "backend is closed")

    def _observe_robot_time(self, state: Any) -> None:
        raw_time = getattr(state, "time", None)
        if raw_time is None:
            raise TypeError("pylibfranka RobotState has no time")
        robot_time_s = _duration_seconds(raw_time)
        if (
            self._last_robot_time_s is not None
            and robot_time_s <= self._last_robot_time_s
        ):
            raise ValueError(
                "pylibfranka RobotState.time did not increase strictly: "
                f"previous={self._last_robot_time_s:.9f}s, "
                f"actual={robot_time_s:.9f}s"
            )
        self._last_robot_time_s = robot_time_s

    def start_joint_position_session(self) -> _PylibfrankaJointPositionControl:
        """Create the sole persistent joint-position control handle."""

        self._require_open()
        if self._start_attempted:
            raise PylibfrankaBackendError(
                "start_control", "a control-handle start was already attempted"
            )
        if self._stop_attempted:
            raise PylibfrankaBackendError(
                "start_control", "control cannot start after a stop attempt"
            )
        self._start_attempted = True
        self._gc_was_enabled = gc.isenabled()
        if self._gc_was_enabled:
            # Match the reviewed sequence driver: collect before the handle,
            # then prevent an unrelated cyclic-GC pause inside the hard
            # readOnce -> writeOnce interval. Reference counting remains live.
            gc.collect()
            gc.disable()
        try:
            controller_mode = self._pylibfranka.ControllerMode.JointImpedance
            handle = self._robot.start_joint_position_control(controller_mode)
            if handle is None:
                raise TypeError("start_joint_position_control returned no handle")
            control = _PylibfrankaJointPositionControl(
                handle,
                self._pylibfranka,
                self._owner_ident,
                self._observe_robot_time,
            )
        except (KeyboardInterrupt, SystemExit):
            if self._gc_was_enabled:
                gc.enable()
            self._gc_was_enabled = None
            raise
        except Exception as exc:
            if self._gc_was_enabled:
                gc.enable()
            self._gc_was_enabled = None
            raise _backend_error("start_control", exc) from exc
        self._control = control
        return control

    def request_stop(self) -> None:
        """Call ``Robot.stop()`` once, on every normal or fault cleanup path."""

        self._require_open()
        if self._stop_attempted:
            raise PylibfrankaBackendError(
                "robot_stop", "Robot.stop() was already attempted"
            )
        # Mark before the call: even a throwing Robot.stop() is an attempted
        # stop, and post-stop reads remain available so the session can gather
        # direct evidence instead of treating an exception as physical state.
        self._stop_attempted = True
        try:
            self._robot.stop()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise _backend_error("robot_stop", exc) from exc

    def read_post_stop_state(self) -> Any:
        """Return one fresh ``Robot.read_once()`` sample after stop attempt."""

        self._require_open()
        if not self._stop_attempted:
            raise PylibfrankaBackendError(
                "post_stop_read", "post-stop verification preceded Robot.stop()"
            )
        try:
            state = self._robot.read_once()
            if state is None:
                raise TypeError("Robot.read_once() returned no state")
            self._observe_robot_time(state)
            return state
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise _backend_error("post_stop_read", exc) from exc

    def close(self) -> None:
        """Release references after stop verification without issuing commands."""

        self._require_open()
        if not self._stop_attempted:
            raise PylibfrankaBackendError(
                "backend_close", "backend cannot close before a stop attempt"
            )
        if self._control is not None:
            self._control.release()
        # pylibfranka Robot 0.21.2 exposes no explicit close().  Releasing the
        # uniquely owned references is therefore the reviewed teardown path;
        # never guess at a binding method or issue automatic recovery here.
        self._control = None
        self._robot = None
        self._pylibfranka = None
        self._last_robot_time_s = None
        if self._gc_was_enabled:
            gc.enable()
        self._gc_was_enabled = None
        self._closed = True


class PylibfrankaBackendFactory:
    """Single-use, delayed real-backend factory for a gated session.

    Constructing this object only validates local strings/callables.  The
    module import and ``Robot(..., RealtimeConfig.kEnforce)`` occur on the sole
    invocation.  ``FrankaPersistentSession`` must own that invocation; no CLI
    or module-level singleton calls the factory eagerly.
    """

    def __init__(
        self,
        robot_ip: str,
        *,
        module_loader: Callable[[str], Any] = importlib.import_module,
        required_version: str = AUDITED_PYLIBFRANKA_VERSION,
    ) -> None:
        address = str(robot_ip).strip()
        if not address:
            raise ValueError("robot_ip must be non-empty")
        if not callable(module_loader):
            raise TypeError("module_loader must be callable")
        version = str(required_version).strip()
        if not version:
            raise ValueError("required_version must be non-empty")
        self.robot_ip = address
        self.required_version = version
        self._module_loader = module_loader
        self._lock = threading.Lock()
        self._creation_attempted = False

    @property
    def creation_attempted(self) -> bool:
        with self._lock:
            return self._creation_attempted

    def __call__(self) -> PylibfrankaPersistentBackend:
        with self._lock:
            if self._creation_attempted:
                raise PylibfrankaBackendError(
                    "backend_factory", "the delayed factory is single-use"
                )
            # A failed import/connection is terminal for this run.  Retrying
            # would be an implicit reconnect outside the sealed session audit.
            self._creation_attempted = True
        try:
            module = self._module_loader("pylibfranka")
            actual_version = str(getattr(module, "__version__", "")).strip()
            if actual_version != self.required_version:
                raise TypeError(
                    "pylibfranka version differs from the audited adapter: "
                    f"expected={self.required_version!r}, actual={actual_version!r}"
                )
            robot_type = getattr(module, "Robot", None)
            joint_type = getattr(module, "JointPositions", None)
            realtime = getattr(
                getattr(module, "RealtimeConfig", None), "kEnforce", None
            )
            controller = getattr(
                getattr(module, "ControllerMode", None), "JointImpedance", None
            )
            if not callable(robot_type):
                raise TypeError("pylibfranka.Robot is unavailable")
            if not callable(joint_type):
                raise TypeError("pylibfranka.JointPositions is unavailable")
            if realtime is None:
                raise TypeError("pylibfranka.RealtimeConfig.kEnforce is unavailable")
            if controller is None:
                raise TypeError(
                    "pylibfranka.ControllerMode.JointImpedance is unavailable"
                )
            robot = robot_type(self.robot_ip, realtime)
            if robot is None:
                raise TypeError("pylibfranka.Robot returned no object")
            for method_name in (
                "start_joint_position_control",
                "stop",
                "read_once",
            ):
                if not callable(getattr(robot, method_name, None)):
                    raise TypeError(f"pylibfranka Robot lacks {method_name}()")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise _backend_error("backend_factory", exc) from exc
        return PylibfrankaPersistentBackend(
            _seal=_BACKEND_SEAL,
            robot=robot,
            pylibfranka=module,
            owner_ident=threading.get_ident(),
        )


__all__ = [
    "AUDITED_PYLIBFRANKA_VERSION",
    "PYLIBFRANKA_STATE_FIELDS",
    "PylibfrankaBackendError",
    "PylibfrankaBackendFactory",
    "PylibfrankaPersistentBackend",
]
