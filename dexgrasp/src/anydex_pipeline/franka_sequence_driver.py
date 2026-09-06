"""Reviewed, dependency-injected Franka motion backend for staged grasps.

Importing this module never imports :mod:`pylibfranka`, opens an FCI
connection, or changes robot configuration.  Callers normally inject both an
already-connected ``robot`` and the matching pylibfranka module.  The optional
:meth:`FrankaSequenceDriver.connect` constructor performs the import and
connection only when explicitly invoked.

Transforms use the ``T_A_B`` convention and Franka's 16-element state/command
poses are converted as column-major matrices.  The driver never changes
``F_T_EE``.  It exposes one explicit, audit-bound external-load transition
after closure and a matching clear after audited setdown; both are serialized
against control handles and verified from fresh Franka state.
"""

from __future__ import annotations

import importlib
import gc
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, List, Optional, Sequence, Tuple

import numpy as np


FR3_JOINT_LIMITS_RAD = np.asarray(
    [
        [-2.9007, 2.9007],
        [-1.8361, 1.8361],
        [-2.9007, 2.9007],
        [-3.0770, -0.1169],
        [-2.8763, 2.8763],
        [0.4398, 4.6216],
        [-3.0508, 3.0508],
    ],
    dtype=np.float64,
)

MIN_CONTROL_PERIOD_S = 1.0e-6
MAX_CONTROL_PERIOD_S = 0.020
FCI_READ_TO_WRITE_BUDGET_NS = 500_000
CONTROL_SUCCESS_RATE_ABSOLUTE_TOLERANCE = 1.0e-6
CONTROL_SUCCESS_QUALIFICATION_MIN_POSITIVE_CYCLES = 100
COSINE_PEAK_VELOCITY_FACTOR = math.pi / 2.0
COSINE_PEAK_ACCELERATION_FACTOR = math.pi * math.pi / 2.0
MINIMUM_JERK_PEAK_VELOCITY_FACTOR = 1.875
# Robot-state transforms can carry single-precision-sized orthonormality error.
# Leave capped Cartesian steps 0.1 ppm inside their hard bound so an SO(3)
# log/exp round trip cannot turn a mathematically exact boundary into a small
# numerical overshoot.  The post-construction retraction below still enforces
# the exact configured limits if a larger valid-input error is encountered.
CARTESIAN_SEGMENT_INWARD_RELATIVE_MARGIN = 1.0e-7
CARTESIAN_SEGMENT_MAX_RETRACTIONS = 8


class FrankaStopUnconfirmed(RuntimeError):
    """A stop command was attempted but fresh robot states did not prove rest."""


@dataclass(frozen=True)
class FrankaControlLoopTelemetry:
    """Allocation-free-in-loop timing summary for the last control handle."""

    kind: str
    samples: int
    max_read_to_write_ns: int
    read_to_write_overruns: int
    success_qualified: bool
    success_qualification_positive_writes: int
    success_qualification_control_time_s: float
    success_qualification_wall_time_s: float
    success_qualification_rate: Optional[float]

    @property
    def max_read_to_write_us(self) -> float:
        return float(self.max_read_to_write_ns) / 1_000.0


@dataclass(frozen=True)
class FrankaMotionLimits:
    """Conservative motion envelope and required tool-frame provenance.

    ``expected_F_T_EE`` has no default on purpose.  It must come from the
    commissioned adapter/hand configuration.  Cartesian workspace bounds may
    be omitted for joint-only commissioning, but :meth:`move_pose` and the
    generic :meth:`verify_settled` then fail closed.  The separate read-only
    audited-joint settle verifier checks an artifact-bound q/EEF pair and does
    not authorize Cartesian interpolation.
    """

    expected_F_T_EE: np.ndarray
    expected_m_ee_kg: Optional[float] = None
    expected_F_x_Cee_m: Optional[np.ndarray] = None
    expected_I_ee_kg_m2: Optional[np.ndarray] = None
    mass_tolerance_kg: float = 1.0e-5
    center_of_mass_tolerance_m: float = 1.0e-6
    inertia_tolerance_kg_m2: float = 1.0e-6
    joint_limits_rad: np.ndarray = field(
        default_factory=lambda: FR3_JOINT_LIMITS_RAD.copy()
    )
    joint_limit_margin_rad: float = 0.05
    max_joint_speed_rad_s: float = 0.08
    max_joint_segment_rad: float = 0.35
    min_joint_duration_s: float = 4.0
    workspace_min_m: Optional[np.ndarray] = None
    workspace_max_m: Optional[np.ndarray] = None
    max_cartesian_speed_m_s: float = 0.010
    max_angular_speed_rad_s: float = 0.030
    max_segment_translation_m: float = 0.030
    max_segment_rotation_rad: float = 0.10
    min_cartesian_duration_s: float = 3.0
    cartesian_endpoint_hold_s: float = 0.0
    cartesian_pose_controller_mode: str = "cartesian_impedance"
    joint_arrival_tolerance_rad: float = 0.025
    max_continuous_joint_tracking_error_rad: Optional[float] = None
    translation_arrival_tolerance_m: float = 0.008
    rotation_arrival_tolerance_rad: float = 0.030
    settle_time_s: float = 0.25
    settle_timeout_s: float = 2.0
    settle_poll_s: float = 0.010
    settle_max_dq_rad_s: float = 0.030
    stop_verify_timeout_s: float = 2.0
    stop_verify_poll_s: float = 0.020
    stop_verify_max_dq_rad_s: float = 0.020
    stop_verify_consecutive_samples: int = 3
    min_control_success_rate: float = 0.95
    control_success_warmup_s: float = 0.10
    control_success_hard_floor: float = 0.80
    control_success_evaluation_window_s: float = 0.50
    F_T_EE_tolerance: float = 1.0e-8
    max_dynamic_segments: int = 100
    wall_deadline_slack_s: float = 2.0
    wall_deadline_fraction: float = 0.25

    def __post_init__(self) -> None:
        expected = _coerce_pose_matrix(self.expected_F_T_EE, "expected_F_T_EE")
        dynamics = (
            self.expected_m_ee_kg,
            self.expected_F_x_Cee_m,
            self.expected_I_ee_kg_m2,
        )
        if any(value is not None for value in dynamics) and not all(
            value is not None for value in dynamics
        ):
            raise ValueError(
                "expected_m_ee_kg, expected_F_x_Cee_m, and "
                "expected_I_ee_kg_m2 must be provided together"
            )
        expected_mass = None
        expected_com = None
        expected_inertia = None
        if self.expected_m_ee_kg is not None:
            expected_mass = float(self.expected_m_ee_kg)
            if not np.isfinite(expected_mass) or expected_mass < 0.0:
                raise ValueError("expected_m_ee_kg must be finite and non-negative")
            expected_com = _finite_vector(
                self.expected_F_x_Cee_m, 3, "expected_F_x_Cee_m"
            )
            expected_inertia = _coerce_inertia_matrix(
                self.expected_I_ee_kg_m2, "expected_I_ee_kg_m2"
            )
        joint_limits = np.asarray(self.joint_limits_rad, dtype=np.float64)
        if joint_limits.shape != (7, 2) or not np.all(np.isfinite(joint_limits)):
            raise ValueError("joint_limits_rad must be a finite (7, 2) array")
        if np.any(joint_limits[:, 0] >= joint_limits[:, 1]):
            raise ValueError("joint_limits_rad lower bounds must be below upper bounds")

        workspace_min = self.workspace_min_m
        workspace_max = self.workspace_max_m
        if (workspace_min is None) != (workspace_max is None):
            raise ValueError(
                "workspace_min_m and workspace_max_m must be provided together"
            )
        if workspace_min is not None:
            workspace_min = _finite_vector(workspace_min, 3, "workspace_min_m")
            workspace_max = _finite_vector(workspace_max, 3, "workspace_max_m")
            if np.any(workspace_min >= workspace_max):
                raise ValueError("workspace_min_m must be below workspace_max_m")

        positive_fields = (
            "joint_limit_margin_rad",
            "max_joint_speed_rad_s",
            "max_joint_segment_rad",
            "min_joint_duration_s",
            "max_cartesian_speed_m_s",
            "max_angular_speed_rad_s",
            "max_segment_translation_m",
            "max_segment_rotation_rad",
            "min_cartesian_duration_s",
            "joint_arrival_tolerance_rad",
            "translation_arrival_tolerance_m",
            "rotation_arrival_tolerance_rad",
            "settle_time_s",
            "settle_timeout_s",
            "settle_poll_s",
            "settle_max_dq_rad_s",
            "stop_verify_timeout_s",
            "stop_verify_poll_s",
            "stop_verify_max_dq_rad_s",
            "F_T_EE_tolerance",
            "mass_tolerance_kg",
            "center_of_mass_tolerance_m",
            "inertia_tolerance_kg_m2",
            "control_success_evaluation_window_s",
            "wall_deadline_slack_s",
            "wall_deadline_fraction",
        )
        for name in positive_fields:
            _require_positive(getattr(self, name), name)
        if self.settle_timeout_s < self.settle_time_s:
            raise ValueError("settle_timeout_s must be at least settle_time_s")
        stop_samples = self.stop_verify_consecutive_samples
        try:
            stop_samples_numeric = float(stop_samples)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "stop_verify_consecutive_samples must be an integer >= 2"
            ) from exc
        if (
            isinstance(stop_samples, (bool, np.bool_))
            or not np.isfinite(stop_samples_numeric)
            or not stop_samples_numeric.is_integer()
            or int(stop_samples_numeric) < 2
        ):
            raise ValueError(
                "stop_verify_consecutive_samples must be an integer >= 2"
            )
        success_rate = float(self.min_control_success_rate)
        if not np.isfinite(success_rate) or not 0.0 <= success_rate <= 1.0:
            raise ValueError("min_control_success_rate must be finite in [0, 1]")
        hard_floor = float(self.control_success_hard_floor)
        if (
            not np.isfinite(hard_floor)
            or not 0.0 <= hard_floor <= success_rate
        ):
            raise ValueError(
                "control_success_hard_floor must be finite in "
                "[0, min_control_success_rate]"
            )
        warmup = float(self.control_success_warmup_s)
        if not np.isfinite(warmup) or warmup < 0.0:
            raise ValueError("control_success_warmup_s must be finite and non-negative")
        endpoint_hold = float(self.cartesian_endpoint_hold_s)
        if not np.isfinite(endpoint_hold) or endpoint_hold < 0.0:
            raise ValueError(
                "cartesian_endpoint_hold_s must be finite and non-negative"
            )
        if (
            endpoint_hold > 0.0
            and endpoint_hold + 1.0e-12 < float(self.settle_time_s)
        ):
            raise ValueError(
                "cartesian_endpoint_hold_s must be at least settle_time_s when "
                "endpoint convergence gating is enabled"
            )
        controller_mode = self.cartesian_pose_controller_mode
        if not isinstance(controller_mode, str) or controller_mode not in (
            "cartesian_impedance",
            "joint_impedance",
        ):
            raise ValueError(
                "cartesian_pose_controller_mode must be either "
                "'cartesian_impedance' or 'joint_impedance'"
            )
        complete_quality_observation_s = (
            warmup + float(self.control_success_evaluation_window_s)
        )
        if (
            float(self.min_joint_duration_s) + 1.0e-12
            < complete_quality_observation_s
            or float(self.min_cartesian_duration_s) + 1.0e-12
            < complete_quality_observation_s
        ):
            raise ValueError(
                "minimum Franka segment durations must each cover the control "
                "success warmup plus one complete evaluation window"
            )
        if isinstance(self.max_dynamic_segments, (bool, np.bool_)) or int(
            self.max_dynamic_segments
        ) <= 0:
            raise ValueError("max_dynamic_segments must be a positive integer")
        continuous_tracking = self.max_continuous_joint_tracking_error_rad
        if continuous_tracking is not None:
            continuous_tracking = float(continuous_tracking)
            if (
                not np.isfinite(continuous_tracking)
                or not 0.0 < continuous_tracking <= 0.01
            ):
                raise ValueError(
                    "max_continuous_joint_tracking_error_rad must be in (0,0.01]"
                )

        # The requested margin must leave a non-empty safe interval per joint.
        widths = joint_limits[:, 1] - joint_limits[:, 0]
        if np.any(2.0 * float(self.joint_limit_margin_rad) >= widths):
            raise ValueError("joint_limit_margin_rad consumes a joint's safe range")

        object.__setattr__(self, "expected_F_T_EE", expected)
        object.__setattr__(self, "expected_m_ee_kg", expected_mass)
        object.__setattr__(self, "expected_F_x_Cee_m", expected_com)
        object.__setattr__(self, "expected_I_ee_kg_m2", expected_inertia)
        object.__setattr__(self, "joint_limits_rad", _readonly(joint_limits))
        object.__setattr__(self, "workspace_min_m", workspace_min)
        object.__setattr__(self, "workspace_max_m", workspace_max)
        object.__setattr__(self, "max_dynamic_segments", int(self.max_dynamic_segments))
        object.__setattr__(
            self,
            "stop_verify_consecutive_samples",
            int(stop_samples_numeric),
        )
        object.__setattr__(
            self,
            "max_continuous_joint_tracking_error_rad",
            continuous_tracking,
        )
        object.__setattr__(self, "cartesian_endpoint_hold_s", endpoint_hold)
        object.__setattr__(self, "cartesian_pose_controller_mode", controller_mode)


class _ControlSuccessWatchdog:
    """Evaluate libfranka's rolling delivery metric without double-counting it.

    ``control_command_success_rate`` already describes the last 100 control
    commands (about 100 ms at the FCI's 1 kHz rate).  Treating one value below
    the quality threshold as an immediate fault therefore double-counts a
    short packet-loss burst for many consecutive state samples.  This
    watchdog still fails closed in either of two cases:

    * one sample crosses the independently configured hard floor; or
    * a complete, time-weighted sliding observation window falls below the
      quality threshold.

    This mirrors the official communication test's use of an aggregate while
    retaining a project-conservative severe-loss stop.  Robot mode, errors,
    contacts/collisions, joint margins, tracking and control-period checks
    remain per-sample gates outside this helper; tool/load provenance is bound
    at each control-handle boundary.
    """

    def __init__(self, limits: FrankaMotionLimits) -> None:
        self._threshold = float(limits.min_control_success_rate)
        self._hard_floor = float(limits.control_success_hard_floor)
        self._window_limit_s = float(
            limits.control_success_evaluation_window_s
        )
        # Adjacent libfranka rolling-success samples are commonly identical.
        # Coalesce such runs so the normal 1 kHz path does not allocate one
        # tuple per state sample.  A new two-float list is allocated only when
        # the reported rate changes; partial window trimming mutates that list
        # in place.
        self._samples: Deque[List[float]] = deque()
        self._window_duration_s = 0.0
        self._weighted_success = 0.0

    def observe(self, success: float, period_s: float) -> None:
        if (
            success
            < self._hard_floor - CONTROL_SUCCESS_RATE_ABSOLUTE_TOLERANCE
        ):
            raise RuntimeError(
                "Franka control command success rate crossed hard floor: "
                f"actual={success:.9f}, hard_floor={self._hard_floor:.9f}, "
                f"quality_threshold={self._threshold:.9f}"
            )

        if period_s <= 0.0:
            return

        if self._samples and self._samples[-1][0] == success:
            self._samples[-1][1] += period_s
        else:
            self._samples.append([success, period_s])
        self._window_duration_s += period_s
        self._weighted_success += success * period_s

        excess = self._window_duration_s - self._window_limit_s
        while excess > 1.0e-12:
            oldest_success, oldest_duration = self._samples[0]
            removed = min(oldest_duration, excess)
            self._window_duration_s -= removed
            self._weighted_success -= oldest_success * removed
            excess -= removed
            if removed + 1.0e-12 >= oldest_duration:
                self._samples.popleft()
            else:
                self._samples[0][1] = oldest_duration - removed

        if self._window_duration_s + 1.0e-12 < self._window_limit_s:
            return

        average = self._weighted_success / self._window_duration_s
        if average < self._threshold - CONTROL_SUCCESS_RATE_ABSOLUTE_TOLERANCE:
            # This O(window-size) scan occurs only on the failure path; the
            # normal 1 kHz path remains O(1) amortized.
            minimum = min(value for value, _ in self._samples)
            raise RuntimeError(
                "Franka control command success rate window average is below "
                "threshold: "
                f"average={average:.9f}, minimum={minimum:.9f}, "
                f"latest={success:.9f}, threshold={self._threshold:.9f}, "
                f"window={self._window_duration_s:.9f}s"
            )


class _ControlSuccessStartupQualification:
    """Hold a new control handle still until its rolling metric is usable."""

    __slots__ = (
        "_threshold",
        "_deadline_s",
        "positive_writes",
        "control_time_s",
        "wall_time_s",
        "qualified",
        "qualification_rate",
    )

    def __init__(self, limits: FrankaMotionLimits) -> None:
        self._threshold = float(limits.min_control_success_rate)
        self._deadline_s = float(limits.control_success_evaluation_window_s)
        self.positive_writes = 0
        self.control_time_s = 0.0
        self.wall_time_s = 0.0
        self.qualified = False
        self.qualification_rate: Optional[float] = None

    def observe_read(
        self, success: float, period_s: float, wall_time_s: float
    ) -> bool:
        """Return whether this read may qualify after one more hold write."""

        if self.qualified:
            raise RuntimeError("internal error: startup qualification already complete")
        if period_s > 0.0:
            self.control_time_s += period_s
        self.wall_time_s = max(self.wall_time_s, float(wall_time_s))
        within_deadline = (
            self.control_time_s
            <= self._deadline_s + CONTROL_SUCCESS_RATE_ABSOLUTE_TOLERANCE
            and self.wall_time_s
            <= self._deadline_s + CONTROL_SUCCESS_RATE_ABSOLUTE_TOLERANCE
        )
        candidate = (
            self.positive_writes
            >= CONTROL_SUCCESS_QUALIFICATION_MIN_POSITIVE_CYCLES
            and period_s > 0.0
            and success
            >= self._threshold - CONTROL_SUCCESS_RATE_ABSOLUTE_TOLERANCE
            and within_deadline
        )
        if not candidate and (
            self.control_time_s + 1.0e-12 >= self._deadline_s
            or self.wall_time_s + 1.0e-12 >= self._deadline_s
        ):
            raise RuntimeError(
                "Franka control command success startup qualification timed out: "
                f"latest={success:.9f}, threshold={self._threshold:.9f}, "
                f"positive_writes={self.positive_writes}, "
                f"required_positive_writes="
                f"{CONTROL_SUCCESS_QUALIFICATION_MIN_POSITIVE_CYCLES}, "
                f"control_time={self.control_time_s:.9f}s, "
                f"wall_time={self.wall_time_s:.9f}s, "
                f"deadline={self._deadline_s:.9f}s"
            )
        return candidate

    def record_hold_write(
        self,
        *,
        positive_period: bool,
        candidate: bool,
        success: float,
        wall_time_s: float,
    ) -> None:
        """Record a completed exact-start write and arm on a valid candidate."""

        if positive_period:
            self.positive_writes += 1
        self.wall_time_s = max(self.wall_time_s, float(wall_time_s))
        if not candidate:
            return
        if self.wall_time_s > (
            self._deadline_s + CONTROL_SUCCESS_RATE_ABSOLUTE_TOLERANCE
        ):
            raise RuntimeError(
                "Franka control command success startup qualification exceeded "
                "its wall-clock deadline during the qualifying hold write: "
                f"wall_time={self.wall_time_s:.9f}s, "
                f"deadline={self._deadline_s:.9f}s"
            )
        self.qualified = True
        self.qualification_rate = float(success)


_REALTIME_VECTOR_FIELDS: Tuple[Tuple[str, int], ...] = (
    ("cartesian_contact", 6),
    ("cartesian_collision", 6),
    ("joint_contact", 7),
    ("joint_collision", 7),
)


class _RealtimeStateValidator:
    """Allocation-light validator for one active FCI control handle.

    A full state validation immediately before the handle starts binds the
    commissioned ``F_T_EE`` and active end-effector/load dynamics.  This
    application's motion process is the sole owner of the Robot; its one
    audit-bound ``set_load`` path is serialized against control handles.  Full
    validation also runs after every successful segment.  Under that ownership
    contract, repeating
    pose orthonormality checks, inertia eigendecompositions, and load
    comparisons at 1 kHz only steals deadline budget.  This validator keeps
    every value that *can* indicate an unsafe active motion on the per-sample
    path:

    * robot mode and every current-error bit;
    * Cartesian/joint contact and collision vectors;
    * finite joint positions and commissioned joint-limit margins;
    * optional command-to-state tracking error; and
    * finite, bounded command-success feedback.

    Control-period, wall-deadline and success-window checks remain in the
    enclosing loops because they need the matching period/clock sample.
    """

    __slots__ = (
        "_limits",
        "_joint_lower",
        "_joint_upper",
        "_errors_type",
        "_errors_strategy",
        "_error_fields",
    )

    def __init__(self, limits: FrankaMotionLimits, reference_state: Any) -> None:
        self._limits = limits
        self._joint_lower = tuple(float(value) for value in limits.joint_limits_rad[:, 0])
        self._joint_upper = tuple(float(value) for value in limits.joint_limits_rad[:, 1])

        errors = getattr(reference_state, "current_errors", None)
        if errors is None:
            raise RuntimeError("Franka state has no current_errors")
        self._errors_type = type(errors)
        self._error_fields: Tuple[str, ...] = ()
        if isinstance(errors, dict):
            self._errors_strategy = "dict"
        elif (
            self._errors_type.__name__ == "Errors"
            and self._errors_type.__module__.startswith("pylibfranka")
            and getattr(self._errors_type, "__bool__", None) is not None
        ):
            # pylibfranka::Errors implements __bool__ as an allocation-free
            # native any-error check.  Expand names only on the fault path.
            self._errors_strategy = "pylibfranka_bool"
        elif hasattr(errors, "__dict__"):
            # SimpleNamespace and Python test doubles can gain fields after
            # preflight, so inspect their existing mapping without copying it.
            self._errors_strategy = "namespace"
        else:
            self._errors_strategy = "fields"
            self._error_fields = tuple(
                name
                for name in dir(errors)
                if not name.startswith("_")
                and _is_boolean_error_value(_safe_getattr(errors, name))
            )

    def validate(
        self,
        state: Any,
        *,
        require_idle: bool,
        enforce_joint_limit_margin: bool = True,
        previous_command_q: Optional[Sequence[float]] = None,
        tracking_limit_rad: Optional[float] = None,
    ) -> float:
        if not enforce_joint_limit_margin and not require_idle:
            raise RuntimeError(
                "joint-limit margin may be omitted only for a passive Idle read"
            )
        mode = _robot_mode_name(getattr(state, "robot_mode", None))
        if require_idle and mode != "idle":
            raise RuntimeError(f"Franka must be Idle before motion, got {mode}")
        if not require_idle and mode not in ("move", "idle"):
            raise RuntimeError(f"Franka mode became unsafe during motion: {mode}")

        self._validate_errors(getattr(state, "current_errors", None))
        for field_name, expected_length in _REALTIME_VECTOR_FIELDS:
            _validate_realtime_flag_vector(
                getattr(state, field_name, None), field_name, expected_length
            )

        q = getattr(state, "q", None)
        try:
            if len(q) != 7 or (
                getattr(q, "shape", (7,)) != (7,)
            ):
                raise RuntimeError("Franka state.q is missing or malformed")
        except TypeError as exc:
            raise RuntimeError("Franka state.q is missing or malformed") from exc

        if (previous_command_q is None) != (tracking_limit_rad is None):
            raise RuntimeError(
                "internal error: tracking command and limit must be supplied together"
            )
        if previous_command_q is not None:
            try:
                if len(previous_command_q) != 7:
                    raise RuntimeError("previous Franka joint command is malformed")
            except TypeError as exc:
                raise RuntimeError("previous Franka joint command is malformed") from exc

        minimum_margin = float("inf")
        minimum_index = 0
        maximum_tracking_error = 0.0
        for index in range(7):
            try:
                value = float(q[index])
            except (TypeError, ValueError, IndexError) as exc:
                raise RuntimeError("Franka state.q is missing or malformed") from exc
            if not math.isfinite(value):
                raise RuntimeError("Franka state.q contains NaN or infinity")
            margin = min(
                value - self._joint_lower[index],
                self._joint_upper[index] - value,
            )
            if margin < minimum_margin:
                minimum_margin = margin
                minimum_index = index
            if previous_command_q is not None:
                try:
                    commanded = float(previous_command_q[index])
                except (TypeError, ValueError, IndexError) as exc:
                    raise RuntimeError(
                        "previous Franka joint command is malformed"
                    ) from exc
                if not math.isfinite(commanded):
                    raise RuntimeError(
                        "previous Franka joint command contains NaN or infinity"
                    )
                maximum_tracking_error = max(
                    maximum_tracking_error, abs(value - commanded)
                )

        if (
            enforce_joint_limit_margin
            and minimum_margin < float(self._limits.joint_limit_margin_rad)
        ):
            raise RuntimeError(
                f"Franka joint {minimum_index + 1} has only "
                f"{minimum_margin:.5f}rad limit margin"
            )
        if (
            tracking_limit_rad is not None
            and maximum_tracking_error > float(tracking_limit_rad)
        ):
            raise RuntimeError(
                "continuous joint tracking error {:.6f}rad exceeds audited "
                "{:.6f}rad bound".format(
                    maximum_tracking_error, float(tracking_limit_rad)
                )
            )

        success_value = getattr(state, "control_command_success_rate", None)
        if success_value is None:
            raise RuntimeError("Franka state has no control_command_success_rate")
        try:
            success = float(success_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Franka control command success rate is non-numeric"
            ) from exc
        if not math.isfinite(success):
            raise RuntimeError("Franka control command success rate is non-finite")
        if not 0.0 <= success <= 1.0:
            raise RuntimeError(
                "Franka control command success rate is outside [0, 1]: "
                f"actual={success:.9f}"
            )
        return success

    def _validate_errors(self, errors: Any) -> None:
        if errors is None:
            raise RuntimeError("Franka state has no current_errors")
        if type(errors) is not self._errors_type:
            raise RuntimeError("Franka current_errors schema changed during motion")

        active: Optional[List[str]] = None
        if self._errors_strategy == "pylibfranka_bool":
            if bool(errors):
                active = _active_error_names(errors)
                if not active:
                    active = ["unknown pylibfranka error"]
        elif self._errors_strategy == "dict":
            if any(bool(value) for value in errors.values()):
                active = sorted(
                    str(name) for name, value in errors.items() if bool(value)
                )
        elif self._errors_strategy == "namespace":
            mapping = vars(errors)
            for name, value in mapping.items():
                if _is_boolean_error_value(value) and bool(value):
                    active = [str(name)]
                    break
        else:
            for name in self._error_fields:
                value = _safe_getattr(errors, name)
                if not _is_boolean_error_value(value):
                    raise RuntimeError(
                        f"Franka current_errors field {name} changed type"
                    )
                if bool(value):
                    active = [name]
                    break
        if active:
            raise RuntimeError(f"Franka current_errors are active: {active}")


class _PreparedCartesianInterpolation:
    """One-time SE(3) preparation plus an allocation-light command sampler."""

    __slots__ = (
        "_rotation_start",
        "_rotation_sin",
        "_rotation_one_minus_cos",
        "_rotation_angle",
        "_translation_start",
        "_translation_delta",
        "target_values",
    )

    def __init__(self, start: np.ndarray, target: np.ndarray) -> None:
        start_pose = _coerce_pose_matrix(start, "Cartesian interpolation start")
        target_pose = _coerce_pose_matrix(target, "Cartesian interpolation target")
        relative_rotvec = so3_log(
            target_pose[:3, :3] @ start_pose[:3, :3].T
        )
        angle = float(np.linalg.norm(relative_rotvec))
        if angle > 1.0e-12:
            axis = relative_rotvec / angle
            skew = _skew(axis)
            skew_squared = skew @ skew
            rotation_sin = skew @ start_pose[:3, :3]
            rotation_one_minus_cos = skew_squared @ start_pose[:3, :3]
        else:
            rotation_sin = np.zeros((3, 3), dtype=np.float64)
            rotation_one_minus_cos = np.zeros((3, 3), dtype=np.float64)
        self._rotation_start = tuple(
            float(value) for value in start_pose[:3, :3].reshape(9, order="F")
        )
        self._rotation_sin = tuple(
            float(value) for value in rotation_sin.reshape(9, order="F")
        )
        self._rotation_one_minus_cos = tuple(
            float(value)
            for value in rotation_one_minus_cos.reshape(9, order="F")
        )
        self._rotation_angle = angle
        self._translation_start = tuple(
            float(value) for value in start_pose[:3, 3]
        )
        self._translation_delta = tuple(
            float(target_pose[index, 3] - start_pose[index, 3])
            for index in range(3)
        )
        self.target_values = _matrix_to_franka_pose(target_pose)

    def command_values(self, alpha: float) -> List[float]:
        values = [0.0] * 16
        self.fill_command_values(alpha, values)
        return values

    def fill_command_values(
        self, alpha: float, output: List[float]
    ) -> None:
        """Fill one preallocated column-major command list in place."""

        if len(output) != 16:
            raise RuntimeError("Cartesian command buffer must contain 16 values")
        blend = minimum_jerk_blend(alpha)
        phase = self._rotation_angle * blend
        sine = math.sin(phase)
        one_minus_cosine = 1.0 - math.cos(phase)
        rotation_index = 0
        while rotation_index < 9:
            output_index = rotation_index + rotation_index // 3
            output[output_index] = (
                self._rotation_start[rotation_index]
                + sine * self._rotation_sin[rotation_index]
                + one_minus_cosine
                * self._rotation_one_minus_cos[rotation_index]
            )
            rotation_index += 1
        output[3] = 0.0
        output[7] = 0.0
        output[11] = 0.0
        index = 0
        while index < 3:
            output[12 + index] = (
                self._translation_start[index]
                + blend * self._translation_delta[index]
            )
            index += 1
        output[15] = 1.0


class FrankaSequenceDriver:
    """Synchronous, fail-closed Franka backend expected by the grasp sequencer."""

    def __init__(
        self,
        robot: Any,
        pylibfranka: Any,
        limits: FrankaMotionLimits,
        *,
        monotonic: Optional[Callable[[], float]] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        if robot is None:
            raise ValueError("robot must be injected")
        if pylibfranka is None:
            raise ValueError("pylibfranka must be injected")
        if not isinstance(limits, FrankaMotionLimits):
            raise TypeError("limits must be FrankaMotionLimits")
        self.robot = robot
        self.pylibfranka = pylibfranka
        self.limits = limits
        self._monotonic = monotonic if monotonic is not None else time.monotonic
        self._sleep = sleep if sleep is not None else time.sleep
        # Kept separate from the injectable wall clock: perf_counter_ns is a
        # direct, monotonic integer clock suitable for sub-millisecond loop
        # telemetry and allocates no timing objects in the hot path.
        self._control_clock_ns: Callable[[], int] = time.perf_counter_ns
        self.last_control_loop_telemetry: Optional[
            FrankaControlLoopTelemetry
        ] = None
        self._configuration_lock = threading.RLock()
        self._control_handle_active = False
        self._expected_external_load_mass_kg = 0.0
        self._expected_F_x_Cload_m = np.zeros(3, dtype=np.float64)
        self._expected_I_load_kg_m2 = np.zeros((3, 3), dtype=np.float64)
        self._external_load_binding_sha256: Optional[str] = None
        self._load_configuration_uncertain = False
        # Optional single-owner native telemetry tap.  It is injected only by
        # the hardware executor after all offline gates pass.  The default
        # remains the exact original pylibfranka readOnce path.
        self._telemetry_tap: Optional[Any] = None

    @property
    def external_load_binding_sha256(self) -> Optional[str]:
        return self._external_load_binding_sha256

    @property
    def expected_external_load_mass_kg(self) -> float:
        return float(self._expected_external_load_mass_kg)

    @classmethod
    def connect(
        cls,
        robot_ip: str,
        limits: FrankaMotionLimits,
        *,
        pylibfranka: Optional[Any] = None,
        enforce_realtime: bool = False,
        monotonic: Optional[Callable[[], float]] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> "FrankaSequenceDriver":
        """Explicitly import pylibfranka (if needed) and open one FCI Robot."""

        module = (
            pylibfranka
            if pylibfranka is not None
            else importlib.import_module("pylibfranka")
        )
        realtime = (
            module.RealtimeConfig.kEnforce
            if enforce_realtime
            else module.RealtimeConfig.kIgnore
        )
        robot = module.Robot(str(robot_ip), realtime)
        return cls(
            robot,
            module,
            limits,
            monotonic=monotonic,
            sleep=sleep,
        )

    def _coerce_bound_external_load(
        self, payload: Any
    ) -> Tuple[float, np.ndarray, np.ndarray, str]:
        if self.limits.expected_m_ee_kg is None:
            raise RuntimeError(
                "external load requires commissioned end-effector mass/CoM/inertia"
            )
        mass = float(getattr(payload, "mass_kg", np.nan))
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("external payload mass_kg must be finite and positive")
        center = _finite_vector(
            getattr(payload, "F_x_Cload_m", None), 3, "payload.F_x_Cload_m"
        )
        inertia = _coerce_inertia_matrix(
            getattr(payload, "I_load_kg_m2", None), "payload.I_load_kg_m2"
        )
        moments = np.linalg.eigvalsh(inertia)
        if float(moments[0]) <= 0.0:
            raise ValueError("external payload inertia must be positive definite")
        if float(moments[2]) > float(moments[0] + moments[1]) + 1.0e-12:
            raise ValueError("external payload inertia violates triangle inequality")
        binding = str(getattr(payload, "binding_sha256", ""))
        if len(binding) != 64 or any(
            character not in "0123456789abcdef" for character in binding
        ):
            raise ValueError(
                "external payload requires a lowercase audit binding SHA-256"
            )
        return mass, center.copy(), inertia.copy(), binding

    def _set_expected_external_load(
        self,
        mass_kg: float,
        center_m: np.ndarray,
        inertia_kg_m2: np.ndarray,
        binding_sha256: Optional[str],
    ) -> None:
        self._expected_external_load_mass_kg = float(mass_kg)
        self._expected_F_x_Cload_m = np.asarray(center_m, dtype=np.float64).copy()
        self._expected_I_load_kg_m2 = np.asarray(
            inertia_kg_m2, dtype=np.float64
        ).copy()
        self._external_load_binding_sha256 = binding_sha256

    def _require_configuration_idle(self) -> None:
        if self._control_handle_active:
            raise RuntimeError(
                "Franka configuration change is forbidden during a control handle"
            )

    def install_native_telemetry_tap(self, tap: Any) -> None:
        """Install one fused ``readOnce`` publisher while no handle is active.

        The tap must not create another Robot connection.  During active
        control it replaces, rather than supplements, ``control.readOnce`` so
        the unique FCI owner and one-read/one-write cadence are preserved.
        """

        if tap is None:
            raise ValueError("native telemetry tap is required")
        for method_name in ("read_once_tapped", "synchronize_and_publish_arm"):
            if not callable(getattr(tap, method_name, None)):
                raise TypeError(
                    "native telemetry tap lacks {}".format(method_name)
                )
        with self._configuration_lock:
            self._require_configuration_idle()
            if self._telemetry_tap is not None:
                raise RuntimeError("a native Franka telemetry tap is already installed")
            self._telemetry_tap = tap

    def remove_native_telemetry_tap(self, tap: Any) -> None:
        """Remove the exact installed tap after active control has ended."""

        with self._configuration_lock:
            self._require_configuration_idle()
            if self._telemetry_tap is not tap:
                raise RuntimeError("native Franka telemetry tap identity mismatch")
            self._telemetry_tap = None

    def publish_native_telemetry_state(self, state: Any) -> Optional[int]:
        """Publish an already-read and subsequently validated idle state.

        This method never reads the robot.  It is used by the existing unique
        FCI owner at boundaries and during RH56 safety polling.
        """

        tap = self._telemetry_tap
        if tap is None:
            return None
        return int(
            tap.synchronize_and_publish_arm(
                state,
                int(time.time_ns()),
                int(time.monotonic_ns()),
            )
        )

    def _call_set_load(
        self, mass_kg: float, center_m: np.ndarray, inertia_kg_m2: np.ndarray
    ) -> None:
        setter = getattr(self.robot, "set_load", None)
        if not callable(setter):
            raise RuntimeError("connected Franka API does not expose Robot.set_load")
        setter(
            float(mass_kg),
            np.asarray(center_m, dtype=np.float64).tolist(),
            np.asarray(inertia_kg_m2, dtype=np.float64)
            .reshape(9, order="F")
            .tolist(),
        )

    def apply_external_load_and_verify(self, payload: Any) -> None:
        """Apply one audit-bound payload after closure and verify fresh state."""

        mass, center, inertia, binding = self._coerce_bound_external_load(payload)
        zeros3 = np.zeros(3, dtype=np.float64)
        zeros33 = np.zeros((3, 3), dtype=np.float64)
        with self._configuration_lock:
            self._require_configuration_idle()
            if self._load_configuration_uncertain:
                raise RuntimeError(
                    "Franka load configuration is uncertain; explicit recovery required"
                )
            if self._external_load_binding_sha256 is not None or not np.isclose(
                self._expected_external_load_mass_kg, 0.0, atol=0.0, rtol=0.0
            ):
                raise RuntimeError("an external Franka load is already active")
            self._validate_state(self.robot.read_once(), require_idle=True)
            try:
                self._call_set_load(mass, center, inertia)
                self._set_expected_external_load(mass, center, inertia, binding)
                self._validate_state(self.robot.read_once(), require_idle=True)
                return
            except BaseException as apply_error:
                try:
                    self._call_set_load(0.0, zeros3, zeros33)
                    self._set_expected_external_load(0.0, zeros3, zeros33, None)
                    self._validate_state(self.robot.read_once(), require_idle=True)
                except BaseException as rollback_error:
                    self._load_configuration_uncertain = True
                    raise RuntimeError(
                        "LOAD CONFIGURATION UNCONFIRMED: payload apply failed and "
                        f"zero-load rollback was not verified: apply={apply_error}; "
                        f"rollback={rollback_error}"
                    ) from apply_error
                raise RuntimeError(
                    "external payload apply was not verified; zero-load rollback "
                    f"was verified: {apply_error}"
                ) from apply_error

    def clear_external_load_and_verify(self) -> None:
        """Clear the payload only after audited setdown and verify fresh state."""

        zeros3 = np.zeros(3, dtype=np.float64)
        zeros33 = np.zeros((3, 3), dtype=np.float64)
        with self._configuration_lock:
            self._require_configuration_idle()
            if not self._load_configuration_uncertain:
                self._validate_state(self.robot.read_once(), require_idle=True)
                if self._external_load_binding_sha256 is None and np.isclose(
                    self._expected_external_load_mass_kg,
                    0.0,
                    atol=0.0,
                    rtol=0.0,
                ):
                    return
            try:
                self._call_set_load(0.0, zeros3, zeros33)
                self._set_expected_external_load(0.0, zeros3, zeros33, None)
                self._load_configuration_uncertain = False
                self._validate_state(self.robot.read_once(), require_idle=True)
            except BaseException as exc:
                self._load_configuration_uncertain = True
                raise RuntimeError(
                    "LOAD CLEAR UNCONFIRMED: Franka zero external load was not "
                    f"verified: {exc}"
                ) from exc

    def verify_idle_state(self) -> None:
        """Refresh the full static/dynamic safety gate between control handles."""

        with self._configuration_lock:
            self._require_configuration_idle()
            state = self.robot.read_once()
            self._validate_state(
                state, require_idle=True, enforce_success=False
            )
            self.publish_native_telemetry_state(state)

    def move_joints(self, q: Sequence[float]) -> np.ndarray:
        """Move through bounded cosine joint segments and verify every segment."""

        return self._move_joints_with_time_law(
            q,
            max_joint_velocity_rad_s=self.limits.max_joint_speed_rad_s,
            max_joint_acceleration_rad_s2=None,
            max_dynamic_segment_rad=self.limits.max_joint_segment_rad,
            min_segment_duration_s=self.limits.min_joint_duration_s,
        )

    def move_loaded_joints(
        self,
        q: Sequence[float],
        *,
        max_joint_velocity_rad_s: float,
        max_joint_acceleration_rad_s2: float,
        max_dynamic_segment_rad: float,
        min_segment_duration_s: float,
    ) -> np.ndarray:
        """Execute the loaded suffix with its exact audit-bound cosine law."""

        velocity = float(max_joint_velocity_rad_s)
        acceleration = float(max_joint_acceleration_rad_s2)
        segment = float(max_dynamic_segment_rad)
        minimum_duration = float(min_segment_duration_s)
        values = (velocity, acceleration, segment, minimum_duration)
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("loaded joint time-law values must be finite and positive")
        if velocity > self.limits.max_joint_speed_rad_s + 1.0e-15:
            raise ValueError("loaded joint velocity exceeds the driver safety limit")
        if segment > self.limits.max_joint_segment_rad + 1.0e-15:
            raise ValueError("loaded dynamic segment exceeds the driver safety limit")
        if minimum_duration + 1.0e-15 < self.limits.min_joint_duration_s:
            raise ValueError("loaded minimum segment duration is below the driver limit")
        with self._configuration_lock:
            self._require_configuration_idle()
            if (
                self._load_configuration_uncertain
                or self._external_load_binding_sha256 is None
                or self._expected_external_load_mass_kg <= 0.0
            ):
                raise RuntimeError(
                    "loaded joint motion requires a verified active external load"
                )
        return self._move_joints_with_time_law(
            q,
            max_joint_velocity_rad_s=velocity,
            max_joint_acceleration_rad_s2=acceleration,
            max_dynamic_segment_rad=segment,
            min_segment_duration_s=minimum_duration,
        )

    def _move_joints_with_time_law(
        self,
        q: Sequence[float],
        *,
        max_joint_velocity_rad_s: float,
        max_joint_acceleration_rad_s2: Optional[float],
        max_dynamic_segment_rad: float,
        min_segment_duration_s: float,
    ) -> np.ndarray:
        """Shared stop-to-stop cosine implementation with explicit limits."""

        try:
            target = _finite_vector(q, 7, "joint target")
            self._validate_joint_target(target)
            completed = 0
            while True:
                state = self.robot.read_once()
                self._validate_state(state, require_idle=True)
                start = _finite_vector(getattr(state, "q", None), 7, "state.q")
                remaining = target - start
                max_delta = float(np.max(np.abs(remaining)))
                # Reuse the already-read, fully validated idle boundary
                # sample.  This is deliberately not another Robot.read_once.
                self.publish_native_telemetry_state(state)
                if max_delta <= self.limits.joint_arrival_tolerance_rad:
                    return start
                completed += 1
                if completed > self.limits.max_dynamic_segments:
                    raise RuntimeError("joint motion exceeded dynamic segment limit")
                fraction = min(
                    1.0, float(max_dynamic_segment_rad) / max_delta
                )
                segment_target = start + fraction * remaining
                self._validate_joint_target(segment_target)
                segment_delta = float(np.max(np.abs(segment_target - start)))
                duration = max(
                    float(min_segment_duration_s),
                    COSINE_PEAK_VELOCITY_FACTOR
                    * segment_delta
                    / float(max_joint_velocity_rad_s),
                )
                if max_joint_acceleration_rad_s2 is not None:
                    duration = max(
                        duration,
                        math.sqrt(
                            COSINE_PEAK_ACCELERATION_FACTOR
                            * segment_delta
                            / float(max_joint_acceleration_rad_s2)
                        ),
                    )
                self._run_joint_segment(start, segment_target, duration)
                final_state = self.robot.read_once()
                self._validate_state(final_state, require_idle=True)
                actual = _finite_vector(
                    getattr(final_state, "q", None), 7, "state.q"
                )
                tracking_error = float(np.max(np.abs(actual - segment_target)))
                if tracking_error > self.limits.joint_arrival_tolerance_rad:
                    raise RuntimeError(
                        "joint segment tracking error "
                        f"{tracking_error:.5f}rad exceeds "
                        f"{self.limits.joint_arrival_tolerance_rad:.5f}rad"
                    )
                # Publish only after the final sample has also passed the
                # segment-specific target tracking check.
                self.publish_native_telemetry_state(final_state)
        except BaseException:
            self._safe_stop()
            raise

    def move_pose(self, T: np.ndarray) -> np.ndarray:
        """Move to ``T_robot_base_EE`` using dynamically bounded SE(3) segments."""

        try:
            self._require_cartesian_workspace()
            target = _coerce_pose_matrix(T, "T_robot_base_EE target")
            self._validate_workspace(target[:3, 3], "Cartesian target")
            completed = 0
            while True:
                state = self.robot.read_once()
                self._validate_state(state, require_idle=True)
                start = _state_pose(state, "O_T_EE")
                self._validate_workspace(start[:3, 3], "actual EEF start")
                translation_error, rotation_error = pose_error(start, target)
                # Preserve the one-read boundary cadence: this publishes the
                # state above and never performs a second FCI read.
                self.publish_native_telemetry_state(state)
                if (
                    translation_error
                    <= self.limits.translation_arrival_tolerance_m
                    and rotation_error <= self.limits.rotation_arrival_tolerance_rad
                ):
                    return start
                completed += 1
                if completed > self.limits.max_dynamic_segments:
                    raise RuntimeError("Cartesian motion exceeded dynamic segment limit")
                segment_target = bounded_pose_step(start, target, self.limits)
                self._validate_workspace(
                    segment_target[:3, 3], "Cartesian segment target"
                )
                segment_translation, segment_rotation = pose_error(
                    start, segment_target
                )
                if (
                    segment_translation
                    > self.limits.max_segment_translation_m
                    or segment_rotation
                    > self.limits.max_segment_rotation_rad
                ):
                    raise RuntimeError("internal error: Cartesian segment exceeds limit")
                duration = max(
                    self.limits.min_cartesian_duration_s,
                    MINIMUM_JERK_PEAK_VELOCITY_FACTOR
                    * segment_translation
                    / self.limits.max_cartesian_speed_m_s,
                    MINIMUM_JERK_PEAK_VELOCITY_FACTOR
                    * segment_rotation
                    / self.limits.max_angular_speed_rad_s,
                )
                self._run_pose_segment(start, segment_target, duration)
                final_state = self.robot.read_once()
                self._validate_state(final_state, require_idle=True)
                actual = _state_pose(final_state, "O_T_EE")
                self._validate_workspace(actual[:3, 3], "actual EEF pose")
                translation_tracking, rotation_tracking = pose_error(
                    actual, segment_target
                )
                if (
                    translation_tracking
                    > self.limits.translation_arrival_tolerance_m
                    or rotation_tracking > self.limits.rotation_arrival_tolerance_rad
                ):
                    raise RuntimeError(
                        "Cartesian segment tracking error exceeds tolerance: "
                        f"translation={translation_tracking:.6f}m, "
                        f"rotation={rotation_tracking:.6f}rad"
                    )
                # The sample is safe, in workspace, and within the exact
                # segment tracking tolerances before it becomes observable.
                self.publish_native_telemetry_state(final_state)
        except BaseException:
            self._safe_stop()
            raise

    def hold_pose_control(self, T: np.ndarray, duration_s: float) -> np.ndarray:
        """Exercise one active Cartesian handle without commanding displacement.

        This diagnostic path is intentionally distinct from :meth:`move_pose`:
        ``move_pose(current_pose)`` returns before opening an FCI handle and
        therefore cannot qualify or observe command delivery.  Here the live
        stationary pose must already match the reviewed target, then the
        normal Cartesian loop holds its first active ``O_T_EE_c`` for the
        requested interval.  Every startup, realtime, workspace, contact,
        collision, success-rate, endpoint and stop gate remains unchanged.
        """

        try:
            self._require_cartesian_workspace()
            duration = float(duration_s)
            if not np.isfinite(duration) or not 1.0 <= duration <= 10.0:
                raise ValueError(
                    "Cartesian hold diagnostic duration must be in [1, 10] seconds"
                )
            reviewed = _coerce_pose_matrix(T, "Cartesian hold reviewed pose")
            self._validate_workspace(reviewed[:3, 3], "Cartesian hold reviewed pose")
            state = self.robot.read_once()
            self._validate_state(state, require_idle=True)
            live = _state_pose(state, "O_T_EE")
            self._validate_workspace(live[:3, 3], "Cartesian hold live pose")
            self._validate_cartesian_start_tracking(
                reviewed,
                live,
                "Cartesian hold live pose versus reviewed pose",
            )
            self.publish_native_telemetry_state(state)
            # Both endpoints are the exact live pose.  The active loop will
            # independently bind and command its first O_T_EE_c, so this does
            # not introduce a correction toward the reviewed YAML pose.
            self._run_pose_segment(live, live, duration)
            final_state = self.robot.read_once()
            self._validate_state(final_state, require_idle=True)
            final_pose = _state_pose(final_state, "O_T_EE")
            self._validate_workspace(
                final_pose[:3, 3], "Cartesian hold final pose"
            )
            self._validate_cartesian_start_tracking(
                live,
                final_pose,
                "Cartesian hold final pose versus live start",
            )
            self.publish_native_telemetry_state(final_state)
            return final_pose
        except BaseException:
            self._safe_stop()
            raise

    def verify_settled(self, T: np.ndarray, tolerances: Any = None) -> bool:
        """Require consecutive safe, accurate, low-velocity samples for a duration."""

        try:
            self._require_cartesian_workspace()
            target = _coerce_pose_matrix(T, "T_robot_base_EE settle target")
            self._validate_workspace(target[:3, 3], "settle target")
            return self._verify_settle_samples(
                target,
                tolerances,
                target_q=None,
                q_tolerance_rad=None,
                validate_workspace=True,
            )
        except BaseException:
            self._safe_stop()
            raise

    def verify_audited_joint_settled(
        self,
        target_q: Sequence[float],
        target_pose: np.ndarray,
        q_tolerance_rad: float,
        tolerances: Any = None,
    ) -> bool:
        """Verify an audit-bound joint/FK/EEF target without Cartesian motion.

        This method is intentionally distinct from :meth:`verify_settled`.
        It may be used only after an exact schema-v2 installed-tool audit has
        bound ``target_q`` to ``target_pose``.  It sends no command: each state
        sample must independently match both the bound joint vector and the
        corresponding EEF pose while remaining stationary.  Cartesian motion
        still requires explicit workspace bounds through :meth:`move_pose`.
        """

        try:
            q = _finite_vector(target_q, 7, "audited settle target_q")
            self._validate_joint_target(q)
            pose = _coerce_pose_matrix(
                target_pose, "audited settle T_robot_base_EE target"
            )
            q_tolerance = float(q_tolerance_rad)
            commissioned = self.limits.max_continuous_joint_tracking_error_rad
            if (
                not np.isfinite(q_tolerance)
                or not 0.0 < q_tolerance <= 0.01
                or commissioned is None
                or not np.isclose(
                    q_tolerance, float(commissioned), atol=1.0e-15, rtol=0.0
                )
            ):
                raise ValueError(
                    "audited settle q tolerance must exactly equal the "
                    "audit-bound continuous tracking tolerance"
                )
            return self._verify_settle_samples(
                pose,
                tolerances,
                target_q=q,
                q_tolerance_rad=q_tolerance,
                validate_workspace=False,
            )
        except BaseException:
            self._safe_stop()
            raise

    def _verify_settle_samples(
        self,
        target_pose: np.ndarray,
        tolerances: Any,
        *,
        target_q: Optional[np.ndarray],
        q_tolerance_rad: Optional[float],
        validate_workspace: bool,
    ) -> bool:
        (
            position_tolerance,
            orientation_tolerance,
            stable_seconds,
            linear_speed_tolerance,
            angular_speed_tolerance,
        ) = self._settle_parameters(tolerances)
        start_wall = self._clock()
        deadline = start_wall + max(
            self.limits.settle_timeout_s,
            stable_seconds + self.limits.settle_poll_s,
        )
        good_since: Optional[float] = None
        while True:
            now = self._clock()
            if now > deadline:
                return False
            state = self.robot.read_once()
            self._validate_state(state, require_idle=True)
            actual_pose = _state_pose(state, "O_T_EE")
            if validate_workspace:
                self._validate_workspace(
                    actual_pose[:3, 3], "settle actual EEF"
                )
            translation_error, rotation_error = pose_error(
                actual_pose, target_pose
            )
            actual_q = _finite_vector(getattr(state, "q", None), 7, "state.q")
            joint_good = True
            if target_q is not None:
                assert q_tolerance_rad is not None
                joint_good = (
                    float(np.max(np.abs(actual_q - target_q)))
                    <= q_tolerance_rad
                )
            dq = _finite_vector(getattr(state, "dq", None), 7, "state.dq")
            cartesian_speed_good = True
            twist_value = getattr(state, "O_dP_EE_c", None)
            if twist_value is not None and (
                linear_speed_tolerance is not None
                or angular_speed_tolerance is not None
            ):
                twist = _finite_vector(twist_value, 6, "state.O_dP_EE_c")
                if linear_speed_tolerance is not None:
                    cartesian_speed_good = cartesian_speed_good and (
                        float(np.linalg.norm(twist[:3]))
                        <= linear_speed_tolerance
                    )
                if angular_speed_tolerance is not None:
                    cartesian_speed_good = cartesian_speed_good and (
                        float(np.linalg.norm(twist[3:]))
                        <= angular_speed_tolerance
                    )
            good = (
                joint_good
                and translation_error <= position_tolerance
                and rotation_error <= orientation_tolerance
                and float(np.max(np.abs(dq)))
                <= self.limits.settle_max_dq_rad_s
                and cartesian_speed_good
            )
            # ``good`` may be false because the arm has not settled yet, but
            # every structural, static, dynamic, workspace, and finite-value
            # validation for this measured sample has passed.  Publishing it
            # here keeps the EEF-settling display live without another read.
            self.publish_native_telemetry_state(state)
            now = self._clock()
            if good:
                if good_since is None:
                    good_since = now
                if now - good_since >= stable_seconds:
                    return True
            else:
                good_since = None
            if now >= deadline:
                return False
            self._sleep(min(self.limits.settle_poll_s, deadline - now))

    def stop(self) -> None:
        """Command a stop, then prove physical rest from fresh state samples.

        Calling ``Robot.stop()`` is unconditional and always precedes every
        read-only verification step.  A successful return from libfranka is
        not by itself treated as proof that the robot is stationary: several
        consecutive fresh states must be Idle, free of errors/contact/
        collision, and below the configured joint-speed ceiling.  Verification
        failure never suppresses or delays the initial stop command and is
        surfaced with an explicit ``STOP UNCONFIRMED`` marker.
        """

        try:
            self.robot.stop()
        except BaseException as exc:
            raise FrankaStopUnconfirmed(
                "STOP UNCONFIRMED: Franka Robot.stop() failed: {}".format(exc)
            ) from exc
        try:
            self._verify_physical_stop()
        except FrankaStopUnconfirmed:
            raise
        except BaseException as exc:
            raise FrankaStopUnconfirmed(
                "STOP UNCONFIRMED: Franka post-stop verification aborted: {}".format(
                    exc
                )
            ) from exc

    def _verify_physical_stop(self) -> None:
        """Require consecutive safe, stationary states after ``Robot.stop()``."""

        required = int(self.limits.stop_verify_consecutive_samples)
        deadline = self._clock() + float(self.limits.stop_verify_timeout_s)
        consecutive = 0
        samples = 0
        last_problem = "no fresh Franka state was received"
        last_max_dq = float("nan")

        while True:
            samples += 1
            try:
                state = self.robot.read_once()
                last_max_dq = self._validate_stopped_state(state)
                # Post-stop telemetry is exposed only after the direct
                # physical-rest evidence for this exact sample has passed.
                self.publish_native_telemetry_state(state)
            except Exception as exc:
                consecutive = 0
                last_problem = str(exc) or type(exc).__name__
            else:
                consecutive += 1
                last_problem = ""
                if consecutive >= required:
                    return

            now = self._clock()
            if now >= deadline:
                dq_detail = (
                    "n/a"
                    if not np.isfinite(last_max_dq)
                    else "{:.9f}rad/s".format(last_max_dq)
                )
                raise FrankaStopUnconfirmed(
                    "STOP UNCONFIRMED: Franka physical rest was not verified; "
                    "required={} consecutive samples, obtained={}, samples={}, "
                    "max_dq_limit={:.9f}rad/s, last_max_dq={}, last_problem={}".format(
                        required,
                        consecutive,
                        samples,
                        float(self.limits.stop_verify_max_dq_rad_s),
                        dq_detail,
                        last_problem or "none",
                    )
                )
            self._sleep(min(float(self.limits.stop_verify_poll_s), deadline - now))

    def _validate_stopped_state(self, state: Any) -> float:
        """Validate only direct physical-stop evidence, not motion provenance."""

        mode_value = getattr(state, "robot_mode", None)
        mode = str(getattr(mode_value, "name", mode_value or "unknown")).lower()
        mode = mode.rsplit(".", 1)[-1]
        if mode in ("kidle", "kmove"):
            mode = mode[1:]
        if mode != "idle":
            raise RuntimeError(
                "Franka post-stop robot mode is {}, expected idle".format(mode)
            )

        errors = getattr(state, "current_errors", None)
        if errors is None:
            raise RuntimeError("Franka post-stop state has no current_errors")
        active_errors = _active_error_names(errors)
        if active_errors:
            raise RuntimeError(
                "Franka post-stop current_errors are active: {}".format(
                    active_errors
                )
            )

        for field_name, expected_length in (
            ("cartesian_contact", 6),
            ("cartesian_collision", 6),
            ("joint_contact", 7),
            ("joint_collision", 7),
        ):
            values = getattr(state, field_name, None)
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (expected_length,) or not np.all(np.isfinite(array)):
                raise RuntimeError(
                    "Franka post-stop {} is missing or malformed".format(field_name)
                )
            if np.any(array > 0.5):
                raise RuntimeError(
                    "Franka post-stop reports {}: {}".format(
                        field_name, array.tolist()
                    )
                )

        try:
            dq = _finite_vector(getattr(state, "dq", None), 7, "state.dq")
        except (TypeError, ValueError) as exc:
            raise RuntimeError("invalid Franka post-stop state.dq: {}".format(exc)) from exc
        max_dq = float(np.max(np.abs(dq)))
        if max_dq > float(self.limits.stop_verify_max_dq_rad_s):
            raise RuntimeError(
                "Franka post-stop max |dq| {:.9f}rad/s exceeds {:.9f}rad/s".format(
                    max_dq, float(self.limits.stop_verify_max_dq_rad_s)
                )
            )
        return max_dq

    def _run_joint_segment(
        self, start: np.ndarray, target: np.ndarray, duration: float
    ) -> None:
        with self._configuration_lock:
            self._require_configuration_idle()
            self._control_handle_active = True
            try:
                self._run_joint_segment_locked(start, target, duration)
            finally:
                self._control_handle_active = False

    def _run_joint_segment_locked(
        self, start: np.ndarray, target: np.ndarray, duration: float
    ) -> None:
        # Static tool/load provenance and all dynamic preconditions are bound
        # immediately before every new control handle.  Only values that can
        # change through an active handle stay on the 1 kHz path below.
        preflight = self.robot.read_once()
        realtime_validator, _ = self._prepare_realtime_validation(
            preflight, require_idle=True
        )
        self.publish_native_telemetry_state(preflight)
        trajectory_elapsed = 0.0
        allowed_initial_zero = True
        wall_start = self._clock()
        deadline = self._motion_deadline(
            wall_start,
            duration + self.limits.control_success_evaluation_window_s,
        )
        planned_start_values = tuple(float(value) for value in start)
        target_values = tuple(float(value) for value in target)
        active_start_values: Optional[Tuple[float, ...]] = None
        command_delta: Optional[Tuple[float, ...]] = None
        command_q = list(planned_start_values)
        previous_command_q: Sequence[float] = planned_start_values
        command_type = self.pylibfranka.JointPositions
        reuse_command = _is_native_pylibfranka_command_type(
            command_type, "JointPositions"
        )
        reusable_command = command_type(command_q) if reuse_command else None
        # A new libfranka handle can initially expose an unfilled rolling-100
        # delivery metric.  Hold its exact active q_d until at least 100
        # positive-period commands have completed and a subsequent state
        # reports the commissioned quality threshold.  Only then may the
        # trajectory clock and normal watchdog start.
        success_qualification = _ControlSuccessStartupQualification(self.limits)
        success_watchdog = _ControlSuccessWatchdog(self.limits)
        sample_count = 0
        maximum_read_to_write_ns = 0
        read_to_write_overruns = 0
        control_clock_ns = self._control_clock_ns
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            # Collect before the handle exists, then prevent an unrelated
            # cyclic-GC pause inside readOnce -> writeOnce.  Reference counting
            # remains active and the prior GC state is restored in all paths.
            gc.collect()
            gc.disable()
        try:
            control = self.robot.start_joint_position_control(
                self.pylibfranka.ControllerMode.JointImpedance
            )
        except BaseException:
            if gc_was_enabled:
                gc.enable()
            raise
        try:
            telemetry_tap = self._telemetry_tap
            qualification_wall_start = self._clock()
            while trajectory_elapsed < duration:
                if telemetry_tap is None:
                    state, period = control.readOnce()
                else:
                    state, period = telemetry_tap.read_once_tapped(control)
                cycle_started_ns = control_clock_ns()
                wall_now = self._clock()
                if wall_now > deadline:
                    raise RuntimeError(
                        "Franka joint segment exceeded wall-clock deadline"
                    )
                dt, allowed_initial_zero = _validated_control_period(
                    period, allowed_initial_zero
                )
                tracking_limit = self.limits.max_continuous_joint_tracking_error_rad
                success = realtime_validator.validate(
                    state,
                    require_idle=False,
                    previous_command_q=(
                        previous_command_q if tracking_limit is not None else None
                    ),
                    tracking_limit_rad=tracking_limit,
                )
                if active_start_values is None:
                    active_start = _finite_vector(
                        getattr(state, "q_d", None), 7, "state.q_d"
                    )
                    self._validate_joint_target(active_start)
                    measured_start = _finite_vector(
                        getattr(state, "q", None), 7, "state.q"
                    )
                    planned_error = float(
                        np.max(
                            np.abs(
                                active_start
                                - np.asarray(planned_start_values, dtype=np.float64)
                            )
                        )
                    )
                    active_tracking_limit = (
                        float(tracking_limit)
                        if tracking_limit is not None
                        else float(self.limits.joint_arrival_tolerance_rad)
                    )
                    measured_error = float(np.max(np.abs(measured_start - active_start)))
                    if planned_error > float(self.limits.joint_arrival_tolerance_rad):
                        raise RuntimeError(
                            "active desired joint start versus planned start exceeds "
                            "joint arrival bound: "
                            f"error={planned_error:.9f}rad, bound="
                            f"{self.limits.joint_arrival_tolerance_rad:.9f}rad"
                        )
                    if measured_error > active_tracking_limit:
                        raise RuntimeError(
                            "active desired joint start versus measured start exceeds "
                            "joint tracking bound: "
                            f"error={measured_error:.9f}rad, "
                            f"bound={active_tracking_limit:.9f}rad"
                        )
                    active_start_values = tuple(float(value) for value in active_start)
                    command_delta = tuple(
                        target_values[index] - active_start_values[index]
                        for index in range(7)
                    )
                    command_q[:] = active_start_values
                    previous_command_q = active_start_values

                if not success_qualification.qualified:
                    qualification_candidate = success_qualification.observe_read(
                        success,
                        dt,
                        wall_now - qualification_wall_start,
                    )
                    command_q[:] = active_start_values
                    if reusable_command is not None:
                        reusable_command.q = command_q
                        control.writeOnce(reusable_command)
                    else:
                        control.writeOnce(command_type(command_q))
                    cycle_duration_ns = control_clock_ns() - cycle_started_ns
                    if cycle_duration_ns < 0:
                        raise RuntimeError("control timing clock moved backwards")
                    sample_count += 1
                    if cycle_duration_ns > maximum_read_to_write_ns:
                        maximum_read_to_write_ns = cycle_duration_ns
                    if cycle_duration_ns > FCI_READ_TO_WRITE_BUDGET_NS:
                        read_to_write_overruns += 1
                    success_qualification.record_hold_write(
                        positive_period=dt > 0.0,
                        candidate=qualification_candidate,
                        success=success,
                        wall_time_s=self._clock() - qualification_wall_start,
                    )
                    previous_command_q = active_start_values
                    continue

                if dt > 0.0:
                    success_watchdog.observe(success, dt)
                    trajectory_elapsed += dt
                alpha = min(1.0, trajectory_elapsed / duration)
                blend = 0.5 - 0.5 * math.cos(math.pi * alpha)
                assert active_start_values is not None and command_delta is not None
                index = 0
                while index < 7:
                    command_q[index] = (
                        active_start_values[index] + blend * command_delta[index]
                    )
                    index += 1
                if reusable_command is not None:
                    reusable_command.q = command_q
                    control.writeOnce(reusable_command)
                else:
                    control.writeOnce(command_type(command_q))
                cycle_duration_ns = control_clock_ns() - cycle_started_ns
                if cycle_duration_ns < 0:
                    raise RuntimeError("control timing clock moved backwards")
                sample_count += 1
                if cycle_duration_ns > maximum_read_to_write_ns:
                    maximum_read_to_write_ns = cycle_duration_ns
                if cycle_duration_ns > FCI_READ_TO_WRITE_BUDGET_NS:
                    read_to_write_overruns += 1
                previous_command_q = command_q
            if reusable_command is not None:
                reusable_command.q = list(target_values)
                final_command = reusable_command
            else:
                final_command = command_type(list(target_values))
            final_command.motion_finished = True
            control.writeOnce(final_command)
        finally:
            self._publish_control_loop_telemetry(
                "joint",
                sample_count,
                maximum_read_to_write_ns,
                read_to_write_overruns,
                success_qualification,
            )
            if gc_was_enabled:
                gc.enable()

    def _run_pose_segment(
        self, start: np.ndarray, target: np.ndarray, duration: float
    ) -> None:
        with self._configuration_lock:
            self._require_configuration_idle()
            self._control_handle_active = True
            try:
                self._run_pose_segment_locked(start, target, duration)
            finally:
                self._control_handle_active = False

    def _run_pose_segment_locked(
        self, start: np.ndarray, target: np.ndarray, duration: float
    ) -> None:
        planned_start = _coerce_pose_matrix(start, "Cartesian segment planned start")
        target_pose = _coerce_pose_matrix(target, "Cartesian interpolation target")
        target_values = _matrix_to_franka_pose(target_pose)
        preflight = self.robot.read_once()
        realtime_validator, _ = self._prepare_realtime_validation(
            preflight, require_idle=True
        )
        self.publish_native_telemetry_state(preflight)
        interpolation: Optional[_PreparedCartesianInterpolation] = None
        active_start_values: Optional[List[float]] = None
        command_values = [0.0] * 16
        command_type = self.pylibfranka.CartesianPose
        reuse_command = _is_native_pylibfranka_command_type(
            command_type, "CartesianPose"
        )
        reusable_command = (
            command_type(target_values) if reuse_command else None
        )
        trajectory_elapsed = 0.0
        endpoint_elapsed = 0.0
        endpoint_stable_elapsed = 0.0
        endpoint_wall_start: Optional[float] = None
        allowed_initial_zero = True
        wall_start = self._clock()
        # A non-zero hold is an absolute convergence timeout, not a blind
        # dwell.  During that interval the exact target remains commanded
        # until measured pose and dq have stayed inside the configured arrival
        # and settle bounds continuously for settle_time_s.
        endpoint_timeout = self.limits.cartesian_endpoint_hold_s
        deadline = self._motion_deadline(
            wall_start,
            duration
            + endpoint_timeout
            + self.limits.control_success_evaluation_window_s,
        )
        # Hold the exact first active O_T_EE_c while the new handle's
        # rolling-100 delivery metric fills; see the joint-loop comment.
        success_qualification = _ControlSuccessStartupQualification(self.limits)
        success_watchdog = _ControlSuccessWatchdog(self.limits)
        sample_count = 0
        maximum_read_to_write_ns = 0
        read_to_write_overruns = 0
        control_clock_ns = self._control_clock_ns
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.collect()
            gc.disable()
        controller_mode = (
            self.pylibfranka.ControllerMode.CartesianImpedance
            if self.limits.cartesian_pose_controller_mode == "cartesian_impedance"
            else self.pylibfranka.ControllerMode.JointImpedance
        )
        try:
            control = self.robot.start_cartesian_pose_control(controller_mode)
        except BaseException:
            if gc_was_enabled:
                gc.enable()
            raise
        try:
            telemetry_tap = self._telemetry_tap
            qualification_wall_start = self._clock()
            while True:
                if telemetry_tap is None:
                    state, period = control.readOnce()
                else:
                    state, period = telemetry_tap.read_once_tapped(control)
                cycle_started_ns = control_clock_ns()
                wall_now = self._clock()
                if wall_now > deadline:
                    raise RuntimeError(
                        "Franka Cartesian segment exceeded wall-clock deadline"
                    )
                dt, allowed_initial_zero = _validated_control_period(
                    period, allowed_initial_zero
                )
                success = realtime_validator.validate(state, require_idle=False)
                # Keep the exact buffer whose 16 entries were just checked for
                # finiteness and whose translation passed the workspace gate.
                # pylibfranka exposes std::array fields through a Python
                # sequence conversion, so reading O_T_EE again in the endpoint
                # hold creates another converted buffer unnecessarily.
                actual_pose_values = self._validate_realtime_cartesian_translation(
                    state
                )
                if interpolation is None:
                    # The active handle's first commanded pose is the only
                    # authoritative interpolation origin.  Binding to the
                    # outer read alone can introduce a discontinuity when
                    # libfranka's command low-pass filter is lagging actual
                    # O_T_EE.  All three start relationships must still fit
                    # the existing segment tracking envelope.
                    commanded_start = _state_pose(state, "O_T_EE_c")
                    measured_start = _state_pose(state, "O_T_EE")
                    self._validate_workspace(
                        commanded_start[:3, 3], "active commanded Cartesian start"
                    )
                    self._validate_cartesian_start_tracking(
                        planned_start,
                        commanded_start,
                        "active commanded start versus planned start",
                    )
                    self._validate_cartesian_start_tracking(
                        planned_start,
                        measured_start,
                        "active measured start versus planned start",
                    )
                    self._validate_cartesian_start_tracking(
                        measured_start,
                        commanded_start,
                        "active commanded start versus measured start",
                    )
                    interpolation = _PreparedCartesianInterpolation(
                        commanded_start, target_pose
                    )
                    active_start_values = _matrix_to_franka_pose(commanded_start)

                if not success_qualification.qualified:
                    qualification_candidate = success_qualification.observe_read(
                        success,
                        dt,
                        wall_now - qualification_wall_start,
                    )
                    assert active_start_values is not None
                    command_values[:] = active_start_values
                    self._validate_workspace_scalars(
                        command_values[12],
                        command_values[13],
                        command_values[14],
                        "Cartesian startup hold command",
                    )
                    if reusable_command is not None:
                        reusable_command.O_T_EE = command_values
                        control.writeOnce(reusable_command)
                    else:
                        control.writeOnce(command_type(command_values))
                    cycle_duration_ns = control_clock_ns() - cycle_started_ns
                    if cycle_duration_ns < 0:
                        raise RuntimeError("control timing clock moved backwards")
                    sample_count += 1
                    if cycle_duration_ns > maximum_read_to_write_ns:
                        maximum_read_to_write_ns = cycle_duration_ns
                    if cycle_duration_ns > FCI_READ_TO_WRITE_BUDGET_NS:
                        read_to_write_overruns += 1
                    success_qualification.record_hold_write(
                        positive_period=dt > 0.0,
                        candidate=qualification_candidate,
                        success=success,
                        wall_time_s=self._clock() - qualification_wall_start,
                    )
                    continue

                if dt > 0.0:
                    success_watchdog.observe(success, dt)

                if trajectory_elapsed < duration:
                    if dt > 0.0:
                        trajectory_elapsed += dt
                    alpha = min(1.0, trajectory_elapsed / duration)
                    interpolation.fill_command_values(alpha, command_values)
                else:
                    if endpoint_timeout <= 0.0:
                        break
                    if endpoint_wall_start is None:
                        endpoint_wall_start = self._clock()
                    if dt > 0.0:
                        endpoint_elapsed += dt
                    translation_error, rotation_error = (
                        _realtime_endpoint_pose_error(
                            actual_pose_values,
                            target_values,
                        )
                    )
                    max_dq = _validated_realtime_max_abs_dq(
                        getattr(state, "dq", None)
                    )
                    endpoint_good = (
                        translation_error
                        <= self.limits.translation_arrival_tolerance_m
                        and rotation_error
                        <= self.limits.rotation_arrival_tolerance_rad
                        and max_dq <= self.limits.settle_max_dq_rad_s
                    )
                    if endpoint_good:
                        endpoint_stable_elapsed += dt
                        if (
                            endpoint_stable_elapsed + 1.0e-12
                            >= self.limits.settle_time_s
                        ):
                            break
                    else:
                        endpoint_stable_elapsed = 0.0
                    endpoint_wall_elapsed = self._clock() - endpoint_wall_start
                    if (
                        endpoint_elapsed + 1.0e-12 >= endpoint_timeout
                        or endpoint_wall_elapsed + 1.0e-12 >= endpoint_timeout
                    ):
                        raise RuntimeError(
                            "Franka Cartesian endpoint convergence timed out: "
                            f"translation={translation_error:.6f}m, "
                            f"rotation={rotation_error:.6f}rad, "
                            f"max_dq={max_dq:.6f}rad/s, "
                            f"stable={endpoint_stable_elapsed:.6f}s, "
                            f"timeout={endpoint_timeout:.6f}s"
                        )
                    command_values[:] = target_values

                self._validate_workspace_scalars(
                    command_values[12],
                    command_values[13],
                    command_values[14],
                    "Cartesian command",
                )
                if reusable_command is not None:
                    reusable_command.O_T_EE = command_values
                    control.writeOnce(reusable_command)
                else:
                    control.writeOnce(command_type(command_values))
                cycle_duration_ns = control_clock_ns() - cycle_started_ns
                if cycle_duration_ns < 0:
                    raise RuntimeError("control timing clock moved backwards")
                sample_count += 1
                if cycle_duration_ns > maximum_read_to_write_ns:
                    maximum_read_to_write_ns = cycle_duration_ns
                if cycle_duration_ns > FCI_READ_TO_WRITE_BUDGET_NS:
                    read_to_write_overruns += 1
                if (
                    trajectory_elapsed >= duration
                    and endpoint_timeout > 0.0
                    and endpoint_wall_start is None
                ):
                    endpoint_wall_start = self._clock()
            if reusable_command is not None:
                reusable_command.O_T_EE = target_values
                final_command = reusable_command
            else:
                final_command = command_type(target_values)
            final_command.motion_finished = True
            control.writeOnce(final_command)
        finally:
            self._publish_control_loop_telemetry(
                "cartesian",
                sample_count,
                maximum_read_to_write_ns,
                read_to_write_overruns,
                success_qualification,
            )
            if gc_was_enabled:
                gc.enable()

    def _validate_cartesian_start_tracking(
        self, reference: np.ndarray, candidate: np.ndarray, label: str
    ) -> None:
        translation, rotation = pose_error(reference, candidate)
        if (
            translation > self.limits.translation_arrival_tolerance_m
            or rotation > self.limits.rotation_arrival_tolerance_rad
        ):
            raise RuntimeError(
                f"{label} exceeds Cartesian tracking bounds: "
                f"translation={translation:.6f}m, "
                f"rotation={rotation:.6f}rad"
            )

    def _validate_joint_target(self, q: np.ndarray) -> None:
        lower = self.limits.joint_limits_rad[:, 0] + self.limits.joint_limit_margin_rad
        upper = self.limits.joint_limits_rad[:, 1] - self.limits.joint_limit_margin_rad
        outside = np.flatnonzero((q < lower) | (q > upper))
        if len(outside):
            index = int(outside[0])
            raise ValueError(
                f"joint {index + 1} target {q[index]:.5f}rad is outside "
                f"[{lower[index]:.5f}, {upper[index]:.5f}]"
            )

    def _validate_state(
        self,
        state: Any,
        *,
        require_idle: bool,
        enforce_success: bool = False,
        enforce_joint_limit_margin: bool = True,
    ) -> float:
        _, success = self._prepare_realtime_validation(
            state,
            require_idle=require_idle,
            enforce_joint_limit_margin=enforce_joint_limit_margin,
        )
        success_threshold = float(self.limits.min_control_success_rate)
        if (
            enforce_success
            and success
            < success_threshold - CONTROL_SUCCESS_RATE_ABSOLUTE_TOLERANCE
        ):
            raise RuntimeError(
                "Franka control command success rate is below threshold: "
                f"actual={success:.9f}, threshold={success_threshold:.9f}, "
                "absolute_tolerance="
                f"{CONTROL_SUCCESS_RATE_ABSOLUTE_TOLERANCE:.9f}"
            )
        return success

    def _prepare_realtime_validation(
        self,
        state: Any,
        *,
        require_idle: bool,
        enforce_joint_limit_margin: bool = True,
    ) -> Tuple[_RealtimeStateValidator, float]:
        """Bind static provenance and return a validator for one handle.

        The caller must invoke this immediately before creating the libfranka
        control handle.  The application must retain sole Robot ownership and
        must not expose a concurrent configuration mutator; every genuinely
        dynamic safety value remains checked by
        :class:`_RealtimeStateValidator` for every state sample.
        """

        if not enforce_joint_limit_margin and not require_idle:
            raise RuntimeError(
                "joint-limit margin may be omitted only for a passive Idle read"
            )
        validator = _RealtimeStateValidator(self.limits, state)
        success = validator.validate(
            state,
            require_idle=require_idle,
            enforce_joint_limit_margin=enforce_joint_limit_margin,
        )
        self._validate_static_state(state)
        return validator, success

    def _validate_static_state(self, state: Any) -> None:
        """Validate full poses and commissioned tool/load provenance."""

        _state_pose(state, "O_T_EE")
        actual_F_T_EE = _state_pose(state, "F_T_EE")
        if not np.allclose(
            actual_F_T_EE,
            self.limits.expected_F_T_EE,
            atol=self.limits.F_T_EE_tolerance,
            rtol=0.0,
        ):
            max_error = float(
                np.max(np.abs(actual_F_T_EE - self.limits.expected_F_T_EE))
            )
            raise RuntimeError(
                "Franka F_T_EE differs from the commissioned transform; "
                f"max element error={max_error:.3e}"
            )

        if self.limits.expected_m_ee_kg is not None:
            actual_mass = float(getattr(state, "m_ee", np.nan))
            if not np.isfinite(actual_mass) or not np.isclose(
                actual_mass,
                self.limits.expected_m_ee_kg,
                atol=self.limits.mass_tolerance_kg,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "Franka m_ee differs from commissioned adapter + hand mass: "
                    f"expected={self.limits.expected_m_ee_kg:.8g}kg, "
                    f"actual={actual_mass!r}kg"
                )
            actual_com = _finite_vector(
                getattr(state, "F_x_Cee", None), 3, "state.F_x_Cee"
            )
            if not np.allclose(
                actual_com,
                self.limits.expected_F_x_Cee_m,
                atol=self.limits.center_of_mass_tolerance_m,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "Franka F_x_Cee differs from the commissioned adapter + hand CoM"
                )
            actual_inertia = _coerce_inertia_matrix(
                getattr(state, "I_ee", None), "state.I_ee"
            )
            if not np.allclose(
                actual_inertia,
                self.limits.expected_I_ee_kg_m2,
                atol=self.limits.inertia_tolerance_kg_m2,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "Franka I_ee differs from the commissioned adapter + hand inertia"
                )
            actual_load_mass = float(getattr(state, "m_load", np.nan))
            expected_load_mass = float(self._expected_external_load_mass_kg)
            if self._load_configuration_uncertain:
                raise RuntimeError(
                    "Franka external load configuration is unverified"
                )
            if not np.isfinite(actual_load_mass) or not np.isclose(
                actual_load_mass,
                expected_load_mass,
                atol=self.limits.mass_tolerance_kg,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "Franka external load mass differs from the active audited load: "
                    f"m_load={actual_load_mass!r}kg; "
                    f"expected={expected_load_mass:.8g}kg"
                )
            if expected_load_mass > 0.0:
                actual_load_com = _finite_vector(
                    getattr(state, "F_x_Cload", None), 3, "state.F_x_Cload"
                )
                if not np.allclose(
                    actual_load_com,
                    self._expected_F_x_Cload_m,
                    atol=self.limits.center_of_mass_tolerance_m,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        "Franka F_x_Cload differs from the active audited payload"
                    )
                actual_load_inertia = _coerce_inertia_matrix(
                    getattr(state, "I_load", None), "state.I_load"
                )
                if not np.allclose(
                    actual_load_inertia,
                    self._expected_I_load_kg_m2,
                    atol=self.limits.inertia_tolerance_kg_m2,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        "Franka I_load differs from the active audited payload"
                    )
            actual_total_mass = float(getattr(state, "m_total", np.nan))
            expected_total_mass = (
                float(self.limits.expected_m_ee_kg) + expected_load_mass
            )
            if not np.isfinite(actual_total_mass) or not np.isclose(
                actual_total_mass,
                expected_total_mass,
                atol=self.limits.mass_tolerance_kg,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "Franka total mass differs from the commissioned end effector "
                    "plus active audited external load: "
                    f"expected={expected_total_mass:.8g}kg, "
                    f"actual={actual_total_mass!r}kg"
                )


    def _validate_realtime_cartesian_translation(
        self, state: Any
    ) -> Sequence[float]:
        """Check live EEF translation and return the validated 16-value buffer.

        The returned sequence is the same flattened buffer inspected here when
        pylibfranka already supplied a 16-vector.  Endpoint convergence can
        therefore retain the per-cycle finite/workspace gate without fetching
        and converting ``O_T_EE`` a second time.
        """

        values = getattr(state, "O_T_EE", None)
        try:
            shape = getattr(values, "shape", None)
            if shape == (4, 4):
                flattened = [
                    float(values[row, column])
                    for column in range(4)
                    for row in range(4)
                ]
            elif len(values) == 16 and shape in (None, (16,)):
                flattened = values
            else:
                raise RuntimeError("Franka O_T_EE is missing or malformed")
            # Validate all entries for finite transport corruption, but avoid
            # matrix products/determinants in the deadline-sensitive path.
            for index in range(16):
                if not math.isfinite(float(flattened[index])):
                    raise RuntimeError("Franka O_T_EE contains NaN or infinity")
            x = float(flattened[12])
            y = float(flattened[13])
            z = float(flattened[14])
        except (TypeError, ValueError, IndexError) as exc:
            raise RuntimeError("Franka O_T_EE is missing or malformed") from exc
        self._validate_workspace_scalars(x, y, z, "actual Cartesian EEF")
        return flattened

    def _require_cartesian_workspace(self) -> None:
        if (
            self.limits.workspace_min_m is None
            or self.limits.workspace_max_m is None
        ):
            raise RuntimeError(
                "Cartesian motion requires commissioned workspace_min_m and "
                "workspace_max_m"
            )

    def _settle_parameters(
        self, tolerances: Any
    ) -> Tuple[float, float, float, Optional[float], Optional[float]]:
        if tolerances is None:
            return (
                self.limits.translation_arrival_tolerance_m,
                self.limits.rotation_arrival_tolerance_rad,
                self.limits.settle_time_s,
                None,
                None,
            )

        def finite_nonnegative(name: str, fallback: Optional[float]) -> Optional[float]:
            value = getattr(tolerances, name, fallback)
            if value is None:
                return None
            numeric = float(value)
            if not np.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"settle tolerance {name} must be finite and non-negative")
            return numeric

        position = finite_nonnegative(
            "position_m", self.limits.translation_arrival_tolerance_m
        )
        orientation = finite_nonnegative(
            "orientation_rad", self.limits.rotation_arrival_tolerance_rad
        )
        stable = float(getattr(tolerances, "stable_seconds", self.limits.settle_time_s))
        if not np.isfinite(stable) or stable <= 0.0:
            raise ValueError("settle tolerance stable_seconds must be finite and positive")
        linear = finite_nonnegative("linear_speed_m_s", None)
        angular = finite_nonnegative("angular_speed_rad_s", None)
        # position/orientation cannot be None due to their finite fallbacks.
        return float(position), float(orientation), stable, linear, angular

    def _validate_workspace(self, xyz: np.ndarray, label: str) -> None:
        self._require_cartesian_workspace()
        point = _finite_vector(xyz, 3, label)
        if np.any(point < self.limits.workspace_min_m) or np.any(
            point > self.limits.workspace_max_m
        ):
            raise RuntimeError(
                f"{label} is outside the commissioned Cartesian workspace: "
                f"{point.tolist()}"
            )

    def _validate_workspace_scalars(
        self, x: float, y: float, z: float, label: str
    ) -> None:
        self._require_cartesian_workspace()
        x_value = float(x)
        y_value = float(y)
        z_value = float(z)
        if not (
            math.isfinite(x_value)
            and math.isfinite(y_value)
            and math.isfinite(z_value)
        ):
            raise RuntimeError(f"{label} contains NaN or infinity")
        lower = self.limits.workspace_min_m
        upper = self.limits.workspace_max_m
        assert lower is not None and upper is not None
        outside = (
            x_value < float(lower[0])
            or x_value > float(upper[0])
            or y_value < float(lower[1])
            or y_value > float(upper[1])
            or z_value < float(lower[2])
            or z_value > float(upper[2])
        )
        if outside:
            raise RuntimeError(
                f"{label} is outside the commissioned Cartesian workspace: "
                f"{[x_value, y_value, z_value]}"
            )

    def _publish_control_loop_telemetry(
        self,
        kind: str,
        samples: int,
        maximum_read_to_write_ns: int,
        read_to_write_overruns: int,
        success_qualification: _ControlSuccessStartupQualification,
    ) -> None:
        """Publish after a handle without ever masking its primary outcome."""

        try:
            self.last_control_loop_telemetry = FrankaControlLoopTelemetry(
                kind=str(kind),
                samples=int(samples),
                max_read_to_write_ns=int(maximum_read_to_write_ns),
                read_to_write_overruns=int(read_to_write_overruns),
                success_qualified=bool(success_qualification.qualified),
                success_qualification_positive_writes=int(
                    success_qualification.positive_writes
                ),
                success_qualification_control_time_s=float(
                    success_qualification.control_time_s
                ),
                success_qualification_wall_time_s=float(
                    success_qualification.wall_time_s
                ),
                success_qualification_rate=success_qualification.qualification_rate,
            )
        except Exception:
            # Timing is diagnostic, never a motion authorization or stop gate;
            # failure to allocate its one post-loop summary must not replace a
            # primary safety exception raised by the control path.
            self.last_control_loop_telemetry = None

    def _motion_deadline(self, start: float, duration: float) -> float:
        if not np.isfinite(duration) or duration <= 0.0:
            raise RuntimeError("computed Franka motion duration is invalid")
        return start + duration + max(
            self.limits.wall_deadline_slack_s,
            self.limits.wall_deadline_fraction * duration,
        )

    def _clock(self) -> float:
        value = float(self._monotonic())
        if not math.isfinite(value):
            raise RuntimeError("monotonic clock returned a non-finite value")
        return value

    def _safe_stop(self) -> None:
        # Do not suppress a failed stop acknowledgement.  Motion entry points
        # call this from their exception handlers, so a verification failure
        # deliberately replaces the less urgent motion error while preserving
        # it as exception context and exposing STOP UNCONFIRMED to every caller.
        self.stop()


def bounded_pose_step(
    T_start: np.ndarray, T_target: np.ndarray, limits: FrankaMotionLimits
) -> np.ndarray:
    """Return one synchronized pose step bounded in translation and rotation."""

    start = _coerce_pose_matrix(T_start, "T_start")
    target = _coerce_pose_matrix(T_target, "T_target")
    translation, rotation = pose_error(start, target)
    fractions = [1.0]
    if translation > 0.0:
        fractions.append(limits.max_segment_translation_m / translation)
    if rotation > 0.0:
        fractions.append(limits.max_segment_rotation_rad / rotation)
    fraction = min(1.0, max(0.0, min(fractions)))
    if fraction >= 1.0:
        return interpolate_pose_linear_se3(start, target, fraction)

    # Do not aim exactly at a hard segment boundary.  In particular, Franka
    # state rotations are commonly represented with float32-sized residual
    # non-orthonormality.  Re-evaluating the interpolated angle can therefore
    # differ from the angle used to compute ``fraction`` by a few nanoradians.
    fraction *= 1.0 - CARTESIAN_SEGMENT_INWARD_RELATIVE_MARGIN
    for _ in range(CARTESIAN_SEGMENT_MAX_RETRACTIONS):
        candidate = interpolate_pose_linear_se3(start, target, fraction)
        candidate_translation, candidate_rotation = pose_error(start, candidate)
        correction = 1.0
        if candidate_translation > limits.max_segment_translation_m:
            correction = min(
                correction,
                limits.max_segment_translation_m / candidate_translation,
            )
        if candidate_rotation > limits.max_segment_rotation_rad:
            correction = min(
                correction,
                limits.max_segment_rotation_rad / candidate_rotation,
            )
        if correction >= 1.0:
            return candidate
        fraction *= correction * (
            1.0 - CARTESIAN_SEGMENT_INWARD_RELATIVE_MARGIN
        )

    raise RuntimeError(
        "internal error: unable to construct Cartesian segment within limits"
    )


def cosine_interpolate(
    start: np.ndarray, target: np.ndarray, alpha: float
) -> np.ndarray:
    """Cosine joint interpolation with zero endpoint velocity."""

    start_array = np.asarray(start, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if start_array.shape != target_array.shape:
        raise ValueError("cosine interpolation endpoints must have equal shape")
    if not np.all(np.isfinite(start_array)) or not np.all(np.isfinite(target_array)):
        raise ValueError("cosine interpolation endpoints must be finite")
    bounded = min(1.0, max(0.0, float(alpha)))
    blend = 0.5 - 0.5 * math.cos(math.pi * bounded)
    return start_array + blend * (target_array - start_array)


def minimum_jerk_blend(alpha: float) -> float:
    bounded = min(1.0, max(0.0, float(alpha)))
    return bounded ** 3 * (10.0 + bounded * (-15.0 + 6.0 * bounded))


def interpolate_pose_minimum_jerk(
    T_start: np.ndarray, T_target: np.ndarray, alpha: float
) -> np.ndarray:
    """Synchronously interpolate translation and SO(3) with minimum jerk."""

    return interpolate_pose_linear_se3(
        T_start, T_target, minimum_jerk_blend(alpha)
    )


def interpolate_pose_linear_se3(
    T_start: np.ndarray, T_target: np.ndarray, fraction: float
) -> np.ndarray:
    """Interpolate translation and the shortest SO(3) relative rotation."""

    start = _coerce_pose_matrix(T_start, "T_start")
    target = _coerce_pose_matrix(T_target, "T_target")
    alpha = min(1.0, max(0.0, float(fraction)))
    relative_rotvec = so3_log(target[:3, :3] @ start[:3, :3].T)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = so3_exp(alpha * relative_rotvec) @ start[:3, :3]
    result[:3, 3] = start[:3, 3] + alpha * (
        target[:3, 3] - start[:3, 3]
    )
    return _coerce_pose_matrix(result, "interpolated pose")


def pose_error(T_actual: np.ndarray, T_target: np.ndarray) -> Tuple[float, float]:
    """Return Euclidean translation and geodesic SO(3) error."""

    actual = _coerce_pose_matrix(T_actual, "T_actual")
    target = _coerce_pose_matrix(T_target, "T_target")
    translation = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
    rotation = float(
        np.linalg.norm(so3_log(target[:3, :3] @ actual[:3, :3].T))
    )
    return translation, rotation


def so3_exp(rotvec: Sequence[float]) -> np.ndarray:
    """Rodrigues exponential map from a rotation vector to SO(3)."""

    vector = _finite_vector(rotvec, 3, "rotation vector")
    theta = float(np.linalg.norm(vector))
    skew = _skew(vector)
    if theta < 1.0e-8:
        # Stable Taylor expansion through second order.
        return np.eye(3) + skew + 0.5 * (skew @ skew)
    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    rotation = np.eye(3) + a * skew + b * (skew @ skew)
    return _validate_rotation(rotation, "so3_exp result")


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Shortest rotation vector for a proper rotation matrix, including pi."""

    matrix = _validate_rotation(rotation, "rotation")
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cosine)
    if theta < 1.0e-8:
        return _readonly(
            0.5
            * np.asarray(
                [
                    matrix[2, 1] - matrix[1, 2],
                    matrix[0, 2] - matrix[2, 0],
                    matrix[1, 0] - matrix[0, 1],
                ]
            )
        )
    if math.pi - theta < 1.0e-6:
        # (R + I) / 2 is axis*axis.T at pi.  The dominant eigenvector avoids
        # dividing by sin(theta), which is numerically zero there.
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + np.eye(3)))
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        skew_vector = np.asarray(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
        )
        if float(np.dot(axis, skew_vector)) < 0.0:
            axis = -axis
        return _readonly(theta * axis / np.linalg.norm(axis))
    factor = theta / (2.0 * math.sin(theta))
    vector = factor * np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    return _readonly(vector)


def _validate_realtime_rotation_scalars(
    r00: float,
    r10: float,
    r20: float,
    r01: float,
    r11: float,
    r21: float,
    r02: float,
    r12: float,
    r22: float,
    name: str,
) -> None:
    """Apply the reviewed SO(3) gates without temporary NumPy arrays."""

    # These are exactly the six independent entries of R.T @ R.  The other
    # three off-diagonal entries are symmetric and therefore carry no
    # additional information.  Keep the same absolute tolerance and zero
    # relative tolerance as _validate_rotation.
    if (
        abs(r00 * r00 + r10 * r10 + r20 * r20 - 1.0) > 1.0e-6
        or abs(r01 * r01 + r11 * r11 + r21 * r21 - 1.0) > 1.0e-6
        or abs(r02 * r02 + r12 * r12 + r22 * r22 - 1.0) > 1.0e-6
        or abs(r00 * r01 + r10 * r11 + r20 * r21) > 1.0e-6
        or abs(r00 * r02 + r10 * r12 + r20 * r22) > 1.0e-6
        or abs(r01 * r02 + r11 * r12 + r21 * r22) > 1.0e-6
    ):
        raise RuntimeError(f"{name} is not orthonormal")

    determinant = (
        r00 * (r11 * r22 - r12 * r21)
        - r01 * (r10 * r22 - r12 * r20)
        + r02 * (r10 * r21 - r11 * r20)
    )
    if abs(determinant - 1.0) > 1.0e-6:
        raise RuntimeError(f"{name} determinant must be +1")


def _realtime_endpoint_pose_error(
    actual_values: Sequence[float], target_values: Sequence[float]
) -> Tuple[float, float]:
    """Return endpoint SE(3) error while retaining all rigid-pose gates.

    ``actual_values`` is the finite 16-value column-major buffer returned by
    :meth:`_validate_realtime_cartesian_translation`.  ``target_values`` was
    fully validated once before the control handle started.  This helper keeps
    the actual homogeneous-row, SO(3), relative-SO(3), translation-error and
    geodesic-rotation-error checks scalar so the 1 kHz endpoint hold does not
    repeatedly allocate and factorize tiny NumPy matrices.
    """

    # The finite/shape gate has already run on this exact buffer.  Read each
    # rotation entry once; float(float_value) is allocation-free on CPython and
    # also accepts the NumPy scalar form used by tests/adapters.
    a00 = float(actual_values[0])
    a10 = float(actual_values[1])
    a20 = float(actual_values[2])
    a01 = float(actual_values[4])
    a11 = float(actual_values[5])
    a21 = float(actual_values[6])
    a02 = float(actual_values[8])
    a12 = float(actual_values[9])
    a22 = float(actual_values[10])

    if (
        abs(float(actual_values[3])) > 1.0e-8
        or abs(float(actual_values[7])) > 1.0e-8
        or abs(float(actual_values[11])) > 1.0e-8
        or abs(float(actual_values[15]) - 1.0) > 1.0e-8
    ):
        raise RuntimeError(
            "invalid Franka O_T_EE: state.O_T_EE has invalid homogeneous bottom row"
        )
    _validate_realtime_rotation_scalars(
        a00,
        a10,
        a20,
        a01,
        a11,
        a21,
        a02,
        a12,
        a22,
        "invalid Franka O_T_EE: state.O_T_EE rotation",
    )

    t00 = target_values[0]
    t10 = target_values[1]
    t20 = target_values[2]
    t01 = target_values[4]
    t11 = target_values[5]
    t21 = target_values[6]
    t02 = target_values[8]
    t12 = target_values[9]
    t22 = target_values[10]

    # Relative rotation R_target @ R_actual.T in column-major scalar form.
    relative00 = t00 * a00 + t01 * a01 + t02 * a02
    relative10 = t10 * a00 + t11 * a01 + t12 * a02
    relative20 = t20 * a00 + t21 * a01 + t22 * a02
    relative01 = t00 * a10 + t01 * a11 + t02 * a12
    relative11 = t10 * a10 + t11 * a11 + t12 * a12
    relative21 = t20 * a10 + t21 * a11 + t22 * a12
    relative02 = t00 * a20 + t01 * a21 + t02 * a22
    relative12 = t10 * a20 + t11 * a21 + t12 * a22
    relative22 = t20 * a20 + t21 * a21 + t22 * a22
    _validate_realtime_rotation_scalars(
        relative00,
        relative10,
        relative20,
        relative01,
        relative11,
        relative21,
        relative02,
        relative12,
        relative22,
        "rotation",
    )

    dx = target_values[12] - float(actual_values[12])
    dy = target_values[13] - float(actual_values[13])
    dz = target_values[14] - float(actual_values[14])
    translation = math.sqrt(dx * dx + dy * dy + dz * dz)
    cosine = 0.5 * (relative00 + relative11 + relative22 - 1.0)
    cosine = min(1.0, max(-1.0, cosine))
    return translation, math.acos(cosine)


def _validated_realtime_max_abs_dq(values: Any) -> float:
    """Validate the seven live joint velocities and return max absolute dq."""

    try:
        shape = getattr(values, "shape", None)
        if len(values) != 7 or shape not in (None, (7,)):
            raise RuntimeError("Franka state.dq is missing or malformed")
    except TypeError as exc:
        raise RuntimeError("Franka state.dq is missing or malformed") from exc

    maximum = 0.0
    for index in range(7):
        try:
            value = float(values[index])
        except (TypeError, ValueError, IndexError) as exc:
            raise RuntimeError("Franka state.dq is missing or malformed") from exc
        if not math.isfinite(value):
            raise RuntimeError("Franka state.dq contains NaN or infinity")
        magnitude = abs(value)
        if magnitude > maximum:
            maximum = magnitude
    return maximum


def _validated_control_period(period: Any, allow_zero: bool) -> Tuple[float, bool]:
    raw = period.to_sec() if hasattr(period, "to_sec") else period
    dt = float(raw)
    if not math.isfinite(dt) or dt < 0.0 or dt > MAX_CONTROL_PERIOD_S:
        raise RuntimeError(f"unsafe Franka control period dt={dt!r}s")
    if dt == 0.0:
        if not allow_zero:
            raise RuntimeError("repeated zero Franka control period")
        return 0.0, False
    if dt <= MIN_CONTROL_PERIOD_S:
        raise RuntimeError(f"unsafe tiny Franka control period dt={dt!r}s")
    return dt, False


def _state_pose(state: Any, field_name: str) -> np.ndarray:
    try:
        value = getattr(state, field_name)
    except AttributeError as exc:
        raise RuntimeError(f"Franka state has no {field_name}") from exc
    try:
        return _coerce_pose_matrix(value, f"state.{field_name}")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid Franka {field_name}: {exc}") from exc


def _coerce_pose_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (16,):
        array = array.reshape((4, 4), order="F")
    if array.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4) or (16,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    if not np.allclose(array[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8, rtol=0.0):
        raise ValueError(f"{name} has invalid homogeneous bottom row")
    _validate_rotation(array[:3, :3], f"{name} rotation")
    return _readonly(array)


def _coerce_inertia_matrix(value: Any, name: str) -> np.ndarray:
    """Accept Franka's column-major 9-vector or one explicit 3x3 matrix."""

    array = np.asarray(value, dtype=np.float64)
    if array.shape == (9,):
        array = array.reshape((3, 3), order="F")
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3x3 matrix or 9-vector")
    if not np.allclose(array, array.T, atol=1.0e-10, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(array)
    if float(np.min(eigenvalues)) < -1.0e-10:
        raise ValueError(f"{name} must be positive semidefinite")
    return _readonly(array)


def _validate_rotation(value: Any, name: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must be a finite (3, 3) matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0):
        raise ValueError(f"{name} is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6, rtol=0.0):
        raise ValueError(f"{name} determinant must be +1")
    return _readonly(rotation)


def _matrix_to_franka_pose(transform: np.ndarray) -> List[float]:
    matrix = _coerce_pose_matrix(transform, "Franka command pose")
    return matrix.reshape(16, order="F").tolist()


def _active_error_names(errors: Any) -> List[str]:
    if isinstance(errors, dict):
        return sorted(str(name) for name, value in errors.items() if bool(value))
    active = []
    for name in dir(errors):
        if name.startswith("_"):
            continue
        try:
            value = getattr(errors, name)
        except Exception:
            continue
        if isinstance(value, (bool, np.bool_)) and bool(value):
            active.append(name)
    return active


_MISSING_ATTRIBUTE = object()


def _safe_getattr(value: Any, name: str) -> Any:
    try:
        return getattr(value, name)
    except Exception:
        return _MISSING_ATTRIBUTE


def _is_boolean_error_value(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _is_native_pylibfranka_command_type(command_type: Any, name: str) -> bool:
    """Allow command-object reuse only for the reviewed native binding.

    Test doubles often retain command object identities to inspect the whole
    trajectory; reusing those objects would corrupt their histories.  The
    native writeOnce binding consumes/copies the command synchronously and its
    vector properties are explicitly writable.
    """

    return (
        getattr(command_type, "__name__", "") == name
        and str(getattr(command_type, "__module__", "")).startswith(
            "pylibfranka"
        )
    )


def _robot_mode_name(mode_value: Any) -> str:
    """Normalize pylibfranka/test enum names with a fast common-case path."""

    value = getattr(mode_value, "name", mode_value)
    if value == "Idle":
        return "idle"
    if value == "Move":
        return "move"
    if value is None:
        return "unknown"
    mode = str(value).lower().rsplit(".", 1)[-1]
    if mode in ("kidle", "kmove"):
        mode = mode[1:]
    return mode


def _validate_realtime_flag_vector(
    values: Any, field_name: str, expected_length: int
) -> None:
    """Validate one state flag vector without NumPy temporary arrays."""

    try:
        if len(values) != expected_length or (
            getattr(values, "shape", (expected_length,)) != (expected_length,)
        ):
            raise RuntimeError(f"Franka {field_name} is missing or malformed")
    except TypeError as exc:
        raise RuntimeError(f"Franka {field_name} is missing or malformed") from exc
    for index in range(expected_length):
        try:
            value = float(values[index])
        except (TypeError, ValueError, IndexError) as exc:
            raise RuntimeError(
                f"Franka {field_name} is missing or malformed"
            ) from exc
        if not math.isfinite(value):
            raise RuntimeError(f"Franka {field_name} is missing or malformed")
        if value < 0.0 or value > 1.0:
            raise RuntimeError(
                f"Franka {field_name} contains a value outside [0, 1]"
            )
        if value > 0.5:
            # Build the diagnostic vector only on the failure path.
            try:
                diagnostic = [float(values[item]) for item in range(expected_length)]
            except Exception:
                diagnostic = [f"index {index} active"]
            raise RuntimeError(f"Franka reports {field_name}: {diagnostic}")


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return _readonly(array)


def _require_positive(value: Any, name: str) -> None:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


__all__ = [
    "FR3_JOINT_LIMITS_RAD",
    "FrankaMotionLimits",
    "FrankaSequenceDriver",
    "FrankaStopUnconfirmed",
    "bounded_pose_step",
    "cosine_interpolate",
    "interpolate_pose_linear_se3",
    "interpolate_pose_minimum_jerk",
    "minimum_jerk_blend",
    "pose_error",
    "so3_exp",
    "so3_log",
]
