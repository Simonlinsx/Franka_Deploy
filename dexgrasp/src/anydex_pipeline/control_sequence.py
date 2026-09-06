"""Hardware-independent staged execution for one planned dexterous grasp.

This module deliberately knows nothing about libfranka, serial ports, or RH56
registers.  It coordinates two injected drivers and enforces the one safety
invariant that matters at this layer: neither thumb preshape nor finger closure
may start until the commanded grasp pose has been independently verified as
settled.

The sequence enters :class:`SequenceState.HOLDING`.  A successful no-contact
air run must then explicitly call :meth:`StagedGraspSequence.return_air_hand_to_open`
to replay the monitored reverse hand path.  Loaded lift is a separate audited
round trip: once an object may be suspended, faults stop Franka while retaining
the numeric Inspire hold.  Hand output is never disabled until the audited
setdown endpoint has been reached and settled.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from .types import rigid_transform
from .pregrasp_only_audit import (
    normalize_pregrasp_waypoints,
    pregrasp_prefix_contract_sha256,
)


OPEN_HAND_TARGETS: Tuple[int, ...] = (1000, 1000, 1000, 1000, 1000, 1000)
DISABLED_HAND_TARGETS: Tuple[int, ...] = (-1, -1, -1, -1, -1, -1)


def _strict_finite_real(value: Any, name: str) -> float:
    """Accept only an actual JSON-like numeric scalar, never bool or text."""

    if isinstance(value, (bool, np.bool_, str, bytes)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise SequencePlanError(f"{name} must be a finite numeric scalar")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise SequencePlanError(f"{name} must be a finite numeric scalar")
    return numeric


class SequenceState(str, Enum):
    DISARMED = "disarmed"
    OPENING_HAND = "opening_hand"
    OPEN_VERIFIED = "open_verified"
    MOVING_FRANKA_DEFAULT = "moving_franka_default"
    DEFAULT_VERIFIED = "default_verified"
    MOVING_PREGRASP = "moving_pregrasp"
    PREGRASP_VERIFIED = "pregrasp_verified"
    MOVING_GRASP = "moving_grasp"
    EEF_SETTLING = "eef_settling"
    THUMB_PRESHAPE = "thumb_preshape"
    CLOSING_BENDS = "closing_bends"
    HOLDING = "holding"
    MOVING_LIFT = "moving_lift"
    LIFT_SETTLING = "lift_settling"
    LIFTED_HOLDING = "lifted_holding"
    LOWERING_LOAD = "lowering_load"
    SETDOWN_SETTLING = "setdown_settling"
    SETDOWN_HOLDING = "setdown_holding"
    SETDOWN_COMPLETE = "setdown_complete"
    REOPENING_HAND = "reopening_hand"
    RETURNED_OPEN = "returned_open"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAULT_LATCHED = "fault_latched"


@dataclass(frozen=True)
class SettleTolerances:
    """Generic EEF arrival limits interpreted by the injected arm driver."""

    position_m: float = 0.005
    orientation_rad: float = 0.05
    linear_speed_m_s: float = 0.01
    angular_speed_rad_s: float = 0.05
    stable_seconds: float = 0.30

    def __post_init__(self) -> None:
        values = (
            self.position_m,
            self.orientation_rad,
            self.linear_speed_m_s,
            self.angular_speed_rad_s,
            self.stable_seconds,
        )
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("settle tolerances must be finite")
        if any(float(value) < 0.0 for value in values[:-1]):
            raise ValueError("settle error and speed tolerances cannot be negative")
        if float(self.stable_seconds) <= 0.0:
            raise ValueError("stable_seconds must be positive")


@runtime_checkable
class ArmSequenceDriver(Protocol):
    """Minimal Franka-side contract required by the state machine."""

    def move_joints(self, target_q: Sequence[float]) -> Sequence[float]:
        ...

    def move_loaded_joints(
        self,
        target_q: Sequence[float],
        *,
        max_joint_velocity_rad_s: float,
        max_joint_acceleration_rad_s2: float,
        max_dynamic_segment_rad: float,
        min_segment_duration_s: float,
    ) -> Sequence[float]:
        ...

    def move_pose(self, target: np.ndarray) -> None:
        ...

    def verify_settled(self, target: np.ndarray, tolerances: Any) -> bool:
        ...

    def verify_audited_joint_settled(
        self,
        target_q: Sequence[float],
        target_pose: np.ndarray,
        q_tolerance_rad: float,
        tolerances: Any,
    ) -> bool:
        ...

    def stop(self) -> None:
        ...

    def apply_external_load_and_verify(self, payload: Any) -> None:
        ...

    def clear_external_load_and_verify(self) -> None:
        ...

    def verify_idle_state(self) -> None:
        ...


@runtime_checkable
class HandSequenceDriver(Protocol):
    """Minimal Inspire-side contract required by the state machine."""

    def open_and_verify(self, targets: Sequence[int]) -> None:
        ...

    def preshape_thumb(self, target_q6: int) -> None:
        ...

    def close_bends_and_hold(self, targets: Sequence[int]) -> None:
        ...

    def close_bends_no_contact_and_hold(self, targets: Sequence[int]) -> None:
        ...

    def return_no_contact_hand_to_open(self) -> None:
        ...

    def verify_loaded_hold(
        self, targets: Sequence[int], minimum_contact_axes: int
    ) -> None:
        ...

    def verify_bounded_hold(
        self, targets: Sequence[int], *, no_contact: bool
    ) -> None:
        ...

    def disable_and_verify(self) -> None:
        ...


@dataclass(frozen=True)
class SequencePlan:
    """Small standalone plan accepted directly by :class:`StagedGraspSequence`.

    Other plan objects are accepted by duck typing; see :func:`_normalize_plan`.
    ``thumb_preshape_target`` is optional.  When present, it must equal element
    six of ``hand_target6`` so the preshape and final hold cannot disagree.
    """

    execution_eligible: bool
    default_q: np.ndarray
    pregrasp_pose: np.ndarray
    grasp_pose: np.ndarray
    hand_target6: Tuple[int, ...]
    settle_tolerances: Any = SettleTolerances()
    thumb_preshape_target: Optional[int] = None
    ineligibility_reason: str = ""

    def __post_init__(self) -> None:
        normalized = _normalize_plan_fields(
            execution_eligible=self.execution_eligible,
            default_q=self.default_q,
            pregrasp_pose=self.pregrasp_pose,
            grasp_pose=self.grasp_pose,
            hand_target6=self.hand_target6,
            settle_tolerances=self.settle_tolerances,
            thumb_preshape_target=self.thumb_preshape_target,
            ineligibility_reason=self.ineligibility_reason,
        )
        object.__setattr__(self, "execution_eligible", normalized.execution_eligible)
        object.__setattr__(self, "default_q", normalized.default_q)
        object.__setattr__(self, "pregrasp_pose", normalized.pregrasp_pose)
        object.__setattr__(self, "grasp_pose", normalized.grasp_pose)
        object.__setattr__(self, "hand_target6", normalized.hand_target6)
        object.__setattr__(self, "settle_tolerances", normalized.settle_tolerances)
        object.__setattr__(
            self, "thumb_preshape_target", normalized.thumb_preshape_target
        )
        object.__setattr__(
            self, "ineligibility_reason", normalized.ineligibility_reason
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        default_q: Sequence[float],
        pregrasp_pose: Sequence[Sequence[float]],
        grasp_pose: Sequence[Sequence[float]],
        hand_target6: Sequence[int],
        execution_eligible: bool = True,
        settle_tolerances: Any = None,
        thumb_preshape_target: Optional[int] = None,
        ineligibility_reason: str = "",
    ) -> "SequencePlan":
        return cls(
            execution_eligible=execution_eligible,
            default_q=np.asarray(default_q, dtype=np.float64),
            pregrasp_pose=np.asarray(pregrasp_pose, dtype=np.float64),
            grasp_pose=np.asarray(grasp_pose, dtype=np.float64),
            hand_target6=tuple(hand_target6),
            settle_tolerances=(
                SettleTolerances()
                if settle_tolerances is None
                else settle_tolerances
            ),
            thumb_preshape_target=thumb_preshape_target,
            ineligibility_reason=ineligibility_reason,
        )


@dataclass(frozen=True)
class DefaultSequencePlan:
    """Independent permission for the open-hand + Franka-default stage only."""

    default_execution_eligible: bool
    default_q: np.ndarray
    open_targets: Tuple[int, ...] = OPEN_HAND_TARGETS
    ineligibility_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.default_execution_eligible, (bool, np.bool_)):
            raise SequencePlanError("default_execution_eligible must be boolean")
        targets = _hand_targets(self.open_targets)
        if targets != OPEN_HAND_TARGETS:
            raise SequencePlanError("default stage must open all six axes to 1000")
        object.__setattr__(self, "default_q", _default_joints(self.default_q))
        object.__setattr__(self, "open_targets", targets)
        object.__setattr__(
            self, "ineligibility_reason", str(self.ineligibility_reason or "")
        )


@dataclass(frozen=True)
class AuditedJointSequencePlan:
    """Exact named joint polyline bound by an installed-tool audit artifact.

    The first waypoint is the read-only ``current`` state and is never sent as
    a motion command.  Every following named waypoint is passed to
    ``move_joints`` in order.  The arm driver must use the artifact's stricter
    tracking tolerance and the caller must verify the live current q before
    allowing the hand to open.
    """

    execution_eligible: bool
    mode: str
    contact_and_lift_forbidden: bool
    audit_schema_version: int
    joint_pose_binding_verified: bool
    joint_waypoints: Tuple[Tuple[str, np.ndarray], ...]
    pregrasp_pose: np.ndarray
    grasp_pose: np.ndarray
    hand_target6: Tuple[int, ...]
    max_q_tracking_error_rad: float
    settle_tolerances: Any = SettleTolerances()
    thumb_preshape_target: Optional[int] = None
    ineligibility_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.execution_eligible, (bool, np.bool_)):
            raise SequencePlanError("execution_eligible must be boolean")
        if self.mode not in ("loaded_grasp", "air_grasp"):
            raise SequencePlanError("mode must be loaded_grasp or air_grasp")
        if not isinstance(self.contact_and_lift_forbidden, (bool, np.bool_)):
            raise SequencePlanError("contact_and_lift_forbidden must be boolean")
        if bool(self.contact_and_lift_forbidden) != (self.mode == "air_grasp"):
            raise SequencePlanError(
                "contact_and_lift_forbidden must be true exactly for air_grasp"
            )
        if type(self.audit_schema_version) is not int or self.audit_schema_version != 2:
            raise SequencePlanError(
                "audited joint execution requires schema-v2 evidence"
            )
        if self.joint_pose_binding_verified is not True:
            raise SequencePlanError(
                "audited joint execution requires verified q/FK/EEF pose binding"
            )
        try:
            raw_waypoints = tuple(self.joint_waypoints)
        except TypeError as exc:
            raise SequencePlanError("joint_waypoints must be a sequence") from exc
        if len(raw_waypoints) < 4:
            raise SequencePlanError("joint_waypoints must contain at least four entries")
        normalized = []
        for index, item in enumerate(raw_waypoints):
            try:
                name, value = item
            except (TypeError, ValueError) as exc:
                raise SequencePlanError(
                    "joint_waypoints[{}] must be a (name, q) pair".format(index)
                ) from exc
            if not isinstance(name, str) or not name:
                raise SequencePlanError("joint waypoint name must be non-empty")
            normalized.append((name, _default_joints(value)))
        names = [item[0] for item in normalized]
        if names[0] != "current" or names[-2:] != ["pregrasp", "grasp"]:
            raise SequencePlanError(
                "joint waypoint order must be current, optional default_transit_N, "
                "default, optional approach_transit_N, pregrasp, grasp"
            )
        if names.count("default") != 1:
            raise SequencePlanError("joint waypoints must contain exactly one default")
        default_index = names.index("default")
        if default_index < 1 or default_index > len(names) - 3:
            raise SequencePlanError("default waypoint is out of order")
        expected_transits = [
            "default_transit_{}".format(index)
            for index in range(default_index - 1)
        ]
        if names[1:default_index] != expected_transits:
            raise SequencePlanError("default transit waypoint names are out of order")
        expected_approach_transits = [
            "approach_transit_{}".format(index)
            for index in range(len(names) - default_index - 3)
        ]
        if names[default_index + 1 : -2] != expected_approach_transits:
            raise SequencePlanError("approach transit waypoint names are out of order")
        if len(set(names)) != len(names):
            raise SequencePlanError("joint waypoint names must be unique")
        targets = _hand_targets(self.hand_target6)
        tracking_error = float(self.max_q_tracking_error_rad)
        if not np.isfinite(tracking_error) or not 0.0 < tracking_error <= 0.01:
            raise SequencePlanError(
                "max_q_tracking_error_rad must be finite in (0,0.01]"
            )
        thumb = self.thumb_preshape_target
        if thumb is not None:
            thumb = _strict_integral(thumb, "thumb_preshape_target")
            if thumb != targets[5]:
                raise SequencePlanError(
                    "thumb_preshape_target must equal hand_target6[5]"
                )
        object.__setattr__(self, "joint_waypoints", tuple(normalized))
        object.__setattr__(
            self, "contact_and_lift_forbidden", bool(self.contact_and_lift_forbidden)
        )
        object.__setattr__(self, "pregrasp_pose", _pose(self.pregrasp_pose, "pregrasp_pose"))
        object.__setattr__(self, "grasp_pose", _pose(self.grasp_pose, "grasp_pose"))
        object.__setattr__(self, "hand_target6", targets)
        object.__setattr__(self, "max_q_tracking_error_rad", tracking_error)
        object.__setattr__(self, "thumb_preshape_target", thumb)
        object.__setattr__(self, "ineligibility_reason", str(self.ineligibility_reason or ""))


@dataclass(frozen=True)
class LoadedLiftPayload:
    """Artifact-bound external load passed to Franka only after closure.

    Franka expects the CoM and inertia in flange frame ``F`` and inertia in
    column-major order at the API boundary.  The matrix remains a 3x3 tensor
    here so physical validity can be checked before any driver is called.
    """

    mass_kg: float
    F_x_Cload_m: np.ndarray
    I_load_kg_m2: np.ndarray
    binding_sha256: str

    def __post_init__(self) -> None:
        mass = _strict_finite_real(self.mass_kg, "loaded payload mass_kg")
        if mass <= 0.0:
            raise SequencePlanError("loaded payload mass_kg must be finite and positive")
        center = np.asarray(self.F_x_Cload_m, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise SequencePlanError("loaded payload F_x_Cload_m must have shape (3,)")
        inertia = np.asarray(self.I_load_kg_m2, dtype=np.float64)
        if inertia.shape == (9,):
            inertia = inertia.reshape((3, 3), order="F")
        if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
            raise SequencePlanError("loaded payload inertia must be finite 3x3")
        if not np.allclose(inertia, inertia.T, atol=1e-12, rtol=0.0):
            raise SequencePlanError("loaded payload inertia must be symmetric")
        moments = np.linalg.eigvalsh(inertia)
        if np.any(moments <= 0.0):
            raise SequencePlanError("loaded payload inertia must be positive definite")
        if float(moments[-1]) > float(moments[0] + moments[1]) + 1e-12:
            raise SequencePlanError("loaded payload inertia violates triangle inequality")
        digest = str(self.binding_sha256)
        if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
            raise SequencePlanError("loaded payload binding_sha256 must be lowercase SHA-256")
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "F_x_Cload_m", center.copy())
        object.__setattr__(self, "I_load_kg_m2", inertia.copy())
        object.__setattr__(self, "binding_sha256", digest)


@dataclass(frozen=True)
class AuditedLoadedLiftSequencePlan:
    """A separate loaded-lift permission layered on a loaded-grasp audit.

    An ordinary :class:`AuditedJointSequencePlan` can never satisfy this type.
    The lift suffix starts at the exact audited grasp q and ends at ``lift``;
    successful CLI use must replay it in reverse and settle at the supported
    grasp pose before hand output can be disabled.
    """

    execution_eligible: bool
    loaded_lift_audit_schema_version: int
    loaded_lift_binding_verified: bool
    loaded_lift_artifact_sha256: str
    grasp_plan: AuditedJointSequencePlan
    lift_waypoints: Tuple[Tuple[str, np.ndarray], ...]
    lift_pose: np.ndarray
    payload: LoadedLiftPayload
    minimum_contact_axes: int
    round_trip_setdown_required: bool
    max_q_tracking_error_rad: float
    time_law_max_joint_velocity_rad_s: float
    time_law_max_joint_acceleration_rad_s2: float
    time_law_max_dynamic_segment_rad: float
    time_law_min_segment_duration_s: float
    settle_tolerances: Any = SettleTolerances()
    ineligibility_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.execution_eligible, (bool, np.bool_)):
            raise SequencePlanError("execution_eligible must be boolean")
        if type(self.loaded_lift_audit_schema_version) is not int or (
            self.loaded_lift_audit_schema_version != 1
        ):
            raise SequencePlanError("loaded lift execution requires schema-v1 evidence")
        if self.loaded_lift_binding_verified is not True:
            raise SequencePlanError("loaded lift requires verified independent audit binding")
        digest = str(self.loaded_lift_artifact_sha256)
        if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
            raise SequencePlanError("loaded_lift_artifact_sha256 must be lowercase SHA-256")
        if not isinstance(self.grasp_plan, AuditedJointSequencePlan):
            raise SequencePlanError("grasp_plan must be an AuditedJointSequencePlan")
        if self.grasp_plan.mode != "loaded_grasp" or self.grasp_plan.contact_and_lift_forbidden:
            raise SequencePlanError("loaded lift requires a loaded_grasp base plan")
        if self.round_trip_setdown_required is not True:
            raise SequencePlanError("loaded lift requires audited round-trip setdown")
        try:
            raw = tuple(self.lift_waypoints)
        except TypeError as exc:
            raise SequencePlanError("lift_waypoints must be a sequence") from exc
        if len(raw) < 2:
            raise SequencePlanError("lift_waypoints must contain grasp and lift")
        normalized = []
        for index, item in enumerate(raw):
            try:
                name, q = item
            except (TypeError, ValueError) as exc:
                raise SequencePlanError(
                    "lift_waypoints[{}] must be a (name, q) pair".format(index)
                ) from exc
            if not isinstance(name, str) or not name:
                raise SequencePlanError("lift waypoint name must be non-empty")
            normalized.append((name, _default_joints(q)))
        names = [name for name, _q in normalized]
        expected = [
            "lift_transit_{}".format(index) for index in range(len(names) - 2)
        ]
        if names[0] != "grasp" or names[-1] != "lift" or names[1:-1] != expected:
            raise SequencePlanError(
                "lift waypoint order must be grasp, optional lift_transit_N, lift"
            )
        if not np.array_equal(normalized[0][1], self.grasp_plan.joint_waypoints[-1][1]):
            raise SequencePlanError("lift start q must exactly equal base grasp q")
        tracking = _strict_finite_real(
            self.max_q_tracking_error_rad, "loaded lift tracking error"
        )
        if not 0.0 < tracking <= 0.01:
            raise SequencePlanError("loaded lift tracking error must be in (0,0.01]")
        contacts = self.minimum_contact_axes
        if (
            isinstance(contacts, (bool, np.bool_))
            or not isinstance(contacts, (int, np.integer))
            or not 1 <= int(contacts) <= 5
        ):
            raise SequencePlanError("minimum_contact_axes must be an integer in 1..5")
        if not isinstance(self.payload, LoadedLiftPayload):
            raise SequencePlanError("payload must be LoadedLiftPayload")
        try:
            time_law_values = {
                "time_law_max_joint_velocity_rad_s": (
                    _strict_finite_real(
                        self.time_law_max_joint_velocity_rad_s,
                        "time_law_max_joint_velocity_rad_s",
                    ),
                    0.20,
                ),
                "time_law_max_joint_acceleration_rad_s2": (
                    _strict_finite_real(
                        self.time_law_max_joint_acceleration_rad_s2,
                        "time_law_max_joint_acceleration_rad_s2",
                    ),
                    0.50,
                ),
                "time_law_max_dynamic_segment_rad": (
                    _strict_finite_real(
                        self.time_law_max_dynamic_segment_rad,
                        "time_law_max_dynamic_segment_rad",
                    ),
                    0.35,
                ),
                "time_law_min_segment_duration_s": (
                    _strict_finite_real(
                        self.time_law_min_segment_duration_s,
                        "time_law_min_segment_duration_s",
                    ),
                    30.0,
                ),
            }
        except (TypeError, ValueError) as exc:
            raise SequencePlanError("loaded lift time law is malformed") from exc
        for name, (value, upper) in time_law_values.items():
            if not np.isfinite(value) or not 0.0 < value <= upper:
                raise SequencePlanError(
                    "{} must be finite in (0,{}]".format(name, upper)
                )
        settle = self.settle_tolerances
        try:
            settle_values = {
                "position_m": float(getattr(settle, "position_m")),
                "orientation_rad": float(getattr(settle, "orientation_rad")),
                "linear_speed_m_s": float(getattr(settle, "linear_speed_m_s")),
                "angular_speed_rad_s": float(
                    getattr(settle, "angular_speed_rad_s")
                ),
                "stable_seconds": float(getattr(settle, "stable_seconds")),
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise SequencePlanError(
                "loaded lift settle tolerances are incomplete"
            ) from exc
        settle_bounds = {
            "position_m": (0.0, 0.01),
            "orientation_rad": (0.0, 0.10),
            "linear_speed_m_s": (0.0, 0.02),
            "angular_speed_rad_s": (0.0, 0.10),
            "stable_seconds": (0.20, 2.0),
        }
        for name, value in settle_values.items():
            lower, upper = settle_bounds[name]
            if not np.isfinite(value) or not lower < value <= upper:
                raise SequencePlanError(
                    "loaded lift {} must be in ({},{}]".format(
                        name, lower, upper
                    )
                )
        object.__setattr__(self, "execution_eligible", bool(self.execution_eligible))
        object.__setattr__(self, "loaded_lift_artifact_sha256", digest)
        object.__setattr__(self, "lift_waypoints", tuple(normalized))
        object.__setattr__(self, "lift_pose", _pose(self.lift_pose, "lift_pose"))
        object.__setattr__(self, "minimum_contact_axes", int(contacts))
        object.__setattr__(self, "max_q_tracking_error_rad", tracking)
        for name, (value, _upper) in time_law_values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "ineligibility_reason", str(self.ineligibility_reason or ""))


@dataclass(frozen=True)
class AuditedPregraspSequencePlan:
    """Dedicated open-hand joint prefix that cannot represent a grasp stage."""

    execution_eligible: bool
    audit_schema_version: int
    audit_artifact_sha256: str
    prefix_contract_sha256: str
    joint_path_samples_sha256: str
    joint_waypoints: Tuple[Tuple[str, np.ndarray], ...]
    pregrasp_pose: np.ndarray
    max_joint_step_rad: float
    max_q_tracking_error_rad: float
    settle_tolerances: Any = SettleTolerances()
    ineligibility_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.execution_eligible, (bool, np.bool_)):
            raise SequencePlanError("execution_eligible must be boolean")
        if type(self.audit_schema_version) is not int or self.audit_schema_version != 2:
            raise SequencePlanError("pregrasp-only execution requires schema-v2 evidence")
        for name in (
            "audit_artifact_sha256",
            "prefix_contract_sha256",
            "joint_path_samples_sha256",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise SequencePlanError("{} must be a lowercase SHA-256 digest".format(name))
        try:
            normalized = normalize_pregrasp_waypoints(self.joint_waypoints)
        except (TypeError, ValueError) as exc:
            raise SequencePlanError(str(exc)) from exc
        pose = _pose(self.pregrasp_pose, "pregrasp_pose")
        maximum = float(self.max_joint_step_rad)
        tracking = float(self.max_q_tracking_error_rad)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise SequencePlanError("max_joint_step_rad must be finite and positive")
        if not np.isfinite(tracking) or not 0.0 < tracking <= 0.01:
            raise SequencePlanError("max_q_tracking_error_rad must be in (0,0.01]")
        expected = pregrasp_prefix_contract_sha256(
            waypoints=normalized,
            pregrasp_pose=pose,
            max_joint_step_rad=maximum,
            max_q_tracking_error_rad=tracking,
            samples_sha256=str(self.joint_path_samples_sha256),
        )
        if str(self.prefix_contract_sha256) != expected:
            raise SequencePlanError("pregrasp-only prefix contract SHA-256 mismatch")
        object.__setattr__(self, "execution_eligible", bool(self.execution_eligible))
        object.__setattr__(self, "joint_waypoints", tuple(normalized))
        object.__setattr__(self, "pregrasp_pose", pose)
        object.__setattr__(self, "max_joint_step_rad", maximum)
        object.__setattr__(self, "max_q_tracking_error_rad", tracking)
        object.__setattr__(self, "ineligibility_reason", str(self.ineligibility_reason or ""))


@dataclass(frozen=True)
class _NormalizedPlan:
    execution_eligible: bool
    default_q: np.ndarray
    pregrasp_pose: np.ndarray
    grasp_pose: np.ndarray
    hand_target6: Tuple[int, ...]
    settle_tolerances: Any
    thumb_preshape_target: Optional[int]
    ineligibility_reason: str


@dataclass(frozen=True)
class _NormalizedDefaultPlan:
    execution_eligible: bool
    default_q: np.ndarray
    ineligibility_reason: str


class SequencePlanError(ValueError):
    """The plan was rejected before any driver method was called."""


class SequenceStateError(RuntimeError):
    """The caller requested an operation that is invalid in the current state."""


class EEFNotSettledError(RuntimeError):
    """Independent EEF arrival verification did not pass."""


class SequenceExecutionError(RuntimeError):
    """Execution failed and the coordinated stop path was attempted.

    ``original_error`` is always retained.  ``stop_confirmed`` is true only when
    both ``arm.stop()`` and ``hand.disable_and_verify()`` returned successfully.
    Failures from either stop driver are retained separately in ``stop_errors``.
    """

    def __init__(
        self,
        original_error: BaseException,
        *,
        failure_state: SequenceState,
        stop_errors: Sequence[BaseException] = (),
        manual_load_recovery_required: bool = False,
    ) -> None:
        self.original_error = original_error
        self.failure_state = failure_state
        self.stop_errors = tuple(stop_errors)
        self.manual_load_recovery_required = bool(manual_load_recovery_required)
        self.stop_confirmed = not self.stop_errors and not self.manual_load_recovery_required
        if self.manual_load_recovery_required:
            suffix = (
                "LOADED HOLD CONTAINMENT ACTIVE: Franka stop requested; Inspire "
                "numeric hold intentionally remains enabled; support the object and "
                "use the explicit loaded recovery procedure"
            )
            if self.stop_errors:
                suffix += "; CONTAINMENT UNCONFIRMED: " + "; ".join(
                    str(error) for error in self.stop_errors
                )
        elif self.stop_confirmed:
            suffix = "stop confirmed"
        else:
            suffix = "STOP UNCONFIRMED: " + "; ".join(
                str(error) for error in self.stop_errors
            )
        super().__init__(
            f"sequence failed in {failure_state.value}: {original_error}; {suffix}"
        )


_MISSING = object()


def _first_attribute(plan: object, names: Sequence[str], default: Any = _MISSING) -> Any:
    for name in names:
        if hasattr(plan, name):
            return getattr(plan, name)
    if default is not _MISSING:
        return default
    raise SequencePlanError(
        "plan is missing required attribute; expected one of " + ", ".join(names)
    )


def _strict_integral(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise SequencePlanError(f"{name} must be an integer, not boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise SequencePlanError(f"{name} must be an integer") from exc
    if not np.isfinite(numeric) or numeric != round(numeric):
        raise SequencePlanError(f"{name} must be a finite integer")
    return int(numeric)


def _hand_targets(value: Any) -> Tuple[int, ...]:
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise SequencePlanError("hand_target6 must be a six-value sequence") from exc
    if len(raw) != 6:
        raise SequencePlanError("hand_target6 must contain exactly six values")
    targets = tuple(
        _strict_integral(item, f"hand_target6[{index}]")
        for index, item in enumerate(raw)
    )
    if any(target < 0 or target > 1000 for target in targets):
        raise SequencePlanError("hand_target6 values must be in 0..1000")
    return targets


def _default_joints(value: Any) -> np.ndarray:
    joints = np.asarray(value, dtype=np.float64)
    if joints.shape != (7,):
        raise SequencePlanError(f"default_q must have shape (7,), got {joints.shape}")
    if not np.isfinite(joints).all():
        raise SequencePlanError("default_q must contain only finite values")
    return joints.copy()


def _pose(value: Any, name: str) -> np.ndarray:
    try:
        return rigid_transform(np.asarray(value, dtype=np.float64), name)
    except (TypeError, ValueError) as exc:
        raise SequencePlanError(str(exc)) from exc


def _normalize_plan_fields(
    *,
    execution_eligible: Any,
    default_q: Any,
    pregrasp_pose: Any,
    grasp_pose: Any,
    hand_target6: Any,
    settle_tolerances: Any,
    thumb_preshape_target: Any,
    ineligibility_reason: Any,
) -> _NormalizedPlan:
    if not isinstance(execution_eligible, (bool, np.bool_)):
        raise SequencePlanError("execution_eligible must be boolean")
    targets = _hand_targets(hand_target6)
    thumb_target = None
    if thumb_preshape_target is not None:
        thumb_target = _strict_integral(
            thumb_preshape_target, "thumb_preshape_target"
        )
        if not 0 <= thumb_target <= 1000:
            raise SequencePlanError("thumb_preshape_target must be in 0..1000")
        if thumb_target != targets[5]:
            raise SequencePlanError(
                "thumb_preshape_target must equal hand_target6[5]"
            )
    if settle_tolerances is None:
        settle_tolerances = SettleTolerances()
    return _NormalizedPlan(
        execution_eligible=bool(execution_eligible),
        default_q=_default_joints(default_q),
        pregrasp_pose=_pose(pregrasp_pose, "pregrasp_pose"),
        grasp_pose=_pose(grasp_pose, "grasp_pose"),
        hand_target6=targets,
        settle_tolerances=settle_tolerances,
        thumb_preshape_target=thumb_target,
        ineligibility_reason=str(ineligibility_reason or ""),
    )


def _normalize_plan(plan: object) -> _NormalizedPlan:
    """Normalize SequencePlan or a likely GraspExecutionPlan by duck typing."""

    if isinstance(plan, SequencePlan):
        return _NormalizedPlan(
            execution_eligible=plan.execution_eligible,
            default_q=plan.default_q.copy(),
            pregrasp_pose=plan.pregrasp_pose.copy(),
            grasp_pose=plan.grasp_pose.copy(),
            hand_target6=plan.hand_target6,
            settle_tolerances=plan.settle_tolerances,
            thumb_preshape_target=plan.thumb_preshape_target,
            ineligibility_reason=plan.ineligibility_reason,
        )

    # ``control_plan.GraspExecutionPlan`` intentionally keeps executable values
    # inside immutable named stages.  Parse it structurally instead of importing
    # that module, so this executor stays decoupled and also accepts equivalent
    # plans from downstream applications.
    if hasattr(plan, "stages"):
        stages = tuple(getattr(plan, "stages"))

        def stage_named(expected: str, *, required: bool = True) -> Any:
            matches = []
            for stage in stages:
                raw_name = getattr(stage, "name", "")
                name = getattr(raw_name, "value", raw_name)
                if str(name) == expected:
                    matches.append(stage)
            if len(matches) > 1:
                raise SequencePlanError(
                    f"plan contains duplicate {expected} stages"
                )
            if not matches:
                if required:
                    raise SequencePlanError(f"plan has no {expected} stage")
                return None
            return matches[0]

        default_stage = stage_named("FRANKA_DEFAULT")
        open_stage = stage_named("INSPIRE_OPEN")
        pregrasp_stage = stage_named("FRANKA_PREGRASP")
        grasp_stage = stage_named("FRANKA_GRASP")
        settle_stage = stage_named("EEF_SETTLE_GATE")
        close_stage = stage_named("INSPIRE_CLOSE")
        thumb_stage = stage_named("THUMB_PRESHAPE", required=False)
        if not bool(getattr(settle_stage, "is_verification_gate", False)):
            raise SequencePlanError("EEF_SETTLE_GATE must be a verification gate")
        settle_pose = getattr(settle_stage, "T_reference_EE", None)
        grasp_pose = getattr(grasp_stage, "T_reference_EE", None)
        if settle_pose is None or grasp_pose is None or not np.allclose(
            np.asarray(settle_pose), np.asarray(grasp_pose), atol=1e-9, rtol=0.0
        ):
            raise SequencePlanError(
                "EEF_SETTLE_GATE target must equal the FRANKA_GRASP target"
            )
        open_targets = _hand_targets(getattr(open_stage, "inspire_angles", None))
        if open_targets != OPEN_HAND_TARGETS:
            raise SequencePlanError(
                "INSPIRE_OPEN must command [1000, 1000, 1000, 1000, 1000, 1000]"
            )
        close_targets_raw = getattr(close_stage, "inspire_angles", None)
        thumb_target = None
        if thumb_stage is not None:
            thumb_angles = _hand_targets(
                getattr(thumb_stage, "inspire_angles", None)
            )
            if thumb_angles[:5] != OPEN_HAND_TARGETS[:5]:
                raise SequencePlanError(
                    "THUMB_PRESHAPE must leave the first five axes fully open"
                )
            thumb_target = thumb_angles[5]
        blockers = _first_attribute(
            plan,
            ("execution_blockers", "ineligibility_reason"),
            (),
        )
        if isinstance(blockers, str):
            reason = blockers
        else:
            try:
                reason = "; ".join(str(item) for item in blockers)
            except TypeError:
                reason = str(blockers)
        return _normalize_plan_fields(
            execution_eligible=_first_attribute(plan, ("execution_eligible",)),
            default_q=getattr(default_stage, "franka_q", None),
            pregrasp_pose=getattr(pregrasp_stage, "T_reference_EE", None),
            grasp_pose=grasp_pose,
            hand_target6=close_targets_raw,
            settle_tolerances=_first_attribute(
                plan,
                ("settle_tolerances", "eef_settle_tolerances", "tolerances"),
                SettleTolerances(),
            ),
            thumb_preshape_target=thumb_target,
            ineligibility_reason=reason,
        )

    eligible = _first_attribute(plan, ("execution_eligible",))
    default_q = _first_attribute(
        plan, ("default_q", "franka_default_q", "default_joint_positions")
    )
    pregrasp = _first_attribute(
        plan,
        (
            "pregrasp_pose",
            "T_base_pregrasp",
            "T_base_eef_pregrasp",
            "T_base_flange_pregrasp",
        ),
    )
    grasp = _first_attribute(
        plan,
        (
            "grasp_pose",
            "T_base_grasp",
            "T_base_eef_grasp",
            "T_base_flange_grasp",
        ),
    )
    hand_targets = _first_attribute(
        plan,
        ("hand_target6", "hand_targets", "inspire_targets", "hand_angles"),
    )
    tolerances = _first_attribute(
        plan,
        ("settle_tolerances", "eef_settle_tolerances", "tolerances"),
        SettleTolerances(),
    )
    thumb_target = _first_attribute(
        plan, ("thumb_preshape_target", "thumb_preshape_q6"), None
    )
    if thumb_target is None and bool(
        _first_attribute(
            plan, ("requires_thumb_preshape", "preshape_thumb"), False
        )
    ):
        thumb_target = tuple(hand_targets)[5]
    reason = _first_attribute(
        plan,
        ("ineligibility_reason", "execution_ineligibility_reason", "reject_reason"),
        "",
    )
    return _normalize_plan_fields(
        execution_eligible=eligible,
        default_q=default_q,
        pregrasp_pose=pregrasp,
        grasp_pose=grasp,
        hand_target6=hand_targets,
        settle_tolerances=tolerances,
        thumb_preshape_target=thumb_target,
        ineligibility_reason=reason,
    )


def _normalize_default_plan(plan: object) -> _NormalizedDefaultPlan:
    """Read only the independently-authorized open/default portion of a plan."""

    if isinstance(plan, DefaultSequencePlan):
        return _NormalizedDefaultPlan(
            execution_eligible=plan.default_execution_eligible,
            default_q=plan.default_q.copy(),
            ineligibility_reason=plan.ineligibility_reason,
        )
    if isinstance(plan, SequencePlan):
        return _NormalizedDefaultPlan(
            execution_eligible=plan.execution_eligible,
            default_q=plan.default_q.copy(),
            ineligibility_reason=plan.ineligibility_reason,
        )

    default_q = None
    if hasattr(plan, "stages"):
        open_stages = []
        default_stages = []
        for stage in tuple(getattr(plan, "stages")):
            raw_name = getattr(stage, "name", "")
            name = str(getattr(raw_name, "value", raw_name))
            if name == "INSPIRE_OPEN":
                open_stages.append(stage)
            elif name == "FRANKA_DEFAULT":
                default_stages.append(stage)
        if len(open_stages) != 1 or len(default_stages) != 1:
            raise SequencePlanError(
                "default plan requires exactly one INSPIRE_OPEN and FRANKA_DEFAULT stage"
            )
        if _hand_targets(getattr(open_stages[0], "inspire_angles", None)) != OPEN_HAND_TARGETS:
            raise SequencePlanError("INSPIRE_OPEN must command [1000]*6")
        default_q = getattr(default_stages[0], "franka_q", None)
    if default_q is None:
        default_q = _first_attribute(
            plan, ("default_q", "franka_default_q", "default_joint_positions")
        )

    eligible = _first_attribute(
        plan,
        ("default_execution_eligible", "execution_eligible"),
    )
    blockers = _first_attribute(
        plan,
        (
            "default_execution_blockers",
            "default_ineligibility_reason",
            "ineligibility_reason",
            "execution_blockers",
        ),
        "",
    )
    if isinstance(blockers, str):
        reason = blockers
    else:
        try:
            reason = "; ".join(str(item) for item in blockers)
        except TypeError:
            reason = str(blockers)
    if not isinstance(eligible, (bool, np.bool_)):
        raise SequencePlanError("default execution eligibility must be boolean")
    return _NormalizedDefaultPlan(
        execution_eligible=bool(eligible),
        default_q=_default_joints(default_q),
        ineligibility_reason=reason,
    )


_ALLOWED_TRANSITIONS = {
    SequenceState.DISARMED: {SequenceState.OPENING_HAND, SequenceState.STOPPING},
    SequenceState.OPENING_HAND: {SequenceState.OPEN_VERIFIED, SequenceState.STOPPING},
    SequenceState.OPEN_VERIFIED: {
        SequenceState.MOVING_FRANKA_DEFAULT,
        SequenceState.STOPPING,
    },
    SequenceState.MOVING_FRANKA_DEFAULT: {
        SequenceState.DEFAULT_VERIFIED,
        SequenceState.STOPPING,
    },
    SequenceState.DEFAULT_VERIFIED: {
        SequenceState.MOVING_PREGRASP,
        SequenceState.STOPPING,
    },
    SequenceState.MOVING_PREGRASP: {
        SequenceState.PREGRASP_VERIFIED,
        SequenceState.STOPPING,
    },
    SequenceState.PREGRASP_VERIFIED: {
        SequenceState.MOVING_GRASP,
        SequenceState.STOPPING,
    },
    SequenceState.MOVING_GRASP: {
        SequenceState.EEF_SETTLING,
        SequenceState.STOPPING,
    },
    SequenceState.EEF_SETTLING: {
        SequenceState.THUMB_PRESHAPE,
        SequenceState.CLOSING_BENDS,
        SequenceState.STOPPING,
    },
    SequenceState.THUMB_PRESHAPE: {
        SequenceState.CLOSING_BENDS,
        SequenceState.STOPPING,
    },
    SequenceState.CLOSING_BENDS: {
        SequenceState.HOLDING,
        SequenceState.STOPPING,
    },
    SequenceState.HOLDING: {
        SequenceState.MOVING_LIFT,
        SequenceState.REOPENING_HAND,
        SequenceState.STOPPING,
    },
    SequenceState.MOVING_LIFT: {
        SequenceState.LIFT_SETTLING,
        SequenceState.STOPPING,
    },
    SequenceState.LIFT_SETTLING: {
        SequenceState.LIFTED_HOLDING,
        SequenceState.STOPPING,
    },
    SequenceState.LIFTED_HOLDING: {
        SequenceState.LOWERING_LOAD,
        SequenceState.STOPPING,
    },
    SequenceState.LOWERING_LOAD: {
        SequenceState.SETDOWN_SETTLING,
        SequenceState.STOPPING,
    },
    SequenceState.SETDOWN_SETTLING: {
        SequenceState.SETDOWN_HOLDING,
        SequenceState.STOPPING,
    },
    SequenceState.SETDOWN_HOLDING: {
        SequenceState.SETDOWN_COMPLETE,
        SequenceState.STOPPING,
    },
    SequenceState.SETDOWN_COMPLETE: {SequenceState.STOPPING},
    SequenceState.REOPENING_HAND: {
        SequenceState.RETURNED_OPEN,
        SequenceState.STOPPING,
    },
    SequenceState.RETURNED_OPEN: {SequenceState.STOPPING},
    SequenceState.STOPPING: {
        SequenceState.STOPPED,
        SequenceState.FAULT_LATCHED,
    },
    SequenceState.STOPPED: set(),
    SequenceState.FAULT_LATCHED: set(),
}


class StagedGraspSequence:
    """Execute one open-hand Franka approach and staged Inspire closure."""

    def __init__(
        self,
        arm: ArmSequenceDriver,
        hand: HandSequenceDriver,
        *,
        settle_tolerances: Any = None,
        transition_observer: Optional[Callable[[SequenceState], None]] = None,
        boundary_observer: Optional[
            Callable[
                [
                    str,
                    Optional[np.ndarray],
                    Optional[np.ndarray],
                    Optional[Tuple[int, ...]],
                ],
                None,
            ]
        ] = None,
    ) -> None:
        self.arm = arm
        self.hand = hand
        self.settle_tolerances = settle_tolerances
        if transition_observer is not None and not callable(transition_observer):
            raise TypeError("transition_observer must be callable or None")
        if boundary_observer is not None and not callable(boundary_observer):
            raise TypeError("boundary_observer must be callable or None")
        self._transition_observer = transition_observer
        self._transition_observer_error: Optional[BaseException] = None
        self._boundary_observer = boundary_observer
        self.state = SequenceState.DISARMED
        self.state_history = [self.state]
        self._prepared_default_q: Optional[np.ndarray] = None
        self._eef_settle_verified = False
        self._air_no_contact_hold = False
        self._bounded_hold_targets: Optional[Tuple[int, ...]] = None
        self._loaded_object_suspended = False
        self._loaded_payload_active = False
        self._loaded_lift_artifact_sha256: Optional[str] = None
        self._loaded_hold_targets: Optional[Tuple[int, ...]] = None
        self._loaded_minimum_contact_axes: Optional[int] = None
        self.last_error: Optional[SequenceExecutionError] = None

    @property
    def requires_manual_load_recovery(self) -> bool:
        """True when generic cleanup must not disable the grasping hand."""

        return bool(self._loaded_object_suspended)

    @property
    def bounded_hold_targets(self) -> Optional[Tuple[int, ...]]:
        """Exact six-axis target accepted by the successful close operation."""

        return self._bounded_hold_targets

    def _transition(self, new_state: SequenceState) -> None:
        if new_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise SequenceStateError(
                f"invalid sequence transition {self.state.value} -> {new_state.value}"
            )
        self.state = new_state
        self.state_history.append(new_state)
        observer = self._transition_observer
        if observer is None:
            return
        try:
            observer(new_state)
        except BaseException as exc:
            # A display/telemetry observer is fail-closed during ordinary
            # execution, but it must never recursively prevent the STOP or
            # suspended-load containment path.  Disable it before propagating
            # an ordinary-stage failure; terminal cleanup notifications are
            # best effort after recording the error.
            self._transition_observer = None
            self._transition_observer_error = exc
            if new_state in (
                SequenceState.STOPPING,
                SequenceState.STOPPED,
                SequenceState.FAULT_LATCHED,
            ):
                return
            raise

    def _observe_boundary(
        self,
        stage: str,
        target_q: Optional[np.ndarray] = None,
        target_pose: Optional[np.ndarray] = None,
        hand_targets: Optional[Sequence[int]] = None,
    ) -> None:
        observer = self._boundary_observer
        if observer is None:
            return
        observer(
            str(stage),
            None if target_q is None else np.asarray(target_q, dtype=np.float64).copy(),
            None
            if target_pose is None
            else np.asarray(target_pose, dtype=np.float64).copy(),
            None
            if hand_targets is None
            else tuple(int(value) for value in hand_targets),
        )

    @staticmethod
    def _require_eligible(plan: _NormalizedPlan) -> None:
        if not plan.execution_eligible:
            detail = (
                f": {plan.ineligibility_reason}"
                if plan.ineligibility_reason
                else ""
            )
            raise SequencePlanError("plan is not execution eligible" + detail)

    def _run_to_default(self, plan: Any) -> None:
        self._transition(SequenceState.OPENING_HAND)
        self.hand.open_and_verify(list(OPEN_HAND_TARGETS))
        self._transition(SequenceState.OPEN_VERIFIED)
        self._observe_boundary("open", hand_targets=OPEN_HAND_TARGETS)
        self._transition(SequenceState.MOVING_FRANKA_DEFAULT)
        self.arm.move_joints(plan.default_q.copy())
        self._prepared_default_q = plan.default_q.copy()
        self._transition(SequenceState.DEFAULT_VERIFIED)
        self._observe_boundary("default", target_q=plan.default_q)

    def _run_to_default_joint_waypoints(
        self, plan: AuditedJointSequencePlan
    ) -> None:
        self._transition(SequenceState.OPENING_HAND)
        self.hand.open_and_verify(list(OPEN_HAND_TARGETS))
        self._transition(SequenceState.OPEN_VERIFIED)
        self._observe_boundary("open", hand_targets=OPEN_HAND_TARGETS)
        self._transition(SequenceState.MOVING_FRANKA_DEFAULT)
        # Skip waypoint zero: the caller has just read and strictly matched the
        # live q against the audit before allowing this method to run.
        default_index = next(
            index
            for index, (name, _target) in enumerate(plan.joint_waypoints)
            if name == "default"
        )
        for name, target_q in plan.joint_waypoints[1 : default_index + 1]:
            self._move_audited_waypoint(
                target_q,
                plan.max_q_tracking_error_rad,
                waypoint_name=name,
            )
        self._prepared_default_q = plan.joint_waypoints[default_index][1].copy()
        self._transition(SequenceState.DEFAULT_VERIFIED)

    def _run_pregrasp_only_joint_waypoints(
        self, plan: AuditedPregraspSequencePlan
    ) -> None:
        """Open/verify, latch RH56 disabled, then execute only the prefix."""

        self._transition(SequenceState.OPENING_HAND)
        self.hand.open_and_verify(list(OPEN_HAND_TARGETS))
        # ``open_and_verify`` on the reviewed RH56 backend already ends at
        # ANGLE_SET=[-1]*6.  Repeating the formal disable/idle verification here
        # makes that property part of this generic sequencer contract and
        # permanently latches out every later numeric hand command.
        self.hand.disable_and_verify()
        self._transition(SequenceState.OPEN_VERIFIED)
        self._observe_boundary("open", hand_targets=DISABLED_HAND_TARGETS)

        default_index = next(
            index
            for index, (name, _target) in enumerate(plan.joint_waypoints)
            if name == "default"
        )
        self._transition(SequenceState.MOVING_FRANKA_DEFAULT)
        for name, target_q in plan.joint_waypoints[1 : default_index + 1]:
            self._move_audited_waypoint(
                target_q,
                plan.max_q_tracking_error_rad,
                waypoint_name=name,
            )
        self._prepared_default_q = plan.joint_waypoints[default_index][1].copy()
        self._transition(SequenceState.DEFAULT_VERIFIED)

        self._transition(SequenceState.MOVING_PREGRASP)
        for name, target_q in plan.joint_waypoints[default_index + 1 :]:
            self._move_audited_waypoint(
                target_q,
                plan.max_q_tracking_error_rad,
                waypoint_name=name,
            )
        tolerances = (
            plan.settle_tolerances
            if self.settle_tolerances is None
            else self.settle_tolerances
        )
        self._verify_audited_joint_arrival(
            plan.joint_waypoints[-1][1],
            plan.pregrasp_pose,
            plan.max_q_tracking_error_rad,
            tolerances,
            "pregrasp",
        )
        self._transition(SequenceState.PREGRASP_VERIFIED)
        self._observe_boundary(
            "pregrasp",
            target_q=plan.joint_waypoints[-1][1],
            target_pose=plan.pregrasp_pose,
            hand_targets=DISABLED_HAND_TARGETS,
        )

    def _move_audited_waypoint(
        self,
        target_q: np.ndarray,
        tolerance_rad: float,
        *,
        waypoint_name: Optional[str] = None,
        loaded_time_law: Optional[AuditedLoadedLiftSequencePlan] = None,
    ) -> None:
        if loaded_time_law is None:
            actual = self.arm.move_joints(target_q.copy())
        else:
            actual = self.arm.move_loaded_joints(
                target_q.copy(),
                max_joint_velocity_rad_s=(
                    loaded_time_law.time_law_max_joint_velocity_rad_s
                ),
                max_joint_acceleration_rad_s2=(
                    loaded_time_law.time_law_max_joint_acceleration_rad_s2
                ),
                max_dynamic_segment_rad=(
                    loaded_time_law.time_law_max_dynamic_segment_rad
                ),
                min_segment_duration_s=(
                    loaded_time_law.time_law_min_segment_duration_s
                ),
            )
        feedback = np.asarray(actual, dtype=np.float64)
        if feedback.shape != (7,) or not np.all(np.isfinite(feedback)):
            raise RuntimeError(
                "audited joint waypoint driver returned no valid seven-axis readback"
            )
        error = float(np.max(np.abs(feedback - target_q)))
        if error > float(tolerance_rad):
            raise RuntimeError(
                "audited joint waypoint readback error {:.6f}rad exceeds {:.6f}rad".format(
                    error, float(tolerance_rad)
                )
            )
        if waypoint_name is not None:
            self._observe_boundary(str(waypoint_name), target_q=target_q)

    def _verify_arrival(
        self, target: np.ndarray, tolerances: Any, label: str
    ) -> None:
        verified = self.arm.verify_settled(target.copy(), tolerances)
        if not isinstance(verified, (bool, np.bool_)) or not bool(verified):
            raise EEFNotSettledError(f"{label} EEF settle verification failed")

    def _verify_audited_joint_arrival(
        self,
        target_q: np.ndarray,
        target_pose: np.ndarray,
        q_tolerance_rad: float,
        tolerances: Any,
        label: str,
    ) -> None:
        verified = self.arm.verify_audited_joint_settled(
            target_q.copy(),
            target_pose.copy(),
            float(q_tolerance_rad),
            tolerances,
        )
        if not isinstance(verified, (bool, np.bool_)) or not bool(verified):
            raise EEFNotSettledError(
                f"{label} audited joint/q/EEF settle verification failed"
            )

    def _run_after_default(self, plan: _NormalizedPlan) -> None:
        if self._prepared_default_q is not None and not np.array_equal(
            self._prepared_default_q, plan.default_q
        ):
            raise SequencePlanError(
                "plan default_q differs from the already verified default pose"
            )

        self._transition(SequenceState.MOVING_PREGRASP)
        self.arm.move_pose(plan.pregrasp_pose.copy())
        tolerances = (
            plan.settle_tolerances
            if self.settle_tolerances is None
            else self.settle_tolerances
        )
        self._verify_arrival(plan.pregrasp_pose, tolerances, "pregrasp")
        self._transition(SequenceState.PREGRASP_VERIFIED)
        self._observe_boundary(
            "pregrasp",
            target_pose=plan.pregrasp_pose,
            hand_targets=OPEN_HAND_TARGETS,
        )

        self._transition(SequenceState.MOVING_GRASP)
        self.arm.move_pose(plan.grasp_pose.copy())
        self._transition(SequenceState.EEF_SETTLING)
        self._verify_arrival(plan.grasp_pose, tolerances, "grasp")
        self._eef_settle_verified = True
        self._observe_boundary(
            "grasp",
            target_pose=plan.grasp_pose,
            hand_targets=OPEN_HAND_TARGETS,
        )

        if plan.thumb_preshape_target is not None:
            self._transition(SequenceState.THUMB_PRESHAPE)
            self._require_settle_gate("thumb preshape")
            self.hand.preshape_thumb(plan.thumb_preshape_target)
            self._observe_boundary(
                "thumb_preshape",
                target_pose=plan.grasp_pose,
                hand_targets=(-1, -1, -1, -1, -1, plan.thumb_preshape_target),
            )

        self._transition(SequenceState.CLOSING_BENDS)
        self._require_settle_gate("finger closure")
        self.hand.close_bends_and_hold(list(plan.hand_target6))
        self._bounded_hold_targets = tuple(plan.hand_target6)
        self._transition(SequenceState.HOLDING)
        self._observe_boundary(
            "close",
            target_pose=plan.grasp_pose,
            hand_targets=plan.hand_target6,
        )

    def _run_after_default_joint_waypoints(
        self, plan: AuditedJointSequencePlan
    ) -> None:
        default_index = next(
            index
            for index, (name, _target) in enumerate(plan.joint_waypoints)
            if name == "default"
        )
        default_q = plan.joint_waypoints[default_index][1]
        if self._prepared_default_q is not None and not np.array_equal(
            self._prepared_default_q, default_q
        ):
            raise SequencePlanError(
                "audited default waypoint differs from the already verified default pose"
            )
        tolerances = (
            plan.settle_tolerances
            if self.settle_tolerances is None
            else self.settle_tolerances
        )
        self._transition(SequenceState.MOVING_PREGRASP)
        # Every named waypoint after default through pregrasp changes the
        # certified polyline.  Dispatch all of them in order; the final item
        # in this slice is exactly the pregrasp waypoint.
        for name, target_q in plan.joint_waypoints[default_index + 1 : -1]:
            self._move_audited_waypoint(
                target_q,
                plan.max_q_tracking_error_rad,
                waypoint_name=name,
            )
        self._verify_audited_joint_arrival(
            plan.joint_waypoints[-2][1],
            plan.pregrasp_pose,
            plan.max_q_tracking_error_rad,
            tolerances,
            "pregrasp",
        )
        self._transition(SequenceState.PREGRASP_VERIFIED)
        self._observe_boundary(
            "pregrasp",
            target_q=plan.joint_waypoints[-2][1],
            target_pose=plan.pregrasp_pose,
            hand_targets=OPEN_HAND_TARGETS,
        )

        self._transition(SequenceState.MOVING_GRASP)
        self._move_audited_waypoint(
            plan.joint_waypoints[-1][1],
            plan.max_q_tracking_error_rad,
            waypoint_name=plan.joint_waypoints[-1][0],
        )
        self._transition(SequenceState.EEF_SETTLING)
        self._verify_audited_joint_arrival(
            plan.joint_waypoints[-1][1],
            plan.grasp_pose,
            plan.max_q_tracking_error_rad,
            tolerances,
            "grasp",
        )
        self._eef_settle_verified = True
        self._observe_boundary(
            "grasp",
            target_q=plan.joint_waypoints[-1][1],
            target_pose=plan.grasp_pose,
            hand_targets=OPEN_HAND_TARGETS,
        )

        if plan.thumb_preshape_target is not None:
            self._transition(SequenceState.THUMB_PRESHAPE)
            self._require_settle_gate("thumb preshape")
            self.hand.preshape_thumb(plan.thumb_preshape_target)
            self._observe_boundary(
                "thumb_preshape",
                target_q=plan.joint_waypoints[-1][1],
                target_pose=plan.grasp_pose,
                hand_targets=(-1, -1, -1, -1, -1, plan.thumb_preshape_target),
            )

        self._transition(SequenceState.CLOSING_BENDS)
        self._require_settle_gate("finger closure")
        if plan.contact_and_lift_forbidden:
            self.hand.close_bends_no_contact_and_hold(list(plan.hand_target6))
            self._air_no_contact_hold = True
        else:
            self.hand.close_bends_and_hold(list(plan.hand_target6))
        self._bounded_hold_targets = tuple(plan.hand_target6)
        self._transition(SequenceState.HOLDING)
        self._observe_boundary(
            "close",
            target_q=plan.joint_waypoints[-1][1],
            target_pose=plan.grasp_pose,
            hand_targets=plan.hand_target6,
        )

    def _require_settle_gate(self, operation: str) -> None:
        if not self._eef_settle_verified or self.state not in (
            SequenceState.THUMB_PRESHAPE,
            SequenceState.CLOSING_BENDS,
        ):
            raise SequenceStateError(
                f"{operation} is forbidden before verified grasp-pose settling"
            )

    def _verify_loaded_hold(self) -> None:
        if self._loaded_hold_targets is None or self._loaded_minimum_contact_axes is None:
            raise SequenceStateError("loaded hold verification contract is not armed")
        self.hand.verify_loaded_hold(
            list(self._loaded_hold_targets), self._loaded_minimum_contact_axes
        )

    def verify_bounded_holding(self) -> SequenceState:
        """Refresh the ordinary grasp/air-grasp hold through the hand owner.

        The concrete RH56 owner's dedicated verifier performs the sole read and
        owns tolerance, current-cap, status, envelope, and external Franka-gate
        semantics.  This layer only binds it to the exact target accepted by
        the successful close and routes failure into coordinated cleanup.
        """

        if self.state != SequenceState.HOLDING:
            raise SequenceStateError(
                "bounded hold monitoring requires HOLDING, got "
                + self.state.value
            )
        expected = self._bounded_hold_targets
        if expected is None:
            raise SequenceStateError("bounded hold verification contract is not armed")
        try:
            verifier = getattr(self.hand, "verify_bounded_hold", None)
            if not callable(verifier):
                raise RuntimeError(
                    "hand driver exposes no dedicated bounded-hold verifier"
                )
            verifier(
                list(expected), no_contact=bool(self._air_no_contact_hold)
            )
        except BaseException as exc:
            if isinstance(exc, SequenceExecutionError):
                raise
            raise self._fail(exc) from exc
        return self.state

    def _stop_drivers(self) -> Tuple[BaseException, ...]:
        errors = []
        try:
            self.arm.stop()
        except BaseException as exc:
            errors.append(exc)
        try:
            self.hand.disable_and_verify()
        except BaseException as exc:
            errors.append(exc)
        if self._loaded_payload_active:
            try:
                self.arm.clear_external_load_and_verify()
                self._loaded_payload_active = False
            except BaseException as exc:
                errors.append(exc)
        return tuple(errors)

    def _contain_suspended_load(self) -> Tuple[BaseException, ...]:
        """Stop the arm without ever removing the hand's numeric hold."""

        errors = []
        try:
            self.arm.stop()
        except BaseException as exc:
            errors.append(exc)
        try:
            self._verify_loaded_hold()
        except BaseException as exc:
            errors.append(exc)
        return tuple(errors)

    def _fail(self, error: BaseException) -> "SequenceExecutionError":
        failure_state = self.state
        if self.state not in (SequenceState.STOPPING, SequenceState.STOPPED):
            self._transition(SequenceState.STOPPING)
        manual_recovery = self._loaded_object_suspended
        stop_errors = (
            self._contain_suspended_load() if manual_recovery else self._stop_drivers()
        )
        self._transition(SequenceState.FAULT_LATCHED)
        wrapped = SequenceExecutionError(
            error,
            failure_state=failure_state,
            stop_errors=stop_errors,
            manual_load_recovery_required=manual_recovery,
        )
        self.last_error = wrapped
        return wrapped

    def run_to_default(self, plan: object) -> SequenceState:
        """Open the hand and move Franka to its verified default joint pose."""

        normalized = _normalize_default_plan(plan)
        self._require_eligible(normalized)
        if self.state == SequenceState.DEFAULT_VERIFIED:
            if self._prepared_default_q is not None and not np.array_equal(
                self._prepared_default_q, normalized.default_q
            ):
                raise SequencePlanError(
                    "plan default_q differs from the already verified default pose"
                )
            return self.state
        if self.state != SequenceState.DISARMED:
            raise SequenceStateError(
                f"run_to_default requires DISARMED, got {self.state.value}"
            )
        try:
            self._run_to_default(normalized)
        except BaseException as exc:
            raise self._fail(exc) from exc
        return self.state

    def return_air_hand_to_open(self) -> SequenceState:
        """Success-only monitored reverse hand path; never used for fault recovery."""

        if self.state != SequenceState.HOLDING or not self._air_no_contact_hold:
            raise SequenceStateError(
                "air-hand return requires a successful no-contact HOLDING state"
            )
        try:
            self._transition(SequenceState.REOPENING_HAND)
            self.hand.return_no_contact_hand_to_open()
            self._transition(SequenceState.RETURNED_OPEN)
            self._observe_boundary("open", hand_targets=DISABLED_HAND_TARGETS)
        except BaseException as exc:
            raise self._fail(exc) from exc
        return self.state

    def run_full(self, plan: object) -> SequenceState:
        """Run through closure and return only after entering HOLDING."""

        normalized = _normalize_plan(plan)
        self._require_eligible(normalized)
        if self.state not in (SequenceState.DISARMED, SequenceState.DEFAULT_VERIFIED):
            raise SequenceStateError(
                "run_full requires DISARMED or DEFAULT_VERIFIED, got "
                f"{self.state.value}"
            )
        try:
            if self.state == SequenceState.DISARMED:
                self._run_to_default(normalized)
            self._run_after_default(normalized)
        except BaseException as exc:
            if isinstance(exc, SequenceExecutionError):
                raise
            raise self._fail(exc) from exc
        return self.state

    def run_full_joint_waypoints(
        self, plan: AuditedJointSequencePlan
    ) -> SequenceState:
        """Execute only the exact named joint polyline from a passing audit."""

        if not isinstance(plan, AuditedJointSequencePlan):
            raise SequencePlanError(
                "run_full_joint_waypoints requires AuditedJointSequencePlan"
            )
        self._require_eligible(plan)
        if self.state not in (SequenceState.DISARMED, SequenceState.DEFAULT_VERIFIED):
            raise SequenceStateError(
                "run_full_joint_waypoints requires DISARMED or DEFAULT_VERIFIED, got "
                + self.state.value
            )
        try:
            if self.state == SequenceState.DISARMED:
                self._run_to_default_joint_waypoints(plan)
            self._run_after_default_joint_waypoints(plan)
        except BaseException as exc:
            if isinstance(exc, SequenceExecutionError):
                raise
            raise self._fail(exc) from exc
        return self.state

    def run_loaded_lift_joint_waypoints(
        self, plan: AuditedLoadedLiftSequencePlan
    ) -> SequenceState:
        """Execute an independently audited loaded grasp and lift suffix.

        Success stops in ``LIFTED_HOLDING`` with both the Franka external-load
        model and the Inspire numeric hold active.  The only normal continuation
        is :meth:`return_loaded_lift_to_setdown`; generic abort is deliberately
        a manual-recovery fault while the payload may be suspended.
        """

        if not isinstance(plan, AuditedLoadedLiftSequencePlan):
            raise SequencePlanError(
                "run_loaded_lift_joint_waypoints requires "
                "AuditedLoadedLiftSequencePlan"
            )
        self._require_eligible(plan)
        self._require_eligible(plan.grasp_plan)
        if self.state != SequenceState.DISARMED:
            raise SequenceStateError(
                "run_loaded_lift_joint_waypoints requires DISARMED, got "
                + self.state.value
            )
        self._loaded_lift_artifact_sha256 = plan.loaded_lift_artifact_sha256
        self._loaded_hold_targets = tuple(plan.grasp_plan.hand_target6)
        self._loaded_minimum_contact_axes = plan.minimum_contact_axes
        tolerances = (
            plan.settle_tolerances
            if self.settle_tolerances is None
            else self.settle_tolerances
        )
        try:
            self._run_to_default_joint_waypoints(plan.grasp_plan)
            self._run_after_default_joint_waypoints(plan.grasp_plan)
            self._verify_loaded_hold()
            self.arm.apply_external_load_and_verify(plan.payload)
            self._loaded_payload_active = True
            self._observe_boundary(
                "load_applied",
                target_q=plan.lift_waypoints[0][1],
                target_pose=plan.grasp_plan.grasp_pose,
                hand_targets=plan.grasp_plan.hand_target6,
            )

            self._transition(SequenceState.MOVING_LIFT)
            # Set this before the first arm command: even a partially completed
            # move can remove the object from support.
            self._loaded_object_suspended = True
            for name, target_q in plan.lift_waypoints[1:]:
                self._verify_loaded_hold()
                self._move_audited_waypoint(
                    target_q,
                    plan.max_q_tracking_error_rad,
                    waypoint_name=name,
                    loaded_time_law=plan,
                )
                self._verify_loaded_hold()

            self._transition(SequenceState.LIFT_SETTLING)
            self._verify_audited_joint_arrival(
                plan.lift_waypoints[-1][1],
                plan.lift_pose,
                plan.max_q_tracking_error_rad,
                tolerances,
                "loaded lift",
            )
            self._verify_loaded_hold()
            self._transition(SequenceState.LIFTED_HOLDING)
            self._observe_boundary(
                "lift",
                target_q=plan.lift_waypoints[-1][1],
                target_pose=plan.lift_pose,
                hand_targets=plan.grasp_plan.hand_target6,
            )
        except BaseException as exc:
            if isinstance(exc, SequenceExecutionError):
                raise
            raise self._fail(exc) from exc
        return self.state

    def return_loaded_lift_to_setdown(
        self, plan: AuditedLoadedLiftSequencePlan
    ) -> SequenceState:
        """Replay the audited lift suffix in reverse before releasing the load."""

        if not isinstance(plan, AuditedLoadedLiftSequencePlan):
            raise SequencePlanError(
                "return_loaded_lift_to_setdown requires "
                "AuditedLoadedLiftSequencePlan"
            )
        if self.state != SequenceState.LIFTED_HOLDING:
            raise SequenceStateError(
                "loaded setdown requires LIFTED_HOLDING, got " + self.state.value
            )
        if plan.loaded_lift_artifact_sha256 != self._loaded_lift_artifact_sha256:
            raise SequencePlanError(
                "loaded setdown plan differs from the active lift audit artifact"
            )
        if not self._loaded_object_suspended or not self._loaded_payload_active:
            raise SequenceStateError("loaded setdown contract is not active")
        tolerances = (
            plan.settle_tolerances
            if self.settle_tolerances is None
            else self.settle_tolerances
        )
        try:
            self._transition(SequenceState.LOWERING_LOAD)
            for name, target_q in reversed(plan.lift_waypoints[:-1]):
                self._verify_loaded_hold()
                self._move_audited_waypoint(
                    target_q,
                    plan.max_q_tracking_error_rad,
                    waypoint_name="setdown_{}".format(name),
                    loaded_time_law=plan,
                )
                self._verify_loaded_hold()

            self._transition(SequenceState.SETDOWN_SETTLING)
            self._verify_audited_joint_arrival(
                plan.lift_waypoints[0][1],
                plan.grasp_plan.grasp_pose,
                plan.max_q_tracking_error_rad,
                tolerances,
                "loaded setdown",
            )
            self._verify_loaded_hold()
            self._transition(SequenceState.SETDOWN_HOLDING)
            self._observe_boundary(
                "setdown",
                target_q=plan.lift_waypoints[0][1],
                target_pose=plan.grasp_plan.grasp_pose,
                hand_targets=plan.grasp_plan.hand_target6,
            )

            # The independently audited grasp endpoint is the support endpoint.
            # Only after its q/EEF settle gate passes may releasing be attempted.
            self._loaded_object_suspended = False
            self.hand.disable_and_verify()
            self.arm.clear_external_load_and_verify()
            self._loaded_payload_active = False
            self._transition(SequenceState.SETDOWN_COMPLETE)
            self._observe_boundary(
                "setdown_complete",
                target_q=plan.lift_waypoints[0][1],
                target_pose=plan.grasp_plan.grasp_pose,
                hand_targets=DISABLED_HAND_TARGETS,
            )
        except BaseException as exc:
            if isinstance(exc, SequenceExecutionError):
                raise
            raise self._fail(exc) from exc
        return self.state

    def verify_loaded_lift_holding(self) -> SequenceState:
        """Refresh both loaded-hold proofs between FCI control handles.

        A caller may invoke this at a bounded low rate during the lifted hold.
        Any failed arm or hand read enters the suspended-load containment path:
        Franka is stopped and the last numeric Inspire hold is retained.
        """

        if self.state != SequenceState.LIFTED_HOLDING:
            raise SequenceStateError(
                "loaded hold monitoring requires LIFTED_HOLDING, got "
                + self.state.value
            )
        try:
            self.arm.verify_idle_state()
            self._verify_loaded_hold()
        except BaseException as exc:
            if isinstance(exc, SequenceExecutionError):
                raise
            raise self._fail(exc) from exc
        return self.state

    def run_to_default_joint_waypoints(
        self, plan: AuditedJointSequencePlan
    ) -> SequenceState:
        """Open the hand and execute only the audited prefix through default.

        This is the staged commissioning counterpart of
        :meth:`run_full_joint_waypoints`.  It consumes the exact same immutable
        joint polyline, but deliberately stops before the pregrasp waypoint so
        the caller can disable output, capture a new scene at default, and
        build a new audit from the fresh robot state.
        """

        if not isinstance(plan, AuditedJointSequencePlan):
            raise SequencePlanError(
                "run_to_default_joint_waypoints requires AuditedJointSequencePlan"
            )
        self._require_eligible(plan)
        if self.state == SequenceState.DEFAULT_VERIFIED:
            expected_default = next(
                target
                for name, target in plan.joint_waypoints
                if name == "default"
            )
            if self._prepared_default_q is not None and not np.array_equal(
                self._prepared_default_q, expected_default
            ):
                raise SequencePlanError(
                    "audited default waypoint differs from the already verified pose"
                )
            return self.state
        if self.state != SequenceState.DISARMED:
            raise SequenceStateError(
                "run_to_default_joint_waypoints requires DISARMED, got "
                + self.state.value
            )
        try:
            self._run_to_default_joint_waypoints(plan)
        except BaseException as exc:
            if isinstance(exc, SequenceExecutionError):
                raise
            raise self._fail(exc) from exc
        return self.state

    def run_to_pregrasp_joint_waypoints(
        self, plan: AuditedPregraspSequencePlan
    ) -> SequenceState:
        """Execute exactly one dedicated audited open-hand prefix.

        This method has no grasp pose, q6 target, bend targets, contact stage,
        or continuation API.  On success the only legal next transition is a
        coordinated stop/disable through :meth:`abort`.
        """

        if not isinstance(plan, AuditedPregraspSequencePlan):
            raise SequencePlanError(
                "run_to_pregrasp_joint_waypoints requires AuditedPregraspSequencePlan"
            )
        self._require_eligible(plan)
        if self.state != SequenceState.DISARMED:
            raise SequenceStateError(
                "run_to_pregrasp_joint_waypoints requires DISARMED, got "
                + self.state.value
            )
        try:
            self._run_pregrasp_only_joint_waypoints(plan)
        except BaseException as exc:
            if isinstance(exc, SequenceExecutionError):
                raise
            raise self._fail(exc) from exc
        return self.state

    def abort(self, reason: str = "abort requested") -> SequenceState:
        """Stop arm output first, then disable and verify hand output."""

        if self.state == SequenceState.STOPPED:
            return self.state
        if self.state in (SequenceState.STOPPING, SequenceState.FAULT_LATCHED):
            raise SequenceStateError(f"cannot abort from terminal state {self.state.value}")
        if self._loaded_object_suspended:
            error = RuntimeError(
                str(reason)
                + "; loaded object may be suspended, so Inspire hold was not disabled"
            )
            raise self._fail(error)
        self._transition(SequenceState.STOPPING)
        stop_errors = self._stop_drivers()
        if stop_errors:
            self._transition(SequenceState.FAULT_LATCHED)
            error = SequenceExecutionError(
                RuntimeError(str(reason)),
                failure_state=SequenceState.STOPPING,
                stop_errors=stop_errors,
            )
            self.last_error = error
            raise error
        self._transition(SequenceState.STOPPED)
        return self.state


__all__ = [
    "ArmSequenceDriver",
    "AuditedJointSequencePlan",
    "AuditedLoadedLiftSequencePlan",
    "AuditedPregraspSequencePlan",
    "DefaultSequencePlan",
    "EEFNotSettledError",
    "HandSequenceDriver",
    "LoadedLiftPayload",
    "OPEN_HAND_TARGETS",
    "SequenceExecutionError",
    "SequencePlan",
    "SequencePlanError",
    "SequenceState",
    "SequenceStateError",
    "SettleTolerances",
    "StagedGraspSequence",
]
