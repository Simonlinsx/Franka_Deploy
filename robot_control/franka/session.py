"""Hardware-independent persistent Franka joint-control session contract.

This module intentionally imports no Franka binding and cannot discover or
open a robot by itself.  A future motion-bearing entry point must inject an
explicit backend factory, a run-scoped :class:`MotionAuthorization`, and a
sealed token made from an already-passing commissioning report.  The factory
is not called until all offline, authorization, interlock, command-freshness,
and run-binding checks pass.

The implementation is a deterministic reference for the unique 1 kHz owner:

* exactly one active-handle read is performed per control cycle;
* that state is the sole source for dynamic checks and the pose ring;
* a sequenced 60 Hz target is sampled and held at the FCI rate;
* contract-aware target admission and per-cycle command/state velocity,
  acceleration, jerk, tracking, joint, contact, mode, error, communication,
  and age limits fail closed; and
* every exit requests a stop and verifies consecutive fresh idle states.

No limit in this module is a motion default.  Every numerical envelope comes
from the exact commissioning profile whose SHA-256 is bound by the preflight
report.  The current V94 profile deliberately fails to load while its online
acceleration, jerk, and tracking limits remain uncommissioned (``null``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Deque, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from sim2real.closed_loop_core import (
    AUTHORIZATION_SCOPE,
    ClosedLoopCommand,
    ClosedLoopSafetyGate,
    ExecutedActionLedger,
    MotionAuthorization,
    SafetyState,
)
from sim2real.contracts.actions import (
    LEGACY_FRANKA_ACTION_CONTRACT_ID,
    QD_G015_FRANKA_ACTION_CONTRACT_ID,
    normalize_franka_action_contract_id,
)


class FrankaPersistentSessionError(RuntimeError):
    """A terminal persistent-session validation or control fault."""


class FrankaSessionMode(str, Enum):
    C1_COMMISSIONING = "c1_commissioning"
    C2_V94_POLICY = "c2_v94_policy"
    SUPERVISED_V94 = "supervised_v94_experimental_non_c2"


class FrankaSessionState(str, Enum):
    DISABLED = "disabled"
    RUNNING = "running"
    STOPPED = "stopped"
    FAULT_LATCHED = "fault_latched"


class FrankaTargetSource(str, Enum):
    C1_DETERMINISTIC = "c1_deterministic"
    C2_V94_POLICY = "c2_v94_policy"
    SUPERVISED_V94 = "supervised_v94_experimental_non_c2"


def _is_transactional_policy_mode(mode: FrankaSessionMode) -> bool:
    return mode in (
        FrankaSessionMode.C2_V94_POLICY,
        FrankaSessionMode.SUPERVISED_V94,
    )


def _target_source_for_mode(mode: FrankaSessionMode) -> FrankaTargetSource:
    if mode is FrankaSessionMode.C2_V94_POLICY:
        return FrankaTargetSource.C2_V94_POLICY
    if mode is FrankaSessionMode.SUPERVISED_V94:
        return FrankaTargetSource.SUPERVISED_V94
    return FrankaTargetSource.C1_DETERMINISTIC


_ENVELOPE_SEAL = object()
_PREFLIGHT_TOKEN_SEAL = object()
_SUPERVISED_PREFLIGHT_TOKEN_SEAL = object()
_SHA256_HEX = frozenset("0123456789abcdef")

# The first bootstrap cycle may report a zero Duration and includes one-time
# Python object initialization.  It therefore has a separate bounded deadline;
# every later supervised cycle uses the tighter envelope deadline.
SUPERVISED_BOOTSTRAP_READ_TO_WRITE_DEADLINE_S = 0.0015


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_scalar(value: object, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _readonly_vector(value: object, size: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} finite values") from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    result = result.copy()
    result.setflags(write=False)
    return result


def _positive_vector(value: object, size: int, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive scalar or {size}-vector") from exc
    if raw.shape == ():
        raw = np.full(size, float(raw), dtype=np.float64)
    result = _readonly_vector(raw, size, name)
    if np.any(result <= 0.0):
        raise ValueError(f"{name} must be positive on every axis")
    return result


def _readonly_matrix(
    value: object, shape: Tuple[int, int], name: str
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have finite shape {shape}") from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have finite shape {shape}")
    result = result.copy()
    result.setflags(write=False)
    return result


def _sha256_text(value: str, name: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(character not in _SHA256_HEX for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _load_strict_json_bytes(path: Path) -> Tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read commissioning profile {path}: {exc}") from exc
    if not payload:
        raise ValueError("commissioning profile must be non-empty")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON number is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid commissioning profile JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("commissioning profile must contain a JSON object")
    return decoded, hashlib.sha256(payload).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


@dataclass(frozen=True, init=False)
class CommissionedFrankaEnvelope:
    """Exact, profile-bound limits for one persistent session.

    Instances can only be made by :func:`load_commissioned_franka_envelope`.
    This prevents a production caller from replacing uncommissioned ``null``
    limits with convenient runtime defaults.
    """

    profile_path: str
    profile_sha256: str
    expected_F_T_EE: np.ndarray
    expected_end_effector_mass_kg: float
    expected_end_effector_com_m: np.ndarray
    expected_end_effector_inertia_kg_m2: np.ndarray
    expected_external_load_mass_kg: float
    expected_external_load_com_m: np.ndarray
    expected_external_load_inertia_kg_m2: np.ndarray
    F_T_EE_tolerance: float
    mass_tolerance_kg: float
    center_of_mass_tolerance_m: float
    inertia_tolerance_kg_m2: float
    joint_limits_rad: np.ndarray
    joint_limit_margin_rad: float
    maximum_velocity_rad_s: np.ndarray
    maximum_target_rate_rad_s: np.ndarray
    maximum_acceleration_rad_s2: np.ndarray
    maximum_jerk_rad_s3: np.ndarray
    maximum_measured_acceleration_rad_s2: np.ndarray
    maximum_measured_jerk_rad_s3: np.ndarray
    maximum_tracking_error_rad: np.ndarray
    initial_target_tolerance_rad: np.ndarray
    target_reached_tolerance_rad: np.ndarray
    control_period_min_s: float
    control_period_max_s: float
    policy_command_max_age_s: float
    read_to_write_deadline_s: float
    minimum_control_success_rate: float
    maximum_session_duration_s: float
    stop_maximum_velocity_rad_s: np.ndarray
    stop_consecutive_samples: int
    stop_maximum_samples: int
    pose_ring_capacity: int
    allow_one_initial_zero_period: bool

    def __init__(self, *, _seal: object, **values: Any) -> None:
        if _seal is not _ENVELOPE_SEAL:
            raise TypeError(
                "CommissionedFrankaEnvelope must be loaded from an exact profile"
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def safe_joint_lower_rad(self) -> np.ndarray:
        result = self.joint_limits_rad[:, 0] + self.joint_limit_margin_rad
        result.setflags(write=False)
        return result

    @property
    def safe_joint_upper_rad(self) -> np.ndarray:
        result = self.joint_limits_rad[:, 1] - self.joint_limit_margin_rad
        result.setflags(write=False)
        return result

    @property
    def binding_sha256(self) -> str:
        """Hash the exact numeric runtime envelope, independent of JSON order."""

        value = {
            "profile_sha256": self.profile_sha256,
            "expected_F_T_EE": self.expected_F_T_EE.tolist(),
            "expected_end_effector_mass_kg": self.expected_end_effector_mass_kg,
            "expected_end_effector_com_m": self.expected_end_effector_com_m.tolist(),
            "expected_end_effector_inertia_kg_m2": (
                self.expected_end_effector_inertia_kg_m2.tolist()
            ),
            "expected_external_load_mass_kg": self.expected_external_load_mass_kg,
            "expected_external_load_com_m": self.expected_external_load_com_m.tolist(),
            "expected_external_load_inertia_kg_m2": (
                self.expected_external_load_inertia_kg_m2.tolist()
            ),
            "F_T_EE_tolerance": self.F_T_EE_tolerance,
            "mass_tolerance_kg": self.mass_tolerance_kg,
            "center_of_mass_tolerance_m": self.center_of_mass_tolerance_m,
            "inertia_tolerance_kg_m2": self.inertia_tolerance_kg_m2,
            "joint_limits_rad": self.joint_limits_rad.tolist(),
            "joint_limit_margin_rad": self.joint_limit_margin_rad,
            "maximum_velocity_rad_s": self.maximum_velocity_rad_s.tolist(),
            "maximum_target_rate_rad_s": self.maximum_target_rate_rad_s.tolist(),
            "maximum_acceleration_rad_s2": self.maximum_acceleration_rad_s2.tolist(),
            "maximum_jerk_rad_s3": self.maximum_jerk_rad_s3.tolist(),
            "maximum_measured_acceleration_rad_s2": (
                self.maximum_measured_acceleration_rad_s2.tolist()
            ),
            "maximum_measured_jerk_rad_s3": (
                self.maximum_measured_jerk_rad_s3.tolist()
            ),
            "maximum_tracking_error_rad": self.maximum_tracking_error_rad.tolist(),
            "initial_target_tolerance_rad": self.initial_target_tolerance_rad.tolist(),
            "target_reached_tolerance_rad": self.target_reached_tolerance_rad.tolist(),
            "control_period_min_s": self.control_period_min_s,
            "control_period_max_s": self.control_period_max_s,
            "policy_command_max_age_s": self.policy_command_max_age_s,
            "read_to_write_deadline_s": self.read_to_write_deadline_s,
            "minimum_control_success_rate": self.minimum_control_success_rate,
            "maximum_session_duration_s": self.maximum_session_duration_s,
            "stop_maximum_velocity_rad_s": self.stop_maximum_velocity_rad_s.tolist(),
            "stop_consecutive_samples": self.stop_consecutive_samples,
            "stop_maximum_samples": self.stop_maximum_samples,
            "pose_ring_capacity": self.pose_ring_capacity,
            "allow_one_initial_zero_period": self.allow_one_initial_zero_period,
        }
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_commissioned_franka_envelope(
    profile_path: str | Path,
) -> CommissionedFrankaEnvelope:
    """Load all persistent limits from one immutable profile snapshot.

    The caller cannot pass replacements for missing values.  In particular,
    ``online_max_joint_acceleration_rad_s2``,
    ``online_max_joint_jerk_rad_s3``, and
    ``online_max_tracking_error_rad`` must be commissioned and non-null.
    """

    source = Path(profile_path).expanduser().resolve()
    profile, profile_sha256 = _load_strict_json_bytes(source)
    if profile.get("schema_version") != 1:
        raise ValueError("commissioning profile schema_version must be 1")
    franka = _mapping(profile.get("franka"), "franka")
    session = _mapping(franka.get("persistent_session"), "franka.persistent_session")
    expected_end_effector = _mapping(
        franka.get("expected_end_effector"), "franka.expected_end_effector"
    )
    expected_external_load = _mapping(
        session.get("expected_external_load"),
        "franka.persistent_session.expected_external_load",
    )

    expected_F_T_EE = _readonly_matrix(
        franka.get("expected_F_T_EE"), (4, 4), "franka.expected_F_T_EE"
    )
    if not np.allclose(
        expected_F_T_EE[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("franka.expected_F_T_EE bottom row is invalid")
    expected_ee_mass = _positive_scalar(
        expected_end_effector.get("mass_kg"),
        "franka.expected_end_effector.mass_kg",
    )
    expected_ee_com = _readonly_vector(
        expected_end_effector.get("F_x_Cee_m"),
        3,
        "franka.expected_end_effector.F_x_Cee_m",
    )
    expected_ee_inertia = _readonly_matrix(
        expected_end_effector.get("inertia_kg_m2"),
        (3, 3),
        "franka.expected_end_effector.inertia_kg_m2",
    )
    expected_load_mass = _finite_scalar(
        expected_external_load.get("mass_kg"),
        "franka.persistent_session.expected_external_load.mass_kg",
    )
    if expected_load_mass < 0.0:
        raise ValueError("expected external load mass cannot be negative")
    expected_load_com = _readonly_vector(
        expected_external_load.get("F_x_Cload_m"),
        3,
        "franka.persistent_session.expected_external_load.F_x_Cload_m",
    )
    expected_load_inertia = _readonly_matrix(
        expected_external_load.get("inertia_kg_m2"),
        (3, 3),
        "franka.persistent_session.expected_external_load.inertia_kg_m2",
    )
    for name, inertia in (
        ("franka.expected_end_effector.inertia_kg_m2", expected_ee_inertia),
        (
            "franka.persistent_session.expected_external_load.inertia_kg_m2",
            expected_load_inertia,
        ),
    ):
        if not np.allclose(inertia, inertia.T, atol=1.0e-10, rtol=0.0):
            raise ValueError(f"{name} must be symmetric")
        if float(np.min(np.linalg.eigvalsh(inertia))) < -1.0e-10:
            raise ValueError(f"{name} must be positive semidefinite")
    rotation = expected_F_T_EE[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0):
        raise ValueError("franka.expected_F_T_EE rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6, rtol=0.0):
        raise ValueError("franka.expected_F_T_EE rotation determinant is not +1")

    limits = np.asarray(franka.get("joint_limits_rad"), dtype=np.float64)
    if limits.shape != (7, 2) or not np.all(np.isfinite(limits)):
        raise ValueError("franka.joint_limits_rad must have finite shape (7,2)")
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("franka.joint_limits_rad intervals are invalid")
    limits = limits.copy()
    limits.setflags(write=False)
    margin = _positive_scalar(
        franka.get("joint_limit_margin_rad"), "franka.joint_limit_margin_rad"
    )
    if np.any(2.0 * margin >= limits[:, 1] - limits[:, 0]):
        raise ValueError("joint-limit margin consumes a joint interval")

    maximum_velocity = _positive_vector(
        franka.get("default_max_joint_velocity_rad_s"),
        7,
        "franka.default_max_joint_velocity_rad_s",
    )
    maximum_acceleration = _positive_vector(
        franka.get("online_max_joint_acceleration_rad_s2"),
        7,
        "franka.online_max_joint_acceleration_rad_s2",
    )
    maximum_jerk = _positive_vector(
        franka.get("online_max_joint_jerk_rad_s3"),
        7,
        "franka.online_max_joint_jerk_rad_s3",
    )
    tracking = _positive_vector(
        franka.get("online_max_tracking_error_rad"),
        7,
        "franka.online_max_tracking_error_rad",
    )
    initial_target_tolerance = _positive_vector(
        session.get("initial_target_tolerance_rad"),
        7,
        "franka.persistent_session.initial_target_tolerance_rad",
    )
    target_reached_tolerance = _positive_vector(
        session.get("target_reached_tolerance_rad"),
        7,
        "franka.persistent_session.target_reached_tolerance_rad",
    )
    period_min = _positive_scalar(
        session.get("control_period_min_s"),
        "franka.persistent_session.control_period_min_s",
    )
    period_max = _positive_scalar(
        session.get("control_period_max_s"),
        "franka.persistent_session.control_period_max_s",
    )
    if period_min > period_max:
        raise ValueError("control_period_min_s must not exceed control_period_max_s")
    success_rate = _finite_scalar(
        session.get("minimum_control_success_rate"),
        "franka.persistent_session.minimum_control_success_rate",
    )
    if not 0.0 <= success_rate <= 1.0:
        raise ValueError("minimum_control_success_rate must lie in [0,1]")
    result = CommissionedFrankaEnvelope(
        _seal=_ENVELOPE_SEAL,
        profile_path=str(source),
        profile_sha256=profile_sha256,
        expected_F_T_EE=expected_F_T_EE,
        expected_end_effector_mass_kg=expected_ee_mass,
        expected_end_effector_com_m=expected_ee_com,
        expected_end_effector_inertia_kg_m2=expected_ee_inertia,
        expected_external_load_mass_kg=expected_load_mass,
        expected_external_load_com_m=expected_load_com,
        expected_external_load_inertia_kg_m2=expected_load_inertia,
        F_T_EE_tolerance=_positive_scalar(
            session.get("F_T_EE_tolerance"),
            "franka.persistent_session.F_T_EE_tolerance",
        ),
        mass_tolerance_kg=_positive_scalar(
            session.get("mass_tolerance_kg"),
            "franka.persistent_session.mass_tolerance_kg",
        ),
        center_of_mass_tolerance_m=_positive_scalar(
            session.get("center_of_mass_tolerance_m"),
            "franka.persistent_session.center_of_mass_tolerance_m",
        ),
        inertia_tolerance_kg_m2=_positive_scalar(
            session.get("inertia_tolerance_kg_m2"),
            "franka.persistent_session.inertia_tolerance_kg_m2",
        ),
        joint_limits_rad=limits,
        joint_limit_margin_rad=margin,
        maximum_velocity_rad_s=maximum_velocity,
        maximum_target_rate_rad_s=maximum_velocity,
        maximum_acceleration_rad_s2=maximum_acceleration,
        maximum_jerk_rad_s3=maximum_jerk,
        maximum_measured_acceleration_rad_s2=maximum_acceleration,
        maximum_measured_jerk_rad_s3=maximum_jerk,
        maximum_tracking_error_rad=tracking,
        initial_target_tolerance_rad=initial_target_tolerance,
        target_reached_tolerance_rad=target_reached_tolerance,
        control_period_min_s=period_min,
        control_period_max_s=period_max,
        policy_command_max_age_s=_positive_scalar(
            session.get("policy_command_max_age_s"),
            "franka.persistent_session.policy_command_max_age_s",
        ),
        read_to_write_deadline_s=_positive_scalar(
            session.get("read_to_write_deadline_s"),
            "franka.persistent_session.read_to_write_deadline_s",
        ),
        minimum_control_success_rate=success_rate,
        maximum_session_duration_s=_positive_scalar(
            session.get("maximum_session_duration_s"),
            "franka.persistent_session.maximum_session_duration_s",
        ),
        stop_maximum_velocity_rad_s=_positive_vector(
            session.get("stop_maximum_velocity_rad_s"),
            7,
            "franka.persistent_session.stop_maximum_velocity_rad_s",
        ),
        stop_consecutive_samples=_positive_integer(
            session.get("stop_consecutive_samples"),
            "franka.persistent_session.stop_consecutive_samples",
        ),
        stop_maximum_samples=_positive_integer(
            session.get("stop_maximum_samples"),
            "franka.persistent_session.stop_maximum_samples",
        ),
        pose_ring_capacity=_positive_integer(
            session.get("pose_ring_capacity"),
            "franka.persistent_session.pose_ring_capacity",
        ),
        allow_one_initial_zero_period=_strict_bool(
            session.get("allow_one_initial_zero_period"),
            "franka.persistent_session.allow_one_initial_zero_period",
        ),
    )
    if result.stop_consecutive_samples > result.stop_maximum_samples:
        raise ValueError("stop_consecutive_samples exceeds stop_maximum_samples")
    if result.read_to_write_deadline_s >= result.control_period_max_s:
        raise ValueError("read_to_write_deadline_s must be below control_period_max_s")
    return result


class ExperimentalSupervisedFrankaEnvelope(CommissionedFrankaEnvelope):
    """Explicitly non-commissioned envelope for a <=720-step supervised run.

    It is intentionally a distinct type: callers and audit logs can never
    mistake these conservative, code-pinned limits for profile-commissioned
    C2 limits.  Construction remains sealed to this module.
    """


def load_experimental_supervised_franka_envelope(
    profile_path: str | Path,
) -> ExperimentalSupervisedFrankaEnvelope:
    """Bind the V94 profile while supplying fixed experimental online guards.

    The current V94 commissioning profile leaves persistent acceleration,
    jerk, and tracking fields null.  The operator-supervised runner therefore
    uses this conspicuously non-C2 envelope.  These are ceilings, not claims
    that the same values have passed sustained C2 commissioning.
    """

    source = Path(profile_path).expanduser().resolve()
    profile, profile_sha256 = _load_strict_json_bytes(source)
    if profile.get("schema_version") != 1:
        raise ValueError("commissioning profile schema_version must be 1")
    if profile.get("mode") != "commissioning_locked":
        raise ValueError("supervised V94 requires the commissioning_locked profile")
    franka = _mapping(profile.get("franka"), "franka")
    tool = _mapping(profile.get("tool"), "tool")
    if tool.get("installed_on_franka_verified") is not True:
        raise ValueError("installed RH56/adapter configuration is not verified")
    expected_end_effector = _mapping(
        franka.get("expected_end_effector"), "franka.expected_end_effector"
    )
    expected_F_T_EE = _readonly_matrix(
        franka.get("expected_F_T_EE"), (4, 4), "franka.expected_F_T_EE"
    )
    rotation = expected_F_T_EE[:3, :3]
    if not np.allclose(expected_F_T_EE[3], [0, 0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError("franka.expected_F_T_EE bottom row is invalid")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("franka.expected_F_T_EE rotation is not orthonormal")
    limits = _readonly_matrix(
        franka.get("joint_limits_rad"), (7, 2), "franka.joint_limits_rad"
    )
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("franka.joint_limits_rad intervals are invalid")
    margin = _positive_scalar(
        franka.get("joint_limit_margin_rad"), "franka.joint_limit_margin_rad"
    )
    if np.any(2.0 * margin >= limits[:, 1] - limits[:, 0]):
        raise ValueError("joint-limit margin consumes a joint interval")
    profile_speed = _positive_scalar(
        franka.get("default_max_joint_velocity_rad_s"),
        "franka.default_max_joint_velocity_rad_s",
    )
    # The profile value remains hash-bound provenance.  Online supervised
    # execution has a separately displayed 0.50 rad/s ceiling, which covers
    # the observed 0.405 rad/s peak in the v205 successful simulator trace.
    inertia = _readonly_matrix(
        expected_end_effector.get("inertia_kg_m2"),
        (3, 3),
        "franka.expected_end_effector.inertia_kg_m2",
    )
    zero3 = _readonly_vector([0.0, 0.0, 0.0], 3, "zero external load COM")
    zero33 = _readonly_matrix(np.zeros((3, 3)), (3, 3), "zero external load inertia")
    return ExperimentalSupervisedFrankaEnvelope(
        _seal=_ENVELOPE_SEAL,
        profile_path=str(source),
        profile_sha256=profile_sha256,
        expected_F_T_EE=expected_F_T_EE,
        expected_end_effector_mass_kg=_positive_scalar(
            expected_end_effector.get("mass_kg"),
            "franka.expected_end_effector.mass_kg",
        ),
        expected_end_effector_com_m=_readonly_vector(
            expected_end_effector.get("F_x_Cee_m"),
            3,
            "franka.expected_end_effector.F_x_Cee_m",
        ),
        expected_end_effector_inertia_kg_m2=inertia,
        expected_external_load_mass_kg=0.0,
        expected_external_load_com_m=zero3,
        expected_external_load_inertia_kg_m2=zero33,
        F_T_EE_tolerance=1.0e-8,
        mass_tolerance_kg=1.0e-5,
        center_of_mass_tolerance_m=1.0e-6,
        inertia_tolerance_kg_m2=1.0e-6,
        joint_limits_rad=limits,
        joint_limit_margin_rad=margin,
        maximum_velocity_rad_s=_positive_vector(0.50, 7, "speed"),
        maximum_target_rate_rad_s=_positive_vector(0.50, 7, "target rate"),
        maximum_acceleration_rad_s2=_positive_vector(5.0, 7, "acceleration"),
        maximum_jerk_rad_s3=_positive_vector(250.0, 7, "jerk"),
        # Raw dq finite differences are noisy.  These remain terminal
        # diagnostic bounds, but are intentionally distinct from the much
        # tighter command-path limits above.  This legacy Python
        # production native supervised session uses a separate 0.70 rad/s
        # measured-dq fault boundary above the 0.50 rad/s command ceiling.
        maximum_measured_acceleration_rad_s2=_positive_vector(
            20.0, 7, "measured acceleration diagnostic"
        ),
        maximum_measured_jerk_rad_s3=_positive_vector(
            20000.0, 7, "measured jerk diagnostic"
        ),
        maximum_tracking_error_rad=_positive_vector(0.01, 7, "tracking"),
        initial_target_tolerance_rad=_positive_vector(0.01, 7, "initial target"),
        target_reached_tolerance_rad=_positive_vector(0.05, 7, "target following"),
        control_period_min_s=0.0005,
        control_period_max_s=0.002,
        policy_command_max_age_s=0.05,
        # The first real supervised attempt showed that a 0.4 ms Python-side
        # budget was too small before any write.  Full static provenance is
        # now prevalidated read-only; 0.8 ms remains below the nominal 1 ms
        # FCI cycle and preserves a hard fail-before-write deadline.
        read_to_write_deadline_s=0.0008,
        minimum_control_success_rate=0.90,
        # 720 targets at 60 Hz span 12 seconds.  The remaining three seconds
        # cover the fixed healthy-control bootstrap, first-target handoff, and
        # final acknowledged hold/stop request.
        maximum_session_duration_s=15.0,
        stop_maximum_velocity_rad_s=_positive_vector(0.01, 7, "stop velocity"),
        stop_consecutive_samples=3,
        stop_maximum_samples=20,
        pose_ring_capacity=256,
        allow_one_initial_zero_period=True,
    )


@dataclass(frozen=True, init=False)
class SupervisedFrankaPreflightToken:
    """Run-bound permit for the explicit experimental non-C2 session mode."""

    run_id: str
    stage: FrankaSessionMode
    evidence_run_id: str
    report_sha256: str
    profile_sha256: str
    envelope_sha256: str
    issued_monotonic_s: float
    expires_monotonic_s: float
    classification: str = "experimental_operator_supervised_non_c2"

    def __init__(self, *, _seal: object, **values: Any) -> None:
        if _seal is not _SUPERVISED_PREFLIGHT_TOKEN_SEAL:
            raise TypeError(
                "SupervisedFrankaPreflightToken must come from the interactive runner"
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def require_active(
        self,
        *,
        run_id: str,
        stage: FrankaSessionMode,
        envelope: CommissionedFrankaEnvelope,
        now_monotonic_s: object,
    ) -> None:
        now = _finite_scalar(now_monotonic_s, "now_monotonic_s")
        if stage is not FrankaSessionMode.SUPERVISED_V94:
            raise FrankaPersistentSessionError(
                "supervised token cannot authorize a C1 or C2 session"
            )
        if not isinstance(envelope, ExperimentalSupervisedFrankaEnvelope):
            raise FrankaPersistentSessionError(
                "supervised token requires the explicit experimental envelope"
            )
        if str(run_id).strip() != self.run_id:
            raise FrankaPersistentSessionError("supervised token run ID mismatch")
        if envelope.profile_sha256 != self.profile_sha256:
            raise FrankaPersistentSessionError("supervised profile hash changed")
        if envelope.binding_sha256 != self.envelope_sha256:
            raise FrankaPersistentSessionError("supervised envelope changed")
        if not self.issued_monotonic_s <= now < self.expires_monotonic_s:
            raise FrankaPersistentSessionError("supervised token is not active")


def _issue_supervised_franka_preflight_token(
    *,
    envelope: ExperimentalSupervisedFrankaEnvelope,
    run_id: str,
    confirmed_permit_sha256: str,
    issued_monotonic_s: object,
    expires_monotonic_s: object,
) -> SupervisedFrankaPreflightToken:
    """Internal issuer called only after the CLI seals its interactive permit."""

    if not isinstance(envelope, ExperimentalSupervisedFrankaEnvelope):
        raise TypeError("supervised envelope type is required")
    active_run = str(run_id).strip()
    if not active_run:
        raise ValueError("run_id must be non-empty")
    digest = _sha256_text(confirmed_permit_sha256, "confirmed_permit_sha256")
    issued = _finite_scalar(issued_monotonic_s, "issued_monotonic_s")
    expires = _finite_scalar(expires_monotonic_s, "expires_monotonic_s")
    if expires <= issued:
        raise ValueError("supervised token must expire after issue")
    return SupervisedFrankaPreflightToken(
        _seal=_SUPERVISED_PREFLIGHT_TOKEN_SEAL,
        run_id=active_run,
        stage=FrankaSessionMode.SUPERVISED_V94,
        evidence_run_id=f"interactive:{active_run}",
        report_sha256=digest,
        profile_sha256=envelope.profile_sha256,
        envelope_sha256=envelope.binding_sha256,
        issued_monotonic_s=issued,
        expires_monotonic_s=expires,
        classification="experimental_operator_supervised_non_c2",
    )


@dataclass(frozen=True, init=False)
class VerifiedFrankaPreflightToken:
    """Sealed, expiring binding from a passing offline preflight report."""

    run_id: str
    stage: FrankaSessionMode
    evidence_run_id: str
    report_sha256: str
    profile_sha256: str
    envelope_sha256: str
    issued_monotonic_s: float
    expires_monotonic_s: float

    def __init__(self, *, _seal: object, **values: Any) -> None:
        if _seal is not _PREFLIGHT_TOKEN_SEAL:
            raise TypeError(
                "VerifiedFrankaPreflightToken must come from a passing report"
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def require_active(
        self,
        *,
        run_id: str,
        stage: FrankaSessionMode,
        envelope: CommissionedFrankaEnvelope,
        now_monotonic_s: object,
    ) -> None:
        now = _finite_scalar(now_monotonic_s, "now_monotonic_s")
        if str(run_id).strip() != self.run_id:
            raise FrankaPersistentSessionError("preflight token run ID mismatch")
        if stage is FrankaSessionMode.C2_V94_POLICY and self.stage is not stage:
            raise FrankaPersistentSessionError("C2 session requires a C2 preflight token")
        if envelope.profile_sha256 != self.profile_sha256:
            raise FrankaPersistentSessionError("commissioning profile hash changed")
        if envelope.binding_sha256 != self.envelope_sha256:
            raise FrankaPersistentSessionError("runtime Franka envelope changed")
        if not self.issued_monotonic_s <= now < self.expires_monotonic_s:
            raise FrankaPersistentSessionError("preflight token is not active")


def verified_franka_preflight_token_from_report(
    report: Mapping[str, Any],
    *,
    envelope: CommissionedFrankaEnvelope,
    run_id: str,
    stage: FrankaSessionMode,
    issued_monotonic_s: object,
    expires_monotonic_s: object,
) -> VerifiedFrankaPreflightToken:
    """Seal a passing, artifact-bound C1/C2 preflight for one future run.

    This function does not authorize motion.  It only proves that an immutable
    passing report was checked and that its exact commissioning profile matches
    the numerical envelope loaded above.
    """

    if not isinstance(report, Mapping):
        raise TypeError("preflight report must be a mapping")
    if not isinstance(envelope, CommissionedFrankaEnvelope):
        raise TypeError("envelope must be CommissionedFrankaEnvelope")
    if not isinstance(stage, FrankaSessionMode):
        raise TypeError("stage must be FrankaSessionMode")
    active_run_id = str(run_id).strip()
    if not active_run_id:
        raise ValueError("run_id must be non-empty")
    issued = _finite_scalar(issued_monotonic_s, "issued_monotonic_s")
    expires = _finite_scalar(expires_monotonic_s, "expires_monotonic_s")
    if expires <= issued:
        raise ValueError("preflight token must expire after it is issued")

    expected_readiness = (
        "C2_BOUNDED_CLOSED_LOOP"
        if stage is FrankaSessionMode.C2_V94_POLICY
        else "C1_SUPERVISED_CONTROL_COMMISSIONING"
    )
    boolean_requirements = {
        "offline_only": True,
        "device_access": False,
        "hardware_writes": False,
        "robot_command_writes": False,
        "motion_authorization_created": False,
        "physical_motion_authorized": False,
        "eligible_for_operator_authorization": True,
    }
    failures = []
    if report.get("result") != "PASS":
        failures.append("result")
    if report.get("readiness_level") != expected_readiness:
        failures.append("readiness_level")
    if report.get("arming_state") != SafetyState.DISARMED.value.upper():
        failures.append("arming_state")
    if report.get("future_authorization_scope_required") != AUTHORIZATION_SCOPE:
        failures.append("authorization_scope")
    if report.get("failed_checks") != [] or report.get("blockers") != []:
        failures.append("failed_checks")
    for name, expected in boolean_requirements.items():
        if report.get(name) is not expected:
            failures.append(name)

    inputs = report.get("inputs")
    hashes: Mapping[str, Any] = {}
    if isinstance(inputs, Mapping) and isinstance(inputs.get("sha256"), Mapping):
        hashes = inputs["sha256"]
    if hashes.get("commissioning_profile_sha256") != envelope.profile_sha256:
        failures.append("commissioning_profile_sha256")

    checks = report.get("checks")
    indexed: dict[str, Mapping[str, Any]] = {}
    if isinstance(checks, Sequence) and not isinstance(checks, (str, bytes)):
        for item in checks:
            if isinstance(item, Mapping) and isinstance(item.get("code"), str):
                indexed[str(item["code"])] = item
    required_checks = (
        "evidence_installed_payload_dynamics",
        "evidence_installed_fr3_adapter_rh56_collision_model",
        "evidence_physical_deadman_acceptance",
        "evidence_physical_emergency_stop_acceptance",
        "evidence_franka_persistent_fci_1khz_session",
        "evidence_franka_single_owner_state_source",
        "evidence_franka_online_velocity_acceleration_jerk_limits",
        "evidence_franka_command_watchdog_and_fault_stop",
        "evidence_franka_tracking_and_collision_monitoring",
        "evidence_metric_franka_control_loop_rate_hz",
        "evidence_metric_policy_command_watchdog_timeout_s",
    )
    for code in required_checks:
        if indexed.get(code, {}).get("passed") is not True:
            failures.append(code)
    watchdog_check = indexed.get(
        "evidence_metric_policy_command_watchdog_timeout_s", {}
    )
    try:
        verified_watchdog_s = float(watchdog_check.get("actual"))
    except (TypeError, ValueError):
        verified_watchdog_s = math.nan
    if not math.isfinite(verified_watchdog_s) or not math.isclose(
        verified_watchdog_s,
        envelope.policy_command_max_age_s,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        failures.append("policy_command_watchdog_envelope_binding")
    if failures:
        raise ValueError(
            "preflight report cannot mint a Franka token: "
            + ", ".join(sorted(set(failures)))
        )

    evidence_run_id = str(report.get("run_id") or "").strip()
    if not evidence_run_id:
        raise ValueError("preflight report evidence run ID must be non-empty")
    try:
        encoded = json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("preflight report must be canonical JSON data") from exc
    return VerifiedFrankaPreflightToken(
        _seal=_PREFLIGHT_TOKEN_SEAL,
        run_id=active_run_id,
        stage=stage,
        evidence_run_id=evidence_run_id,
        report_sha256=hashlib.sha256(encoded).hexdigest(),
        profile_sha256=envelope.profile_sha256,
        envelope_sha256=envelope.binding_sha256,
        issued_monotonic_s=issued,
        expires_monotonic_s=expires,
    )


@dataclass(frozen=True)
class FrankaJointTarget:
    sequence: int
    produced_monotonic_s: float
    target_q_rad: np.ndarray
    source: FrankaTargetSource
    source_id: str
    closed_loop_command: Optional[ClosedLoopCommand] = None

    def __post_init__(self) -> None:
        sequence = _positive_integer(self.sequence, "target sequence")
        produced = _finite_scalar(self.produced_monotonic_s, "produced_monotonic_s")
        target = _readonly_vector(self.target_q_rad, 7, "target_q_rad")
        if not isinstance(self.source, FrankaTargetSource):
            raise TypeError("source must be FrankaTargetSource")
        source_id = str(self.source_id).strip()
        if not source_id:
            raise ValueError("source_id must be non-empty")
        if self.source in (
            FrankaTargetSource.C2_V94_POLICY,
            FrankaTargetSource.SUPERVISED_V94,
        ):
            command = self.closed_loop_command
            if not isinstance(command, ClosedLoopCommand):
                raise ValueError("policy target requires its exact ClosedLoopCommand")
            if command.sequence != sequence:
                raise ValueError("policy target sequence differs from ClosedLoopCommand")
            if command.produced_monotonic_s != produced:
                raise ValueError("policy target timestamp differs from ClosedLoopCommand")
            if not np.array_equal(command.franka_target_q_rad, target):
                raise ValueError("policy target differs from ClosedLoopCommand")
        elif self.closed_loop_command is not None:
            raise ValueError("C1 deterministic target cannot carry a policy command")
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "produced_monotonic_s", produced)
        object.__setattr__(self, "target_q_rad", target)
        object.__setattr__(self, "source_id", source_id)

    @classmethod
    def from_closed_loop_command(
        cls,
        command: ClosedLoopCommand,
        *,
        source: FrankaTargetSource = FrankaTargetSource.C2_V94_POLICY,
    ) -> "FrankaJointTarget":
        if not isinstance(command, ClosedLoopCommand):
            raise TypeError("command must be ClosedLoopCommand")
        if source not in (
            FrankaTargetSource.C2_V94_POLICY,
            FrankaTargetSource.SUPERVISED_V94,
        ):
            raise ValueError("closed-loop command requires a policy target source")
        return cls(
            sequence=command.sequence,
            produced_monotonic_s=command.produced_monotonic_s,
            target_q_rad=command.franka_target_q_rad,
            source=source,
            source_id=f"v94:{command.sequence}",
            closed_loop_command=command,
        )


class FrankaTargetSampleHold:
    """Thread-safe, monotone publication for the unique Franka owner."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[FrankaJointTarget] = None

    def publish(self, target: FrankaJointTarget) -> None:
        if not isinstance(target, FrankaJointTarget):
            raise TypeError("target must be FrankaJointTarget")
        with self._lock:
            expected = 1 if self._latest is None else self._latest.sequence + 1
            if target.sequence != expected:
                raise FrankaPersistentSessionError(
                    f"target sequence gap: expected={expected}, actual={target.sequence}"
                )
            if (
                self._latest is not None
                and target.produced_monotonic_s <= self._latest.produced_monotonic_s
            ):
                raise FrankaPersistentSessionError(
                    "target production timestamps must increase strictly"
                )
            self._latest = target

    def peek(self) -> Optional[FrankaJointTarget]:
        """Return the latest immutable target without applying an age policy."""

        with self._lock:
            return self._latest

    def sample(self, *, now_monotonic_s: object, maximum_age_s: object) -> FrankaJointTarget:
        now = _finite_scalar(now_monotonic_s, "now_monotonic_s")
        maximum_age = _positive_scalar(maximum_age_s, "maximum_age_s")
        with self._lock:
            target = self._latest
        if target is None:
            raise FrankaPersistentSessionError("no Franka target is available")
        age = now - target.produced_monotonic_s
        if age < 0.0:
            raise FrankaPersistentSessionError("Franka target timestamp is in the future")
        if age > maximum_age:
            raise FrankaPersistentSessionError(
                f"Franka target is stale: age={age:.6f}s, limit={maximum_age:.6f}s"
            )
        return target


@dataclass(frozen=True)
class FrankaPoseSample:
    cycle: int
    realtime_s: float
    monotonic_s: float
    q_rad: np.ndarray
    dq_rad_s: np.ndarray
    T_base_eef: np.ndarray
    # Native status bits are carried into the policy observation so a lower
    # Franka contact flag can hold only the arm action while RH56 continues.
    # Legacy/non-native producers default to a clear state.
    status_flags: int = 0
    # Exact V258/V65 controller-state input captured in the same native STATE
    # packet as q/dq.  Legacy producers may omit it; 96D policies require it.
    controller_state29: Optional[np.ndarray] = None
    # Coherent software trajectory-generator state captured in the same native
    # STATE packet.  The q_d-relative Student contract requires q_d for the
    # next action mapping; the remaining components are retained explicitly so
    # native producers do not need to reconstruct them from normalized 29D
    # features.  Legacy/non-native publishers may omit all four fields.
    shaper_q_d_rad: Optional[np.ndarray] = None
    shaper_dq_d_rad_s: Optional[np.ndarray] = None
    shaper_ddq_d_rad_s2: Optional[np.ndarray] = None
    held_q_cmd_rad: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        cycle = _positive_integer(self.cycle, "pose cycle")
        realtime = _finite_scalar(self.realtime_s, "realtime_s")
        monotonic = _finite_scalar(self.monotonic_s, "monotonic_s")
        q = _readonly_vector(self.q_rad, 7, "q_rad")
        dq = _readonly_vector(self.dq_rad_s, 7, "dq_rad_s")
        pose = np.asarray(self.T_base_eef, dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("T_base_eef must be a finite 4x4 matrix")
        pose = pose.copy()
        pose.setflags(write=False)
        if isinstance(self.status_flags, (bool, np.bool_)):
            raise ValueError("status_flags must be a non-negative integer")
        status_flags = int(self.status_flags)
        if status_flags != self.status_flags or status_flags < 0:
            raise ValueError("status_flags must be a non-negative integer")
        object.__setattr__(self, "cycle", cycle)
        object.__setattr__(self, "realtime_s", realtime)
        object.__setattr__(self, "monotonic_s", monotonic)
        object.__setattr__(self, "q_rad", q)
        object.__setattr__(self, "dq_rad_s", dq)
        object.__setattr__(self, "T_base_eef", pose)
        object.__setattr__(self, "status_flags", status_flags)
        controller_state = self.controller_state29
        if controller_state is not None:
            state = np.asarray(controller_state, dtype=np.float32)
            if state.shape != (29,) or not np.all(np.isfinite(state)):
                raise ValueError("controller_state29 must be a finite 29D vector")
            if (
                np.any(state[:14] < -4.0)
                or np.any(state[:14] > 4.0)
                or np.any(state[14:28] < -1.0)
                or np.any(state[14:28] > 1.0)
                or state[28] != np.float32(1.0)
            ):
                raise ValueError("controller_state29 violates the V258 contract")
            state = state.copy()
            state.setflags(write=False)
            object.__setattr__(self, "controller_state29", state)
        for field_name in (
            "shaper_q_d_rad",
            "shaper_dq_d_rad_s",
            "shaper_ddq_d_rad_s2",
            "held_q_cmd_rad",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _readonly_vector(value, 7, field_name),
                )


class FrankaPoseRing:
    """Bounded pose history published only from active-handle states."""

    def __init__(self, capacity: int) -> None:
        self.capacity = _positive_integer(capacity, "pose ring capacity")
        if self.capacity < 2:
            raise ValueError("pose ring capacity must be at least two")
        self._lock = threading.Lock()
        self._samples: Deque[FrankaPoseSample] = deque(maxlen=self.capacity)

    def publish(self, sample: FrankaPoseSample) -> None:
        if not isinstance(sample, FrankaPoseSample):
            raise TypeError("sample must be FrankaPoseSample")
        with self._lock:
            if self._samples:
                previous = self._samples[-1]
                if sample.cycle != previous.cycle + 1:
                    raise FrankaPersistentSessionError("pose-ring cycle sequence changed")
                if sample.monotonic_s < previous.monotonic_s:
                    raise FrankaPersistentSessionError("pose-ring monotonic clock regressed")
                if sample.realtime_s < previous.realtime_s:
                    raise FrankaPersistentSessionError("pose-ring realtime clock regressed")
            self._samples.append(sample)

    def snapshot(self) -> Tuple[FrankaPoseSample, ...]:
        with self._lock:
            return tuple(self._samples)

    def bracket_realtime(self, realtime_s: object) -> Tuple[FrankaPoseSample, FrankaPoseSample]:
        requested = _finite_scalar(realtime_s, "realtime_s")
        with self._lock:
            samples = tuple(self._samples)
        if len(samples) < 2:
            raise FrankaPersistentSessionError("pose ring has fewer than two samples")
        if requested < samples[0].realtime_s or requested > samples[-1].realtime_s:
            raise FrankaPersistentSessionError("requested time is outside the pose ring")
        for index in range(1, len(samples)):
            if requested <= samples[index].realtime_s:
                return samples[index - 1], samples[index]
        raise FrankaPersistentSessionError("pose bracket search failed")


class PersistentFrankaControl(Protocol):
    """Active joint-position handle owned by exactly one session thread."""

    def read_once(self) -> Tuple[Any, float]:
        ...

    def write_once(self, q_rad: Sequence[float], *, sequence: int) -> None:
        ...

    def write_bootstrap_hold(self, q_rad: Sequence[float]) -> None:
        """Respond to one active read with an unsequenced measured-q hold."""

        ...

    def finish(self, q_rad: Sequence[float]) -> None:
        ...


class PersistentFrankaBackend(Protocol):
    """Injected adapter; implementations may wrap pylibfranka or a fake."""

    def start_joint_position_session(self) -> PersistentFrankaControl:
        ...

    def request_stop(self) -> None:
        ...

    def read_post_stop_state(self) -> Any:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class FrankaPersistentTelemetry:
    run_id: str
    mode: FrankaSessionMode
    state: FrankaSessionState
    authorization_id: str
    preflight_report_sha256: str
    profile_sha256: str
    envelope_sha256: str
    backend_opened: bool
    static_provenance_verified: bool
    active_read_count: int
    active_write_count: int
    pose_publish_count: int
    distinct_target_count: int
    first_target_sequence: Optional[int]
    last_target_sequence: Optional[int]
    franka_ack_count: int
    maximum_command_age_s: float
    maximum_control_period_s: float
    maximum_read_to_write_s: float
    maximum_tracking_error_rad: float
    maximum_measured_velocity_rad_s: float
    maximum_measured_acceleration_rad_s2: float
    maximum_measured_jerk_rad_s3: float
    maximum_command_velocity_rad_s: float
    maximum_command_acceleration_rad_s2: float
    maximum_command_jerk_rad_s3: float
    maximum_target_rate_rad_s: float
    maximum_target_following_error_rad: float
    stop_requested: bool
    stop_finish_sent: bool
    stop_verified: bool
    stop_verification_samples: int
    fault_reason: Optional[str]
    bootstrap_hold_write_count: int = 0


class _TelemetryBuilder:
    def __init__(
        self,
        *,
        run_id: str,
        mode: FrankaSessionMode,
        authorization: MotionAuthorization,
        token: VerifiedFrankaPreflightToken | SupervisedFrankaPreflightToken,
        envelope: CommissionedFrankaEnvelope,
    ) -> None:
        self.run_id = run_id
        self.mode = mode
        self.authorization_id = authorization.authorization_id
        self.preflight_report_sha256 = token.report_sha256
        self.profile_sha256 = envelope.profile_sha256
        self.envelope_sha256 = envelope.binding_sha256
        self.backend_opened = False
        self.static_provenance_verified = False
        self.active_read_count = 0
        self.active_write_count = 0
        self.pose_publish_count = 0
        self.distinct_target_count = 0
        self.first_target_sequence: Optional[int] = None
        self.last_target_sequence: Optional[int] = None
        self.franka_ack_count = 0
        self.maximum_command_age_s = 0.0
        self.maximum_control_period_s = 0.0
        self.maximum_read_to_write_s = 0.0
        self.maximum_tracking_error_rad = 0.0
        self.maximum_measured_velocity_rad_s = 0.0
        self.maximum_measured_acceleration_rad_s2 = 0.0
        self.maximum_measured_jerk_rad_s3 = 0.0
        self.maximum_command_velocity_rad_s = 0.0
        self.maximum_command_acceleration_rad_s2 = 0.0
        self.maximum_command_jerk_rad_s3 = 0.0
        self.maximum_target_rate_rad_s = 0.0
        self.maximum_target_following_error_rad = 0.0
        self.stop_requested = False
        self.stop_finish_sent = False
        self.stop_verified = False
        self.stop_verification_samples = 0
        self.fault_reason: Optional[str] = None
        self.bootstrap_hold_write_count = 0

    def freeze(self, state: FrankaSessionState) -> FrankaPersistentTelemetry:
        return FrankaPersistentTelemetry(
            run_id=self.run_id,
            mode=self.mode,
            state=state,
            authorization_id=self.authorization_id,
            preflight_report_sha256=self.preflight_report_sha256,
            profile_sha256=self.profile_sha256,
            envelope_sha256=self.envelope_sha256,
            backend_opened=self.backend_opened,
            static_provenance_verified=self.static_provenance_verified,
            active_read_count=self.active_read_count,
            active_write_count=self.active_write_count,
            pose_publish_count=self.pose_publish_count,
            distinct_target_count=self.distinct_target_count,
            first_target_sequence=self.first_target_sequence,
            last_target_sequence=self.last_target_sequence,
            franka_ack_count=self.franka_ack_count,
            maximum_command_age_s=self.maximum_command_age_s,
            maximum_control_period_s=self.maximum_control_period_s,
            maximum_read_to_write_s=self.maximum_read_to_write_s,
            maximum_tracking_error_rad=self.maximum_tracking_error_rad,
            maximum_measured_velocity_rad_s=self.maximum_measured_velocity_rad_s,
            maximum_measured_acceleration_rad_s2=self.maximum_measured_acceleration_rad_s2,
            maximum_measured_jerk_rad_s3=self.maximum_measured_jerk_rad_s3,
            maximum_command_velocity_rad_s=self.maximum_command_velocity_rad_s,
            maximum_command_acceleration_rad_s2=self.maximum_command_acceleration_rad_s2,
            maximum_command_jerk_rad_s3=self.maximum_command_jerk_rad_s3,
            maximum_target_rate_rad_s=self.maximum_target_rate_rad_s,
            maximum_target_following_error_rad=(
                self.maximum_target_following_error_rad
            ),
            stop_requested=self.stop_requested,
            stop_finish_sent=self.stop_finish_sent,
            stop_verified=self.stop_verified,
            stop_verification_samples=self.stop_verification_samples,
            fault_reason=self.fault_reason,
            bootstrap_hold_write_count=self.bootstrap_hold_write_count,
        )


def _robot_mode_name(value: object) -> str:
    raw = getattr(value, "name", value)
    text = str(raw).strip().lower().replace("robotmode", "")
    text = text.replace("_", "").replace(".", "")
    if text.startswith("k"):
        text = text[1:]
    if "idle" in text:
        return "idle"
    if "move" in text:
        return "move"
    return text


def _has_active_errors(errors: object) -> bool:
    if errors is None:
        raise FrankaPersistentSessionError("Franka state has no current_errors")
    if isinstance(errors, Mapping):
        return any(bool(value) for value in errors.values())
    mapping = getattr(errors, "__dict__", None)
    if isinstance(mapping, Mapping):
        return any(
            bool(value)
            for name, value in mapping.items()
            if not str(name).startswith("_")
            and isinstance(value, (bool, np.bool_, int, np.integer))
        )
    try:
        return bool(errors)
    except BaseException as exc:
        raise FrankaPersistentSessionError(
            "Franka current_errors cannot be evaluated"
        ) from exc


def _require_clear_flags(state: object, field_name: str, expected_size: int) -> None:
    value = getattr(state, field_name, None)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FrankaPersistentSessionError(
            f"Franka state.{field_name} is malformed"
        ) from exc
    if array.shape != (expected_size,) or not np.all(np.isfinite(array)):
        raise FrankaPersistentSessionError(
            f"Franka state.{field_name} is malformed"
        )
    active = np.flatnonzero(array != 0.0)
    if len(active):
        raise FrankaPersistentSessionError(
            f"Franka state.{field_name} is active at index {int(active[0])}"
        )


def _state_vectors(state: object) -> Tuple[np.ndarray, np.ndarray]:
    q = _readonly_vector(getattr(state, "q", None), 7, "Franka state.q")
    dq = _readonly_vector(getattr(state, "dq", None), 7, "Franka state.dq")
    return q, dq


def _state_pose(state: object, field_name: str = "O_T_EE") -> np.ndarray:
    raw = np.asarray(getattr(state, field_name, None), dtype=np.float64)
    if raw.shape == (16,):
        pose = raw.reshape((4, 4), order="F")
    elif raw.shape == (4, 4):
        pose = raw
    else:
        raise FrankaPersistentSessionError(
            f"Franka state.{field_name} is malformed"
        )
    if not np.all(np.isfinite(pose)):
        raise FrankaPersistentSessionError(
            f"Franka state.{field_name} is non-finite"
        )
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8, rtol=0.0):
        raise FrankaPersistentSessionError(
            f"Franka state.{field_name} bottom row is invalid"
        )
    result = pose.copy()
    result.setflags(write=False)
    return result


def _state_inertia(state: object, field_name: str) -> np.ndarray:
    value = np.asarray(getattr(state, field_name, None), dtype=np.float64)
    if value.shape == (9,):
        value = value.reshape((3, 3), order="F")
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise FrankaPersistentSessionError(
            f"Franka state.{field_name} is malformed"
        )
    if not np.allclose(value, value.T, atol=1.0e-10, rtol=0.0):
        raise FrankaPersistentSessionError(
            f"Franka state.{field_name} is not symmetric"
        )
    result = value.copy()
    result.setflags(write=False)
    return result


class _JerkLimitedCommandState:
    def __init__(self, q_rad: np.ndarray) -> None:
        self.q = np.asarray(q_rad, dtype=np.float64).copy()
        self.velocity = np.zeros(7, dtype=np.float64)
        self.acceleration = np.zeros(7, dtype=np.float64)

    def step(
        self,
        target_q: np.ndarray,
        dt: float,
        envelope: CommissionedFrankaEnvelope,
        *,
        safe_joint_lower_rad: Optional[np.ndarray] = None,
        safe_joint_upper_rad: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float, float]:
        error = np.asarray(target_q, dtype=np.float64) - self.q
        stopping_velocity = np.sqrt(
            2.0 * envelope.maximum_acceleration_rad_s2 * np.abs(error)
        )
        desired_velocity = np.sign(error) * np.minimum(
            envelope.maximum_velocity_rad_s, stopping_velocity
        )
        desired_acceleration = (desired_velocity - self.velocity) / dt
        lower = np.maximum(
            -envelope.maximum_acceleration_rad_s2,
            self.acceleration - envelope.maximum_jerk_rad_s3 * dt,
        )
        upper = np.minimum(
            envelope.maximum_acceleration_rad_s2,
            self.acceleration + envelope.maximum_jerk_rad_s3 * dt,
        )
        lower = np.maximum(
            lower,
            (-envelope.maximum_velocity_rad_s - self.velocity) / dt,
        )
        upper = np.minimum(
            upper,
            (envelope.maximum_velocity_rad_s - self.velocity) / dt,
        )
        safe_lower = (
            envelope.safe_joint_lower_rad
            if safe_joint_lower_rad is None
            else safe_joint_lower_rad
        )
        safe_upper = (
            envelope.safe_joint_upper_rad
            if safe_joint_upper_rad is None
            else safe_joint_upper_rad
        )
        lower = np.maximum(lower, (safe_lower - self.q - self.velocity * dt) / (dt * dt))
        upper = np.minimum(upper, (safe_upper - self.q - self.velocity * dt) / (dt * dt))
        if np.any(lower > upper + 1.0e-12):
            raise FrankaPersistentSessionError(
                "no jerk/acceleration/velocity/joint-safe command exists"
            )
        new_acceleration = np.minimum(np.maximum(desired_acceleration, lower), upper)
        new_velocity = self.velocity + new_acceleration * dt
        new_q = self.q + new_velocity * dt
        actual_acceleration = (new_velocity - self.velocity) / dt
        actual_jerk = (actual_acceleration - self.acceleration) / dt
        velocity_peak = float(np.max(np.abs(new_velocity)))
        acceleration_peak = float(np.max(np.abs(actual_acceleration)))
        jerk_peak = float(np.max(np.abs(actual_jerk)))
        if np.any(np.abs(new_velocity) > envelope.maximum_velocity_rad_s + 1.0e-10):
            raise FrankaPersistentSessionError("internal command velocity guard failed")
        if np.any(
            np.abs(actual_acceleration)
            > envelope.maximum_acceleration_rad_s2 + 1.0e-9
        ):
            raise FrankaPersistentSessionError("internal command acceleration guard failed")
        if np.any(np.abs(actual_jerk) > envelope.maximum_jerk_rad_s3 + 1.0e-7):
            raise FrankaPersistentSessionError("internal command jerk guard failed")
        if np.any(new_q < safe_lower - 1.0e-12) or np.any(new_q > safe_upper + 1.0e-12):
            raise FrankaPersistentSessionError("internal command joint guard failed")
        self.q = new_q
        self.velocity = new_velocity
        self.acceleration = actual_acceleration
        output = new_q.copy()
        output.setflags(write=False)
        return output, velocity_peak, acceleration_peak, jerk_peak


class FrankaPersistentSession:
    """Single-use, single-thread owner for one persistent Franka handle."""

    def __init__(
        self,
        *,
        run_id: str,
        mode: FrankaSessionMode,
        envelope: CommissionedFrankaEnvelope,
        target_hold: FrankaTargetSampleHold,
        safety_gate: ClosedLoopSafetyGate,
        backend_factory: Optional[Callable[[], PersistentFrankaBackend]] = None,
        action_ledger: Optional[ExecutedActionLedger] = None,
        enable_c2_measured_hold_bootstrap: bool = False,
        supervised_reference_q_rad: Optional[Sequence[float]] = None,
        supervised_maximum_start_error_rad: Optional[float] = None,
        supervised_maximum_tick_target_delta_rad: Optional[float] = None,
        supervised_maximum_episode_delta_rad: Optional[float] = None,
        supervised_static_provenance_prevalidated: bool = False,
        franka_action_contract_id: str = LEGACY_FRANKA_ACTION_CONTRACT_ID,
        monotonic: Callable[[], float] = time.monotonic,
        realtime: Callable[[], float] = time.time,
    ) -> None:
        active_run_id = str(run_id).strip()
        if not active_run_id:
            raise ValueError("run_id must be non-empty")
        if not isinstance(mode, FrankaSessionMode):
            raise TypeError("mode must be FrankaSessionMode")
        if not isinstance(envelope, CommissionedFrankaEnvelope):
            raise TypeError("envelope must be loaded from a commissioning profile")
        if not isinstance(target_hold, FrankaTargetSampleHold):
            raise TypeError("target_hold must be FrankaTargetSampleHold")
        if not isinstance(safety_gate, ClosedLoopSafetyGate):
            raise TypeError("safety_gate must be ClosedLoopSafetyGate")
        if backend_factory is not None and not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        if _is_transactional_policy_mode(mode):
            if not isinstance(action_ledger, ExecutedActionLedger):
                raise TypeError("policy session requires an ExecutedActionLedger")
        elif action_ledger is not None:
            raise ValueError("C1 session cannot acknowledge a policy action ledger")
        bootstrap = _strict_bool(
            enable_c2_measured_hold_bootstrap,
            "enable_c2_measured_hold_bootstrap",
        )
        if bootstrap and not _is_transactional_policy_mode(mode):
            raise ValueError("measured-hold bootstrap is valid only for policy mode")
        action_contract = normalize_franka_action_contract_id(
            franka_action_contract_id
        )
        required_supervised_values = (
            supervised_reference_q_rad,
            supervised_maximum_start_error_rad,
            supervised_maximum_tick_target_delta_rad,
        )
        static_prevalidated = _strict_bool(
            supervised_static_provenance_prevalidated,
            "supervised_static_provenance_prevalidated",
        )
        if mode is FrankaSessionMode.SUPERVISED_V94:
            if any(value is None for value in required_supervised_values):
                raise ValueError(
                    "supervised session requires reference, start, and tick guards"
                )
            reference = _readonly_vector(
                supervised_reference_q_rad, 7, "supervised_reference_q_rad"
            )
            start_limit = _positive_scalar(
                supervised_maximum_start_error_rad,
                "supervised_maximum_start_error_rad",
            )
            tick_limit = _positive_scalar(
                supervised_maximum_tick_target_delta_rad,
                "supervised_maximum_tick_target_delta_rad",
            )
            if action_contract == LEGACY_FRANKA_ACTION_CONTRACT_ID:
                episode_limit = _positive_scalar(
                    supervised_maximum_episode_delta_rad,
                    "supervised_maximum_episode_delta_rad",
                )
            else:
                assert action_contract == QD_G015_FRANKA_ACTION_CONTRACT_ID
                if supervised_maximum_episode_delta_rad is not None:
                    raise ValueError(
                        "q_d g015 supervised session must disable the "
                        "home-centered episode limit"
                    )
                episode_limit = None
            if not static_prevalidated:
                raise ValueError(
                    "supervised session requires read-only static provenance prevalidation"
                )
        else:
            if any(value is not None for value in required_supervised_values) or (
                supervised_maximum_episode_delta_rad is not None
            ):
                raise ValueError("supervised motion guards cannot enter C1/C2")
            if action_contract != LEGACY_FRANKA_ACTION_CONTRACT_ID:
                raise ValueError(
                    "q_d g015 action contract is valid only for supervised mode"
                )
            reference = None
            start_limit = tick_limit = episode_limit = None
            if static_prevalidated:
                raise ValueError("C1/C2 cannot consume supervised static prevalidation")
        if not callable(monotonic) or not callable(realtime):
            raise TypeError("session clocks must be callable")
        self.run_id = active_run_id
        self.mode = mode
        self.envelope = envelope
        self.target_hold = target_hold
        self.safety_gate = safety_gate
        self._backend_factory = backend_factory
        self.action_ledger = action_ledger
        self.franka_action_contract_id = action_contract
        self.enable_c2_measured_hold_bootstrap = bootstrap
        self.supervised_reference_q_rad = reference
        self.supervised_maximum_start_error_rad = start_limit
        self.supervised_maximum_tick_target_delta_rad = tick_limit
        self.supervised_maximum_episode_delta_rad = episode_limit
        self.supervised_static_provenance_prevalidated = static_prevalidated
        # These values are immutable for the lifetime of the sealed envelope.
        # Cache them before the backend is opened so the 1 kHz hot path does
        # not allocate arrays or serialize the envelope binding hash.
        self._envelope_profile_sha256 = envelope.profile_sha256
        self._envelope_binding_sha256 = envelope.binding_sha256
        self._safe_joint_lower_rad = envelope.safe_joint_lower_rad
        self._safe_joint_upper_rad = envelope.safe_joint_upper_rad
        self._monotonic = monotonic
        self._realtime = realtime
        self.pose_ring = FrankaPoseRing(envelope.pose_ring_capacity)
        self._state_lock = threading.Lock()
        self._state = FrankaSessionState.DISABLED
        self._owner_ident: Optional[int] = None
        self._clean_stop = threading.Event()
        self._bootstrap_ready = threading.Event()
        self.last_telemetry: Optional[FrankaPersistentTelemetry] = None

    @property
    def state(self) -> FrankaSessionState:
        with self._state_lock:
            return self._state

    @property
    def owner_ident(self) -> Optional[int]:
        """Thread identity that exclusively owned the active backend calls."""

        with self._state_lock:
            return self._owner_ident

    def request_clean_stop(self) -> None:
        self._clean_stop.set()

    @property
    def c2_bootstrap_ready(self) -> bool:
        return self._bootstrap_ready.is_set()

    def wait_for_c2_bootstrap(self, timeout_s: object) -> bool:
        timeout = _finite_scalar(timeout_s, "bootstrap wait timeout_s")
        if timeout < 0.0:
            raise ValueError("bootstrap wait timeout_s cannot be negative")
        return self._bootstrap_ready.wait(timeout)

    def _set_running_once(self) -> None:
        with self._state_lock:
            if self._state is not FrankaSessionState.DISABLED:
                raise FrankaPersistentSessionError("persistent session is single-use")
            self._state = FrankaSessionState.RUNNING
            self._owner_ident = threading.get_ident()

    def _set_terminal(self, state: FrankaSessionState) -> None:
        with self._state_lock:
            self._state = state

    def _require_preflight_token_type(
        self,
        token: VerifiedFrankaPreflightToken | SupervisedFrankaPreflightToken,
    ) -> None:
        if self.mode is FrankaSessionMode.SUPERVISED_V94:
            if not isinstance(token, SupervisedFrankaPreflightToken):
                raise TypeError(
                    "supervised session requires SupervisedFrankaPreflightToken"
                )
            return
        if not isinstance(token, VerifiedFrankaPreflightToken):
            raise TypeError("C1/C2 session requires VerifiedFrankaPreflightToken")

    def _require_active_token_in_hot_path(
        self,
        token: VerifiedFrankaPreflightToken | SupervisedFrankaPreflightToken,
        now: float,
    ) -> None:
        """Recheck the sealed token without allocating in supervised cycles.

        The full binding check is performed before the backend is opened.
        The supervised envelope is frozen and all NumPy members are read-only,
        so comparing the cached digest here preserves that binding while
        avoiding JSON serialization between ``read_once`` and ``write_once``.
        Formal C1/C2 behavior is deliberately unchanged.
        """

        if self.mode is not FrankaSessionMode.SUPERVISED_V94:
            token.require_active(
                run_id=self.run_id,
                stage=self.mode,
                envelope=self.envelope,
                now_monotonic_s=now,
            )
            return
        if not isinstance(token, SupervisedFrankaPreflightToken):
            raise FrankaPersistentSessionError(
                "supervised session token type changed in the active loop"
            )
        if token.stage is not FrankaSessionMode.SUPERVISED_V94:
            raise FrankaPersistentSessionError("supervised token stage changed")
        if token.run_id != self.run_id:
            raise FrankaPersistentSessionError("supervised token run ID mismatch")
        if token.profile_sha256 != self._envelope_profile_sha256:
            raise FrankaPersistentSessionError("supervised profile hash changed")
        if token.envelope_sha256 != self._envelope_binding_sha256:
            raise FrankaPersistentSessionError("supervised envelope binding changed")
        if not token.issued_monotonic_s <= now < token.expires_monotonic_s:
            raise FrankaPersistentSessionError("supervised token is not active")

    def _validate_authorized_boundary(
        self,
        authorization: MotionAuthorization,
        token: VerifiedFrankaPreflightToken | SupervisedFrankaPreflightToken,
        now: float,
    ) -> FrankaJointTarget:
        if not isinstance(authorization, MotionAuthorization):
            raise TypeError("authorization must be MotionAuthorization")
        self._require_preflight_token_type(token)
        if authorization.run_id != self.run_id:
            raise FrankaPersistentSessionError("motion authorization run ID mismatch")
        if authorization.scope != AUTHORIZATION_SCOPE:
            raise FrankaPersistentSessionError("motion authorization scope mismatch")
        if not authorization.issued_monotonic_s <= now < authorization.expires_monotonic_s:
            raise FrankaPersistentSessionError("motion authorization is not active")
        token.require_active(
            run_id=self.run_id,
            stage=self.mode,
            envelope=self.envelope,
            now_monotonic_s=now,
        )
        if self.safety_gate.state is not SafetyState.ARMED:
            raise FrankaPersistentSessionError("closed-loop safety gate is not armed")
        if self.safety_gate.authorization_id != authorization.authorization_id:
            raise FrankaPersistentSessionError("safety-gate authorization identity mismatch")
        self.safety_gate.require_motion(run_id=self.run_id, now_monotonic_s=now)
        target = self.target_hold.sample(
            now_monotonic_s=now,
            maximum_age_s=self.envelope.policy_command_max_age_s,
        )
        expected_source = _target_source_for_mode(self.mode)
        if target.source is not expected_source:
            raise FrankaPersistentSessionError("target source does not match session mode")
        if _is_transactional_policy_mode(self.mode):
            assert self.action_ledger is not None
            if self.action_ledger.pending_sequence != target.sequence:
                raise FrankaPersistentSessionError(
                    "policy target is not the exact pending dual-device transaction"
                )
            if self.action_ledger.last_committed_sequence != target.sequence - 1:
                raise FrankaPersistentSessionError(
                    "policy action ledger sequence is inconsistent before control"
                )
        return target

    def _validate_bootstrap_authorized_boundary(
        self,
        authorization: MotionAuthorization,
        token: VerifiedFrankaPreflightToken | SupervisedFrankaPreflightToken,
        now: float,
    ) -> None:
        """Validate C2 before an active measured-q hold publishes pose state.

        This boundary permits only the dedicated bootstrap write method on the
        unique active handle.  It does not create/stage a policy sequence and
        cannot acknowledge the action ledger.  The first policy target remains
        subject to every normal C2 target check below.
        """

        if not _is_transactional_policy_mode(self.mode):
            raise FrankaPersistentSessionError("bootstrap boundary requires policy mode")
        if not isinstance(authorization, MotionAuthorization):
            raise TypeError("authorization must be MotionAuthorization")
        self._require_preflight_token_type(token)
        if authorization.run_id != self.run_id:
            raise FrankaPersistentSessionError("motion authorization run ID mismatch")
        if authorization.scope != AUTHORIZATION_SCOPE:
            raise FrankaPersistentSessionError("motion authorization scope mismatch")
        if not authorization.issued_monotonic_s <= now < authorization.expires_monotonic_s:
            raise FrankaPersistentSessionError("motion authorization is not active")
        token.require_active(
            run_id=self.run_id,
            stage=self.mode,
            envelope=self.envelope,
            now_monotonic_s=now,
        )
        if self.safety_gate.state is not SafetyState.ARMED:
            raise FrankaPersistentSessionError("closed-loop safety gate is not armed")
        if self.safety_gate.authorization_id != authorization.authorization_id:
            raise FrankaPersistentSessionError(
                "safety-gate authorization identity mismatch"
            )
        self.safety_gate.require_motion(run_id=self.run_id, now_monotonic_s=now)
        assert self.action_ledger is not None
        if self.action_ledger.last_committed_sequence != 0:
            raise FrankaPersistentSessionError(
                "C2 bootstrap requires a fresh action ledger"
            )
        if self.action_ledger.pending_sequence is not None:
            raise FrankaPersistentSessionError(
                "C2 bootstrap cannot start with a staged policy action"
            )
        if self.target_hold.peek() is not None:
            raise FrankaPersistentSessionError(
                "C2 bootstrap requires an initially empty policy target hold"
            )

    def _validate_dynamic_state(
        self,
        state: object,
        *,
        previous_command_q: Optional[np.ndarray],
        previous_dq: Optional[np.ndarray],
        previous_measured_acceleration: Optional[np.ndarray],
        dt: float,
        telemetry: _TelemetryBuilder,
        allow_low_success_rate: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        mode = _robot_mode_name(getattr(state, "robot_mode", None))
        if mode not in ("idle", "move"):
            raise FrankaPersistentSessionError(
                f"Franka mode became unsafe during control: {mode}"
            )
        if _has_active_errors(getattr(state, "current_errors", None)):
            raise FrankaPersistentSessionError("Franka current_errors are active")
        _require_clear_flags(state, "joint_contact", 7)
        _require_clear_flags(state, "joint_collision", 7)
        _require_clear_flags(state, "cartesian_contact", 6)
        _require_clear_flags(state, "cartesian_collision", 6)
        q, dq = _state_vectors(state)
        lower = self._safe_joint_lower_rad
        upper = self._safe_joint_upper_rad
        outside = np.flatnonzero((q < lower) | (q > upper))
        if len(outside):
            raise FrankaPersistentSessionError(
                f"Franka joint {int(outside[0]) + 1} left its commissioned safe interval"
            )
        measured_velocity = np.abs(dq)
        telemetry.maximum_measured_velocity_rad_s = max(
            telemetry.maximum_measured_velocity_rad_s,
            float(np.max(measured_velocity)),
        )
        velocity_violation = np.flatnonzero(
            measured_velocity > self.envelope.maximum_velocity_rad_s + 1.0e-10
        )
        if len(velocity_violation):
            raise FrankaPersistentSessionError(
                f"measured joint velocity exceeds commissioned limit at axis "
                f"{int(velocity_violation[0]) + 1}"
            )
        measured_acceleration: Optional[np.ndarray] = None
        if previous_dq is not None and dt > 0.0:
            measured_acceleration = (dq - previous_dq) / dt
            telemetry.maximum_measured_acceleration_rad_s2 = max(
                telemetry.maximum_measured_acceleration_rad_s2,
                float(np.max(np.abs(measured_acceleration))),
            )
            violation = np.flatnonzero(
                np.abs(measured_acceleration)
                > self.envelope.maximum_measured_acceleration_rad_s2 + 1.0e-9
            )
            if len(violation):
                raise FrankaPersistentSessionError(
                    f"measured joint acceleration exceeds commissioned limit at axis "
                    f"{int(violation[0]) + 1}"
                )
            if previous_measured_acceleration is not None:
                measured_jerk = (
                    measured_acceleration - previous_measured_acceleration
                ) / dt
                telemetry.maximum_measured_jerk_rad_s3 = max(
                    telemetry.maximum_measured_jerk_rad_s3,
                    float(np.max(np.abs(measured_jerk))),
                )
                violation = np.flatnonzero(
                    np.abs(measured_jerk)
                    > self.envelope.maximum_measured_jerk_rad_s3 + 1.0e-7
                )
                if len(violation):
                    raise FrankaPersistentSessionError(
                        f"measured joint jerk exceeds commissioned limit at axis "
                        f"{int(violation[0]) + 1}"
                    )
        if previous_command_q is not None:
            error = np.abs(q - previous_command_q)
            telemetry.maximum_tracking_error_rad = max(
                telemetry.maximum_tracking_error_rad, float(np.max(error))
            )
            violation = np.flatnonzero(
                error > self.envelope.maximum_tracking_error_rad + 1.0e-10
            )
            if len(violation):
                raise FrankaPersistentSessionError(
                    f"joint tracking error exceeds commissioned limit at axis "
                    f"{int(violation[0]) + 1}"
                )
        success = _finite_scalar(
            getattr(state, "control_command_success_rate", None),
            "Franka control_command_success_rate",
        )
        if not 0.0 <= success <= 1.0:
            raise FrankaPersistentSessionError(
                "Franka control_command_success_rate is outside [0,1]"
            )
        if (
            not allow_low_success_rate
            and success < self.envelope.minimum_control_success_rate
        ):
            raise FrankaPersistentSessionError(
                "Franka control command success rate is below commissioned minimum"
            )
        return q, dq, measured_acceleration

    def _validate_static_provenance(self, state: object) -> None:
        """Bind tool and load configuration before the first active write."""

        actual_O_T_EE = _state_pose(state, "O_T_EE")
        actual_F_T_EE = _state_pose(state, "F_T_EE")
        for field_name, pose in (
            ("O_T_EE", actual_O_T_EE),
            ("F_T_EE", actual_F_T_EE),
        ):
            actual_rotation = pose[:3, :3]
            if not np.allclose(
                actual_rotation.T @ actual_rotation,
                np.eye(3),
                atol=1.0e-6,
                rtol=0.0,
            ) or not np.isclose(
                np.linalg.det(actual_rotation), 1.0, atol=1.0e-6, rtol=0.0
            ):
                raise FrankaPersistentSessionError(
                    f"Franka state.{field_name} is not a rigid transform"
                )
        if not np.allclose(
            actual_F_T_EE,
            self.envelope.expected_F_T_EE,
            atol=self.envelope.F_T_EE_tolerance,
            rtol=0.0,
        ):
            raise FrankaPersistentSessionError(
                "Franka F_T_EE differs from the commissioned transform"
            )
        actual_ee_mass = _finite_scalar(
            getattr(state, "m_ee", None), "Franka state.m_ee"
        )
        if not math.isclose(
            actual_ee_mass,
            self.envelope.expected_end_effector_mass_kg,
            rel_tol=0.0,
            abs_tol=self.envelope.mass_tolerance_kg,
        ):
            raise FrankaPersistentSessionError(
                "Franka m_ee differs from the commissioned adapter and hand"
            )
        actual_ee_com = _readonly_vector(
            getattr(state, "F_x_Cee", None), 3, "Franka state.F_x_Cee"
        )
        if not np.allclose(
            actual_ee_com,
            self.envelope.expected_end_effector_com_m,
            atol=self.envelope.center_of_mass_tolerance_m,
            rtol=0.0,
        ):
            raise FrankaPersistentSessionError(
                "Franka F_x_Cee differs from the commissioned adapter and hand"
            )
        actual_ee_inertia = _state_inertia(state, "I_ee")
        if not np.allclose(
            actual_ee_inertia,
            self.envelope.expected_end_effector_inertia_kg_m2,
            atol=self.envelope.inertia_tolerance_kg_m2,
            rtol=0.0,
        ):
            raise FrankaPersistentSessionError(
                "Franka I_ee differs from the commissioned adapter and hand"
            )
        actual_load_mass = _finite_scalar(
            getattr(state, "m_load", None), "Franka state.m_load"
        )
        if not math.isclose(
            actual_load_mass,
            self.envelope.expected_external_load_mass_kg,
            rel_tol=0.0,
            abs_tol=self.envelope.mass_tolerance_kg,
        ):
            raise FrankaPersistentSessionError(
                "Franka external load mass differs from the commissioned load"
            )
        if self.envelope.expected_external_load_mass_kg > 0.0:
            actual_load_com = _readonly_vector(
                getattr(state, "F_x_Cload", None),
                3,
                "Franka state.F_x_Cload",
            )
            if not np.allclose(
                actual_load_com,
                self.envelope.expected_external_load_com_m,
                atol=self.envelope.center_of_mass_tolerance_m,
                rtol=0.0,
            ):
                raise FrankaPersistentSessionError(
                    "Franka F_x_Cload differs from the commissioned load"
                )
            actual_load_inertia = _state_inertia(state, "I_load")
            if not np.allclose(
                actual_load_inertia,
                self.envelope.expected_external_load_inertia_kg_m2,
                atol=self.envelope.inertia_tolerance_kg_m2,
                rtol=0.0,
            ):
                raise FrankaPersistentSessionError(
                    "Franka I_load differs from the commissioned load"
                )
        actual_total_mass = _finite_scalar(
            getattr(state, "m_total", None), "Franka state.m_total"
        )
        expected_total_mass = (
            self.envelope.expected_end_effector_mass_kg
            + self.envelope.expected_external_load_mass_kg
        )
        if not math.isclose(
            actual_total_mass,
            expected_total_mass,
            rel_tol=0.0,
            abs_tol=self.envelope.mass_tolerance_kg,
        ):
            raise FrankaPersistentSessionError(
                "Franka total mass differs from the commissioned tool and load"
            )

    def _verify_stop(
        self,
        backend: PersistentFrankaBackend,
        telemetry: _TelemetryBuilder,
    ) -> bool:
        consecutive = 0
        for _ in range(self.envelope.stop_maximum_samples):
            state = backend.read_post_stop_state()
            telemetry.stop_verification_samples += 1
            try:
                _q, dq = _state_vectors(state)
                idle = _robot_mode_name(getattr(state, "robot_mode", None)) == "idle"
                low_velocity = bool(
                    np.all(
                        np.abs(dq)
                        <= self.envelope.stop_maximum_velocity_rad_s + 1.0e-10
                    )
                )
                healthy = not _has_active_errors(
                    getattr(state, "current_errors", None)
                )
                _require_clear_flags(state, "joint_contact", 7)
                _require_clear_flags(state, "joint_collision", 7)
                _require_clear_flags(state, "cartesian_contact", 6)
                _require_clear_flags(state, "cartesian_collision", 6)
            except (ValueError, FrankaPersistentSessionError):
                consecutive = 0
                continue
            if idle and low_velocity and healthy:
                consecutive += 1
                if consecutive >= self.envelope.stop_consecutive_samples:
                    return True
            else:
                consecutive = 0
        return False

    def run(
        self,
        *,
        authorization: MotionAuthorization,
        preflight_token: VerifiedFrankaPreflightToken | SupervisedFrankaPreflightToken,
        maximum_cycles: Optional[int] = None,
    ) -> FrankaPersistentTelemetry:
        """Run the injected backend once; never reconnect or recover in place."""

        if maximum_cycles is not None:
            maximum_cycles = _positive_integer(maximum_cycles, "maximum_cycles")
        now = _finite_scalar(self._monotonic(), "monotonic clock")
        initial_target: Optional[FrankaJointTarget]
        if self.enable_c2_measured_hold_bootstrap:
            self._validate_bootstrap_authorized_boundary(
                authorization, preflight_token, now
            )
            initial_target = None
        else:
            initial_target = self._validate_authorized_boundary(
                authorization, preflight_token, now
            )
        if self._backend_factory is None:
            raise FrankaPersistentSessionError(
                "no backend factory was injected; device access remains disabled"
            )
        self._set_running_once()
        telemetry = _TelemetryBuilder(
            run_id=self.run_id,
            mode=self.mode,
            authorization=authorization,
            token=preflight_token,
            envelope=self.envelope,
        )
        backend: Optional[PersistentFrankaBackend] = None
        control: Optional[PersistentFrankaControl] = None
        shaper: Optional[_JerkLimitedCommandState] = None
        previous_command_q: Optional[np.ndarray] = None
        previous_dq: Optional[np.ndarray] = None
        previous_measured_acceleration: Optional[np.ndarray] = None
        current_target: Optional[FrankaJointTarget] = None
        prior_target_q: Optional[np.ndarray] = None
        prior_target_time: Optional[float] = None
        bootstrap_hold_q: Optional[np.ndarray] = None
        supervised_start_q: Optional[np.ndarray] = None
        last_franka_acked_sequence = 0
        initial_zero_available = self.envelope.allow_one_initial_zero_period
        session_start = now
        failure: Optional[BaseException] = None
        stop_failure: Optional[BaseException] = None
        try:
            backend = self._backend_factory()
            if backend is None:
                raise FrankaPersistentSessionError("backend factory returned None")
            telemetry.backend_opened = True
            control = backend.start_joint_position_session()
            if control is None:
                raise FrankaPersistentSessionError("backend returned no control handle")
            while not self._clean_stop.is_set():
                cycle_now = _finite_scalar(self._monotonic(), "monotonic clock")
                self.safety_gate.require_motion(
                    run_id=self.run_id, now_monotonic_s=cycle_now
                )
                self._require_active_token_in_hot_path(preflight_token, cycle_now)
                if cycle_now - session_start > self.envelope.maximum_session_duration_s:
                    raise FrankaPersistentSessionError(
                        "persistent session exceeded commissioned duration"
                    )
                state, raw_dt = control.read_once()
                telemetry.active_read_count += 1
                read_complete = _finite_scalar(self._monotonic(), "monotonic clock")
                state_realtime = _finite_scalar(self._realtime(), "realtime clock")
                dt = _finite_scalar(raw_dt, "Franka control period")
                # Record the state period before enforcing its envelope so a
                # terminal timing fault is represented faithfully in the
                # audit.  Previously the failing sample was omitted, which
                # made a 4 ms fault look like a 1 ms maximum afterwards.
                telemetry.maximum_control_period_s = max(
                    telemetry.maximum_control_period_s, dt
                )
                if dt == 0.0 and initial_zero_available:
                    initial_zero_available = False
                    shaping_dt = self.envelope.control_period_min_s
                else:
                    initial_zero_available = False
                    if not (
                        self.envelope.control_period_min_s
                        <= dt
                        <= self.envelope.control_period_max_s
                    ):
                        raise FrankaPersistentSessionError(
                            f"Franka control period {dt:.9f}s is outside commissioned bounds"
                        )
                    shaping_dt = dt
                if telemetry.active_read_count == 1:
                    if not (
                        self.mode is FrankaSessionMode.SUPERVISED_V94
                        and self.supervised_static_provenance_prevalidated
                    ):
                        self._validate_static_provenance(state)
                    telemetry.static_provenance_verified = True
                q, dq, measured_acceleration = self._validate_dynamic_state(
                    state,
                    previous_command_q=previous_command_q,
                    previous_dq=previous_dq,
                    previous_measured_acceleration=previous_measured_acceleration,
                    dt=shaping_dt,
                    telemetry=telemetry,
                    allow_low_success_rate=(
                        self.mode is FrankaSessionMode.SUPERVISED_V94
                        and cycle_now - session_start <= 0.10
                    ),
                )
                if self.mode is FrankaSessionMode.SUPERVISED_V94:
                    assert self.supervised_reference_q_rad is not None
                    assert self.supervised_maximum_start_error_rad is not None
                    if supervised_start_q is None:
                        start_error = float(
                            np.max(np.abs(q - self.supervised_reference_q_rad))
                        )
                        if start_error > self.supervised_maximum_start_error_rad:
                            raise FrankaPersistentSessionError(
                                "supervised Franka start differs from V94 q_home: "
                                f"Linf={start_error:.9f}rad, limit="
                                f"{self.supervised_maximum_start_error_rad:.9f}rad"
                            )
                        supervised_start_q = q.copy()
                    if (
                        self.franka_action_contract_id
                        == LEGACY_FRANKA_ACTION_CONTRACT_ID
                    ):
                        assert self.supervised_maximum_episode_delta_rad is not None
                        measured_episode_delta = float(
                            np.max(np.abs(q - supervised_start_q))
                        )
                        if (
                            measured_episode_delta
                            > self.supervised_maximum_episode_delta_rad
                        ):
                            raise FrankaPersistentSessionError(
                                "supervised measured-q episode envelope exceeded"
                            )
                pose = _state_pose(state)
                sample = FrankaPoseSample(
                    cycle=telemetry.active_read_count,
                    realtime_s=state_realtime,
                    monotonic_s=read_complete,
                    q_rad=q,
                    dq_rad_s=dq,
                    T_base_eef=pose,
                )
                self.pose_ring.publish(sample)
                telemetry.pose_publish_count += 1
                latest_target = self.target_hold.peek()
                if latest_target is None:
                    if not self.enable_c2_measured_hold_bootstrap:
                        raise FrankaPersistentSessionError(
                            "no Franka target is available"
                        )
                    if current_target is not None:
                        raise FrankaPersistentSessionError(
                            "published C2 target disappeared during control"
                        )
                    target = None
                else:
                    target = self.target_hold.sample(
                        now_monotonic_s=read_complete,
                        maximum_age_s=self.envelope.policy_command_max_age_s,
                    )
                    expected_source = _target_source_for_mode(self.mode)
                    if target.source is not expected_source:
                        raise FrankaPersistentSessionError(
                            "target source changed during persistent session"
                        )
                if target is not None and (
                    current_target is None
                    or target.sequence != current_target.sequence
                ):
                    expected_sequence = (
                        1
                        if current_target is None
                        else current_target.sequence + 1
                    )
                    if target.sequence != expected_sequence:
                        raise FrankaPersistentSessionError(
                            f"sampled target sequence skipped: expected={expected_sequence}, "
                            f"actual={target.sequence}"
                        )
                    if _is_transactional_policy_mode(self.mode):
                        assert self.action_ledger is not None
                        if self.action_ledger.pending_sequence != target.sequence:
                            raise FrankaPersistentSessionError(
                                "new C2 target is not the exact pending "
                                "dual-device transaction"
                            )
                        if (
                            self.action_ledger.last_committed_sequence
                            != target.sequence - 1
                        ):
                            raise FrankaPersistentSessionError(
                                "new C2 target does not follow the committed ledger"
                            )
                    if self.mode is FrankaSessionMode.SUPERVISED_V94:
                        assert supervised_start_q is not None
                        if (
                            self.franka_action_contract_id
                            == LEGACY_FRANKA_ACTION_CONTRACT_ID
                        ):
                            assert (
                                self.supervised_maximum_tick_target_delta_rad
                                is not None
                            )
                            assert self.supervised_maximum_episode_delta_rad is not None
                            prior_guard_target = (
                                supervised_start_q
                                if prior_target_q is None
                                else prior_target_q
                            )
                            tick_delta = float(
                                np.max(
                                    np.abs(
                                        target.target_q_rad - prior_guard_target
                                    )
                                )
                            )
                            if (
                                tick_delta
                                > self.supervised_maximum_tick_target_delta_rad
                            ):
                                raise FrankaPersistentSessionError(
                                    "supervised policy target tick delta exceeded: "
                                    f"Linf={tick_delta:.9f}rad"
                                )
                            episode_delta = float(
                                np.max(
                                    np.abs(
                                        target.target_q_rad - supervised_start_q
                                    )
                                )
                            )
                            if (
                                episode_delta
                                > self.supervised_maximum_episode_delta_rad
                            ):
                                raise FrankaPersistentSessionError(
                                    "supervised policy target episode envelope exceeded"
                                )
                    lower = self._safe_joint_lower_rad
                    upper = self._safe_joint_upper_rad
                    outside = np.flatnonzero(
                        (target.target_q_rad < lower) | (target.target_q_rad > upper)
                    )
                    if len(outside):
                        raise FrankaPersistentSessionError(
                            f"target joint {int(outside[0]) + 1} is outside "
                            "commissioned safe limits"
                        )
                    if current_target is None:
                        if (
                            self.franka_action_contract_id
                            == LEGACY_FRANKA_ACTION_CONTRACT_ID
                        ):
                            initial_error = np.abs(target.target_q_rad - q)
                            violation = np.flatnonzero(
                                initial_error
                                > self.envelope.initial_target_tolerance_rad
                                + 1.0e-10
                            )
                            if len(violation):
                                raise FrankaPersistentSessionError(
                                    "initial target is not a stationary hold at axis "
                                    f"{int(violation[0]) + 1}"
                                )
                        if shaper is None:
                            shaper = _JerkLimitedCommandState(q)
                    else:
                        assert prior_target_q is not None
                        assert prior_target_time is not None
                        target_dt = target.produced_monotonic_s - prior_target_time
                        if target_dt <= 0.0:
                            raise FrankaPersistentSessionError(
                                "target timestamps did not increase"
                            )
                        target_rate = np.abs(target.target_q_rad - prior_target_q) / target_dt
                        telemetry.maximum_target_rate_rad_s = max(
                            telemetry.maximum_target_rate_rad_s,
                            float(np.max(target_rate)),
                        )
                        if (
                            self.franka_action_contract_id
                            == LEGACY_FRANKA_ACTION_CONTRACT_ID
                        ):
                            violation = np.flatnonzero(
                                target_rate
                                > self.envelope.maximum_target_rate_rad_s
                                + 1.0e-10
                            )
                            if len(violation):
                                raise FrankaPersistentSessionError(
                                    "policy target rate exceeds commissioned limit "
                                    f"at axis {int(violation[0]) + 1}"
                                )
                    current_target = target
                    prior_target_q = target.target_q_rad.copy()
                    prior_target_time = target.produced_monotonic_s
                    telemetry.distinct_target_count += 1
                    telemetry.last_target_sequence = target.sequence
                    if telemetry.first_target_sequence is None:
                        telemetry.first_target_sequence = target.sequence
                if target is None:
                    if bootstrap_hold_q is None:
                        bootstrap_hold_q = q.copy()
                        shaper = _JerkLimitedCommandState(q)
                    active_target_q = bootstrap_hold_q
                else:
                    assert current_target is not None
                    active_target_q = current_target.target_q_rad
                assert shaper is not None
                command_q, velocity_peak, acceleration_peak, jerk_peak = shaper.step(
                    active_target_q,
                    shaping_dt,
                    self.envelope,
                    safe_joint_lower_rad=self._safe_joint_lower_rad,
                    safe_joint_upper_rad=self._safe_joint_upper_rad,
                )
                if (
                    self.mode is FrankaSessionMode.SUPERVISED_V94
                    and self.franka_action_contract_id
                    == LEGACY_FRANKA_ACTION_CONTRACT_ID
                ):
                    assert supervised_start_q is not None
                    assert self.supervised_maximum_episode_delta_rad is not None
                    command_episode_delta = float(
                        np.max(np.abs(command_q - supervised_start_q))
                    )
                    if command_episode_delta > self.supervised_maximum_episode_delta_rad:
                        raise FrankaPersistentSessionError(
                            "supervised Franka command episode envelope exceeded"
                        )
                target_following_error = np.abs(
                    active_target_q - command_q
                )
                telemetry.maximum_target_following_error_rad = max(
                    telemetry.maximum_target_following_error_rad,
                    float(np.max(target_following_error)),
                )
                if (
                    self.mode is not FrankaSessionMode.SUPERVISED_V94
                    or self.franka_action_contract_id
                    == LEGACY_FRANKA_ACTION_CONTRACT_ID
                ):
                    following_violation = np.flatnonzero(
                        target_following_error
                        > self.envelope.target_reached_tolerance_rad + 1.0e-10
                    )
                    if len(following_violation):
                        raise FrankaPersistentSessionError(
                            "jerk-limited command cannot follow the policy target "
                            "within the commissioned bound at axis "
                            f"{int(following_violation[0]) + 1}"
                        )
                telemetry.maximum_command_velocity_rad_s = max(
                    telemetry.maximum_command_velocity_rad_s, velocity_peak
                )
                telemetry.maximum_command_acceleration_rad_s2 = max(
                    telemetry.maximum_command_acceleration_rad_s2, acceleration_peak
                )
                telemetry.maximum_command_jerk_rad_s3 = max(
                    telemetry.maximum_command_jerk_rad_s3, jerk_peak
                )
                before_write = _finite_scalar(self._monotonic(), "monotonic clock")
                # Interlocks/authorization may change while read/validation/
                # shaping is in progress.  Recheck at the final no-write
                # boundary, not only before the blocking active-handle read.
                self.safety_gate.require_motion(
                    run_id=self.run_id, now_monotonic_s=before_write
                )
                self._require_active_token_in_hot_path(preflight_token, before_write)
                if target is not None:
                    assert current_target is not None
                    command_age = before_write - current_target.produced_monotonic_s
                    if (
                        command_age < 0.0
                        or command_age > self.envelope.policy_command_max_age_s
                    ):
                        raise FrankaPersistentSessionError(
                            "Franka target crossed its age watchdog before write"
                        )
                    telemetry.maximum_command_age_s = max(
                        telemetry.maximum_command_age_s, command_age
                    )
                read_to_write = before_write - read_complete
                if read_to_write < 0.0:
                    raise FrankaPersistentSessionError("control timing clock regressed")
                read_to_write_deadline_s = (
                    SUPERVISED_BOOTSTRAP_READ_TO_WRITE_DEADLINE_S
                    if (
                        self.mode is FrankaSessionMode.SUPERVISED_V94
                        and telemetry.active_read_count == 1
                    )
                    else self.envelope.read_to_write_deadline_s
                )
                telemetry.maximum_read_to_write_s = max(
                    telemetry.maximum_read_to_write_s, read_to_write
                )
                if read_to_write > read_to_write_deadline_s:
                    raise FrankaPersistentSessionError(
                        "Franka read-to-write deadline expired: "
                        f"actual={read_to_write:.9f}s "
                        f"limit={read_to_write_deadline_s:.9f}s"
                    )
                if target is None:
                    bootstrap_writer = getattr(
                        control, "write_bootstrap_hold", None
                    )
                    if not callable(bootstrap_writer):
                        raise FrankaPersistentSessionError(
                            "active backend has no measured-hold bootstrap method"
                        )
                    bootstrap_writer(command_q)
                    telemetry.bootstrap_hold_write_count += 1
                else:
                    assert current_target is not None
                    control.write_once(
                        command_q, sequence=current_target.sequence
                    )
                telemetry.active_write_count += 1
                after_write = _finite_scalar(self._monotonic(), "monotonic clock")
                total_read_to_write = after_write - read_complete
                telemetry.maximum_read_to_write_s = max(
                    telemetry.maximum_read_to_write_s, total_read_to_write
                )
                if total_read_to_write > read_to_write_deadline_s:
                    raise FrankaPersistentSessionError(
                        "Franka write completion exceeded read-to-write deadline: "
                        f"actual={total_read_to_write:.9f}s "
                        f"limit={read_to_write_deadline_s:.9f}s"
                    )
                if (
                    target is not None
                    and _is_transactional_policy_mode(self.mode)
                    and self.action_ledger is not None
                    and current_target.sequence > last_franka_acked_sequence
                ):
                    self.action_ledger.acknowledge(
                        "franka",
                        sequence=current_target.sequence,
                        now_monotonic_s=after_write,
                    )
                    last_franka_acked_sequence = current_target.sequence
                    telemetry.franka_ack_count += 1
                if target is None:
                    self._bootstrap_ready.set()
                previous_command_q = command_q
                previous_dq = dq
                previous_measured_acceleration = measured_acceleration
                if maximum_cycles is not None and telemetry.active_write_count >= maximum_cycles:
                    break
        except BaseException as exc:
            failure = exc
            telemetry.fault_reason = f"{type(exc).__name__}: {exc}"
            if _is_transactional_policy_mode(self.mode):
                assert self.action_ledger is not None
                failed_sequence = (
                    current_target.sequence
                    if current_target is not None
                    else (
                        self.action_ledger.pending_sequence
                        or self.action_ledger.last_committed_sequence + 1
                    )
                )
                self.action_ledger.fail(
                    "franka",
                    sequence=failed_sequence,
                    reason=telemetry.fault_reason,
                )
            self.safety_gate.latch_fault(telemetry.fault_reason)
        finally:
            # A terminal position write is allowed only after a clean loop
            # completion.  If any interlock, state, timing, command or binding
            # fault has already occurred, issuing even a motion-finished hold
            # would cross the final no-write boundary and delay Robot.stop().
            # Fault cleanup therefore goes directly to the unconditional stop
            # request below.
            if (
                failure is None
                and control is not None
                and previous_command_q is not None
            ):
                try:
                    control.finish(previous_command_q)
                    telemetry.stop_finish_sent = True
                except BaseException as exc:
                    stop_failure = exc
            if backend is not None:
                try:
                    backend.request_stop()
                    telemetry.stop_requested = True
                except BaseException as exc:
                    if stop_failure is None:
                        stop_failure = exc
                try:
                    telemetry.stop_verified = self._verify_stop(backend, telemetry)
                    if not telemetry.stop_verified and stop_failure is None:
                        stop_failure = FrankaPersistentSessionError(
                            "Franka stop was not verified"
                        )
                except BaseException as exc:
                    if stop_failure is None:
                        stop_failure = exc
                try:
                    backend.close()
                except BaseException as exc:
                    if stop_failure is None:
                        stop_failure = exc
            if stop_failure is not None:
                detail = f"{type(stop_failure).__name__}: {stop_failure}"
                if telemetry.fault_reason is None:
                    telemetry.fault_reason = detail
                else:
                    telemetry.fault_reason += f"; stop={detail}"
                if _is_transactional_policy_mode(self.mode):
                    assert self.action_ledger is not None
                    failed_sequence = (
                        current_target.sequence
                        if current_target is not None
                        else (
                            self.action_ledger.pending_sequence
                            or self.action_ledger.last_committed_sequence + 1
                        )
                    )
                    self.action_ledger.fail(
                        "franka",
                        sequence=failed_sequence,
                        reason=telemetry.fault_reason,
                    )
                self.safety_gate.latch_fault(telemetry.fault_reason)
                if failure is None:
                    failure = stop_failure
            terminal = (
                FrankaSessionState.STOPPED
                if failure is None
                else FrankaSessionState.FAULT_LATCHED
            )
            self._set_terminal(terminal)
            self.last_telemetry = telemetry.freeze(terminal)
        if failure is not None:
            if isinstance(failure, (KeyboardInterrupt, SystemExit)):
                raise failure
            raise FrankaPersistentSessionError(
                f"persistent Franka session failed: {failure}"
            ) from failure
        assert self.last_telemetry is not None
        return self.last_telemetry


__all__ = [
    "CommissionedFrankaEnvelope",
    "ExperimentalSupervisedFrankaEnvelope",
    "FrankaJointTarget",
    "FrankaPersistentSession",
    "FrankaPersistentSessionError",
    "FrankaPersistentTelemetry",
    "FrankaPoseRing",
    "FrankaPoseSample",
    "FrankaSessionMode",
    "FrankaSessionState",
    "FrankaTargetSampleHold",
    "FrankaTargetSource",
    "PersistentFrankaBackend",
    "PersistentFrankaControl",
    "SUPERVISED_BOOTSTRAP_READ_TO_WRITE_DEADLINE_S",
    "SupervisedFrankaPreflightToken",
    "VerifiedFrankaPreflightToken",
    "load_commissioned_franka_envelope",
    "load_experimental_supervised_franka_envelope",
    "verified_franka_preflight_token_from_report",
]
