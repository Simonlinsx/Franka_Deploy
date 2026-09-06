"""Typed observation contract shared by collection and policy dry-runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

OBSERVATION_SCHEMA_VERSION = 1
FRANKA_DOF = 7
INSPIRE_AXES: Tuple[str, ...] = (
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotate",
)


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} finite numbers") from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result.copy()


def _integer_vector(value: Any, size: int, name: str) -> np.ndarray:
    try:
        numeric = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} finite integers") from exc
    if (
        numeric.shape != (size,)
        or not np.all(np.isfinite(numeric))
        or not np.array_equal(numeric, np.rint(numeric))
    ):
        raise ValueError(f"{name} must contain {size} finite integers")
    return numeric.astype(np.int32)


def _finite_array(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite numeric array")
    return result.copy()


def _rigid_transform(value: Any, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 4x4 transform") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-7, rtol=0.0):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5, rtol=0.0):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5, rtol=0.0):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix.copy()


def _inertia_matrix(value: Any, name: str) -> np.ndarray:
    flat = _finite_vector(value, 9, name)
    matrix = flat.reshape((3, 3), order="F")
    if not np.allclose(matrix, matrix.T, atol=1.0e-9, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    if float(np.min(np.linalg.eigvalsh(matrix))) < -1.0e-10:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def franka_column_major_pose(value: Any, name: str = "O_T_EE") -> np.ndarray:
    """Convert a libfranka 16-vector into a strict row-major 4x4 matrix."""

    flat = _finite_vector(value, 16, name)
    return _rigid_transform(flat.reshape((4, 4), order="F"), name)


def normalize_robot_mode(value: Any) -> str:
    """Return a stable lower-case robot-mode name across pylibfranka builds."""

    raw = getattr(value, "name", None)
    text = str(raw if raw is not None else value).strip().lower()
    for candidate in (
        "idle",
        "move",
        "reflex",
        "userstopped",
        "automaticerrorrecovery",
    ):
        if candidate in text.replace("_", ""):
            return candidate
    return text


def active_error_names(errors: Any) -> Tuple[str, ...]:
    """Extract active libfranka error flags without relying on one binding ABI."""

    if errors is None:
        return ("missing_current_errors",)
    if isinstance(errors, Mapping):
        return tuple(
            sorted(str(name) for name, active in errors.items() if bool(active))
        )

    names = []
    saw_boolean_field = False
    for name in dir(errors):
        if name.startswith("_"):
            continue
        try:
            value = getattr(errors, name)
        except Exception:
            continue
        if isinstance(value, (bool, np.bool_)):
            saw_boolean_field = True
            if bool(value):
                names.append(name)
    if names:
        return tuple(sorted(names))
    if saw_boolean_field:
        return ()

    # pylibfranka::Errors implements a native bool check.  If it reports an
    # active error that this binding does not expose as named boolean fields,
    # retain a fail-closed sentinel.
    try:
        return ("unknown_active_error",) if bool(errors) else ()
    except Exception:
        return ("unreadable_current_errors",)


@dataclass(frozen=True)
class FrankaObservation:
    captured_at_s: float
    q: np.ndarray
    dq: np.ndarray
    q_desired: np.ndarray
    dq_desired: np.ndarray
    tau_j: np.ndarray
    tau_ext_hat_filtered: np.ndarray
    T_base_ee: np.ndarray
    T_base_ee_desired: np.ndarray
    desired_cartesian_velocity: np.ndarray
    external_wrench_base: np.ndarray
    joint_contact: np.ndarray
    joint_collision: np.ndarray
    cartesian_contact: np.ndarray
    cartesian_collision: np.ndarray
    robot_mode: str
    current_errors: Tuple[str, ...]
    control_command_success_rate: float
    F_T_EE: np.ndarray
    F_x_Cee_m: np.ndarray
    I_ee_kg_m2: np.ndarray
    m_ee_kg: float
    m_load_kg: float
    m_total_kg: float

    @classmethod
    def from_state(cls, state: Any, captured_at_s: float) -> "FrankaObservation":
        timestamp = float(captured_at_s)
        scalar_values = (
            timestamp,
            float(getattr(state, "control_command_success_rate")),
            float(getattr(state, "m_ee")),
            float(getattr(state, "m_load")),
            float(getattr(state, "m_total")),
        )
        if not all(np.isfinite(value) for value in scalar_values):
            raise ValueError("Franka timestamps, rates, and masses must be finite")
        if any(value < 0.0 for value in scalar_values[2:]):
            raise ValueError("Franka masses must be non-negative")
        return cls(
            captured_at_s=timestamp,
            q=_finite_vector(getattr(state, "q", None), 7, "state.q"),
            dq=_finite_vector(getattr(state, "dq", None), 7, "state.dq"),
            q_desired=_finite_vector(getattr(state, "q_d", None), 7, "state.q_d"),
            dq_desired=_finite_vector(getattr(state, "dq_d", None), 7, "state.dq_d"),
            tau_j=_finite_vector(getattr(state, "tau_J", None), 7, "state.tau_J"),
            tau_ext_hat_filtered=_finite_vector(
                getattr(state, "tau_ext_hat_filtered", None),
                7,
                "state.tau_ext_hat_filtered",
            ),
            T_base_ee=franka_column_major_pose(getattr(state, "O_T_EE", None)),
            T_base_ee_desired=franka_column_major_pose(
                getattr(state, "O_T_EE_d", None), "O_T_EE_d"
            ),
            desired_cartesian_velocity=_finite_vector(
                getattr(state, "O_dP_EE_d", None), 6, "state.O_dP_EE_d"
            ),
            external_wrench_base=_finite_vector(
                getattr(state, "O_F_ext_hat_K", None), 6, "state.O_F_ext_hat_K"
            ),
            joint_contact=_finite_vector(
                getattr(state, "joint_contact", None), 7, "state.joint_contact"
            ),
            joint_collision=_finite_vector(
                getattr(state, "joint_collision", None), 7, "state.joint_collision"
            ),
            cartesian_contact=_finite_vector(
                getattr(state, "cartesian_contact", None), 6, "state.cartesian_contact"
            ),
            cartesian_collision=_finite_vector(
                getattr(state, "cartesian_collision", None),
                6,
                "state.cartesian_collision",
            ),
            robot_mode=normalize_robot_mode(getattr(state, "robot_mode", None)),
            current_errors=active_error_names(getattr(state, "current_errors", None)),
            control_command_success_rate=scalar_values[1],
            F_T_EE=franka_column_major_pose(getattr(state, "F_T_EE", None), "F_T_EE"),
            F_x_Cee_m=_finite_vector(
                getattr(state, "F_x_Cee", None), 3, "state.F_x_Cee"
            ),
            I_ee_kg_m2=_inertia_matrix(getattr(state, "I_ee", None), "state.I_ee"),
            m_ee_kg=scalar_values[2],
            m_load_kg=scalar_values[3],
            m_total_kg=scalar_values[4],
        )


@dataclass(frozen=True)
class InspireObservation:
    captured_at_s: float
    angle_targets: np.ndarray
    angles: np.ndarray
    positions: np.ndarray
    forces: np.ndarray
    currents: np.ndarray
    errors: np.ndarray
    statuses: np.ndarray
    temperatures_c: np.ndarray

    @classmethod
    def from_snapshot(
        cls, snapshot: Mapping[str, Any], captured_at_s: float
    ) -> "InspireObservation":
        timestamp = float(captured_at_s)
        if not np.isfinite(timestamp):
            raise ValueError("Inspire captured_at_s must be finite")
        return cls(
            captured_at_s=timestamp,
            angle_targets=_integer_vector(
                snapshot.get("angle_targets"), 6, "angle_targets"
            ),
            angles=_integer_vector(snapshot.get("angles"), 6, "angles"),
            positions=_integer_vector(snapshot.get("positions"), 6, "positions"),
            forces=_integer_vector(snapshot.get("forces"), 6, "forces"),
            currents=_integer_vector(snapshot.get("currents"), 6, "currents"),
            errors=_integer_vector(snapshot.get("errors"), 6, "errors"),
            statuses=_integer_vector(snapshot.get("statuses"), 6, "statuses"),
            temperatures_c=_integer_vector(
                snapshot.get("temperatures"), 6, "temperatures"
            ),
        )


@dataclass(frozen=True)
class ObjectPCDObservation:
    received_at_s: float
    captured_at_s: float
    frame_id: int
    valid: bool
    pcd_current: Optional[np.ndarray]
    pcd_history: Optional[np.ndarray]
    pcd_reference: Optional[np.ndarray]
    center: Optional[np.ndarray]
    velocity: Optional[np.ndarray]
    bbox_xyxy: Optional[np.ndarray]
    reference_frame: str
    point_frame: str
    calibration_id: Optional[str]
    camera_serial: Optional[str]
    T_base_camera: Optional[np.ndarray]
    raw_points: int
    message: str = ""

    @property
    def age_s(self) -> float:
        return float(self.received_at_s - self.captured_at_s)

    @classmethod
    def from_packet(cls, packet: Any, received_at_s: float) -> "ObjectPCDObservation":
        received = float(received_at_s)
        captured = float(getattr(packet, "timestamp"))
        if not np.isfinite(received) or not np.isfinite(captured):
            raise ValueError("object PCD timestamps must be finite")
        valid = bool(getattr(packet, "valid"))
        current = history = reference = center = velocity = bbox = None
        transform = None
        debug = getattr(packet, "debug", None)
        if debug is None:
            debug = {}
        if not isinstance(debug, Mapping):
            raise ValueError("object PCD packet.debug must be a mapping")
        try:
            raw_points = int(debug.get("raw_points", 0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("object PCD raw_points must be an integer") from exc
        if raw_points < 0:
            raise ValueError("object PCD raw_points cannot be negative")
        if valid:
            current = _finite_array(getattr(packet, "pcd_current"), "pcd_current")
            history = _finite_array(getattr(packet, "pcd_history"), "pcd_history")
            reference = _finite_array(getattr(packet, "pcd_reference"), "pcd_reference")
            center = _finite_vector(getattr(packet, "center"), 3, "center").astype(
                np.float32
            )
            velocity = _finite_vector(
                getattr(packet, "velocity"), 3, "velocity"
            ).astype(np.float32)
            bbox = _finite_vector(getattr(packet, "bbox_xyxy"), 4, "bbox_xyxy").astype(
                np.float32
            )
            transform = _rigid_transform(
                getattr(packet, "T_base_camera", None), "packet.T_base_camera"
            ).astype(np.float32)
        else:
            leaked = [
                name
                for name in (
                    "pcd_current",
                    "pcd_history",
                    "pcd_reference",
                    "center",
                    "velocity",
                    "bbox_xyxy",
                )
                if getattr(packet, name, None) is not None
            ]
            if leaked:
                raise ValueError(
                    "invalid object PCD packet retained payload fields: "
                    + ", ".join(leaked)
                )
        return cls(
            received_at_s=received,
            captured_at_s=captured,
            frame_id=int(getattr(packet, "frame_id")),
            valid=valid,
            pcd_current=current,
            pcd_history=history,
            pcd_reference=reference,
            center=center,
            velocity=velocity,
            bbox_xyxy=bbox,
            reference_frame=str(getattr(packet, "reference_frame", "")),
            point_frame=str(getattr(packet, "point_frame", "")),
            calibration_id=(
                None
                if getattr(packet, "calibration_id", None) is None
                else str(getattr(packet, "calibration_id"))
            ),
            camera_serial=(
                None
                if getattr(packet, "camera_serial", None) is None
                else str(getattr(packet, "camera_serial"))
            ),
            T_base_camera=transform,
            raw_points=raw_points,
            message=str(getattr(packet, "message", "")),
        )


@dataclass(frozen=True)
class ObservationSpec:
    require_franka: bool = True
    require_inspire: bool = True
    require_object_pcd: bool = True
    object_history_shape: Tuple[int, int, int] = (4, 1024, 3)
    object_current_shape: Tuple[int, int] = (1024, 3)
    object_reference_frame: str = "robot_base"
    object_point_frame: str = "robot_base"
    calibration_id: Optional[str] = None
    camera_serial: Optional[str] = None
    expected_T_base_camera: Optional[np.ndarray] = None
    expected_F_T_EE: Optional[np.ndarray] = None
    expected_m_ee_kg: Optional[float] = None
    expected_F_x_Cee_m: Optional[np.ndarray] = None
    expected_I_ee_kg_m2: Optional[np.ndarray] = None
    max_object_age_s: float = 0.15
    min_raw_points: int = 10
    max_center_reference_error_m: float = 0.020
    max_future_skew_s: float = 0.05
    max_capture_span_s: float = 0.50
    max_hand_temperature_c: int = 59
    minimum_control_success_rate: float = 0.95
    joint_limits_rad: Optional[np.ndarray] = None
    joint_limit_margin_rad: float = 0.05

    def __post_init__(self) -> None:
        if len(self.object_history_shape) != 3 or any(
            int(value) <= 0 for value in self.object_history_shape
        ):
            raise ValueError(
                "object_history_shape must contain three positive integers"
            )
        if len(self.object_current_shape) != 2 or any(
            int(value) <= 0 for value in self.object_current_shape
        ):
            raise ValueError("object_current_shape must contain two positive integers")
        if tuple(self.object_history_shape[1:]) != tuple(self.object_current_shape):
            raise ValueError("history point shape must match object_current_shape")
        for name in (
            "max_object_age_s",
            "max_future_skew_s",
            "max_capture_span_s",
            "max_center_reference_error_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        success = float(self.minimum_control_success_rate)
        if not np.isfinite(success) or not 0.0 <= success <= 1.0:
            raise ValueError("minimum_control_success_rate must be in [0,1]")
        if (
            isinstance(self.min_raw_points, (bool, np.bool_))
            or int(self.min_raw_points) < 10
        ):
            raise ValueError("min_raw_points must be an integer >= 10")
        object.__setattr__(self, "min_raw_points", int(self.min_raw_points))
        if self.expected_T_base_camera is not None:
            object.__setattr__(
                self,
                "expected_T_base_camera",
                _rigid_transform(self.expected_T_base_camera, "expected_T_base_camera"),
            )
        end_effector_values = (
            self.expected_m_ee_kg,
            self.expected_F_x_Cee_m,
            self.expected_I_ee_kg_m2,
        )
        if any(value is not None for value in end_effector_values) and not all(
            value is not None for value in end_effector_values
        ):
            raise ValueError(
                "expected_m_ee_kg, expected_F_x_Cee_m, and "
                "expected_I_ee_kg_m2 must be provided together"
            )
        if self.expected_F_T_EE is not None:
            object.__setattr__(
                self,
                "expected_F_T_EE",
                _rigid_transform(self.expected_F_T_EE, "expected_F_T_EE"),
            )
        if self.expected_m_ee_kg is not None:
            expected_mass = float(self.expected_m_ee_kg)
            if not np.isfinite(expected_mass) or expected_mass < 0.0:
                raise ValueError("expected_m_ee_kg must be finite and non-negative")
            object.__setattr__(self, "expected_m_ee_kg", expected_mass)
            object.__setattr__(
                self,
                "expected_F_x_Cee_m",
                _finite_vector(self.expected_F_x_Cee_m, 3, "expected_F_x_Cee_m"),
            )
            expected_inertia = np.asarray(self.expected_I_ee_kg_m2, dtype=np.float64)
            if expected_inertia.shape == (3, 3):
                expected_inertia = _inertia_matrix(
                    expected_inertia.reshape(9, order="F"),
                    "expected_I_ee_kg_m2",
                )
            else:
                expected_inertia = _inertia_matrix(
                    expected_inertia, "expected_I_ee_kg_m2"
                )
            object.__setattr__(self, "expected_I_ee_kg_m2", expected_inertia)
        if self.joint_limits_rad is not None:
            limits = np.asarray(self.joint_limits_rad, dtype=np.float64)
            if limits.shape != (7, 2) or not np.all(np.isfinite(limits)):
                raise ValueError("joint_limits_rad must be a finite (7,2) array")
            if np.any(limits[:, 0] >= limits[:, 1]):
                raise ValueError("joint limit lower bounds must be below upper bounds")
            object.__setattr__(self, "joint_limits_rad", limits.copy())


@dataclass(frozen=True)
class ObservationSample:
    captured_at_s: float
    franka: Optional[FrankaObservation]
    inspire: Optional[InspireObservation]
    object_pcd: Optional[ObjectPCDObservation]
    invalid_reasons: Tuple[str, ...] = field(default_factory=tuple)
    motion_blockers: Tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = OBSERVATION_SCHEMA_VERSION

    @property
    def valid(self) -> bool:
        return not self.invalid_reasons

    @property
    def motion_safe(self) -> bool:
        return self.valid and not self.motion_blockers

    def summary(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "captured_at_s": self.captured_at_s,
            "valid": self.valid,
            "motion_safe": self.motion_safe,
            "invalid_reasons": list(self.invalid_reasons),
            "motion_blockers": list(self.motion_blockers),
            "franka_mode": None if self.franka is None else self.franka.robot_mode,
            "franka_q": None if self.franka is None else self.franka.q.tolist(),
            "eef_xyz_m": (
                None if self.franka is None else self.franka.T_base_ee[:3, 3].tolist()
            ),
            "inspire_angles": (
                None if self.inspire is None else self.inspire.angles.tolist()
            ),
            "object_valid": (
                None if self.object_pcd is None else self.object_pcd.valid
            ),
            "object_age_s": (
                None if self.object_pcd is None else self.object_pcd.age_s
            ),
            "object_center_m": (
                None
                if self.object_pcd is None or self.object_pcd.center is None
                else self.object_pcd.center.tolist()
            ),
            "object_history_shape": (
                None
                if self.object_pcd is None or self.object_pcd.pcd_history is None
                else list(self.object_pcd.pcd_history.shape)
            ),
        }

    def to_npz_payload(self) -> Mapping[str, np.ndarray]:
        """Flatten one sample into a checkpoint-debug-friendly NPZ payload."""

        payload: dict[str, np.ndarray] = {
            "schema_version": np.asarray(self.schema_version, dtype=np.int32),
            "captured_at_s": np.asarray(self.captured_at_s, dtype=np.float64),
            "valid": np.asarray(self.valid, dtype=np.bool_),
            "motion_safe": np.asarray(self.motion_safe, dtype=np.bool_),
            "invalid_reasons": np.asarray(self.invalid_reasons, dtype=np.str_),
            "motion_blockers": np.asarray(self.motion_blockers, dtype=np.str_),
        }
        if self.franka is not None:
            robot = self.franka
            payload.update(
                {
                    "robot_timestamp_s": np.asarray(robot.captured_at_s),
                    "robot_q": robot.q.astype(np.float32),
                    "robot_dq": robot.dq.astype(np.float32),
                    "robot_q_desired": robot.q_desired.astype(np.float32),
                    "robot_dq_desired": robot.dq_desired.astype(np.float32),
                    "robot_tau_j": robot.tau_j.astype(np.float32),
                    "robot_tau_ext_hat_filtered": robot.tau_ext_hat_filtered.astype(
                        np.float32
                    ),
                    "robot_T_base_ee": robot.T_base_ee.astype(np.float32),
                    "robot_T_base_ee_desired": robot.T_base_ee_desired.astype(
                        np.float32
                    ),
                    "robot_desired_cartesian_velocity": (
                        robot.desired_cartesian_velocity.astype(np.float32)
                    ),
                    "robot_external_wrench_base": robot.external_wrench_base.astype(
                        np.float32
                    ),
                    "robot_mode": np.asarray(robot.robot_mode),
                    "robot_current_errors": np.asarray(
                        robot.current_errors, dtype=np.str_
                    ),
                    "robot_control_success_rate": np.asarray(
                        robot.control_command_success_rate, dtype=np.float32
                    ),
                    "robot_F_T_EE": robot.F_T_EE.astype(np.float32),
                    "robot_F_x_Cee_m": robot.F_x_Cee_m.astype(np.float32),
                    "robot_I_ee_kg_m2": robot.I_ee_kg_m2.astype(np.float32),
                    "robot_masses_kg": np.asarray(
                        [robot.m_ee_kg, robot.m_load_kg, robot.m_total_kg],
                        dtype=np.float32,
                    ),
                    "robot_joint_contact": robot.joint_contact.astype(np.float32),
                    "robot_joint_collision": robot.joint_collision.astype(np.float32),
                    "robot_cartesian_contact": robot.cartesian_contact.astype(
                        np.float32
                    ),
                    "robot_cartesian_collision": robot.cartesian_collision.astype(
                        np.float32
                    ),
                }
            )
        if self.inspire is not None:
            hand = self.inspire
            payload.update(
                {
                    "hand_timestamp_s": np.asarray(hand.captured_at_s),
                    "hand_angle_targets": hand.angle_targets,
                    "hand_angles": hand.angles,
                    "hand_positions": hand.positions,
                    "hand_forces": hand.forces,
                    "hand_currents": hand.currents,
                    "hand_errors": hand.errors,
                    "hand_statuses": hand.statuses,
                    "hand_temperatures_c": hand.temperatures_c,
                }
            )
        if self.object_pcd is not None:
            obj = self.object_pcd
            payload.update(
                {
                    "object_received_at_s": np.asarray(obj.received_at_s),
                    "object_timestamp_s": np.asarray(obj.captured_at_s),
                    "object_frame_id": np.asarray(obj.frame_id, dtype=np.int64),
                    "object_valid": np.asarray(obj.valid, dtype=np.bool_),
                    "object_reference_frame": np.asarray(obj.reference_frame),
                    "object_point_frame": np.asarray(obj.point_frame),
                    "object_calibration_id": np.asarray(obj.calibration_id or ""),
                    "object_camera_serial": np.asarray(obj.camera_serial or ""),
                    "object_raw_points": np.asarray(obj.raw_points, dtype=np.int32),
                    "object_message": np.asarray(obj.message),
                }
            )
            if obj.valid:
                assert obj.pcd_current is not None
                assert obj.pcd_history is not None
                assert obj.pcd_reference is not None
                assert obj.center is not None
                assert obj.velocity is not None
                assert obj.bbox_xyxy is not None
                assert obj.T_base_camera is not None
                payload.update(
                    {
                        "object_pcd": obj.pcd_current.astype(np.float32),
                        "object_pcd_history": obj.pcd_history.astype(np.float32),
                        "object_pcd_reference": obj.pcd_reference.astype(np.float32),
                        "object_center": obj.center.astype(np.float32),
                        "object_velocity": obj.velocity.astype(np.float32),
                        "object_bbox_xyxy": obj.bbox_xyxy.astype(np.float32),
                        "object_T_base_camera": obj.T_base_camera.astype(np.float32),
                    }
                )
        return payload


def assemble_observation(
    *,
    captured_at_s: float,
    spec: ObservationSpec,
    franka: Optional[FrankaObservation],
    inspire: Optional[InspireObservation],
    object_pcd: Optional[ObjectPCDObservation],
) -> ObservationSample:
    """Validate asynchronous sensor samples and retain explicit stop reasons."""

    now = float(captured_at_s)
    if not np.isfinite(now):
        raise ValueError("captured_at_s must be finite")
    invalid = []
    blockers = []
    timestamps = []

    if franka is None:
        if spec.require_franka:
            invalid.append("missing Franka observation")
    else:
        timestamps.append(franka.captured_at_s)
        if franka.current_errors:
            blockers.append("Franka current_errors=" + ",".join(franka.current_errors))
        if franka.robot_mode not in ("idle", "move"):
            blockers.append(f"Franka robot_mode={franka.robot_mode}")
        if np.any(franka.joint_collision) or np.any(franka.cartesian_collision):
            blockers.append("Franka collision flag is active")
        if np.any(franka.joint_contact) or np.any(franka.cartesian_contact):
            blockers.append("Franka contact flag is active")
        if (
            franka.robot_mode == "move"
            and franka.control_command_success_rate < spec.minimum_control_success_rate
        ):
            blockers.append(
                "Franka control success rate "
                f"{franka.control_command_success_rate:.3f}<"
                f"{spec.minimum_control_success_rate:.3f}"
            )
        if spec.joint_limits_rad is not None:
            lower = spec.joint_limits_rad[:, 0] + spec.joint_limit_margin_rad
            upper = spec.joint_limits_rad[:, 1] - spec.joint_limit_margin_rad
            if np.any(franka.q <= lower) or np.any(franka.q >= upper):
                blockers.append("Franka q violates commissioned joint-limit margin")
        if spec.expected_F_T_EE is not None and not np.allclose(
            franka.F_T_EE, spec.expected_F_T_EE, atol=1.0e-8, rtol=0.0
        ):
            blockers.append("Franka F_T_EE differs from commissioning profile")
        if spec.expected_m_ee_kg is not None:
            if not np.isclose(
                franka.m_ee_kg,
                spec.expected_m_ee_kg,
                atol=1.0e-5,
                rtol=0.0,
            ):
                blockers.append("Franka m_ee differs from commissioning profile")
            if not np.allclose(
                franka.F_x_Cee_m,
                spec.expected_F_x_Cee_m,
                atol=1.0e-6,
                rtol=0.0,
            ):
                blockers.append("Franka F_x_Cee differs from commissioning profile")
            if not np.allclose(
                franka.I_ee_kg_m2,
                spec.expected_I_ee_kg_m2,
                atol=1.0e-6,
                rtol=0.0,
            ):
                blockers.append("Franka I_ee differs from commissioning profile")

    if inspire is None:
        if spec.require_inspire:
            invalid.append("missing Inspire observation")
    else:
        timestamps.append(inspire.captured_at_s)
        if np.any(inspire.errors):
            blockers.append(f"Inspire errors={inspire.errors.tolist()}")
        protection_statuses = inspire.statuses[
            np.isin(inspire.statuses, np.asarray([5, 6, 7], dtype=np.int32))
        ]
        if protection_statuses.size:
            blockers.append(
                "Inspire protection/fault status=" f"{inspire.statuses.tolist()}"
            )
        known_statuses = np.asarray([0, 1, 2, 3, 5, 6, 7, 255], dtype=np.int32)
        if not np.all(np.isin(inspire.statuses, known_statuses)):
            invalid.append(f"Inspire has unknown status={inspire.statuses.tolist()}")
        if np.max(inspire.temperatures_c) > spec.max_hand_temperature_c:
            blockers.append(
                "Inspire temperature exceeds "
                f"{spec.max_hand_temperature_c} C: {inspire.temperatures_c.tolist()}"
            )
        if np.any(inspire.angles < 0) or np.any(inspire.angles > 1000):
            invalid.append("Inspire ANGLE_ACT is outside 0..1000")
        if np.any(inspire.angle_targets < -1) or np.any(inspire.angle_targets > 1000):
            invalid.append("Inspire ANGLE_SET is outside -1..1000")

    if object_pcd is None:
        if spec.require_object_pcd:
            invalid.append("missing object PCD observation")
    else:
        timestamps.append(object_pcd.captured_at_s)
        if not object_pcd.valid:
            invalid.append("object PCD packet is invalid")
        else:
            if object_pcd.age_s > spec.max_object_age_s:
                invalid.append(
                    f"object PCD is stale: {object_pcd.age_s:.3f}s>"
                    f"{spec.max_object_age_s:.3f}s"
                )
            if object_pcd.age_s < -spec.max_future_skew_s:
                invalid.append(
                    f"object PCD timestamp is {-object_pcd.age_s:.3f}s in the future"
                )
            if object_pcd.pcd_history is None or tuple(
                object_pcd.pcd_history.shape
            ) != tuple(spec.object_history_shape):
                actual = (
                    None
                    if object_pcd.pcd_history is None
                    else tuple(object_pcd.pcd_history.shape)
                )
                invalid.append(
                    f"object PCD history shape {actual}!={spec.object_history_shape}"
                )
            if object_pcd.pcd_current is None or tuple(
                object_pcd.pcd_current.shape
            ) != tuple(spec.object_current_shape):
                actual = (
                    None
                    if object_pcd.pcd_current is None
                    else tuple(object_pcd.pcd_current.shape)
                )
                invalid.append(
                    f"object PCD current shape {actual}!={spec.object_current_shape}"
                )
            if object_pcd.pcd_reference is None or tuple(
                object_pcd.pcd_reference.shape
            ) != tuple(spec.object_current_shape):
                actual = (
                    None
                    if object_pcd.pcd_reference is None
                    else tuple(object_pcd.pcd_reference.shape)
                )
                invalid.append(
                    f"object PCD reference shape {actual}!={spec.object_current_shape}"
                )
            if object_pcd.raw_points < spec.min_raw_points:
                invalid.append(
                    f"object PCD raw_points {object_pcd.raw_points}<"
                    f"{spec.min_raw_points}"
                )
            if (
                object_pcd.pcd_reference is not None
                and object_pcd.pcd_reference.ndim == 2
                and object_pcd.pcd_reference.shape[1] >= 3
                and object_pcd.center is not None
            ):
                reference_center = np.median(object_pcd.pcd_reference[:, :3], axis=0)
                center_error = float(
                    np.linalg.norm(reference_center - object_pcd.center)
                )
                if center_error > spec.max_center_reference_error_m:
                    invalid.append(
                        "object PCD center/reference mismatch "
                        f"{center_error:.4f}m>"
                        f"{spec.max_center_reference_error_m:.4f}m"
                    )
            if object_pcd.reference_frame != spec.object_reference_frame:
                invalid.append(
                    "object PCD reference_frame "
                    f"{object_pcd.reference_frame!r}!={spec.object_reference_frame!r}"
                )
            if object_pcd.point_frame != spec.object_point_frame:
                invalid.append(
                    "object PCD point_frame "
                    f"{object_pcd.point_frame!r}!={spec.object_point_frame!r}"
                )
            if (
                spec.calibration_id is not None
                and object_pcd.calibration_id != spec.calibration_id
            ):
                invalid.append(
                    "object PCD calibration_id "
                    f"{object_pcd.calibration_id!r}!={spec.calibration_id!r}"
                )
            if (
                spec.camera_serial is not None
                and object_pcd.camera_serial != spec.camera_serial
            ):
                invalid.append(
                    "object PCD camera_serial "
                    f"{object_pcd.camera_serial!r}!={spec.camera_serial!r}"
                )
            if spec.expected_T_base_camera is not None:
                if object_pcd.T_base_camera is None:
                    invalid.append("object PCD has no T_base_camera")
                elif not np.allclose(
                    object_pcd.T_base_camera,
                    spec.expected_T_base_camera,
                    atol=1.0e-6,
                    rtol=0.0,
                ):
                    invalid.append(
                        "object PCD T_base_camera does not match calibration"
                    )

    if timestamps and max(timestamps) - min(timestamps) > spec.max_capture_span_s:
        invalid.append(
            "sensor capture span "
            f"{max(timestamps) - min(timestamps):.3f}s>"
            f"{spec.max_capture_span_s:.3f}s"
        )

    return ObservationSample(
        captured_at_s=now,
        franka=franka,
        inspire=inspire,
        object_pcd=object_pcd,
        invalid_reasons=tuple(invalid),
        motion_blockers=tuple(blockers),
    )
