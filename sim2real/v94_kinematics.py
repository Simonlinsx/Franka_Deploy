"""Mechanical transforms and feedback kinematics for the V94 observation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np

from sim2real.contracts.v94 import (
    POLICY_HAND_ORDER,
    Q_HAND_SEMANTIC_CLOSE_RAD,
    REGISTER_HAND_ORDER,
    V94Contract,
)
from .observation.model import _rigid, pose_from_position_quaternion_wxyz

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ANYDEX_ROOT = WORKSPACE_ROOT / "dexgrasp" / "third_party" / "AnyDexGrasp"


def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _register_to_policy(value: np.ndarray) -> np.ndarray:
    registers = np.asarray(value)
    if registers.shape != (6,):
        raise ValueError("register-order vector must have shape (6,)")
    by_name = dict(zip(REGISTER_HAND_ORDER, registers.tolist()))
    return np.asarray([by_name[name] for name in POLICY_HAND_ORDER])


@dataclass(frozen=True)
class RH56VirtualFeedback:
    q_policy_order_rad: np.ndarray
    close_fraction_policy_order: np.ndarray


class RH56FeedbackMapper:
    """Map manufacturer ANGLE_ACT feedback to six semantic virtual joints."""

    def __init__(
        self,
        *,
        open_endpoints_register_order: np.ndarray = np.full(6, 1000),
        close_endpoints_register_order: np.ndarray = np.zeros(6),
        q_hand_close_rad: np.ndarray = Q_HAND_SEMANTIC_CLOSE_RAD,
    ) -> None:
        self.open_register = _vector(
            open_endpoints_register_order, 6, "open_endpoints_register_order"
        )
        self.close_register = _vector(
            close_endpoints_register_order, 6, "close_endpoints_register_order"
        )
        if np.any(np.isclose(self.open_register, self.close_register)):
            raise ValueError("RH56 open and close endpoints must differ on every axis")
        self.q_close = _vector(q_hand_close_rad, 6, "q_hand_close_rad")
        if np.any(self.q_close <= 0.0):
            raise ValueError("q_hand_close_rad must be positive")

    def map(self, angle_act_register_order: np.ndarray) -> RH56VirtualFeedback:
        actual = _vector(angle_act_register_order, 6, "angle_act_register_order")
        fraction_register = (self.open_register - actual) / (
            self.open_register - self.close_register
        )
        fraction_policy = np.clip(_register_to_policy(fraction_register), 0.0, 1.0)
        return RH56VirtualFeedback(
            q_policy_order_rad=(fraction_policy * self.q_close).astype(np.float32),
            close_fraction_policy_order=fraction_policy.astype(np.float32),
        )


def T_base_policy_palm_from_franka(
    *,
    T_base_ee: np.ndarray,
    F_T_EE: np.ndarray,
    T_flange_policy_palm: np.ndarray,
) -> np.ndarray:
    """Compose base-to-policy-palm while respecting configured Franka EE."""

    base_ee = _rigid(T_base_ee, "T_base_ee")
    flange_ee = _rigid(F_T_EE, "F_T_EE")
    flange_palm = _rigid(T_flange_policy_palm, "T_flange_policy_palm")
    base_flange = base_ee @ np.linalg.inv(flange_ee)
    return _rigid(base_flange @ flange_palm, "T_base_policy_palm")


@dataclass(frozen=True)
class KinematicVelocity:
    linear_base_m_s: np.ndarray
    angular_base_rad_s: np.ndarray
    hand_policy_rad_s: np.ndarray
    dt_s: Optional[float]


class KinematicVelocityTracker:
    """Finite-difference base-frame palm twist and semantic hand velocity."""

    def __init__(self, *, maximum_dt_s: float = 0.25) -> None:
        self.maximum_dt_s = float(maximum_dt_s)
        if not np.isfinite(self.maximum_dt_s) or self.maximum_dt_s <= 0.0:
            raise ValueError("maximum_dt_s must be finite and positive")
        self._timestamp: Optional[float] = None
        self._palm: Optional[np.ndarray] = None
        self._hand: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._timestamp = None
        self._palm = None
        self._hand = None

    def update(
        self,
        *,
        captured_at_s: float,
        T_base_palm: np.ndarray,
        hand_q_policy_order_rad: np.ndarray,
    ) -> KinematicVelocity:
        timestamp = float(captured_at_s)
        if not np.isfinite(timestamp):
            raise ValueError("captured_at_s must be finite")
        palm = _rigid(T_base_palm, "T_base_palm")
        hand = _vector(hand_q_policy_order_rad, 6, "hand_q_policy_order_rad")
        if self._timestamp is None:
            result = KinematicVelocity(
                linear_base_m_s=np.zeros(3, dtype=np.float32),
                angular_base_rad_s=np.zeros(3, dtype=np.float32),
                hand_policy_rad_s=np.zeros(6, dtype=np.float32),
                dt_s=None,
            )
        else:
            dt = timestamp - self._timestamp
            if not np.isfinite(dt) or dt <= 0.0 or dt > self.maximum_dt_s:
                raise ValueError(f"unsafe kinematic finite-difference dt={dt!r}s")
            linear = (palm[:3, 3] - self._palm[:3, 3]) / dt
            relative = palm[:3, :3] @ self._palm[:3, :3].T
            cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
            angle = float(np.arccos(cosine))
            if angle < 1.0e-8:
                rotation_vector = 0.5 * np.asarray(
                    [
                        relative[2, 1] - relative[1, 2],
                        relative[0, 2] - relative[2, 0],
                        relative[1, 0] - relative[0, 1],
                    ]
                )
            else:
                axis = np.asarray(
                    [
                        relative[2, 1] - relative[1, 2],
                        relative[0, 2] - relative[2, 0],
                        relative[1, 0] - relative[0, 1],
                    ]
                ) / (2.0 * np.sin(angle))
                rotation_vector = axis * angle
            result = KinematicVelocity(
                linear_base_m_s=linear.astype(np.float32),
                angular_base_rad_s=(rotation_vector / dt).astype(np.float32),
                hand_policy_rad_s=((hand - self._hand) / dt).astype(np.float32),
                dt_s=float(dt),
            )
        self._timestamp = timestamp
        self._palm = palm.copy()
        self._hand = hand.copy()
        return result


class RH56FingertipKinematics:
    """Official RH56 FK with tip points anchored by the packaged open pose.

    The handoff does not include the simulator URDF's named fingertip frames.
    We therefore bind fixed points in the official AnyDex distal links using
    the supplied open-pose alignment.  This produces a useful read-only FK,
    but execution must remain locked until these five points are measured on
    the installed hand over its commissioned motion range.
    """

    DISTAL_LINKS: Tuple[str, ...] = ("Link53", "Link11", "Link22", "Link33", "Link44")

    def __init__(
        self,
        contract: V94Contract,
        *,
        anydex_root: str | Path = ANYDEX_ROOT,
    ) -> None:
        source_root = Path(anydex_root).expanduser().resolve()
        source_path = WORKSPACE_ROOT / "dexgrasp" / "src"
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
        from anydex_pipeline.inspire_hand_model import (
            OFFICIAL_SOURCE_MESH_OFFSET_M,
            InspireHandModel,
        )
        from anydex_pipeline.rh56_actuator_mapping import OfficialRH56ActuatorMapper

        self.contract = contract
        self.model = InspireHandModel.from_anydex_root(source_root)
        self.mapper = OfficialRH56ActuatorMapper.from_anydex_root(source_root)
        self.source_offset = np.eye(4, dtype=np.float64)
        self.source_offset[:3, 3] = OFFICIAL_SOURCE_MESH_OFFSET_M

        # Audited bridge: AnyDex hand_source -> simulator hand_base is Rz(+90deg).
        cosine, sine = 0.0, 1.0
        self.T_source_hand_base = np.asarray(
            [
                [cosine, -sine, 0.0, 0.0],
                [sine, cosine, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        hand_base_palm = np.eye(4, dtype=np.float64)
        hand_base_palm[:3, 3] = contract.palm_offset_hand_base_m
        self.T_source_palm = self.T_source_hand_base @ hand_base_palm

        reference_palm = pose_from_position_quaternion_wxyz(
            contract.reference_palm_position_base_m,
            contract.reference_palm_quaternion_base_wxyz,
        )
        reference_tips_palm = (
            contract.reference_fingertip_positions_base_m - reference_palm[:3, 3]
        ) @ reference_palm[:3, :3]
        open_joints = self.mapper.feedback_to_joint_positions_rad([1000] * 6)
        fk = self.model.forward_kinematics(open_joints)
        local_points = []
        for link, tip_palm in zip(self.DISTAL_LINKS, reference_tips_palm):
            T_palm_link = (
                np.linalg.inv(self.T_source_palm) @ self.source_offset @ fk[link]
            )
            homogeneous = np.concatenate([tip_palm, [1.0]])
            local = np.linalg.inv(T_palm_link) @ homogeneous
            local_points.append(local[:3])
        self.tip_points_in_distal_links = np.asarray(local_points)
        if np.any(np.linalg.norm(self.tip_points_in_distal_links, axis=1) > 0.20):
            raise ValueError(
                "derived RH56 fingertip point is implausibly far from its link"
            )

        reproduced = self.positions_base(
            angle_act_register_order=np.full(6, 1000),
            T_base_palm=reference_palm,
        )
        if not np.allclose(
            reproduced,
            contract.reference_fingertip_positions_base_m,
            atol=1.0e-9,
            rtol=0.0,
        ):
            raise RuntimeError("RH56 open-pose fingertip anchor did not reproduce")

    def positions_base(
        self,
        *,
        angle_act_register_order: np.ndarray,
        T_base_palm: np.ndarray,
    ) -> np.ndarray:
        palm = _rigid(T_base_palm, "T_base_palm")
        angles = np.asarray(angle_act_register_order, dtype=np.float64)
        if (
            angles.shape != (6,)
            or not np.all(np.isfinite(angles))
            or not np.array_equal(angles, np.rint(angles))
            or np.any(angles < 0)
            or np.any(angles > 1000)
        ):
            raise ValueError("ANGLE_ACT must contain six integers in [0,1000]")
        joints = self.mapper.feedback_to_joint_positions_rad(angles.astype(int))
        fk = self.model.forward_kinematics(joints)
        output = []
        for link, local_point in zip(
            self.DISTAL_LINKS, self.tip_points_in_distal_links
        ):
            T_palm_link = (
                np.linalg.inv(self.T_source_palm) @ self.source_offset @ fk[link]
            )
            point_palm = T_palm_link @ np.concatenate([local_point, [1.0]])
            point_base = palm @ point_palm
            output.append(point_base[:3])
        return np.asarray(output, dtype=np.float64)


__all__ = [
    "ANYDEX_ROOT",
    "KinematicVelocity",
    "KinematicVelocityTracker",
    "RH56FeedbackMapper",
    "RH56FingertipKinematics",
    "RH56VirtualFeedback",
    "T_base_policy_palm_from_franka",
]
