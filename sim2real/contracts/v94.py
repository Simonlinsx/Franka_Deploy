"""Structured V94 constants sourced from the transferred deployment bundle."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Tuple

import numpy as np

from sim2real.deployment.bundle import DeployBundle

POLICY_HAND_ORDER: Tuple[str, ...] = (
    "thumb_rotation",
    "thumb_bending",
    "index",
    "middle",
    "ring",
    "little",
)
REGISTER_HAND_ORDER: Tuple[str, ...] = (
    "little",
    "ring",
    "middle",
    "index",
    "thumb_bending",
    "thumb_rotation",
)
FINGERTIP_ORDER: Tuple[str, ...] = (
    "thumb",
    "index",
    "middle",
    "ring",
    "little",
)
Q_HAND_SEMANTIC_CLOSE_RAD = np.asarray(
    [1.25, 0.599, 0.95, 0.95, 1.05, 1.10], dtype=np.float32
)
INITIAL_PREVIOUS_ACTION13 = np.asarray([0.0] * 7 + [-1.0] * 6, dtype=np.float32)
for _constant in (Q_HAND_SEMANTIC_CLOSE_RAD, INITIAL_PREVIOUS_ACTION13):
    _constant.setflags(write=False)


def _array(value: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values with shape {shape}")
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class V94Contract:
    bundle_status: str
    control_dt_s: float
    camera_rate_hz: float
    camera_width: int
    camera_height: int
    camera_serial: str
    calibration_id: str
    camera_K: np.ndarray
    depth_scale_m_per_unit: float
    depth_range_m: Tuple[float, float]
    T_base_camera_optical: np.ndarray
    T_flange_hand_base: np.ndarray
    palm_offset_hand_base_m: np.ndarray
    q_home_rad: np.ndarray
    joint_limits_rad: np.ndarray
    q_hand_close_rad: np.ndarray
    reference_palm_position_base_m: np.ndarray
    reference_palm_quaternion_base_wxyz: np.ndarray
    reference_fingertip_positions_base_m: np.ndarray

    @classmethod
    def from_bundle(cls, bundle: DeployBundle) -> "V94Contract":
        scene = bundle.read_json("calibration/scene_manifest.json")
        validation = bundle.read_json(
            "validation_video/rolling_student_dynamic_20trial.config.json"
        )
        initial = bundle.read_json("alignment/reset_idle_open/initial_pose.json")

        camera = _mapping(scene.get("camera"), "scene.camera")
        identity = _mapping(camera.get("identity"), "scene.camera.identity")
        stream = _mapping(camera.get("stream"), "scene.camera.stream")
        intrinsics = _mapping(
            camera.get("pinhole_intrinsics"), "scene.camera.pinhole_intrinsics"
        )
        depth = _mapping(camera.get("depth"), "scene.camera.depth")
        extrinsics = _mapping(camera.get("extrinsics"), "scene.camera.extrinsics")
        robot = _mapping(scene.get("robot"), "scene.robot")
        end_effector = _mapping(
            robot.get("operational_end_effector"),
            "scene.robot.operational_end_effector",
        )
        position_limits = _mapping(
            robot.get("candidate_description_joint_limits"),
            "scene.robot.candidate_description_joint_limits",
        )
        environment = _mapping(validation.get("environment"), "validation.environment")
        init_q = _mapping(
            environment.get("robot_init_joint_pos"),
            "validation.environment.robot_init_joint_pos",
        )

        hand_names = tuple(str(value) for value in environment["hand_joint_names"])
        expected_hand_names = (
            "thumb_proximal_yaw_joint",
            "thumb_proximal_pitch_joint",
            "index_proximal_joint",
            "middle_proximal_joint",
            "ring_proximal_joint",
            "pinky_proximal_joint",
        )
        if hand_names != expected_hand_names:
            raise ValueError(f"unexpected V94 policy hand joints: {hand_names}")
        if environment.get("action_contract") != "inspire_semantic_13d":
            raise ValueError("validation action_contract is not inspire_semantic_13d")
        if environment.get("palm_body_name") != "hand_base_link":
            raise ValueError("validation palm_body_name is not hand_base_link")

        lower = _array(position_limits["position_lower"], (7,), "joint lower")
        upper = _array(position_limits["position_upper"], (7,), "joint upper")
        if np.any(lower >= upper):
            raise ValueError("candidate Franka joint limits are invalid")
        q_home = np.asarray(
            [init_q[f"panda_joint{index}"] for index in range(1, 8)],
            dtype=np.float32,
        )
        control_dt = float(initial["control_dt_s"])
        if not np.isclose(control_dt, 1.0 / 60.0, atol=1.0e-12, rtol=0.0):
            raise ValueError(f"unexpected V94 control period {control_dt}")

        serial_entry = _mapping(identity.get("serial"), "scene.camera.identity.serial")
        depth_scale_entry = _mapping(
            depth.get("scale_m_per_unit"), "scene.camera.depth.scale_m_per_unit"
        )
        depth_range = _mapping(
            depth.get("configured_processing_range"),
            "scene.camera.depth.configured_processing_range",
        )
        contract = cls(
            bundle_status=str(bundle.manifest.get("status", "")),
            control_dt_s=control_dt,
            camera_rate_hz=float(stream["fps"]),
            camera_width=int(stream["width_px"]),
            camera_height=int(stream["height_px"]),
            camera_serial=str(serial_entry["value"]),
            calibration_id=str(extrinsics["calibration_id"]),
            camera_K=_array(intrinsics["K"], (3, 3), "camera K"),
            depth_scale_m_per_unit=float(depth_scale_entry["value"]),
            depth_range_m=(float(depth_range["min"]), float(depth_range["max"])),
            T_base_camera_optical=_array(
                extrinsics["T_base_camera_color_optical"],
                (4, 4),
                "T_base_camera_color_optical",
            ),
            T_flange_hand_base=_array(
                end_effector["T_flange_tool"], (4, 4), "T_flange_tool"
            ),
            palm_offset_hand_base_m=_array(
                environment["palm_offset"], (3,), "palm_offset"
            ),
            q_home_rad=q_home,
            joint_limits_rad=np.stack([lower, upper], axis=1),
            q_hand_close_rad=Q_HAND_SEMANTIC_CLOSE_RAD.copy(),
            reference_palm_position_base_m=_array(
                initial["palm_position_base_m"], (3,), "reference palm position"
            ),
            reference_palm_quaternion_base_wxyz=_array(
                initial["palm_quaternion_base_wxyz"],
                (4,),
                "reference palm quaternion",
            ),
            reference_fingertip_positions_base_m=_array(
                initial["fingertip_positions_base_m"],
                (5, 3),
                "reference fingertip positions",
            ),
        )
        if contract.camera_width != 848 or contract.camera_height != 480:
            raise ValueError("V94 real camera contract must be 848x480")
        if not np.isclose(float(environment["joint_target_arm_max_delta"]), 0.015):
            raise ValueError("unexpected V94 arm target delta limit")
        if not np.isclose(float(environment["joint_target_hand_max_delta"]), 0.05):
            raise ValueError("unexpected V94 hand target delta limit")
        return contract

    @property
    def policy_rate_hz(self) -> float:
        return 1.0 / self.control_dt_s

    def with_runtime_policy_rate_hz(self, policy_rate_hz: object) -> "V94Contract":
        """Return the same observation/action geometry at a selected tick rate.

        The transferred bundle remains the authority for every spatial and
        normalization field.  Runtime rate selection is separately bound to
        the selected checkpoint's ``control_dt`` before hardware access.
        """

        if isinstance(policy_rate_hz, (bool, np.bool_)):
            raise ValueError("runtime policy rate must be exactly 20 or 60 Hz")
        try:
            rate = float(policy_rate_hz)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "runtime policy rate must be exactly 20 or 60 Hz"
            ) from exc
        if not math.isfinite(rate) or not any(
            math.isclose(rate, allowed, rel_tol=0.0, abs_tol=1.0e-12)
            for allowed in (20.0, 60.0)
        ):
            raise ValueError("runtime policy rate must be exactly 20 or 60 Hz")
        return replace(self, control_dt_s=1.0 / rate)

    @property
    def maximum_nominal_arm_velocity_rad_s(self) -> float:
        return 0.003 / self.control_dt_s

    @property
    def T_flange_policy_palm(self) -> np.ndarray:
        offset = np.eye(4, dtype=np.float64)
        offset[:3, 3] = self.palm_offset_hand_base_m
        return self.T_flange_hand_base @ offset


__all__ = [
    "FINGERTIP_ORDER",
    "INITIAL_PREVIOUS_ACTION13",
    "POLICY_HAND_ORDER",
    "Q_HAND_SEMANTIC_CLOSE_RAD",
    "REGISTER_HAND_ORDER",
    "V94Contract",
]
