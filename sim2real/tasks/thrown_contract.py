"""Validated V57 thrown-object task geometry shared by sim and real launchers.

The fixed-camera calibration owns image geometry and ``T_base_camera``.  This
module separately owns the V57 robot reset, ballistic release/target ranges,
object envelopes and selected curriculum.  Keeping those contracts separate
prevents a camera YAML from silently becoming a robot/task configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import yaml

from sim2real.contracts.v94 import V94Contract


POLICY_HAND_ORDER = (
    "thumb_rotation",
    "thumb_bending",
    "index",
    "middle",
    "ring",
    "little",
)
REGISTER_HAND_ORDER = (
    "little",
    "ring",
    "middle",
    "index",
    "thumb_bending",
    "thumb_rotation",
)
SUPPORTED_CURRICULA = ("alpha_0_5", "alpha_1_0")


class ThrownTaskContractError(ValueError):
    """The selected thrown-object task contract is missing or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThrownTaskContractError(f"{name} must be a mapping")
    return value


def _finite_vector(value: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ThrownTaskContractError(
            f"{name} must contain exactly {size} finite values"
        )
    return result


def _finite_range(value: object, name: str) -> tuple[float, float]:
    result = _finite_vector(value, 2, name)
    if not float(result[0]) < float(result[1]):
        raise ThrownTaskContractError(f"{name} must be strictly increasing")
    return float(result[0]), float(result[1])


def _axis_box(value: object, name: str) -> tuple[np.ndarray, np.ndarray]:
    mapping = _mapping(value, name)
    ranges = [_finite_range(mapping.get(axis), f"{name}.{axis}") for axis in "xyz"]
    return (
        np.asarray([item[0] for item in ranges], dtype=np.float64),
        np.asarray([item[1] for item in ranges], dtype=np.float64),
    )


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class V57Curriculum:
    name: str
    release_minimum_base_m: np.ndarray
    release_maximum_base_m: np.ndarray
    target_minimum_base_m: np.ndarray
    target_maximum_base_m: np.ndarray
    target_minimum_offset_norm_m: float
    flight_time_range_s: tuple[float, float]


@dataclass(frozen=True)
class V57ThrownTaskContract:
    source_path: Path
    source_sha256: str
    teacher: str
    control_hz: float
    rh56_speed_set: int
    franka_q_home_rad: np.ndarray
    rh56_policy_q_home_rad: np.ndarray
    rh56_speed_register_order: np.ndarray
    rh56_open_register_order: np.ndarray
    catch_reference_center_base_m: np.ndarray
    object_half_extent_base_m: np.ndarray
    selected_curriculum: V57Curriculum
    curricula: Mapping[str, V57Curriculum]


@dataclass(frozen=True)
class V57CameraVisibility:
    curriculum: str
    target_center_vertices_inside: int
    target_center_vertex_count: int
    target_center_camera_z_min_m: float
    target_center_camera_z_max_m: float
    target_center_u_min_px: float
    target_center_u_max_px: float
    target_center_v_min_px: float
    target_center_v_max_px: float
    object_support_camera_z_min_m: float
    object_support_camera_z_max_m: float
    depth_range_m: tuple[float, float]

    @property
    def target_center_box_fully_visible(self) -> bool:
        return self.target_center_vertices_inside == self.target_center_vertex_count

    @property
    def object_support_depth_covered(self) -> bool:
        return bool(
            self.object_support_camera_z_min_m >= self.depth_range_m[0]
            and self.object_support_camera_z_max_m <= self.depth_range_m[1]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "curriculum": self.curriculum,
            "target_center_vertices_inside": self.target_center_vertices_inside,
            "target_center_vertex_count": self.target_center_vertex_count,
            "target_center_box_fully_visible": self.target_center_box_fully_visible,
            "target_center_camera_z_range_m": [
                self.target_center_camera_z_min_m,
                self.target_center_camera_z_max_m,
            ],
            "target_center_u_range_px": [
                self.target_center_u_min_px,
                self.target_center_u_max_px,
            ],
            "target_center_v_range_px": [
                self.target_center_v_min_px,
                self.target_center_v_max_px,
            ],
            "object_support_camera_z_range_m": [
                self.object_support_camera_z_min_m,
                self.object_support_camera_z_max_m,
            ],
            "depth_range_m": list(self.depth_range_m),
            "object_support_depth_covered": self.object_support_depth_covered,
        }


def load_v57_thrown_task_contract(
    path: Path,
    *,
    selected_curriculum: str,
    expected_sha256: Optional[str] = None,
) -> V57ThrownTaskContract:
    resolved = Path(path).expanduser().resolve(strict=True)
    actual_sha256 = _sha256(resolved)
    if expected_sha256 is not None and actual_sha256 != str(expected_sha256).lower():
        raise ThrownTaskContractError("V57 task contract bytes differ from SHA pin")
    try:
        document = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ThrownTaskContractError(f"cannot load V57 contract {resolved}: {exc}") from exc
    root = _mapping(document, "V57 task contract")
    task = _mapping(root.get("task"), "task")
    if str(task.get("family", "")) != "thrown_object_catch":
        raise ThrownTaskContractError("V57 task family is not thrown_object_catch")
    teacher = str(task.get("teacher", "")).strip()
    if not teacher:
        raise ThrownTaskContractError("V57 task teacher is empty")
    control_hz = float(task.get("control_hz", math.nan))
    if not math.isfinite(control_hz) or control_hz != 20.0:
        raise ThrownTaskContractError("V57 control_hz must be exactly 20")
    speed_set = int(task.get("rh56_speed_set", -1))
    if speed_set != 600:
        raise ThrownTaskContractError("V57 RH56 SPEED_SET must be exactly 600")

    reset = _mapping(root.get("robot_reset"), "robot_reset")
    franka_order = tuple(str(value) for value in reset.get("franka_joint_order", ()))
    if franka_order != tuple(f"panda_joint{index}" for index in range(1, 8)):
        raise ThrownTaskContractError("V57 Franka joint order is not panda_joint1..7")
    franka_q = _finite_vector(
        reset.get("franka_joint_position_rad"), 7, "Franka reset q"
    )
    velocity = _finite_vector(
        reset.get("joint_velocity_rad_s"), 7, "Franka reset dq"
    )
    if not np.array_equal(velocity, np.zeros(7, dtype=np.float64)):
        raise ThrownTaskContractError("V57 reset joint velocity is not exactly zero")
    if float(reset.get("reset_arm_position_noise_rad", math.nan)) != 0.0:
        raise ThrownTaskContractError("V57 reset arm noise must be disabled")
    if reset.get("canonical_reset_curriculum_enabled") is not False:
        raise ThrownTaskContractError("V57 canonical reset curriculum must be disabled")
    if tuple(str(value) for value in reset.get("rh56_policy_joint_order", ())) != (
        POLICY_HAND_ORDER
    ):
        raise ThrownTaskContractError("V57 RH56 policy joint order changed")
    hand_q = _finite_vector(
        reset.get("rh56_policy_joint_position_rad"), 6, "RH56 reset q"
    )
    if not np.array_equal(hand_q, np.zeros(6, dtype=np.float64)):
        raise ThrownTaskContractError("V57 RH56 policy reset is not fully open")
    if tuple(str(value) for value in reset.get("rh56_register_order", ())) != (
        REGISTER_HAND_ORDER
    ):
        raise ThrownTaskContractError("V57 RH56 register order changed")
    speed_registers = _finite_vector(
        reset.get("rh56_speed_set_register_order"), 6, "RH56 SPEED_SET registers"
    )
    if not np.array_equal(speed_registers, np.full(6, 600.0)):
        raise ThrownTaskContractError("V57 RH56 SPEED_SET registers differ from 600")
    open_registers = _finite_vector(
        reset.get("rh56_angle_set_open_register_order"),
        6,
        "RH56 open registers",
    )
    if not np.array_equal(open_registers, np.full(6, 1000.0)):
        raise ThrownTaskContractError("V57 RH56 open registers differ from 1000")

    curricula: dict[str, V57Curriculum] = {}
    for name in SUPPORTED_CURRICULA:
        value = _mapping(root.get(name), name)
        release_minimum, release_maximum = _axis_box(
            value.get("release_position_box_base_m"),
            f"{name}.release_position_box_base_m",
        )
        target_minimum, target_maximum = _axis_box(
            value.get("target_position_box_base_m"),
            f"{name}.target_position_box_base_m",
        )
        offset = _mapping(
            value.get("target_offset_from_catch_reference_m"),
            f"{name}.target_offset_from_catch_reference_m",
        )
        catch_reference = _finite_vector(
            root.get("catch_reference_center_base_m"),
            3,
            "catch_reference_center_base_m",
        )
        expected_minimum = catch_reference + np.asarray(
            [float(offset[axis][0]) for axis in "xyz"], dtype=np.float64
        )
        expected_maximum = catch_reference + np.asarray(
            [float(offset[axis][1]) for axis in "xyz"], dtype=np.float64
        )
        if not np.allclose(target_minimum, expected_minimum, atol=1.0e-12, rtol=0.0):
            raise ThrownTaskContractError(f"{name} target minimum differs from offset")
        if not np.allclose(target_maximum, expected_maximum, atol=1.0e-12, rtol=0.0):
            raise ThrownTaskContractError(f"{name} target maximum differs from offset")
        flight_key = (
            "flight_time_range_s"
            if name == "alpha_0_5"
            else "configured_flight_time_range_s"
        )
        curricula[name] = V57Curriculum(
            name=name,
            release_minimum_base_m=_readonly(release_minimum),
            release_maximum_base_m=_readonly(release_maximum),
            target_minimum_base_m=_readonly(target_minimum),
            target_maximum_base_m=_readonly(target_maximum),
            target_minimum_offset_norm_m=float(
                value.get("target_minimum_offset_norm_m")
            ),
            flight_time_range_s=_finite_range(value.get(flight_key), f"{name}.{flight_key}"),
        )
    if selected_curriculum not in curricula:
        raise ThrownTaskContractError(
            f"selected curriculum must be one of {', '.join(SUPPORTED_CURRICULA)}"
        )

    objects = _mapping(root.get("object_contract"), "object_contract")
    half_extents = []
    for name in ("sphere60", "triangle_sandbag80", "cube_sandbag50"):
        item = _mapping(objects.get(name), f"object_contract.{name}")
        size = _finite_vector(item.get("size_m"), 3, f"object_contract.{name}.size_m")
        if np.any(size <= 0.0):
            raise ThrownTaskContractError(f"object_contract.{name}.size_m is invalid")
        half_extents.append(size / 2.0)
    maximum_half_extent = np.max(np.stack(half_extents, axis=0), axis=0)

    return V57ThrownTaskContract(
        source_path=resolved,
        source_sha256=actual_sha256,
        teacher=teacher,
        control_hz=control_hz,
        rh56_speed_set=speed_set,
        franka_q_home_rad=_readonly(franka_q.astype(np.float32)),
        rh56_policy_q_home_rad=_readonly(hand_q.astype(np.float32)),
        rh56_speed_register_order=_readonly(speed_registers.astype(np.int32)),
        rh56_open_register_order=_readonly(open_registers.astype(np.int32)),
        catch_reference_center_base_m=_readonly(catch_reference),
        object_half_extent_base_m=_readonly(maximum_half_extent),
        selected_curriculum=curricula[selected_curriculum],
        curricula=dict(curricula),
    )


def resolve_v57_thrown_task_contract(
    resolved_config_path: Path,
) -> Optional[V57ThrownTaskContract]:
    """Resolve the V57 contract pinned by a materialized thrown task profile."""

    from sim2real.observation.camera_profile import load_resolved_task_config

    _config, profile = load_resolved_task_config(resolved_config_path)
    if not profile or str(profile.get("name", "")) not in (
        "thrown_object",
        "thrown_object_v60",
        "thrown_object_v61",
    ):
        return None
    path_value = str(profile.get("simulation_task_contract_file", "")).strip()
    expected_sha256 = str(
        profile.get("simulation_task_contract_sha256", "")
    ).strip().lower()
    curriculum = str(profile.get("simulation_curriculum", "")).strip()
    if not path_value or len(expected_sha256) != 64 or not curriculum:
        raise ThrownTaskContractError(
            "resolved thrown task profile does not pin its simulation contract"
        )
    return load_v57_thrown_task_contract(
        Path(path_value),
        selected_curriculum=curriculum,
        expected_sha256=expected_sha256,
    )


def _vertices(minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    return np.asarray(
        list(itertools.product(*zip(minimum.tolist(), maximum.tolist()))),
        dtype=np.float64,
    )


def assess_v57_camera_visibility(
    task: V57ThrownTaskContract,
    camera: V94Contract,
    *,
    curriculum: Optional[str] = None,
) -> V57CameraVisibility:
    name = task.selected_curriculum.name if curriculum is None else str(curriculum)
    if name not in task.curricula:
        raise ThrownTaskContractError(f"unknown V57 curriculum {name!r}")
    selected = task.curricula[name]
    transform = np.asarray(camera.T_base_camera_optical, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ThrownTaskContractError("camera T_base_camera is invalid")
    camera_from_base = np.linalg.inv(transform)
    K = np.asarray(camera.camera_K, dtype=np.float64)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ThrownTaskContractError("camera intrinsic matrix is invalid")

    def project(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        homogeneous = np.concatenate(
            [vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1
        )
        camera_points = (camera_from_base @ homogeneous.T).T[:, :3]
        z = camera_points[:, 2]
        if np.any(z <= 0.0):
            raise ThrownTaskContractError("V57 target box crosses behind the camera")
        u = K[0, 0] * camera_points[:, 0] / z + K[0, 2]
        v = K[1, 1] * camera_points[:, 1] / z + K[1, 2]
        return z, u, v

    center_z, center_u, center_v = project(
        _vertices(selected.target_minimum_base_m, selected.target_maximum_base_m)
    )
    inside = (
        (center_u >= 0.0)
        & (center_u < float(camera.camera_width))
        & (center_v >= 0.0)
        & (center_v < float(camera.camera_height))
    )
    support_z, _support_u, _support_v = project(
        _vertices(
            selected.target_minimum_base_m - task.object_half_extent_base_m,
            selected.target_maximum_base_m + task.object_half_extent_base_m,
        )
    )
    return V57CameraVisibility(
        curriculum=name,
        target_center_vertices_inside=int(np.count_nonzero(inside)),
        target_center_vertex_count=int(inside.size),
        target_center_camera_z_min_m=float(center_z.min()),
        target_center_camera_z_max_m=float(center_z.max()),
        target_center_u_min_px=float(center_u.min()),
        target_center_u_max_px=float(center_u.max()),
        target_center_v_min_px=float(center_v.min()),
        target_center_v_max_px=float(center_v.max()),
        object_support_camera_z_min_m=float(support_z.min()),
        object_support_camera_z_max_m=float(support_z.max()),
        depth_range_m=tuple(float(value) for value in camera.depth_range_m),
    )


def apply_v57_task_contract(
    camera_contract: V94Contract,
    resolved_config_path: Path,
) -> V94Contract:
    """Bind a pinned thrown reset and validate its selected camera volume.

    An explicitly non-executable V60 candidate may retain two recorded
    simulator/bundle mismatches for read-only diagnostics.  The sealed V60
    supervised rollout may reuse those bounds while every current policy
    point still has to pass the ordinary runtime depth gate.
    """

    task = resolve_v57_thrown_task_contract(resolved_config_path)
    if task is None:
        return camera_contract
    from sim2real.observation.camera_profile import load_resolved_task_config

    _resolved, profile = load_resolved_task_config(resolved_config_path)
    candidate_relaxations = profile.get("candidate_shadow_relaxations")
    first_motion_limits = profile.get("first_motion_execution_limits")
    rollout_limits = profile.get("rollout_execution_limits")
    diagnostic_candidate = bool(
        str(profile.get("name", "")) == "thrown_object_v60"
        and str(profile.get("commissioning_status", ""))
        == "candidate_shadow_only"
        and profile.get("robot_execution_enabled") is False
        and isinstance(candidate_relaxations, Mapping)
        and candidate_relaxations.get("robot_hardware_writes") is False
    )
    first_motion_candidate = bool(
        str(profile.get("name", "")) == "thrown_object_v60"
        and str(profile.get("commissioning_status", ""))
        == "first_motion_accepted"
        and profile.get("robot_execution_enabled") is True
        and profile.get("maximum_supervised_execute_steps") == 1
        and isinstance(first_motion_limits, Mapping)
        and first_motion_limits.get("maximum_commanded_policy_ticks") == 1
        and first_motion_limits.get(
            "require_current_policy_points_inside_processing_depth"
        )
        is True
        and first_motion_limits.get("robot_hardware_writes") is True
    )
    rollout_candidate = bool(
        str(profile.get("name", "")) == "thrown_object_v60"
        and str(profile.get("commissioning_status", "")) == "accepted"
        and profile.get("robot_execution_enabled") is True
        and profile.get("maximum_supervised_execute_steps") == 72
        and isinstance(rollout_limits, Mapping)
        and rollout_limits.get("maximum_commanded_policy_ticks") == 72
        and rollout_limits.get("simulator_episode_horizon_ticks") == 72
        and rollout_limits.get(
            "require_current_policy_points_inside_processing_depth"
        )
        is True
        and rollout_limits.get("robot_hardware_writes") is True
    )
    relaxation_contract = (
        rollout_limits
        if rollout_candidate
        else first_motion_limits
        if first_motion_candidate
        else candidate_relaxations
    )
    allow_joint_mismatch = bool(
        (diagnostic_candidate or first_motion_candidate or rollout_candidate)
        and relaxation_contract.get(
            "allow_q_home_outside_base_bundle_joint_limits"
        )
        is True
    )
    allow_depth_gap = bool(
        diagnostic_candidate
        and relaxation_contract.get(
            "allow_object_support_outside_processing_depth"
        )
        is True
        or (first_motion_candidate or rollout_candidate)
        and relaxation_contract.get(
            "allow_global_object_support_outside_processing_depth"
        )
        is True
    )
    effective_joint_limits = np.asarray(
        camera_contract.joint_limits_rad, dtype=np.float64
    )
    if allow_joint_mismatch:
        candidate_limits = np.asarray(
            relaxation_contract.get("fr3_system_joint_limits_rad"),
            dtype=np.float64,
        )
        raw_margin = relaxation_contract.get(
            "fr3_system_joint_limit_margin_rad"
        )
        if (
            candidate_limits.shape != (7, 2)
            or not np.all(np.isfinite(candidate_limits))
            or not np.all(candidate_limits[:, 0] < candidate_limits[:, 1])
            or isinstance(raw_margin, bool)
            or not isinstance(raw_margin, (int, float))
            or not math.isfinite(float(raw_margin))
            or float(raw_margin) <= 0.0
        ):
            raise ThrownTaskContractError(
                "V60 candidate FR3 joint-limit override is malformed"
            )
        effective_joint_limits = candidate_limits
    if not math.isclose(
        camera_contract.policy_rate_hz,
        task.control_hz,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ThrownTaskContractError(
            "runtime policy rate differs from V57 task control_hz"
        )
    q_home_outside_base_limits = bool(
        task.franka_q_home_rad.shape != (7,)
        or np.any(task.franka_q_home_rad < camera_contract.joint_limits_rad[:, 0])
        or np.any(task.franka_q_home_rad > camera_contract.joint_limits_rad[:, 1])
    )
    if q_home_outside_base_limits and not allow_joint_mismatch:
        raise ThrownTaskContractError("thrown Franka reset lies outside joint limits")
    if allow_joint_mismatch:
        margin = float(
            relaxation_contract["fr3_system_joint_limit_margin_rad"]
        )
        if np.any(task.franka_q_home_rad <= effective_joint_limits[:, 0] + margin) or np.any(
            task.franka_q_home_rad >= effective_joint_limits[:, 1] - margin
        ):
            raise ThrownTaskContractError(
                "V60 Franka reset violates its candidate FR3 system-limit margin"
            )
    visibility = assess_v57_camera_visibility(task, camera_contract)
    if not visibility.target_center_box_fully_visible:
        raise ThrownTaskContractError(
            f"selected {visibility.curriculum} target-center box is not fully visible: "
            f"{visibility.target_center_vertices_inside}/"
            f"{visibility.target_center_vertex_count} corners"
        )
    if not visibility.object_support_depth_covered and not allow_depth_gap:
        raise ThrownTaskContractError(
            "camera depth range does not cover the selected thrown target plus "
            "maximum object half-extent"
        )
    return replace(
        camera_contract,
        q_home_rad=np.asarray(task.franka_q_home_rad, dtype=np.float32).copy(),
        joint_limits_rad=np.asarray(effective_joint_limits, dtype=np.float64).copy(),
    )


def v57_task_summary(
    resolved_config_path: Path,
    camera_contract: Optional[V94Contract] = None,
) -> Optional[dict[str, object]]:
    task = resolve_v57_thrown_task_contract(resolved_config_path)
    if task is None:
        return None
    selected = task.selected_curriculum
    result: dict[str, object] = {
        "schema": "v57_real_task_contract_v1",
        "source": str(task.source_path),
        "source_sha256": task.source_sha256,
        "teacher": task.teacher,
        "selected_curriculum": selected.name,
        "control_hz": task.control_hz,
        "rh56_speed_set_register_order": task.rh56_speed_register_order.tolist(),
        "rh56_open_register_order": task.rh56_open_register_order.tolist(),
        "franka_q_home_rad": task.franka_q_home_rad.tolist(),
        "catch_reference_center_base_m": task.catch_reference_center_base_m.tolist(),
        "target_minimum_base_m": selected.target_minimum_base_m.tolist(),
        "target_maximum_base_m": selected.target_maximum_base_m.tolist(),
        "release_minimum_base_m": selected.release_minimum_base_m.tolist(),
        "release_maximum_base_m": selected.release_maximum_base_m.tolist(),
        "maximum_object_half_extent_base_m": task.object_half_extent_base_m.tolist(),
    }
    if camera_contract is not None:
        result["camera_visibility"] = assess_v57_camera_visibility(
            task, camera_contract
        ).as_dict()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument(
        "--recorder-lines",
        action="store_true",
        help="print curriculum, minimum, maximum and source SHA on four lines",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    task = resolve_v57_thrown_task_contract(args.resolved_config)
    if task is None:
        raise ThrownTaskContractError("resolved config is not a V57 thrown task")
    selected = task.selected_curriculum
    if args.recorder_lines:
        print(selected.name)
        print(" ".join(format(value, ".10g") for value in selected.target_minimum_base_m))
        print(" ".join(format(value, ".10g") for value in selected.target_maximum_base_m))
        print(task.source_sha256)
        return 0
    print(json.dumps(v57_task_summary(args.resolved_config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ThrownTaskContractError",
    "V57CameraVisibility",
    "V57Curriculum",
    "V57ThrownTaskContract",
    "apply_v57_task_contract",
    "assess_v57_camera_visibility",
    "load_v57_thrown_task_contract",
    "resolve_v57_thrown_task_contract",
    "v57_task_summary",
]
