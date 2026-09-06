"""Load the small sim2real config together with its commissioning profile."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np
import yaml

from .contracts import ObservationSpec

DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config.json")


def _load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON number is forbidden: {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON config {path} must contain an object")
    return payload


def _resolve_relative(source: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = source.parent / candidate
    return candidate.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load YAML config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config {path} must contain a mapping")
    return payload


def load_runtime_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> Tuple[dict[str, Any], dict[str, Any], Path]:
    """Return runtime config, commissioning profile, and resolved config path."""

    source = Path(path).expanduser().resolve()
    runtime = _load_json(source)
    if runtime.get("schema_version") != 1:
        raise ValueError("sim2real config schema_version must be 1")
    profile_value = runtime.get("commissioning_profile")
    if not isinstance(profile_value, str) or not profile_value.strip():
        raise ValueError("commissioning_profile must be a non-empty path")
    profile = _load_json(_resolve_relative(source, profile_value))
    if profile.get("schema_version") != 1:
        raise ValueError("commissioning profile schema_version must be 1")
    object_cfg = runtime.get("object_pcd")
    if not isinstance(object_cfg, dict):
        raise ValueError("object_pcd must be an object")
    calibration_file = object_cfg.get("calibration_file")
    if not isinstance(calibration_file, str) or not calibration_file.strip():
        raise ValueError("object_pcd.calibration_file must be a non-empty path")
    object_cfg["calibration_file"] = str(_resolve_relative(source, calibration_file))
    return runtime, profile, source


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def observation_spec_from_config(
    runtime: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    require_franka: bool | None = None,
    require_inspire: bool | None = None,
    require_object_pcd: bool | None = None,
) -> ObservationSpec:
    franka_runtime = _mapping(runtime.get("franka"), "franka")
    inspire_runtime = _mapping(runtime.get("inspire"), "inspire")
    object_cfg = _mapping(runtime.get("object_pcd"), "object_pcd")
    observation_cfg = _mapping(runtime.get("observation"), "observation")
    franka_profile = _mapping(profile.get("franka"), "profile.franka")
    end_effector = _mapping(
        franka_profile.get("expected_end_effector"),
        "profile.franka.expected_end_effector",
    )
    calibration = _mapping(profile.get("calibration"), "profile.calibration")
    perception_calibration = _load_yaml(Path(str(object_cfg["calibration_file"])))
    if perception_calibration.get("schema_version") != 1:
        raise ValueError("perception calibration schema_version must be 1")
    if perception_calibration.get("kind") != "camera_robot_extrinsic_calibration":
        raise ValueError("object_pcd.calibration_file has the wrong kind")
    perception_id = str(perception_calibration.get("calibration_id", ""))
    perception_camera = _mapping(
        perception_calibration.get("camera"), "perception calibration.camera"
    )
    perception_serial = str(perception_camera.get("serial", ""))
    if perception_id != str(calibration["id"]):
        raise ValueError(
            "commissioning and perception calibration IDs disagree: "
            f"{calibration['id']!r}!={perception_id!r}"
        )
    if perception_serial != str(calibration["camera_serial"]):
        raise ValueError(
            "commissioning and perception camera serials disagree: "
            f"{calibration['camera_serial']!r}!={perception_serial!r}"
        )

    limits = np.asarray(franka_profile.get("joint_limits_rad"), dtype=np.float64)
    return ObservationSpec(
        require_franka=(
            bool(franka_runtime.get("enabled", True))
            if require_franka is None
            else bool(require_franka)
        ),
        require_inspire=(
            bool(inspire_runtime.get("enabled", True))
            if require_inspire is None
            else bool(require_inspire)
        ),
        require_object_pcd=(
            bool(object_cfg.get("enabled", True))
            if require_object_pcd is None
            else bool(require_object_pcd)
        ),
        object_history_shape=tuple(int(value) for value in object_cfg["history_shape"]),
        object_current_shape=tuple(int(value) for value in object_cfg["current_shape"]),
        object_reference_frame=str(object_cfg["reference_frame"]),
        object_point_frame=str(object_cfg["point_frame"]),
        calibration_id=perception_id,
        camera_serial=perception_serial,
        expected_T_base_camera=np.asarray(
            perception_calibration.get("T_base_camera"), dtype=np.float64
        ),
        expected_F_T_EE=np.asarray(
            franka_profile.get("expected_F_T_EE"), dtype=np.float64
        ),
        expected_m_ee_kg=float(end_effector["mass_kg"]),
        expected_F_x_Cee_m=np.asarray(end_effector["F_x_Cee_m"], dtype=np.float64),
        expected_I_ee_kg_m2=np.asarray(end_effector["inertia_kg_m2"], dtype=np.float64),
        max_object_age_s=float(object_cfg["max_age_s"]),
        min_raw_points=int(object_cfg["min_raw_points"]),
        max_center_reference_error_m=float(object_cfg["max_center_reference_error_m"]),
        max_future_skew_s=float(observation_cfg["max_future_skew_s"]),
        max_capture_span_s=float(observation_cfg["max_capture_span_s"]),
        max_hand_temperature_c=int(observation_cfg["max_hand_temperature_c"]),
        minimum_control_success_rate=float(
            observation_cfg["minimum_control_success_rate"]
        ),
        joint_limits_rad=limits,
        joint_limit_margin_rad=float(franka_profile["joint_limit_margin_rad"]),
    )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "load_runtime_config",
    "observation_spec_from_config",
]
