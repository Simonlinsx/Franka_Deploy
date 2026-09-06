"""Resolve the camera-only part of a V94 task contract from one sealed config.

The policy/action/reset/robot contract remains owned by the verified deployment
bundle.  A task profile may replace only the fixed external RGB-D camera
identity, stream geometry, calibrated transform and depth range.  This keeps
tabletop and thrown-object deployment on one implementation without pretending
that two physical cameras share an intrinsic or extrinsic calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional

import numpy as np
import yaml

from sim2real.contracts.v94 import V94Contract


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DYNAMIC_PCD_ROOT = WORKSPACE_ROOT / "perception"


class RuntimeCameraProfileError(ValueError):
    """A task camera profile is missing, mutable, or internally inconsistent."""


@dataclass(frozen=True)
class TabletopReleaseHeightGateConfig:
    """Require a released object to reach the flat tabletop before rollout."""

    reference_center_base_z_m: float
    tolerance_m: float
    confirm_frames: int
    minimum_valid_points: int


@dataclass(frozen=True)
class ThrownRolloutTriggerConfig:
    """Task-owned camera trigger that starts a motion-synchronised rollout.

    The trigger is evaluated while only the camera owner is open.  It cannot
    command Franka or RH56.  The display fields affect console wording only;
    all admission thresholds remain sealed numeric task data.
    """

    mode: str
    maximum_wait_s: float
    poll_interval_s: float
    minimum_mask_area_px: int
    absent_arm_frames: int
    stable_arm_frames: int
    stable_max_centroid_speed_px_s: float
    trigger_min_centroid_speed_px_s: float
    trigger_min_displacement_px: float
    minimum_area_ratio: float
    maximum_area_ratio: float
    allow_absent_entry: bool = True
    display_name: str = "Throw trigger"
    ready_instruction: str = "throw now"
    tabletop_release_height_gate: Optional[
        TabletopReleaseHeightGateConfig
    ] = None


def _dynamic_imports() -> tuple[Any, Any, Any]:
    if str(DYNAMIC_PCD_ROOT) not in sys.path:
        sys.path.insert(0, str(DYNAMIC_PCD_ROOT))
    from dynamic_pcd.calibration.io import load_calibration, resolve_extrinsics
    from dynamic_pcd.config import load_config

    return load_config, load_calibration, resolve_extrinsics


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeCameraProfileError(f"{name} must be a mapping")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise RuntimeCameraProfileError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeCameraProfileError(
            f"{name} must be a positive integer"
        ) from exc
    if result <= 0 or str(result) != str(value).strip():
        raise RuntimeCameraProfileError(f"{name} must be a positive integer")
    return result


def _positive_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeCameraProfileError(f"{name} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeCameraProfileError(f"{name} must be finite and positive")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_resolved_task_config(path: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Load a materialized task config and return its immutable task metadata."""

    resolved = Path(path).expanduser().resolve()
    # Legacy callers and unit-test doubles may hand the camera path to this
    # resolver before the provider performs its own config admission.  Only a
    # real YAML mapping that explicitly carries task_profile opts into the
    # override contract; everything else remains the sealed bundle camera.
    if not resolved.is_file():
        return {}, {}
    try:
        raw_document = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeCameraProfileError(
            f"cannot inspect task camera profile {resolved}: {exc}"
        ) from exc
    if not isinstance(raw_document, Mapping) or "task_profile" not in raw_document:
        return {}, {}
    load_config, _load_calibration, _resolve_extrinsics = _dynamic_imports()
    config = load_config(str(resolved))
    raw_profile = config.get("task_profile")
    if raw_profile is None:
        return config, {}
    profile = _mapping(raw_profile, "task_profile")
    if profile.get("resolved") is not True:
        raise RuntimeCameraProfileError(
            "task_profile must be materialized by sim2real.tasks.launcher"
        )
    name = str(profile.get("name", "")).strip()
    if name not in (
        "tabletop",
        "thrown_object",
        "thrown_object_v60",
        "thrown_object_v61",
    ):
        raise RuntimeCameraProfileError(f"unsupported task_profile.name={name!r}")
    config_sha256 = _sha256(resolved)
    expected_filename = f"{name}-{config_sha256[:16]}.yaml"
    if resolved.name != expected_filename:
        raise RuntimeCameraProfileError(
            "resolved task config filename does not match its content hash"
        )
    for label, path_key, digest_key in (
        (
            "source task config",
            "source_task_config",
            "source_task_config_sha256",
        ),
        ("base config", "base_config", "base_config_sha256"),
    ):
        source_path = Path(str(profile.get(path_key, ""))).expanduser().resolve()
        expected_sha256 = str(profile.get(digest_key, "")).strip().lower()
        if (
            not source_path.is_file()
            or len(expected_sha256) != 64
            or _sha256(source_path) != expected_sha256
        ):
            raise RuntimeCameraProfileError(
                f"resolved task profile {label} provenance changed"
            )
    return config, profile


def task_profile_allows_robot_execution(path: Path) -> bool:
    """Return the explicit commissioning latch for a resolved task profile."""

    _config, profile = load_resolved_task_config(path)
    if not profile:
        # Preserve the historical commissioned tabletop CLI when no task
        # wrapper is used.
        return True
    if profile.get("robot_execution_enabled") is not True:
        return False
    status = str(profile.get("commissioning_status", "")).strip()
    if status == "accepted":
        return True
    limits = profile.get("first_motion_execution_limits")
    v60_first_motion = bool(
        profile.get("name") == "thrown_object_v60"
        and status == "first_motion_accepted"
        and profile.get("maximum_supervised_execute_steps") == 1
        and isinstance(limits, Mapping)
        and limits.get("maximum_commanded_policy_ticks") == 1
        and limits.get("require_current_policy_points_inside_processing_depth")
        is True
        and limits.get("robot_hardware_writes") is True
    )
    limits = profile.get("supervised_execution_limits")
    v61_forty_tick = bool(
        profile.get("name") == "thrown_object_v61"
        and status == "supervised_40_tick_authorized"
        and profile.get("maximum_supervised_execute_steps") == 40
        and isinstance(limits, Mapping)
        and limits.get("maximum_commanded_policy_ticks") == 40
        and limits.get("require_current_policy_points_inside_processing_depth")
        is True
        and limits.get("robot_hardware_writes") is True
    )
    return v60_first_motion or v61_forty_tick


def task_profile_maximum_supervised_execute_steps(path: Path) -> Optional[int]:
    """Return a task-owned hard execution cap, if the task declares one."""

    _config, profile = load_resolved_task_config(path)
    if not profile:
        return None
    value = profile.get("maximum_supervised_execute_steps")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeCameraProfileError(
            "task_profile.maximum_supervised_execute_steps must be a positive integer"
        )
    return int(value)


def task_profile_policy_rgbd_resolution(path: Path) -> Optional[str]:
    _config, profile = load_resolved_task_config(path)
    if not profile:
        return None
    value = str(profile.get("policy_rgbd_resolution", "")).strip().lower()
    if value != "424x240":
        raise RuntimeCameraProfileError(
            "both commissioned task profiles require policy_rgbd_resolution=424x240"
        )
    return value


def task_profile_policy_rate_hz(path: Path) -> Optional[float]:
    """Return a task-owned policy rate when the resolved profile defines one."""

    _config, profile = load_resolved_task_config(path)
    if not profile or str(profile.get("name", "")) not in (
        "thrown_object",
        "thrown_object_v60",
        "thrown_object_v61",
    ):
        return None
    from sim2real.tasks.thrown_contract import resolve_v57_thrown_task_contract

    task = resolve_v57_thrown_task_contract(path)
    if task is None:
        raise RuntimeCameraProfileError("thrown task has no V57 contract")
    return float(task.control_hz)


def task_profile_rollout_trigger(
    path: Path,
) -> Optional[ThrownRolloutTriggerConfig]:
    """Return a task-sealed camera trigger, if the selected task owns one."""

    _config, profile = load_resolved_task_config(path)
    if not profile:
        return None
    profile_name = str(profile.get("name", ""))
    if profile_name not in (
        "tabletop",
        "thrown_object",
        "thrown_object_v60",
        "thrown_object_v61",
    ):
        return None
    raw = _mapping(profile.get("rollout_trigger"), "task_profile.rollout_trigger")
    if raw.get("enabled") is not True:
        raise RuntimeCameraProfileError(
            "task rollout_trigger.enabled must be true"
        )
    mode = str(raw.get("mode", "")).strip()
    allowed_modes = (
        {"object_motion_only", "object_motion_then_tabletop_height"}
        if profile_name == "tabletop"
        else {"object_motion_or_entry"}
    )
    if mode not in allowed_modes:
        raise RuntimeCameraProfileError(
            f"{profile_name} rollout trigger mode must be one of "
            f"{sorted(allowed_modes)}"
        )
    minimum_area_ratio = _positive_float(
        raw.get("minimum_area_ratio"),
        "rollout_trigger.minimum_area_ratio",
    )
    maximum_area_ratio = _positive_float(
        raw.get("maximum_area_ratio"),
        "rollout_trigger.maximum_area_ratio",
    )
    if minimum_area_ratio >= 1.0 or maximum_area_ratio <= 1.0:
        raise RuntimeCameraProfileError(
            "rollout trigger area-ratio bounds must straddle one"
        )
    stable_speed = _positive_float(
        raw.get("stable_max_centroid_speed_px_s"),
        "rollout_trigger.stable_max_centroid_speed_px_s",
    )
    trigger_speed = _positive_float(
        raw.get("trigger_min_centroid_speed_px_s"),
        "rollout_trigger.trigger_min_centroid_speed_px_s",
    )
    if stable_speed >= trigger_speed:
        raise RuntimeCameraProfileError(
            "rollout trigger motion speed must exceed the stable-arm speed"
        )
    release_height_gate = None
    raw_release_gate = raw.get("tabletop_release_height_gate")
    if profile_name == "tabletop":
        if mode == "object_motion_only":
            if raw_release_gate is not None:
                raise RuntimeCameraProfileError(
                    "object_motion_only must not configure a release-height gate"
                )
        else:
            release_gate = _mapping(
                raw_release_gate,
                "rollout_trigger.tabletop_release_height_gate",
            )
            if release_gate.get("enabled") is not True:
                raise RuntimeCameraProfileError(
                    "tabletop release-height gate must be explicitly enabled"
                )
            reference_z = float(release_gate.get("reference_center_base_z_m"))
            if not math.isfinite(reference_z) or not -0.25 <= reference_z <= 0.50:
                raise RuntimeCameraProfileError(
                    "tabletop release reference Z must be finite and in -0.25..0.50 m"
                )
            tolerance = _positive_float(
                release_gate.get("tolerance_m"),
                "rollout_trigger.tabletop_release_height_gate.tolerance_m",
            )
            if tolerance > 0.03:
                raise RuntimeCameraProfileError(
                    "tabletop release-height tolerance cannot exceed 0.03 m"
                )
            confirm_frames = _positive_int(
                release_gate.get("confirm_frames"),
                "rollout_trigger.tabletop_release_height_gate.confirm_frames",
            )
            if not 2 <= confirm_frames <= 10:
                raise RuntimeCameraProfileError(
                    "tabletop release-height confirmation must use 2..10 frames"
                )
            minimum_valid_points = _positive_int(
                release_gate.get("minimum_valid_points"),
                "rollout_trigger.tabletop_release_height_gate.minimum_valid_points",
            )
            if not 16 <= minimum_valid_points <= 128:
                raise RuntimeCameraProfileError(
                    "tabletop release-height gate requires 16..128 valid points"
                )
            release_height_gate = TabletopReleaseHeightGateConfig(
                reference_center_base_z_m=reference_z,
                tolerance_m=tolerance,
                confirm_frames=confirm_frames,
                minimum_valid_points=minimum_valid_points,
            )
    elif raw_release_gate is not None:
        raise RuntimeCameraProfileError(
            "tabletop_release_height_gate is valid only for the tabletop task"
        )

    return ThrownRolloutTriggerConfig(
        mode=mode,
        maximum_wait_s=_positive_float(
            raw.get("maximum_wait_s"), "rollout_trigger.maximum_wait_s"
        ),
        poll_interval_s=_positive_float(
            raw.get("poll_interval_s"), "rollout_trigger.poll_interval_s"
        ),
        minimum_mask_area_px=_positive_int(
            raw.get("minimum_mask_area_px"),
            "rollout_trigger.minimum_mask_area_px",
        ),
        absent_arm_frames=_positive_int(
            raw.get("absent_arm_frames"), "rollout_trigger.absent_arm_frames"
        ),
        stable_arm_frames=_positive_int(
            raw.get("stable_arm_frames"), "rollout_trigger.stable_arm_frames"
        ),
        stable_max_centroid_speed_px_s=stable_speed,
        trigger_min_centroid_speed_px_s=trigger_speed,
        trigger_min_displacement_px=_positive_float(
            raw.get("trigger_min_displacement_px"),
            "rollout_trigger.trigger_min_displacement_px",
        ),
        minimum_area_ratio=minimum_area_ratio,
        maximum_area_ratio=maximum_area_ratio,
        allow_absent_entry=mode == "object_motion_or_entry",
        display_name=(
            "Object motion trigger"
            if profile_name == "tabletop"
            else "Throw trigger"
        ),
        ready_instruction=(
            "release the object down the ramp now"
            if profile_name == "tabletop"
            else "throw now"
        ),
        tabletop_release_height_gate=release_height_gate,
    )


def resolve_runtime_camera_contract(
    bundle_contract: V94Contract,
    pcd_config_path: Path,
) -> V94Contract:
    """Apply an explicit task camera override to an otherwise sealed V94 contract."""

    config, profile = load_resolved_task_config(pcd_config_path)
    if not profile or profile.get("camera_contract_override") is not True:
        return bundle_contract

    camera = _mapping(config.get("camera"), "camera")
    extrinsics_cfg = _mapping(config.get("extrinsics"), "extrinsics")
    serial = str(camera.get("serial", "")).strip()
    if not serial:
        raise RuntimeCameraProfileError("task camera.serial is required")
    width = _positive_int(camera.get("width"), "camera.width")
    height = _positive_int(camera.get("height"), "camera.height")
    fps = _positive_float(camera.get("fps"), "camera.fps")
    z_min = _positive_float(camera.get("z_min"), "camera.z_min")
    z_max = _positive_float(camera.get("z_max"), "camera.z_max")
    if z_min >= z_max:
        raise RuntimeCameraProfileError("camera.z_min must be less than camera.z_max")
    if extrinsics_cfg.get("require_calibration") is not True:
        raise RuntimeCameraProfileError("task camera override requires calibration")
    if extrinsics_cfg.get("require_quality_pass") is not True:
        raise RuntimeCameraProfileError("task camera override requires quality pass")
    if extrinsics_cfg.get("strict_camera_serial") is not True:
        raise RuntimeCameraProfileError("task camera override requires strict serial")

    calibration_value = str(extrinsics_cfg.get("calibration_file", "")).strip()
    if not calibration_value:
        raise RuntimeCameraProfileError("task camera override has no calibration_file")
    calibration_path = Path(calibration_value).expanduser().resolve()
    if not calibration_path.is_file():
        raise RuntimeCameraProfileError(
            f"task calibration is missing: {calibration_path}"
        )
    expected_calibration_sha256 = str(
        profile.get("calibration_sha256", "")
    ).strip().lower()
    actual_calibration_sha256 = _sha256(calibration_path)
    if (
        len(expected_calibration_sha256) != 64
        or expected_calibration_sha256 != actual_calibration_sha256
    ):
        raise RuntimeCameraProfileError(
            "task calibration bytes differ from task_profile.calibration_sha256"
        )

    _load_config, load_calibration, resolve_extrinsics = _dynamic_imports()
    calibration = load_calibration(str(calibration_path))
    resolved_extrinsics = resolve_extrinsics(extrinsics_cfg)
    calibration_camera = _mapping(calibration.get("camera"), "calibration.camera")
    calibration_serial = str(calibration_camera.get("serial", "")).strip()
    if calibration_serial != serial or resolved_extrinsics.camera_serial != serial:
        raise RuntimeCameraProfileError(
            "task camera serial differs from its calibration serial"
        )
    if resolved_extrinsics.quality_status != "pass":
        raise RuntimeCameraProfileError("task calibration quality is not pass")
    if resolved_extrinsics.base_frame != "robot_base":
        raise RuntimeCameraProfileError("task calibration base frame is not robot_base")
    if resolved_extrinsics.camera_frame != "camera_color_optical_frame":
        raise RuntimeCameraProfileError(
            "task calibration camera frame is not camera_color_optical_frame"
        )

    intrinsics = _mapping(
        calibration_camera.get("intrinsics"), "calibration.camera.intrinsics"
    )
    calibration_width = _positive_int(intrinsics.get("width"), "intrinsics.width")
    calibration_height = _positive_int(
        intrinsics.get("height"), "intrinsics.height"
    )
    if calibration_width % width != 0 or calibration_height % height != 0:
        raise RuntimeCameraProfileError(
            "runtime image dimensions are not an integer calibration decimation"
        )
    stride_x = calibration_width // width
    stride_y = calibration_height // height
    if stride_x != stride_y or stride_x not in (1, 2):
        raise RuntimeCameraProfileError(
            "runtime camera profile supports only calibration-native or exact 2x size"
        )
    stride = float(stride_x)
    K = np.array(
        [
            [
                _positive_float(intrinsics.get("fx"), "intrinsics.fx") / stride,
                0.0,
                float(intrinsics.get("ppx")) / stride,
            ],
            [
                0.0,
                _positive_float(intrinsics.get("fy"), "intrinsics.fy") / stride,
                float(intrinsics.get("ppy")) / stride,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(K)):
        raise RuntimeCameraProfileError("task camera intrinsics are not finite")
    distortion = np.asarray(intrinsics.get("distortion", ()), dtype=np.float64)
    if distortion.shape != (5,) or not np.allclose(distortion, 0.0, atol=1.0e-12):
        raise RuntimeCameraProfileError(
            "V94 task camera requires zero reported aligned-color distortion"
        )
    depth_scale = _positive_float(
        calibration_camera.get("depth_scale"), "calibration.camera.depth_scale"
    )
    calibration_id = str(resolved_extrinsics.calibration_id or "").strip()
    if not calibration_id:
        raise RuntimeCameraProfileError("task calibration has no calibration_id")

    return replace(
        bundle_contract,
        camera_rate_hz=fps,
        camera_width=width,
        camera_height=height,
        camera_serial=serial,
        calibration_id=calibration_id,
        camera_K=K,
        depth_scale_m_per_unit=depth_scale,
        depth_range_m=(z_min, z_max),
        T_base_camera_optical=np.asarray(
            resolved_extrinsics.T_base_camera, dtype=np.float64
        ).copy(),
    )


def resolve_runtime_task_contract(
    bundle_contract: V94Contract,
    pcd_config_path: Path,
) -> V94Contract:
    """Apply camera geometry and any separately pinned robot/task reset."""

    camera_contract = resolve_runtime_camera_contract(
        bundle_contract, pcd_config_path
    )
    from sim2real.tasks.thrown_contract import apply_v57_task_contract

    return apply_v57_task_contract(camera_contract, pcd_config_path)


__all__ = [
    "RuntimeCameraProfileError",
    "TabletopReleaseHeightGateConfig",
    "ThrownRolloutTriggerConfig",
    "load_resolved_task_config",
    "resolve_runtime_camera_contract",
    "resolve_runtime_task_contract",
    "task_profile_allows_robot_execution",
    "task_profile_maximum_supervised_execute_steps",
    "task_profile_policy_rate_hz",
    "task_profile_policy_rgbd_resolution",
    "task_profile_rollout_trigger",
]
