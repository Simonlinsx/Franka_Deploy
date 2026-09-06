"""Fail-closed configuration checks for FR3 + Inspire grasp commissioning.

This module is deliberately hardware-free.  It validates the values that must
be known before a process is even allowed to create a Franka or serial client.
Unknown measurements remain ``null`` in the checked-in commissioning profile;
they are reported as blockers instead of being replaced with guessed values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .air_audit_readiness import (
    SUPERVISED_RANGE_ACCEPTANCE,
    validate_air_candidate_commissioning,
    validate_air_target_acceptance_policy,
)
from .control_plan import is_official_snapshot, select_eligible_candidate_index
from .snapshot import (
    VisualizationSnapshot,
    has_complete_official_provenance,
    validate_snapshot,
)


PathLike = Union[str, Path]


@dataclass(frozen=True)
class ControlReadiness:
    """Independent readiness results for default-pose and full-grasp motion."""

    default_motion_blockers: Tuple[str, ...]
    full_grasp_blockers: Tuple[str, ...]

    @property
    def default_motion_ready(self) -> bool:
        return not self.default_motion_blockers

    @property
    def full_grasp_ready(self) -> bool:
        return not self.full_grasp_blockers


@dataclass(frozen=True)
class VerifiedAdapterAssets:
    mesh_path: Path
    provenance_path: Path
    mesh_sha256: str


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} finite numbers") from exc
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return array


def _positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def validate_rigid_transform(value: Any, name: str) -> np.ndarray:
    """Return one strict ``T_A_B`` matrix without repairing bad rotations."""

    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 4x4 transform") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix.copy()


def load_control_config(path: PathLike) -> tuple[dict[str, Any], Path]:
    """Load and validate a JSON commissioning profile."""

    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load control config {source}: {exc}") from exc
    validate_control_config(payload)
    return payload, source


def verify_adapter_assets(
    config: Mapping[str, Any], config_path: PathLike
) -> VerifiedAdapterAssets:
    """Resolve the vendored STL and require its recorded SHA-256 provenance."""

    validate_control_config(config)
    source = Path(config_path).expanduser().resolve()
    tool = _mapping(config["tool"], "tool")
    mesh_path = (source.parent / str(tool.get("adapter_asset", ""))).resolve()
    provenance_path = (
        source.parent / str(tool.get("adapter_provenance", ""))
    ).resolve()
    if not mesh_path.is_file():
        raise ValueError(f"adapter mesh does not exist: {mesh_path}")
    if not provenance_path.is_file():
        raise ValueError(f"adapter provenance does not exist: {provenance_path}")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load adapter provenance: {exc}") from exc
    expected = str(provenance.get("asset_sha256", "")).lower()
    if len(expected) != 64:
        raise ValueError("adapter provenance has no valid asset_sha256")
    digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(
            f"adapter mesh SHA-256 mismatch: expected {expected}, got {digest}"
        )
    geometry = _mapping(provenance.get("nominal_geometry_m"), "provenance.nominal_geometry_m")
    checks = {
        "disk_diameter": "adapter_disk_diameter_m",
        "disk_thickness": "adapter_disk_thickness_m",
        "spigot_diameter": "adapter_spigot_diameter_m",
        "spigot_protrusion": "adapter_spigot_protrusion_m",
        "total_axial_envelope": "adapter_total_axial_envelope_m",
        "fr3_face_to_rh56_seating_plane": "fr3_face_to_rh56_seating_plane_m",
    }
    for provenance_key, config_key in checks.items():
        if not np.isclose(
            float(geometry.get(provenance_key)),
            float(tool[config_key]),
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError(
                f"adapter config {config_key} disagrees with checked provenance"
            )
    return VerifiedAdapterAssets(mesh_path, provenance_path, digest)


def validate_control_config(config: Mapping[str, Any]) -> None:
    """Validate types, ranges, and adapter geometry invariants."""

    root = _mapping(config, "config")
    if isinstance(root.get("schema_version"), bool) or root.get("schema_version") != 1:
        raise ValueError("control config schema_version must be 1")
    if root.get("reference_frame") != "robot_base":
        raise ValueError("control reference_frame must be robot_base")

    calibration = _mapping(root.get("calibration"), "calibration")
    if not str(calibration.get("id", "")).strip():
        raise ValueError("calibration.id must be non-empty")
    if not str(calibration.get("camera_serial", "")).strip():
        raise ValueError("calibration.camera_serial must be non-empty")

    franka = _mapping(root.get("franka"), "franka")
    q_default = _finite_vector(franka.get("default_q_rad"), 7, "franka.default_q_rad")
    limits = np.asarray(franka.get("joint_limits_rad"), dtype=np.float64)
    if limits.shape != (7, 2) or not np.all(np.isfinite(limits)):
        raise ValueError("franka.joint_limits_rad must be a finite 7x2 array")
    if not np.all(limits[:, 0] < limits[:, 1]):
        raise ValueError("each Franka lower joint limit must be below its upper limit")
    margin = _positive(franka.get("joint_limit_margin_rad"), "franka.joint_limit_margin_rad")
    if np.any(q_default <= limits[:, 0] + margin) or np.any(
        q_default >= limits[:, 1] - margin
    ):
        raise ValueError("franka.default_q_rad violates configured joint-limit margins")
    default_provenance = franka.get("default_q_provenance")
    if default_provenance is not None:
        provenance = _mapping(
            default_provenance, "franka.default_q_provenance"
        )
        for key in (
            "source_kind",
            "recorded_date",
            "installed_configuration",
            "reason",
            "verification_scope",
        ):
            if not str(provenance.get(key, "")).strip():
                raise ValueError(
                    f"franka.default_q_provenance.{key} must be non-empty"
                )
        if provenance.get("evidence_artifact") is not None:
            raise ValueError(
                "franka.default_q_provenance.evidence_artifact must remain null "
                "until a bound evidence file exists"
            )
        if _boolean(
            provenance.get("motion_authorized"),
            "franka.default_q_provenance.motion_authorized",
        ) is not False:
            raise ValueError(
                "franka.default_q_provenance.motion_authorized must be false"
            )
    for key in (
        "default_max_joint_velocity_rad_s",
        "default_max_joint_segment_rad",
        "default_min_duration_s",
        "default_arrival_tolerance_rad",
        "cartesian_max_segment_translation_m",
        "cartesian_max_segment_rotation_rad",
        "cartesian_max_translation_velocity_m_s",
        "cartesian_max_rotation_velocity_rad_s",
        "cartesian_min_segment_duration_s",
        "cartesian_arrival_position_tolerance_m",
        "cartesian_arrival_rotation_tolerance_rad",
        "settle_time_s",
        "settle_timeout_s",
        "settle_poll_s",
    ):
        _positive(franka.get(key), f"franka.{key}")

    workspace_min = franka.get("cartesian_workspace_min_m")
    workspace_max = franka.get("cartesian_workspace_max_m")
    if (workspace_min is None) != (workspace_max is None):
        raise ValueError("both Cartesian workspace bounds must be set or both must be null")
    if workspace_min is not None:
        lower = _finite_vector(workspace_min, 3, "franka.cartesian_workspace_min_m")
        upper = _finite_vector(workspace_max, 3, "franka.cartesian_workspace_max_m")
        if not np.all(lower < upper):
            raise ValueError("Cartesian workspace minimum must be below maximum")

    for key in ("expected_F_T_EE",):
        if franka.get(key) is not None:
            validate_rigid_transform(franka[key], f"franka.{key}")
    _boolean(
        franka.get("default_path_collision_verified"),
        "franka.default_path_collision_verified",
    )

    end_effector = _mapping(
        franka.get("expected_end_effector"), "franka.expected_end_effector"
    )
    dynamics = tuple(
        end_effector.get(key)
        for key in ("mass_kg", "F_x_Cee_m", "inertia_kg_m2")
    )
    if any(value is not None for value in dynamics) and not all(
        value is not None for value in dynamics
    ):
        raise ValueError(
            "franka.expected_end_effector mass/CoM/inertia must be set together"
        )
    if dynamics[0] is not None:
        _positive(dynamics[0], "franka.expected_end_effector.mass_kg")
        _finite_vector(
            dynamics[1], 3, "franka.expected_end_effector.F_x_Cee_m"
        )
        inertia = np.asarray(dynamics[2], dtype=np.float64)
        if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
            raise ValueError(
                "franka.expected_end_effector.inertia_kg_m2 must be a finite 3x3 matrix"
            )
        if not np.allclose(inertia, inertia.T, atol=1e-10, rtol=0.0):
            raise ValueError(
                "franka.expected_end_effector.inertia_kg_m2 must be symmetric"
            )
        if float(np.min(np.linalg.eigvalsh(inertia))) < -1e-10:
            raise ValueError(
                "franka.expected_end_effector.inertia_kg_m2 must be positive semidefinite"
            )

    inspire = _mapping(root.get("inspire"), "inspire")
    open_targets = _finite_vector(inspire.get("open_targets"), 6, "inspire.open_targets")
    if not np.all(open_targets == np.round(open_targets)):
        raise ValueError("inspire.open_targets must be integer register values")
    if tuple(open_targets.astype(int)) != (1000,) * 6:
        raise ValueError("commissioned six-axis open target must be [1000]*6")
    for key in (
        "open_speed",
        "close_speed",
        "force_limit_g",
        "motion_timeout_s",
        "arrival_tolerance_units",
    ):
        _positive(inspire.get(key), f"inspire.{key}")
    thumb_range = _finite_vector(
        inspire.get("thumb_rotate_validated_realtime_range"),
        2,
        "inspire.thumb_rotate_validated_realtime_range",
    )
    if not 0 <= thumb_range[0] < thumb_range[1] <= 1000:
        raise ValueError("thumb rotate validated range must lie within 0..1000")
    if _boolean(
        inspire.get("thumb_preshape_required"),
        "inspire.thumb_preshape_required",
    ) is not True:
        raise ValueError(
            "inspire.thumb_preshape_required must be true for six-axis execution"
        )
    six_axis_coupled = _boolean(
        inspire.get("six_axis_coupled_closure_commissioned"),
        "inspire.six_axis_coupled_closure_commissioned",
    )
    acceptance_mode, _supervised_ranges = validate_air_target_acceptance_policy(
        inspire
    )
    if acceptance_mode == SUPERVISED_RANGE_ACCEPTANCE and not six_axis_coupled:
        raise ValueError(
            "supervised air-closure ranges require six-axis coupled closure"
        )
    commissioned_targets = inspire.get("commissioned_air_closure_targets")
    if not isinstance(commissioned_targets, list):
        raise ValueError(
            "inspire.commissioned_air_closure_targets must be a list of exact six-axis vectors"
        )
    normalized_targets = []
    for index, target in enumerate(commissioned_targets):
        values = _finite_vector(
            target, 6, f"inspire.commissioned_air_closure_targets[{index}]"
        )
        if (
            np.any(values < 0)
            or np.any(values > 1000)
            or not np.array_equal(values, np.rint(values))
        ):
            raise ValueError(
                "commissioned air-closure targets must contain six integer registers in 0..1000"
            )
        exact = tuple(int(value) for value in values)
        if exact in normalized_targets:
            raise ValueError("commissioned air-closure targets must be unique")
        normalized_targets.append(exact)
    if (
        inspire["six_axis_coupled_closure_commissioned"]
        and not normalized_targets
    ):
        raise ValueError(
            "six_axis_coupled_closure_commissioned=true requires at least one exact commissioned air-closure target"
        )

    tool = _mapping(root.get("tool"), "tool")
    disk = _positive(tool.get("adapter_disk_thickness_m"), "tool.adapter_disk_thickness_m")
    spigot = _positive(
        tool.get("adapter_spigot_protrusion_m"),
        "tool.adapter_spigot_protrusion_m",
    )
    total = _positive(
        tool.get("adapter_total_axial_envelope_m"),
        "tool.adapter_total_axial_envelope_m",
    )
    mount = _positive(
        tool.get("fr3_face_to_rh56_seating_plane_m"),
        "tool.fr3_face_to_rh56_seating_plane_m",
    )
    _positive(tool.get("adapter_disk_diameter_m"), "tool.adapter_disk_diameter_m")
    _positive(tool.get("adapter_spigot_diameter_m"), "tool.adapter_spigot_diameter_m")
    _positive(tool.get("rh56_nominal_mass_kg"), "tool.rh56_nominal_mass_kg")
    if not np.isclose(disk + spigot, total, atol=1e-9, rtol=0.0):
        raise ValueError("adapter total envelope must equal disk thickness + spigot protrusion")
    if not np.isclose(mount, disk, atol=1e-9, rtol=0.0):
        raise ValueError(
            "RH56 seating plane is the disk top, not the end of the inserted spigot"
        )
    if tool.get("T_EE_hand") is not None:
        validate_rigid_transform(tool["T_EE_hand"], "tool.T_EE_hand")
    basis = validate_rigid_transform(
        tool.get("T_seating_hand_axis_basis"),
        "tool.T_seating_hand_axis_basis",
    )
    expected_basis = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    if not np.array_equal(basis, expected_basis):
        raise ValueError("tool.T_seating_hand_axis_basis disagrees with AnyDex source axes")
    yaw = tool.get("assembled_yaw_rad")
    if yaw is not None and not np.isfinite(float(yaw)):
        raise ValueError("tool.assembled_yaw_rad must be finite or null")
    source_origin = tool.get("seating_to_hand_source_origin_m")
    if source_origin is not None:
        _finite_vector(
            source_origin, 3, "tool.seating_to_hand_source_origin_m"
        )
    for key in (
        "source_origin_datum_verified",
        "mount_transform_commissioned",
        "installed_on_franka_verified",
        "assembled_yaw_verified",
        "installed_collision_model_verified",
        "low_speed_unloaded_commissioning_only",
    ):
        _boolean(tool.get(key), f"tool.{key}")
    if tool["assembled_yaw_verified"] and yaw is None:
        raise ValueError("verified assembled yaw requires tool.assembled_yaw_rad")
    if tool["source_origin_datum_verified"] and source_origin is None:
        raise ValueError(
            "verified source origin requires tool.seating_to_hand_source_origin_m"
        )
    if tool["mount_transform_commissioned"] and tool.get("T_EE_hand") is None:
        raise ValueError("commissioned mount transform requires tool.T_EE_hand")

    grasp = _mapping(root.get("grasp"), "grasp")
    _positive(grasp.get("pregrasp_distance_m"), "grasp.pregrasp_distance_m")
    _positive(
        grasp.get("air_retreat_distance_m"),
        "grasp.air_retreat_distance_m",
    )
    _positive(
        grasp.get("air_pregrasp_distance_m"),
        "grasp.air_pregrasp_distance_m",
    )
    air_scene_margin = _positive(
        grasp.get("air_audit_observed_scene_margin_m"),
        "grasp.air_audit_observed_scene_margin_m",
    )
    if not 0.001 <= air_scene_margin <= 0.005:
        raise ValueError(
            "grasp.air_audit_observed_scene_margin_m must be in [0.001,0.005] m"
        )
    air_hand_self_margin = grasp.get(
        "air_audit_rh56_self_clearance_margin_m", 0.0
    )
    if (
        isinstance(air_hand_self_margin, bool)
        or not isinstance(air_hand_self_margin, (int, float))
        or not np.isfinite(float(air_hand_self_margin))
        or not np.isclose(float(air_hand_self_margin), 0.0, atol=1e-15, rtol=0.0)
    ):
        raise ValueError(
            "grasp.air_audit_rh56_self_clearance_margin_m must be exactly 0 m"
        )
    air_audit_max_scene_age = _positive(
        grasp.get("air_audit_max_scene_age_s"),
        "grasp.air_audit_max_scene_age_s",
    )
    if air_audit_max_scene_age > 120.0:
        raise ValueError("grasp.air_audit_max_scene_age_s must not exceed 120 s")
    insertion = float(grasp.get("extra_final_insertion_m"))
    if not np.isfinite(insertion) or insertion < 0.0:
        raise ValueError("grasp.extra_final_insertion_m must be finite and non-negative")
    _positive(
        grasp.get("upstream_ur_empirical_insertion_reference_m"),
        "grasp.upstream_ur_empirical_insertion_reference_m",
    )
    for key in (
        "snapshot_hand_pose_already_applies_predicted_depth",
        "extra_final_insertion_commissioned",
        "require_official_anydex_backend",
        "require_hand_scene_collision_check",
        "allow_diagnostic_plan_only",
    ):
        _boolean(grasp.get(key), f"grasp.{key}")


def select_execution_candidate(
    config: Mapping[str, Any], snapshot: VisualizationSnapshot
) -> int:
    """Return the highest-score top-k candidate meeting q6/collision gates."""

    validate_control_config(config)
    validate_snapshot(snapshot)
    inspire = _mapping(config["inspire"], "inspire")
    return select_eligible_candidate_index(
        snapshot,
        thumb_rotate_range=inspire["thumb_rotate_validated_realtime_range"],
        require_collision_checked=True,
        require_collision_free=True,
    )


def _snapshot_is_diagnostic(snapshot: VisualizationSnapshot) -> bool:
    name = snapshot.model_name.lower()
    return any(
        marker in name
        for marker in (
            "diagnostic",
            "geometric",
            "commissioning backend",
            "not anydexgrasp",
            "not decision-model output",
        )
    )


def control_readiness(
    config: Mapping[str, Any],
    snapshot: Optional[VisualizationSnapshot] = None,
) -> ControlReadiness:
    """Report every known blocker without opening either hardware device."""

    validate_control_config(config)
    franka = _mapping(config["franka"], "franka")
    inspire = _mapping(config["inspire"], "inspire")
    tool = _mapping(config["tool"], "tool")
    grasp = _mapping(config["grasp"], "grasp")
    calibration = _mapping(config["calibration"], "calibration")

    default_blockers = []
    if not tool["installed_on_franka_verified"]:
        default_blockers.append(
            "adapter + RH56 are not verified as physically mounted on Franka"
        )
    if franka.get("expected_F_T_EE") is None:
        default_blockers.append("Franka expected_F_T_EE has not been captured and verified")
    end_effector = _mapping(franka.get("expected_end_effector"), "franka.expected_end_effector")
    if any(
        end_effector.get(key) is None
        for key in ("mass_kg", "F_x_Cee_m", "inertia_kg_m2")
    ):
        default_blockers.append("installed adapter + RH56 mass/CoM/inertia are unverified")
    if not franka["default_path_collision_verified"]:
        default_blockers.append(
            "open-hand path from the current pose to Franka default_q is not collision-verified"
        )
    if not tool["assembled_yaw_verified"]:
        default_blockers.append("adapter/RH56 assembled yaw is unverified")
    if not tool["source_origin_datum_verified"]:
        default_blockers.append("RH56 seating plane to AnyDex source origin is unverified")
    if not tool["installed_collision_model_verified"]:
        default_blockers.append("installed adapter + hand collision model is unverified")

    full_blockers = list(default_blockers)
    if franka.get("cartesian_workspace_min_m") is None:
        full_blockers.append("Cartesian home/transit/grasp workspace is uncommissioned")
    if tool.get("T_EE_hand") is None or not tool["mount_transform_commissioned"]:
        full_blockers.append("T_EE_hand is missing or not commissioned")
    if tool["low_speed_unloaded_commissioning_only"]:
        full_blockers.append("installed PLA adapter is approved only for low-speed unloaded checks")
    if not inspire["six_axis_coupled_closure_commissioned"]:
        full_blockers.append("six-axis coupled Inspire closure is not commissioned")
    if (
        float(grasp["extra_final_insertion_m"]) > 0.0
        and not grasp["extra_final_insertion_commissioned"]
    ):
        full_blockers.append("final approach insertion offset is uncommissioned")

    if snapshot is None:
        full_blockers.append("no grasp snapshot was supplied")
    else:
        validate_snapshot(snapshot)
        if snapshot.reference_frame != config["reference_frame"]:
            full_blockers.append("snapshot reference frame does not match robot_base")
        if snapshot.calibration_id != calibration["id"]:
            full_blockers.append("snapshot calibration ID does not match the control profile")
        if snapshot.camera_serial != calibration["camera_serial"]:
            full_blockers.append("snapshot camera serial does not match the control profile")
        selected = int(snapshot.grasps.selected_index)
        if selected < 0 or selected >= snapshot.grasps.count:
            full_blockers.append("snapshot has no selected grasp")
        if snapshot.grasps.hand_poses is None or snapshot.grasps.hand_angles is None:
            full_blockers.append("snapshot has no Inspire hand pose/register target")
        elif 0 <= selected < snapshot.grasps.count:
            selected_values = np.asarray(
                snapshot.grasps.hand_angles[selected], dtype=np.float64
            )
            try:
                validate_air_candidate_commissioning(config, selected_values)
            except ValueError as exc:
                is_integer_target = (
                    selected_values.shape == (6,)
                    and np.all(np.isfinite(selected_values))
                    and np.array_equal(selected_values, np.rint(selected_values))
                )
                selected_targets = (
                    tuple(int(value) for value in selected_values)
                    if is_integer_target
                    else ()
                )
                q6_lower, q6_upper = (
                    int(value)
                    for value in inspire["thumb_rotate_validated_realtime_range"]
                )
                if (
                    is_integer_target
                    and not q6_lower <= selected_targets[5] <= q6_upper
                ):
                    full_blockers.append(
                        "selected thumb-rotate target is outside the commissioned "
                        "q6 range {}..{}".format(q6_lower, q6_upper)
                    )
                else:
                    full_blockers.append(str(exc))
                acceptance_mode, _ranges = validate_air_target_acceptance_policy(
                    inspire
                )
                exact_targets = {
                    tuple(int(value) for value in item)
                    for item in inspire.get("commissioned_air_closure_targets", [])
                }
                if (
                    is_integer_target
                    and acceptance_mode != SUPERVISED_RANGE_ACCEPTANCE
                    and selected_targets not in exact_targets
                ):
                    full_blockers.append(
                        "selected six-axis air-closure target has no exact "
                        "commissioning evidence"
                    )
        if (
            grasp["require_hand_scene_collision_check"]
            and 0 <= selected < snapshot.grasps.count
        ):
            if not bool(snapshot.grasps.collision_checked[selected]):
                full_blockers.append(
                    "hand + adapter scene collision result is not attached to selected grasp"
                )
            elif not bool(snapshot.grasps.collision_free[selected]):
                full_blockers.append("selected grasp failed the scene collision check")
        if grasp["require_official_anydex_backend"] and _snapshot_is_diagnostic(
            snapshot
        ):
            full_blockers.append("selected grasp is diagnostic/geometric, not official AnyDex output")
        elif (
            grasp["require_official_anydex_backend"]
            and not is_official_snapshot(snapshot)
        ):
            full_blockers.append(
                "snapshot is not identified as complete official AnyDex output"
            )
        if (
            grasp["require_official_anydex_backend"]
            and not has_complete_official_provenance(snapshot)
        ):
            full_blockers.append(
                "official provenance is incomplete: require representation SHA-256, "
                "8 decision SHA-256 hashes, and source commit"
            )

    return ControlReadiness(tuple(default_blockers), tuple(full_blockers))
