"""Bind one installed-tool audit sidecar to one immutable execution input.

The perception snapshot remains byte-for-byte unchanged.  Only after the
sidecar passes its strict schema, digest, file replay, freshness, and selected
candidate checks does :func:`bind_installed_tool_audit` return an in-memory
snapshot view in which that one candidate carries collision evidence.  The
view is never written back to the official NPZ.

This module is hardware-free.  A passing result is necessary evidence, never
motion authorization; the audit schema itself requires
``decision.motion_authorized == false``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .air_target_poses import derive_air_target_poses
from .control_plan import GraspExecutionPlan
from .installed_tool_audit import (
    V7_ADAPTER_SHA256,
    V7_T_EE_HAND_SHA256,
    load_installed_tool_audit,
)
from .snapshot import GraspCandidates, VisualizationSnapshot, validate_snapshot


@dataclass(frozen=True)
class InstalledAuditBinding:
    path: Optional[Path]
    artifact: Optional[Mapping[str, Any]]
    blockers: Tuple[str, ...]
    evidence_snapshot: Optional[VisualizationSnapshot]

    @property
    def passed(self) -> bool:
        return not self.blockers and self.artifact is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    shape = ",".join(str(item) for item in array.shape)
    digest = hashlib.sha256()
    digest.update(("dtype=<f8;shape={};".format(shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _same_array(left: Any, right: Any, *, atol: float = 0.0) -> bool:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    return first.shape == second.shape and bool(
        np.allclose(first, second, atol=atol, rtol=0.0)
    )


def _resolved_config_asset(config_path: Path, value: Any) -> Path:
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve()


def _snapshot_schema_version(path: Path) -> int:
    with np.load(path, allow_pickle=False) as archive:
        value = np.asarray(archive["schema_version"])
        if value.shape != () or value.dtype.kind not in "iu":
            raise ValueError("snapshot schema_version is malformed")
        return int(value.item())


def _scalar(archive: Any, key: str) -> Any:
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError("{} must be a scalar".format(key))
    return value.item()


def _replay_live_scene_npz(
    path: Path,
    *,
    scene_binding: Mapping[str, Any],
    snapshot: VisualizationSnapshot,
) -> Tuple[str, ...]:
    """Reconstruct the exact filtered scene array bound by the sidecar."""

    blockers = []
    try:
        with np.load(path, allow_pickle=False) as archive:
            point_key = (
                "filtered_scene_points"
                if "filtered_scene_points" in archive.files
                else "scene_points"
            )
            points = np.asarray(archive[point_key])
            if (
                points.ndim != 2
                or points.shape[1:] != (3,)
                or len(points) == 0
                or not np.all(np.isfinite(points))
            ):
                blockers.append("live-scene NPZ points are missing or malformed")
            elif _array_sha256(points) != scene_binding["points_sha256"]:
                blockers.append("live-scene NPZ filtered points differ from the audit")
            if int(len(points)) != int(scene_binding["point_count"]):
                blockers.append("live-scene NPZ point count differs from the audit")
            if "scene_excludes_object" not in archive.files or bool(
                _scalar(archive, "scene_excludes_object")
            ) is not True:
                blockers.append(
                    "live-scene NPZ must explicitly record scene_excludes_object=true"
                )
            if str(_scalar(archive, "reference_frame")) != "robot_base":
                blockers.append("live-scene NPZ reference frame is not robot_base")
            capture_q = np.asarray(archive["capture_q_rad"], dtype=np.float64)
            if not _same_array(capture_q, scene_binding["capture_q_rad"]):
                blockers.append("live-scene NPZ capture q differs from the audit")
            if _array_sha256(capture_q) != scene_binding["capture_q_sha256"]:
                blockers.append("live-scene NPZ capture q SHA-256 differs from the audit")
            captured_at = float(_scalar(archive, "captured_at_unix_s"))
            if not np.isclose(
                captured_at,
                float(scene_binding["captured_at_s"]),
                atol=1e-9,
                rtol=0.0,
            ):
                blockers.append("live-scene NPZ capture timestamp differs from the audit")
            if "calibration_id" in archive.files and str(
                _scalar(archive, "calibration_id")
            ) != snapshot.calibration_id:
                blockers.append("live-scene NPZ calibration ID differs from snapshot")
            if "camera_serial" in archive.files and str(
                _scalar(archive, "camera_serial")
            ) != snapshot.camera_serial:
                blockers.append("live-scene NPZ camera serial differs from snapshot")
            preprocessing = scene_binding["preprocessing"]
            if preprocessing["required"] is True:
                for key in (
                    "filter_evidence_path",
                    "filter_evidence_sha256",
                    "installed_inflation_margin_m",
                ):
                    if key not in archive.files:
                        blockers.append(
                            "live-scene NPZ is missing {}".format(key)
                        )
                if all(
                    key in archive.files
                    for key in (
                        "filter_evidence_path",
                        "filter_evidence_sha256",
                        "installed_inflation_margin_m",
                    )
                ):
                    evidence_path = Path(
                        str(_scalar(archive, "filter_evidence_path"))
                    ).resolve()
                    if evidence_path != Path(preprocessing["path"]).resolve():
                        blockers.append(
                            "live-scene NPZ filter evidence path differs from audit"
                        )
                    if str(_scalar(archive, "filter_evidence_sha256")) != preprocessing[
                        "sha256"
                    ]:
                        blockers.append(
                            "live-scene NPZ filter evidence hash differs from audit"
                        )
                    if not np.isclose(
                        float(_scalar(archive, "installed_inflation_margin_m")),
                        float(preprocessing["installed_inflation_margin_m"]),
                        atol=1e-15,
                        rtol=0.0,
                    ):
                        blockers.append(
                            "live-scene NPZ installed-return margin differs from audit"
                        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        blockers.append("cannot replay live-scene NPZ: {}".format(exc))
    return tuple(blockers)


def _official_hand_paths(workspace_root: Path) -> Tuple[Path, Path]:
    hand_root = (
        workspace_root
        / "dexgrasp/third_party/AnyDexGrasp/generate_mesh_and_pointcloud/inspire_urdf"
    ).resolve()
    return (
        (hand_root / "urdf-five3/robots/urdf-five3.urdf").resolve(),
        (hand_root / "width_12Dangle_6Dangle.json").resolve(),
    )


def bind_installed_tool_audit(
    audit_path: Optional[Path],
    *,
    config: Mapping[str, Any],
    config_path: Path,
    snapshot: VisualizationSnapshot,
    snapshot_path: Path,
    plan: GraspExecutionPlan,
    workspace_root: Path,
    now_s: Optional[float] = None,
) -> InstalledAuditBinding:
    """Return strict sidecar blockers and a collision-evidence snapshot view."""

    validate_snapshot(snapshot)
    if audit_path is None:
        return InstalledAuditBinding(
            None,
            None,
            ("a passing installed-tool collision audit artifact is required",),
            None,
        )
    source = Path(audit_path).expanduser().resolve()
    try:
        artifact = load_installed_tool_audit(
            source, verify_files=True, require_pass=True
        )
    except (OSError, TypeError, ValueError) as exc:
        return InstalledAuditBinding(
            source,
            None,
            ("installed-tool audit load/replay failed: {}".format(exc),),
            None,
        )

    blockers = []
    bindings = artifact["bindings"]
    profile = bindings["control_profile"]
    config_resolved = Path(config_path).expanduser().resolve()
    snapshot_resolved = Path(snapshot_path).expanduser().resolve()
    if Path(profile["path"]).resolve() != config_resolved:
        blockers.append("audit control-profile path differs from the loaded config")
    elif _sha256_file(config_resolved) != profile["sha256"]:
        blockers.append("loaded control-profile SHA-256 differs from the audit")

    snapshot_binding = bindings["snapshot"]
    if Path(snapshot_binding["path"]).resolve() != snapshot_resolved:
        blockers.append("audit snapshot path differs from the loaded snapshot")
    elif _sha256_file(snapshot_resolved) != snapshot_binding["sha256"]:
        blockers.append("loaded snapshot SHA-256 differs from the audit")
    try:
        schema_version = _snapshot_schema_version(snapshot_resolved)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        blockers.append("cannot verify snapshot schema_version: {}".format(exc))
    else:
        if schema_version != snapshot_binding["schema_version"]:
            blockers.append("snapshot schema_version differs from the audit")

    scalar_snapshot_fields = (
        ("reference_frame", snapshot.reference_frame),
        ("frame_id", int(snapshot.frame_id)),
        ("timestamp_s", float(snapshot.timestamp_s)),
        ("calibration_id", snapshot.calibration_id),
        ("camera_serial", snapshot.camera_serial),
        ("model_name", snapshot.model_name),
        (
            "representation_checkpoint_sha256",
            snapshot.representation_checkpoint_sha256,
        ),
        ("official_source_commit", snapshot.official_source_commit),
    )
    for key, actual in scalar_snapshot_fields:
        if snapshot_binding[key] != actual:
            blockers.append("snapshot {} differs from the audit".format(key))
    if tuple(snapshot_binding["decision_checkpoint_sha256s"]) != tuple(
        snapshot.decision_checkpoint_sha256s
    ):
        blockers.append("snapshot decision checkpoint hashes differ from the audit")

    selected = int(snapshot.grasps.selected_index)
    candidate = snapshot_binding["selected_candidate"]
    if selected != int(candidate["index"]):
        blockers.append("selected candidate index differs from the audit")
    elif not 0 <= selected < snapshot.grasps.count:
        blockers.append("loaded snapshot selected candidate is out of range")
    else:
        comparisons = (
            (
                "canonical pose",
                snapshot.grasps.canonical_poses[selected],
                candidate["canonical_pose_base"],
            ),
            (
                "hand pose",
                snapshot.grasps.hand_poses[selected]
                if snapshot.grasps.hand_poses is not None
                else None,
                candidate["hand_pose_base"],
            ),
            (
                "hand targets",
                snapshot.grasps.hand_angles[selected]
                if snapshot.grasps.hand_angles is not None
                else None,
                candidate["hand_targets"],
            ),
        )
        for label, actual, expected in comparisons:
            if actual is None or not _same_array(actual, expected):
                blockers.append("selected candidate {} differs from the audit".format(label))

    scene = bindings["scene"]
    object_binding = bindings["object"]
    scene_path = Path(scene["path"]).expanduser().resolve()
    if scene_path == snapshot_resolved:
        blockers.append("audit scene must be an independent fresh live-scene NPZ")
    else:
        blockers.extend(
            _replay_live_scene_npz(
                scene_path,
                scene_binding=scene,
                snapshot=snapshot,
            )
        )
    if _array_sha256(snapshot.object_points) != object_binding["points_sha256"]:
        blockers.append("snapshot object points differ from the audited object")
    if snapshot.scene_excludes_object is not True or scene["scene_excludes_object"] is not True:
        blockers.append("audited scene must explicitly exclude the target object")

    adapter = bindings["adapter"]
    configured_adapter = _resolved_config_asset(
        config_resolved, config["tool"]["adapter_asset"]
    )
    if Path(adapter["path"]).resolve() != configured_adapter:
        blockers.append("audit adapter path differs from the configured V7 CAD")
    if adapter["sha256"] != V7_ADAPTER_SHA256:
        blockers.append("audit adapter is not the commissioned V7 CAD")
    mount = bindings["mount_transform"]
    if mount["sha256"] != V7_T_EE_HAND_SHA256 or not _same_array(
        mount["T_EE_hand"], config["tool"]["T_EE_hand"]
    ):
        blockers.append("audit T_EE_hand differs from the configured V7 transform")

    hand = bindings["hand_model"]
    expected_urdf, expected_mapping = _official_hand_paths(
        Path(workspace_root).expanduser().resolve()
    )
    if Path(hand["urdf_path"]).resolve() != expected_urdf:
        blockers.append("audit URDF path is not the vendored official Inspire model")
    if Path(hand["mapping_path"]).resolve() != expected_mapping:
        blockers.append("audit mapping path is not the vendored official Inspire mapping")
    if hand["mesh_resolution"] != "full":
        blockers.append("installed-tool execution requires full-resolution hand link meshes")

    execution_plan = bindings["execution_plan"]
    decision = artifact["decision"]
    runtime_workspace_clear_required = bool(
        decision["runtime_operator_workspace_clear_required"]
    )
    if artifact["mode"] == "loaded_grasp" and runtime_workspace_clear_required:
        blockers.append("loaded grasp cannot consume conditional workspace evidence")
    if artifact["mode"] == "air_grasp" and not runtime_workspace_clear_required:
        blockers.append("air grasp audit omitted the runtime workspace-clear condition")
    audited_scene_margin = float(artifact["policies"]["scene"])
    if artifact["mode"] == "air_grasp":
        expected_scene_margin = float(
            config["grasp"]["air_audit_observed_scene_margin_m"]
        )
        if not np.isclose(
            audited_scene_margin,
            expected_scene_margin,
            atol=1e-15,
            rtol=0.0,
        ):
            blockers.append(
                "audited observed-scene margin differs from the air control profile"
            )
        expected_hand_self_margin = float(
            config["grasp"].get("air_audit_rh56_self_clearance_margin_m", 0.0)
        )
        if not np.isclose(
            float(artifact["policies"]["hand_self"]),
            expected_hand_self_margin,
            atol=1e-15,
            rtol=0.0,
        ):
            blockers.append(
                "audited RH56 self-clearance differs from the air control profile"
            )
    elif not np.isclose(
        audited_scene_margin, 0.005, atol=1e-15, rtol=0.0
    ):
        blockers.append("loaded grasp audit must retain the 5 mm scene margin")
    if not 0 <= selected < snapshot.grasps.count:
        blockers.append("cannot bind execution poses without a valid selected candidate")
        expected_pregrasp = np.asarray(
            execution_plan["pregrasp_pose_base_EE"], dtype=np.float64
        )
        expected_grasp = np.asarray(
            execution_plan["grasp_pose_base_EE"], dtype=np.float64
        )
    elif artifact["mode"] == "loaded_grasp":
        expected_pregrasp = plan.T_reference_EE_pregrasp
        expected_grasp = plan.T_reference_EE_grasp
    else:
        air_targets = derive_air_target_poses(
            np.asarray(snapshot.grasps.canonical_poses[selected], dtype=np.float64),
            np.asarray(snapshot.grasps.hand_poses[selected], dtype=np.float64),
            np.asarray(snapshot.grasps.approach_axis_local, dtype=np.float64),
            np.asarray(config["tool"]["T_EE_hand"], dtype=np.float64),
            retreat_distance_m=float(config["grasp"]["air_retreat_distance_m"]),
            pregrasp_extra_distance_m=float(
                config["grasp"]["air_pregrasp_distance_m"]
            ),
        )
        expected_grasp = air_targets.T_reference_EE_final_air
        expected_pregrasp = air_targets.T_reference_EE_pregrasp
        if not _same_array(
            execution_plan["planned_hand_pose_base"],
            air_targets.T_reference_hand_final_air,
            atol=1e-8,
        ):
            blockers.append(
                "planned air hand pose differs from the shared air-target contract"
            )
    if not _same_array(
        execution_plan["pregrasp_pose_base_EE"], expected_pregrasp, atol=1e-8
    ):
        blockers.append("planned pregrasp EE pose differs from the audit mode contract")
    if not _same_array(
        execution_plan["grasp_pose_base_EE"], expected_grasp, atol=1e-8
    ):
        blockers.append("planned grasp EE pose differs from the audit mode contract")
    expected_retreat = (
        float(config["grasp"]["air_retreat_distance_m"])
        if artifact["mode"] == "air_grasp"
        else 0.0
    )
    expected_pregrasp_distance = float(
        config["grasp"][
            "air_pregrasp_distance_m"
            if artifact["mode"] == "air_grasp"
            else "pregrasp_distance_m"
        ]
    )
    if not np.isclose(
        float(execution_plan["air_retreat_distance_m"]),
        expected_retreat,
        atol=1e-12,
        rtol=0.0,
    ):
        blockers.append("audited air retreat differs from the control profile")
    if not np.isclose(
        float(execution_plan["pregrasp_distance_m"]),
        expected_pregrasp_distance,
        atol=1e-12,
        rtol=0.0,
    ):
        blockers.append("audited pregrasp distance differs from the control profile")
    path_binding = bindings["joint_path"]
    waypoint_items = path_binding["waypoints"]
    default_items = [item for item in waypoint_items if item["name"] == "default"]
    if not _same_array(
        default_items[0]["q_rad"] if len(default_items) == 1 else [],
        config["franka"]["default_q_rad"],
    ):
        blockers.append("audited default q differs from the control profile")
    close_targets = np.asarray(candidate["hand_targets"], dtype=np.float64)
    if 0 <= selected < snapshot.grasps.count and snapshot.grasps.hand_angles is not None and not _same_array(
        close_targets, snapshot.grasps.hand_angles[selected]
    ):
        blockers.append("audited closure targets differ from the selected candidate")

    # Wall-clock freshness is intentionally not consumed during offline
    # binding.  Immediately before the first hand/arm command, the executor
    # rechecks scene age and compares a fresh live q to the audited current
    # waypoint within max_q_tracking_error_rad.

    if blockers:
        return InstalledAuditBinding(source, artifact, tuple(blockers), None)

    checked = np.asarray(snapshot.grasps.collision_checked, dtype=np.bool_).copy()
    free = np.asarray(snapshot.grasps.collision_free, dtype=np.bool_).copy()
    checked[selected] = True
    free[selected] = True
    if snapshot.grasps.hand_poses is None:
        return InstalledAuditBinding(
            source,
            artifact,
            ("snapshot has no hand poses for collision evidence overlay",),
            None,
        )
    hand_poses = np.asarray(snapshot.grasps.hand_poses, dtype=np.float64).copy()
    if artifact["mode"] == "air_grasp":
        hand_poses[selected] = np.asarray(
            execution_plan["planned_hand_pose_base"], dtype=np.float64
        )
    evidence_grasps: GraspCandidates = replace(
        snapshot.grasps,
        collision_checked=checked,
        collision_free=free,
        hand_poses=hand_poses,
    )
    evidence_snapshot = replace(snapshot, grasps=evidence_grasps)
    validate_snapshot(evidence_snapshot)
    return InstalledAuditBinding(source, artifact, (), evidence_snapshot)


def trajectory_contract_blocker() -> str:
    """Describe why the current Cartesian executor cannot consume this audit."""

    return (
        "installed-tool audit covers an exact joint-waypoint polyline, but the "
        "current grasp executor uses Cartesian interpolation for default->pregrasp "
        "and pregrasp->grasp; trajectory equivalence is unproven"
    )


__all__ = [
    "InstalledAuditBinding",
    "bind_installed_tool_audit",
    "trajectory_contract_blocker",
]
