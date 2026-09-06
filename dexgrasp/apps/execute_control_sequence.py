#!/usr/bin/env python3
"""Fail-closed FR3 + Inspire staged-control entry point.

``inspect`` is the default operation and is strictly offline.  The two motion
subcommands perform every configuration, provenance, plan, collision-audit,
and operator-confirmation check before importing a hardware driver or opening
either device.

This application never configures Franka ``set_EE``.  Loaded lift is a separate
audit-bound round trip which applies ``set_load`` only after verified closure
and clears it only after verified setdown.  Ordinary grasp evidence can never
authorize that mode.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT / "src", WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Only hardware-free modules are imported above the execution gate.
from anydex_pipeline.adapter_collision import (  # noqa: E402
    AdapterSweepCollisionReport,
    audit_adapter_sweep_against_points,
)
from anydex_pipeline.air_target_poses import derive_air_target_poses  # noqa: E402
from anydex_pipeline.air_audit_readiness import (  # noqa: E402
    validate_air_candidate_commissioning,
)
from anydex_pipeline.control_config import (  # noqa: E402
    ControlReadiness,
    VerifiedAdapterAssets,
    control_readiness,
    load_control_config,
    verify_adapter_assets,
)
from anydex_pipeline.control_plan import (  # noqa: E402
    AdapterGeometry,
    ExecutionConfig,
    GraspExecutionPlan,
    StageName,
    build_control_plan,
    inverse_rigid_transform,
)
from anydex_pipeline.continuous_telemetry_runtime import (  # noqa: E402
    ContinuousTelemetryRuntime,
    ContinuousTelemetryStartError,
    ExecutionTelemetryRequest,
    load_execution_telemetry_request,
)
from anydex_pipeline.execution_audit_gate import (  # noqa: E402
    InstalledAuditBinding,
    bind_installed_tool_audit,
    trajectory_contract_blocker,
)
from anydex_pipeline.installed_tool_audit import (  # noqa: E402
    load_installed_tool_audit,
)
from anydex_pipeline.loaded_lift_audit import (  # noqa: E402
    LoadedLiftAuditBinding,
    bind_loaded_lift_audit,
    load_loaded_lift_audit,
)
from anydex_pipeline.host_network_preflight import (  # noqa: E402
    HostNetworkPreflightError,
    require_uncontended_franka_https_link,
)
from anydex_pipeline.joint_path_sampling import (  # noqa: E402
    CANONICAL_JOINT_SAMPLING_ALGORITHM,
)
from anydex_pipeline.rh56_hand_path import (  # noqa: E402
    validate_rh56_feedback_envelope_policy,
    validate_rh56_hand_execution_path,
)
from anydex_pipeline.pregrasp_only_audit import (  # noqa: E402
    PregraspOnlyAuditBinding,
    bind_pregrasp_only_audit,
    load_pregrasp_only_audit,
)
from anydex_pipeline.snapshot import (  # noqa: E402
    GraspCandidates,
    VisualizationSnapshot,
    load_snapshot_npz,
)


DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"

WORKSPACE_CLEAR_TOKEN = "FR3_RH56_WORKSPACE_CLEAR"
IMMEDIATE_STOP_TOKEN = "IMMEDIATE_STOP_AND_24V_CUT_READY"
PLA_COMMISSIONING_TOKEN = "PLA_LOW_SPEED_UNLOADED_ONLY"
FULL_GRASP_TOKEN = "FR3_RH56_FULL_GRASP"
AIR_GRASP_TOKEN = "FR3_RH56_AIR_GRASP_NO_CONTACT_NO_LIFT"
Q6_PRESHAPE_TOKEN = "RH56_Q6_PRESHAPE_VERIFIED"
COLLISION_MODEL_TOKEN = "INSTALLED_TOOL_COLLISIONS_VERIFIED"
PREGRASP_ONLY_TOKEN = "FR3_RH56_PREGRASP_ONLY"
LOADED_LIFT_TOKEN = "FR3_RH56_LOADED_LIFT_ROUND_TRIP"
LOAD_SUPPORT_TOKEN = "LOAD_SETDOWN_SUPPORT_READY"
LOAD_RATED_TOKEN = "LOAD_RATED_ADAPTER_AND_PAYLOAD_VERIFIED"

MAX_HOLD_SECONDS = 10.0
BOUNDED_HOLD_POLL_SECONDS = 0.20
LOADED_HOLD_POLL_SECONDS = 0.20
EXPECTED_RH56_SPEED = 40
EXPECTED_RH56_FORCE_G = 80


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OfflineInspection:
    config: Dict[str, Any]
    config_path: Path
    assets: VerifiedAdapterAssets
    snapshot: Optional[VisualizationSnapshot]
    readiness: ControlReadiness
    plan: Optional[GraspExecutionPlan]
    adapter_audit: Optional[AdapterSweepCollisionReport]
    installed_audit: Optional[InstalledAuditBinding] = None
    pregrasp_audit: Optional[PregraspOnlyAuditBinding] = None
    loaded_lift_audit: Optional[LoadedLiftAuditBinding] = None
    plan_error: str = ""


def _print_blockers(title: str, blockers: Sequence[str]) -> None:
    print("[{}] {}".format(title, "READY" if not blockers else "LOCKED"))
    for blocker in blockers:
        print("  - {}".format(blocker))


def _adapter_from_config(config: Mapping[str, Any]) -> AdapterGeometry:
    tool = config["tool"]
    return AdapterGeometry(
        disk_diameter_m=float(tool["adapter_disk_diameter_m"]),
        disk_thickness_m=float(tool["adapter_disk_thickness_m"]),
        spigot_diameter_m=float(tool["adapter_spigot_diameter_m"]),
        spigot_protrusion_m=float(tool["adapter_spigot_protrusion_m"]),
        total_height_m=float(tool["adapter_total_axial_envelope_m"]),
        mount_plane_offset_m=float(tool["fr3_face_to_rh56_seating_plane_m"]),
        mesh_source_name=Path(str(tool["adapter_asset"])).name,
    )


def _execution_config(
    config: Mapping[str, Any],
    *,
    preshape_q6: bool,
    air_grasp: bool = False,
) -> ExecutionConfig:
    distance_key = (
        "air_pregrasp_distance_m" if air_grasp else "pregrasp_distance_m"
    )
    return ExecutionConfig(
        default_q=np.asarray(config["franka"]["default_q_rad"], dtype=np.float64),
        inspire_open_angles=np.asarray(
            config["inspire"]["open_targets"], dtype=np.float64
        ),
        pregrasp_distance_m=float(config["grasp"][distance_key]),
        final_insertion_m=float(config["grasp"]["extra_final_insertion_m"]),
        enable_thumb_preshape=bool(preshape_q6),
    )


def _combined_collision_points(snapshot: VisualizationSnapshot) -> np.ndarray:
    scene = np.asarray(snapshot.scene_points, dtype=np.float64)
    object_points = np.asarray(snapshot.object_points, dtype=np.float64)
    if not len(scene):
        return object_points.copy()
    if not len(object_points):
        return scene.copy()
    return np.concatenate((scene, object_points), axis=0)


def _audit_final_approach(
    config: Mapping[str, Any],
    snapshot: VisualizationSnapshot,
    plan: GraspExecutionPlan,
    *,
    margin_m: float,
    samples: int,
) -> AdapterSweepCollisionReport:
    # The conservative adapter collision frame is coincident with the Franka
    # mounting face F.  Since F_T_EE maps EE coordinates into F, EE_T_adapter
    # is its rigid inverse.  The 10 mm seating plane remains part of the mesh;
    # it is not substituted for the full 17.8 mm collision envelope.
    F_T_EE = np.asarray(config["franka"]["expected_F_T_EE"], dtype=np.float64)
    T_EE_adapter = inverse_rigid_transform(F_T_EE)
    return audit_adapter_sweep_against_points(
        _combined_collision_points(snapshot),
        T_reference_EE_start=plan.T_reference_EE_pregrasp,
        T_reference_EE_end=plan.T_reference_EE_grasp,
        T_EE_adapter=T_EE_adapter,
        adapter=plan.adapter,
        margin_m=margin_m,
        trajectory_samples=samples,
    )


def _snapshot_with_selected_index(
    snapshot: VisualizationSnapshot, selected_index: int
) -> VisualizationSnapshot:
    """Return a read-only in-memory candidate-selection view.

    The official NPZ is immutable.  Candidate selection is instead an
    execution input that must agree with the installed-tool sidecar.  This
    helper changes only the dataclass view used for planning and binding.
    """

    if isinstance(selected_index, (bool, np.bool_)) or not isinstance(
        selected_index, (int, np.integer)
    ):
        raise ValueError("selected candidate index must be an integer")
    selected = int(selected_index)
    if not 0 <= selected < snapshot.grasps.count:
        raise ValueError(
            "selected candidate index {} is outside snapshot candidates 0..{}".format(
                selected, snapshot.grasps.count - 1
            )
        )
    grasps: GraspCandidates = replace(
        snapshot.grasps, selected_index=selected
    )
    return replace(snapshot, grasps=grasps)


def _artifact_selected_index(audit_path: Optional[Path]) -> Optional[int]:
    """Read a valid sidecar's bound selection without weakening later gates."""

    if audit_path is None:
        return None
    try:
        artifact = load_installed_tool_audit(
            Path(audit_path).expanduser().resolve(),
            verify_files=True,
            require_pass=False,
        )
    except (OSError, TypeError, ValueError):
        # ``bind_installed_tool_audit`` reports the detailed replay blocker.
        return None
    return int(
        artifact["bindings"]["snapshot"]["selected_candidate"]["index"]
    )


def _pregrasp_artifact_selected_index(audit_path: Optional[Path]) -> Optional[int]:
    if audit_path is None:
        return None
    try:
        artifact = load_pregrasp_only_audit(
            Path(audit_path).expanduser().resolve(),
            verify_files=True,
            require_pass=False,
        )
    except (OSError, TypeError, ValueError):
        return None
    return int(artifact["bindings"]["snapshot"]["selected_candidate_index"])


def inspect_offline(
    config_path: Path,
    snapshot_path: Optional[Path],
    *,
    collision_margin_m: float,
    collision_samples: int,
    installed_audit_path: Optional[Path] = None,
    pregrasp_audit_path: Optional[Path] = None,
    loaded_lift_audit_path: Optional[Path] = None,
    selected_index: Optional[int] = None,
    now_s: Optional[float] = None,
) -> OfflineInspection:
    """Load, validate, plan, and audit without importing either device driver."""

    config, resolved_config_path = load_control_config(config_path)
    assets = verify_adapter_assets(config, resolved_config_path)
    source_snapshot = (
        None if snapshot_path is None else load_snapshot_npz(snapshot_path.expanduser())
    )
    if source_snapshot is None:
        if selected_index is not None:
            raise ValueError("--selected-index requires --snapshot")
        snapshot = None
    else:
        installed_index = _artifact_selected_index(installed_audit_path)
        pregrasp_index = _pregrasp_artifact_selected_index(pregrasp_audit_path)
        if (
            installed_index is not None
            and pregrasp_index is not None
            and installed_index != pregrasp_index
        ):
            raise ValueError("installed-tool and pregrasp-only audits select different candidates")
        bound_index = installed_index if installed_index is not None else pregrasp_index
        if selected_index is not None and bound_index is not None and int(
            selected_index
        ) != bound_index:
            raise ValueError(
                "--selected-index {} differs from installed-tool audit candidate {}".format(
                    int(selected_index), bound_index
                )
            )
        active_index = (
            int(selected_index)
            if selected_index is not None
            else (
                bound_index
                if bound_index is not None
                else int(source_snapshot.grasps.selected_index)
            )
        )
        snapshot = _snapshot_with_selected_index(source_snapshot, active_index)
    readiness = control_readiness(config, snapshot)
    plan = None
    audit = None
    installed_audit = None
    pregrasp_audit = None
    loaded_lift_audit = None
    plan_error = ""
    tool = config["tool"]
    if snapshot is not None:
        if tool.get("T_EE_hand") is None:
            plan_error = "T_EE_hand is absent; staged poses cannot be composed"
        elif not bool(tool.get("mount_transform_commissioned")):
            plan_error = "T_EE_hand exists but mount_transform_commissioned is false"
        else:
            try:
                plan = build_control_plan(
                    snapshot,
                    T_EE_hand=np.asarray(tool["T_EE_hand"], dtype=np.float64),
                    mount_transform_commissioned=True,
                    config=_execution_config(config, preshape_q6=True),
                    adapter=_adapter_from_config(config),
                    allow_diagnostic=bool(
                        config["grasp"].get("allow_diagnostic_plan_only", False)
                    ),
                )
                if config["franka"].get("expected_F_T_EE") is None:
                    plan_error = (
                        "staged plan built, but adapter sweep needs expected_F_T_EE"
                    )
                else:
                    audit = _audit_final_approach(
                        config,
                        snapshot,
                        plan,
                        margin_m=collision_margin_m,
                        samples=collision_samples,
                    )
                installed_audit = bind_installed_tool_audit(
                    installed_audit_path,
                    config=config,
                    config_path=resolved_config_path,
                    snapshot=snapshot,
                    snapshot_path=Path(snapshot_path).expanduser().resolve(),
                    plan=plan,
                    workspace_root=WORKSPACE,
                    now_s=now_s,
                )
                pregrasp_audit = bind_pregrasp_only_audit(
                    pregrasp_audit_path,
                    config=config,
                    config_path=resolved_config_path,
                    snapshot=snapshot,
                    snapshot_path=Path(snapshot_path).expanduser().resolve(),
                )
                if installed_audit.evidence_snapshot is not None:
                    evidence_snapshot = installed_audit.evidence_snapshot
                    evidence_config = copy.deepcopy(config)
                    # These two legacy booleans are superseded only in memory
                    # by the stronger, immutable, passing sidecar.  The bound
                    # control-profile file itself is never rewritten.
                    evidence_config["franka"]["default_path_collision_verified"] = True
                    evidence_config["tool"]["installed_collision_model_verified"] = True
                    readiness = control_readiness(evidence_config, evidence_snapshot)
                    plan = build_control_plan(
                        evidence_snapshot,
                        T_EE_hand=np.asarray(tool["T_EE_hand"], dtype=np.float64),
                        mount_transform_commissioned=True,
                        config=_execution_config(
                            config,
                            preshape_q6=True,
                            air_grasp=(
                                installed_audit.artifact is not None
                                and installed_audit.artifact["mode"] == "air_grasp"
                            ),
                        ),
                        adapter=_adapter_from_config(config),
                        allow_diagnostic=bool(
                            config["grasp"].get("allow_diagnostic_plan_only", False)
                        ),
                    )
                    if config["franka"].get("expected_F_T_EE") is not None:
                        audit = _audit_final_approach(
                            config,
                            evidence_snapshot,
                            plan,
                            margin_m=collision_margin_m,
                            samples=collision_samples,
                        )
            except (TypeError, ValueError) as exc:
                plan_error = str(exc)
    expected_grasp_q = None
    expected_grasp_pose = None
    expected_targets = None
    expected_candidate = None
    if snapshot is not None:
        expected_candidate = int(snapshot.grasps.selected_index)
    if installed_audit is not None and installed_audit.artifact is not None:
        try:
            expected_grasp_q = installed_audit.artifact["bindings"][
                "joint_path"
            ]["waypoints"][-1]["q_rad"]
        except (KeyError, IndexError, TypeError):
            expected_grasp_q = None
    if plan is not None:
        expected_grasp_pose = plan.T_reference_EE_grasp
        close_stages = [
            stage for stage in plan.stages if stage.name == StageName.INSPIRE_CLOSE
        ]
        if len(close_stages) == 1:
            expected_targets = close_stages[0].inspire_angles
    loaded_lift_audit = bind_loaded_lift_audit(
        loaded_lift_audit_path,
        base_loaded_grasp_audit_path=installed_audit_path,
        config=config,
        config_path=resolved_config_path,
        expected_selected_candidate_index=expected_candidate,
        expected_grasp_q_rad=expected_grasp_q,
        expected_grasp_pose_base_EE=expected_grasp_pose,
        expected_closed_hand_targets=expected_targets,
        verify_files=True,
    )
    return OfflineInspection(
        config=config,
        config_path=resolved_config_path,
        assets=assets,
        snapshot=snapshot,
        readiness=readiness,
        plan=plan,
        adapter_audit=audit,
        installed_audit=installed_audit,
        pregrasp_audit=pregrasp_audit,
        loaded_lift_audit=loaded_lift_audit,
        plan_error=plan_error,
    )


def _print_inspection(
    inspection: OfflineInspection, *, command: str = "inspect"
) -> None:
    print("[mode] OFFLINE INSPECT; no hardware driver was imported")
    print("[config] {}".format(inspection.config_path))
    print(
        "[adapter] verified={} sha256={}".format(
            inspection.assets.mesh_path, inspection.assets.mesh_sha256
        )
    )
    tool = inspection.config["tool"]
    print(
        "[adapter] disk=Ø{:.1f}x{:.1f}mm spigot=Ø{:.1f}x{:.1f}mm "
        "mount-plane={:.1f}mm full-envelope={:.1f}mm material={} mounted={}".format(
            1000.0 * float(tool["adapter_disk_diameter_m"]),
            1000.0 * float(tool["adapter_disk_thickness_m"]),
            1000.0 * float(tool["adapter_spigot_diameter_m"]),
            1000.0 * float(tool["adapter_spigot_protrusion_m"]),
            1000.0 * float(tool["fr3_face_to_rh56_seating_plane_m"]),
            1000.0 * float(tool["adapter_total_axial_envelope_m"]),
            tool.get("material", "unknown"),
            bool(tool.get("installed_on_franka_verified", False)),
        )
    )
    if inspection.snapshot is not None:
        snapshot = inspection.snapshot
        print(
            "[snapshot] frame={} calibration={} camera={} model={}".format(
                snapshot.frame_id,
                snapshot.calibration_id,
                snapshot.camera_serial,
                snapshot.model_name,
            )
        )
    _print_blockers(
        "default-pose motion", inspection.readiness.default_motion_blockers
    )
    _print_blockers("full grasp", inspection.readiness.full_grasp_blockers)
    if inspection.plan is not None:
        plan = inspection.plan
        print(
            "[plan] selected={} eligible={} stages={}".format(
                plan.selected_index,
                plan.execution_eligible,
                ",".join(stage.name.value for stage in plan.stages),
            )
        )
        for blocker in plan.execution_blockers:
            print("  - plan blocker: {}".format(blocker))
        if command == "air-grasp" and inspection.snapshot is not None:
            selected = int(plan.selected_index)
            grasps = inspection.snapshot.grasps
            air_targets = derive_air_target_poses(
                np.asarray(grasps.canonical_poses[selected], dtype=np.float64),
                np.asarray(grasps.hand_poses[selected], dtype=np.float64),
                np.asarray(grasps.approach_axis_local, dtype=np.float64),
                np.asarray(inspection.config["tool"]["T_EE_hand"], dtype=np.float64),
                retreat_distance_m=float(
                    inspection.config["grasp"]["air_retreat_distance_m"]
                ),
                pregrasp_extra_distance_m=float(
                    inspection.config["grasp"]["air_pregrasp_distance_m"]
                ),
            )
            print(
                "[plan/nominal-contact-reference] pregrasp_xyz={} contact_xyz={}".format(
                    np.round(plan.T_reference_EE_pregrasp[:3, 3], 6).tolist(),
                    np.round(plan.T_reference_EE_grasp[:3, 3], 6).tolist(),
                )
            )
            print(
                "[plan/air-execution-contract] pregrasp_xyz={} final_air_xyz={} "
                "retreat_m={:.3f} extra_pregrasp_m={:.3f}".format(
                    np.round(
                        air_targets.T_reference_EE_pregrasp[:3, 3], 6
                    ).tolist(),
                    np.round(
                        air_targets.T_reference_EE_final_air[:3, 3], 6
                    ).tolist(),
                    float(inspection.config["grasp"]["air_retreat_distance_m"]),
                    float(inspection.config["grasp"]["air_pregrasp_distance_m"]),
                )
            )
        else:
            print(
                "[plan] pregrasp_xyz={} grasp_xyz={}".format(
                    np.round(plan.T_reference_EE_pregrasp[:3, 3], 6).tolist(),
                    np.round(plan.T_reference_EE_grasp[:3, 3], 6).tolist(),
                )
            )
    if inspection.plan_error:
        print("[plan] unavailable: {}".format(inspection.plan_error))
    if inspection.adapter_audit is not None:
        report = inspection.adapter_audit
        print(
            "[adapter-audit] collision_free={} authoritative={} points={} "
            "samples={} margin_m={:.4f} collisions={}".format(
                report.collision_free,
                report.authoritative,
                report.point_count,
                report.trajectory_samples,
                report.margin_m,
                len(report.colliding_point_indices),
            )
        )
        if report.colliding_point_indices:
            print(
                "[adapter-audit] first colliding point indices={}".format(
                    list(report.colliding_point_indices[:20])
                )
            )
        print(
            "[adapter-audit] sparse point clearance is diagnostic only and never "
            "becomes authoritative collision proof"
        )
    if inspection.installed_audit is not None:
        binding = inspection.installed_audit
        if binding.artifact is None:
            print("[installed-tool-audit] LOCKED path={}".format(binding.path))
        else:
            artifact = binding.artifact
            joint_path = artifact["bindings"]["joint_path"]
            print(
                "[installed-tool-audit] {} path={} artifact_sha256={} "
                "samples={} motion_authorized={}".format(
                    "EVIDENCE-PASS" if binding.passed else "LOCKED",
                    binding.path,
                    artifact["artifact_sha256"],
                    joint_path["sample_count"],
                    artifact["decision"]["motion_authorized"],
                )
            )
        for blocker in binding.blockers:
            print("  - installed audit blocker: {}".format(blocker))
    if inspection.pregrasp_audit is not None:
        binding = inspection.pregrasp_audit
        if binding.artifact is None:
            print("[pregrasp-only-audit] LOCKED path={}".format(binding.path))
        else:
            artifact = binding.artifact
            prefix = artifact["bindings"]["joint_prefix"]
            print(
                "[pregrasp-only-audit] {} path={} artifact_sha256={} "
                "prefix_sha256={} samples={} motion_authorized={}".format(
                    "EVIDENCE-PASS" if binding.passed else "LOCKED",
                    binding.path,
                    artifact["artifact_sha256"],
                    prefix["prefix_contract_sha256"],
                    prefix["sample_count"],
                    artifact["motion_authorized"],
                )
            )
            scene = artifact["bindings"]["filtered_scene"]
            if scene.get("capture_matches_prefix_current") is False:
                print(
                    "[pregrasp-only/advisory] scene capture q differs from "
                    "the independently audited prefix current by {:.7f}rad "
                    "Linf; scene q is provenance only".format(
                        float(scene["capture_to_prefix_current_linf_rad"])
                    )
                )
            if scene.get("age_exceeded_recommended_max") is True:
                print(
                    "[pregrasp-only/advisory] scene age at artifact creation "
                    "was {:.1f}s (recommended {:.1f}s); this is provenance, "
                    "not an execution gate".format(
                        float(scene["age_at_artifact_creation_s"]),
                        float(scene["configured_recommended_max_scene_age_s"]),
                    )
                )
        for blocker in binding.blockers:
            print("  - pregrasp-only audit blocker: {}".format(blocker))
    if inspection.loaded_lift_audit is not None:
        binding = inspection.loaded_lift_audit
        artifact = getattr(binding, "artifact", None)
        if artifact is None:
            print(
                "[loaded-lift-audit] LOCKED path={}".format(
                    getattr(binding, "path", None)
                )
            )
        else:
            print(
                "[loaded-lift-audit] {} path={} artifact_sha256={} "
                "motion_authorized={}".format(
                    "EVIDENCE-PASS"
                    if bool(getattr(binding, "passed", False))
                    else "LOCKED",
                    getattr(binding, "path", None),
                    artifact.get("artifact_sha256", "missing"),
                    artifact.get("decision", {}).get("motion_authorized"),
                )
            )
        for blocker in tuple(getattr(binding, "blockers", ())):
            print("  - loaded-lift audit blocker: {}".format(blocker))


def _print_staged_dry_run(
    inspection: OfflineInspection,
    *,
    command: str,
    trajectory_mode: str,
    stop_after_default: bool = False,
) -> None:
    """Print the exact intended phases without importing a hardware driver."""

    print("[dry-run] HARDWARE DISABLED; staged command preview follows")
    binding = (
        inspection.pregrasp_audit
        if command == "pregrasp"
        else inspection.installed_audit
    )
    if binding is not None and binding.artifact is not None:
        path = binding.artifact["bindings"][
            "joint_prefix" if command == "pregrasp" else "joint_path"
        ]
        for segment in path["segments"]:
            print(
                "[dry-run/path] {} samples={}..{}".format(
                    segment["name"],
                    segment["start_index"],
                    segment["end_index"],
                )
            )
    phases = [
        "verify immutable config/snapshot/V7 CAD/T_EE_hand/URDF/mapping hashes",
        "verify fresh live Franka q equals audited current waypoint",
        "open all six RH56 axes and verify",
        "verify all-six RH56 numeric outputs disabled before Franka motion",
        "execute audited current/transit/default joint waypoints",
    ]
    if command == "pregrasp":
        phases.extend(
            (
                "execute audited default/approach-transit/pregrasp joint waypoints",
                "verify audited pregrasp q and bound FK/EEF readback settled",
                "stop Franka and re-verify all-six RH56 outputs disabled; no grasp/q6/bends",
            )
        )
    elif command == "grasp-lift":
        phases.extend(
            (
                "execute audited default/pregrasp/grasp joint waypoints and settle",
                "preshape q6, close bends, verify audited minimum contact hold",
                "apply and read back artifact-bound Franka payload dynamics",
                "execute exact audited grasp/lift-transit/lift joint suffix",
                "verify lift q/FK/EEF settle while rechecking numeric hand hold",
                "bounded lifted hold; Ctrl+C stops Franka but intentionally keeps hand hold",
                "replay the exact lift suffix in reverse to the audited support pose",
                "verify setdown q/FK/EEF settle, then disable hand and clear Franka load",
            )
        )
    elif command in ("grasp", "air-grasp") and not stop_after_default:
        phases.extend(
            (
                "execute audited default/pregrasp joint waypoints",
                "execute audited pregrasp/grasp joint waypoints",
                "verify audited target q and bound FK/EEF readback settled",
                "preshape RH56 thumb rotation q6",
                "close five bend axes while holding q6",
                "bounded hold, stop Franka, disable all-six RH56 outputs",
            )
        )
    elif command in ("grasp", "air-grasp"):
        phases.append(
            "stop Franka, disable all-six RH56 outputs, then recapture and re-audit"
        )
    for index, phase in enumerate(phases, start=1):
        print("[dry-run/stage {:02d}] {}".format(index, phase))
    if trajectory_mode == "audited-joint":
        print("[dry-run] exact artifact joint-waypoint mode selected")
    else:
        print("[dry-run] {}".format(trajectory_contract_blocker()))


def _audited_joint_settle_binding_blockers(
    inspection: OfflineInspection,
) -> Tuple[str, ...]:
    """Validate the immutable q/FK/EEF binding used by joint-only settling.

    This is deliberately an offline gate.  A collision sidecar alone is not
    enough: the schema-v2 artifact must bind its final two joint waypoints to
    the same poses as a checksum-valid deterministic FK manifest.  Only then
    may execution use read-only q + EEF arrival verification without generic
    Cartesian workspace bounds.
    """

    binding = inspection.installed_audit
    if binding is None or not binding.passed or binding.artifact is None:
        return ("audited-joint settle requires a passing installed-tool audit",)
    artifact = binding.artifact
    blockers: List[str] = []
    if type(artifact.get("schema_version")) is not int or artifact.get(
        "schema_version"
    ) != 2:
        return ("audited-joint settle requires schema-v2 audit evidence",)
    try:
        bindings = artifact["bindings"]
        execution = bindings["execution_plan"]
        path_binding = bindings["joint_path"]
        if execution["trajectory_contract"] != "joint_waypoint_polyline_v1":
            raise ValueError("trajectory contract is not the audited joint polyline")
        if (
            path_binding["sampling_algorithm"]
            != CANONICAL_JOINT_SAMPLING_ALGORITHM
        ):
            raise ValueError("joint path sampling algorithm is not canonical")
        waypoints = path_binding["waypoints"]
        names = [item["name"] for item in waypoints]
        if (
            len(waypoints) < 4
            or names[0] != "current"
            or names[-2:] != ["pregrasp", "grasp"]
            or names.count("default") != 1
        ):
            raise ValueError("audited waypoint order is invalid")
        q_pre = np.asarray(waypoints[-2]["q_rad"], dtype=np.float64)
        q_grasp = np.asarray(waypoints[-1]["q_rad"], dtype=np.float64)
        pose_pre = np.asarray(
            execution["pregrasp_pose_base_EE"], dtype=np.float64
        )
        pose_grasp = np.asarray(
            execution["grasp_pose_base_EE"], dtype=np.float64
        )
        if any(
            value.shape != shape or not np.all(np.isfinite(value))
            for value, shape in (
                (q_pre, (7,)),
                (q_grasp, (7,)),
                (pose_pre, (4, 4)),
                (pose_grasp, (4, 4)),
            )
        ):
            raise ValueError("audited final q/pose values are malformed")

        manifest_binding = bindings["joint_plan_manifest"]
        if (
            manifest_binding.get("required") is not True
            or manifest_binding.get("validation_error") != ""
            or not str(manifest_binding.get("path", ""))
        ):
            raise ValueError("audit has no validated FK joint-plan manifest")
        manifest_path = Path(str(manifest_binding["path"])).expanduser().resolve()
        if (
            not manifest_path.is_file()
            or _sha256_file(manifest_path) != manifest_binding.get("sha256")
            or manifest_binding.get("sha256_after_audit")
            != manifest_binding.get("sha256")
        ):
            raise ValueError("FK joint-plan manifest file/hash binding failed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 1
            or manifest.get("artifact_type")
            != "installed_air_candidate_joint_plan"
            or manifest.get("motion_authorized") is not False
        ):
            raise ValueError("FK joint-plan manifest schema/type is invalid")
        integrity = manifest.get("integrity")
        unsigned = dict(manifest)
        unsigned.pop("integrity", None)
        if (
            not isinstance(integrity, dict)
            or integrity.get("algorithm")
            != "sha256-canonical-json-without-integrity"
            or integrity.get("payload_sha256")
            != _canonical_json_sha256(unsigned)
        ):
            raise ValueError("FK joint-plan manifest checksum failed")

        manifest_plan = manifest["joint_plan"]
        manifest_geometry = manifest["air_geometry"]
        audit_hand_path = validate_rh56_hand_execution_path(
            bindings["hand_execution_path"]
        )
        manifest_hand_path = validate_rh56_hand_execution_path(
            manifest["hand_execution_path"]
        )
        if audit_hand_path.sha256 != manifest_hand_path.sha256:
            raise ValueError(
                "audit RH56 execution path differs from joint-plan manifest"
            )
        if int(artifact["policies"]["hand_arrival_tolerance_units"]) != int(
            inspection.config["inspire"]["arrival_tolerance_units"]
        ):
            raise ValueError(
                "audit RH56 feedback tube tolerance differs from runtime profile"
            )
        validate_rh56_feedback_envelope_policy(
            artifact["policies"].get("hand_feedback_envelope"),
            int(inspection.config["inspire"]["arrival_tolerance_units"]),
        )
        if (
            not np.array_equal(
                np.asarray(
                    manifest_plan["q_pregrasp_rad"], dtype=np.float64
                ),
                q_pre,
            )
            or not np.array_equal(
                np.asarray(
                    manifest_plan["q_final_air_rad"], dtype=np.float64
                ),
                q_grasp,
            )
            or not np.allclose(
                np.asarray(
                    manifest_geometry["pregrasp_pose_base_EE"],
                    dtype=np.float64,
                ),
                pose_pre,
                atol=1.0e-10,
                rtol=0.0,
            )
            or not np.allclose(
                np.asarray(
                    manifest_geometry["final_air_pose_base_EE"],
                    dtype=np.float64,
                ),
                pose_grasp,
                atol=1.0e-10,
                rtol=0.0,
            )
        ):
            raise ValueError("audited joint waypoints are not bound to FK/EEF poses")
        fk = manifest["fk_residual"]
        ik = manifest["planner"]["ik"]
        position_limit = float(ik["position_tolerance_m"])
        rotation_limit = float(ik["rotation_tolerance_rad"])
        if (
            fk.get("passed") is not True
            or not 0.0 < position_limit <= 1.0e-9
            or not 0.0 < rotation_limit <= 1.0e-9
            or any(
                not np.isfinite(float(fk[key]))
                or float(fk[key]) > limit
                for key, limit in (
                    ("pregrasp_position_m", position_limit),
                    ("final_air_position_m", position_limit),
                    ("pregrasp_rotation_rad", rotation_limit),
                    ("final_air_rotation_rad", rotation_limit),
                )
            )
        ):
            raise ValueError("FK residual evidence does not meet its bound tolerance")

        inputs = manifest["inputs"]
        for name, audit_key in (
            ("config", "control_profile"),
            ("snapshot", "snapshot"),
        ):
            if inputs[name]["sha256"] != bindings[audit_key]["sha256"]:
                raise ValueError(
                    "FK manifest {} differs from schema-v2 audit".format(name)
                )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        blockers.append("audited-joint q/FK/EEF binding failed: {}".format(exc))
    return tuple(blockers)


def _runtime_config_blockers(
    inspection: OfflineInspection, *, command: str, trajectory_mode: str = "cartesian"
) -> Tuple[str, ...]:
    """Validate driver construction constraints before loading driver modules."""

    config = inspection.config
    franka = config["franka"]
    inspire = config["inspire"]
    blockers: List[str] = []
    if command == "pregrasp":
        binding = inspection.pregrasp_audit
        if binding is None:
            blockers.append("a dedicated bound pregrasp-only audit artifact is required")
        elif not binding.passed:
            blockers.extend(binding.blockers)
        elif binding.artifact is None:
            blockers.append("passing pregrasp-only audit artifact is unavailable")
        else:
            artifact = binding.artifact
            if artifact.get("schema_version") != 2:
                blockers.append(
                    "pregrasp-only execution requires schema-v2 evidence"
                )
            if artifact.get("execution_scope") != "open_hand_current_to_pregrasp_only":
                blockers.append("pregrasp audit has the wrong execution scope")
            if artifact.get("motion_authorized") is not False:
                blockers.append("pregrasp audit improperly claims motion authorization")
            if artifact["decision"].get("required_geometry_passed") is not True:
                blockers.append("pregrasp authoritative physical/object geometry did not pass")
    else:
        if inspection.installed_audit is None:
            blockers.append("a bound installed-tool audit artifact is required")
        elif not inspection.installed_audit.passed:
            blockers.extend(inspection.installed_audit.blockers)
        elif trajectory_mode != "audited-joint":
            # A Cartesian interpolation is a different geometric path from the
            # exact joint polyline certified by the sidecar.
            blockers.append(trajectory_contract_blocker())
        elif command not in ("grasp", "grasp-lift", "air-grasp"):
            blockers.append(
                "audited-joint execution is currently exposed only for the complete grasp sequence"
            )
        elif trajectory_mode == "audited-joint" and command != "grasp-lift":
            blockers.extend(_audited_joint_settle_binding_blockers(inspection))
    if command == "grasp-lift":
        lift_binding = inspection.loaded_lift_audit
        if lift_binding is None:
            blockers.append(
                "an independent bound loaded-lift audit artifact is required"
            )
        elif not bool(getattr(lift_binding, "passed", False)):
            blockers.extend(tuple(getattr(lift_binding, "blockers", ())))
        elif getattr(lift_binding, "artifact", None) is None:
            blockers.append("passing loaded-lift audit artifact is unavailable")
        else:
            time_law_values = (
                getattr(lift_binding, "approved_max_joint_velocity_rad_s", None),
                getattr(
                    lift_binding, "approved_max_joint_acceleration_rad_s2", None
                ),
                getattr(lift_binding, "time_law_max_joint_velocity_rad_s", None),
                getattr(
                    lift_binding, "time_law_max_joint_acceleration_rad_s2", None
                ),
                getattr(lift_binding, "time_law_max_dynamic_segment_rad", None),
                getattr(lift_binding, "time_law_min_segment_duration_s", None),
            )
            try:
                (
                    approved_velocity,
                    approved_acceleration,
                    time_velocity,
                    time_acceleration,
                    time_segment,
                    time_duration,
                ) = tuple(float(value) for value in time_law_values)
            except (TypeError, ValueError):
                blockers.append("loaded-lift audit has no complete execution time-law")
            else:
                if not all(
                    np.isfinite(value) and value > 0.0
                    for value in (
                        approved_velocity,
                        approved_acceleration,
                        time_velocity,
                        time_acceleration,
                        time_segment,
                        time_duration,
                    )
                ):
                    blockers.append("loaded-lift execution time-law is malformed")
                if time_velocity > approved_velocity + 1.0e-15 or (
                    time_acceleration > approved_acceleration + 1.0e-15
                ):
                    blockers.append("loaded-lift time-law exceeds material approval")
                if time_velocity > float(
                    franka["default_max_joint_velocity_rad_s"]
                ) + 1.0e-15:
                    blockers.append(
                        "loaded-lift time-law exceeds runtime velocity limit"
                    )
                if time_segment > float(
                    franka["default_max_joint_segment_rad"]
                ) + 1.0e-15:
                    blockers.append(
                        "loaded-lift time-law exceeds runtime segment limit"
                    )
                if time_duration + 1.0e-15 < float(
                    franka["default_min_duration_s"]
                ):
                    blockers.append(
                        "loaded-lift time-law duration is below runtime minimum"
                    )
    if not bool(config["tool"].get("installed_on_franka_verified", False)):
        blockers.append(
            "adapter + RH56 are not explicitly verified as mounted on Franka"
        )
    if not str(franka.get("ip", "")).strip():
        blockers.append("Franka IP is empty")
    if not str(inspire.get("port", "")).strip():
        blockers.append("Inspire serial port is empty")

    if float(inspire["open_speed"]) != EXPECTED_RH56_SPEED or float(
        inspire["close_speed"]
    ) != EXPECTED_RH56_SPEED:
        blockers.append("reviewed RH56 driver requires open_speed=close_speed=40")
    if float(inspire["force_limit_g"]) != EXPECTED_RH56_FORCE_G:
        blockers.append("reviewed RH56 driver requires force_limit_g=80")
    timeout = float(inspire["motion_timeout_s"])
    if not 5.0 <= timeout <= 30.0:
        blockers.append("RH56 motion_timeout_s must be in 5..30")
    tolerance = float(inspire["arrival_tolerance_units"])
    if tolerance != round(tolerance) or not 0 <= tolerance <= 100:
        blockers.append("RH56 arrival_tolerance_units must be an integer in 0..100")
    q6_range = np.asarray(
        inspire["thumb_rotate_validated_realtime_range"], dtype=np.float64
    )
    if not np.all(q6_range == np.rint(q6_range)):
        blockers.append("validated q6 range endpoints must be integers")

    expected_ee = franka.get("expected_end_effector", {})
    try:
        mass = float(expected_ee.get("mass_kg"))
        center = np.asarray(expected_ee.get("F_x_Cee_m"), dtype=np.float64)
        inertia = np.asarray(expected_ee.get("inertia_kg_m2"), dtype=np.float64)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError
        if inertia.shape not in ((9,), (3, 3)) or not np.all(np.isfinite(inertia)):
            raise ValueError
    except (TypeError, ValueError):
        blockers.append("installed end-effector mass/CoM/inertia values are malformed")

    if command in ("grasp", "grasp-lift", "air-grasp"):
        # Cartesian interpolation and its generic settle gate still require
        # explicit robot-base workspace bounds.  Exact schema-v2 audited-joint
        # execution instead uses the separate q/FK/EEF read-only settle path
        # checked above, so null Cartesian bounds do not create a late fault.
        workspace_min = franka.get("cartesian_workspace_min_m")
        workspace_max = franka.get("cartesian_workspace_max_m")
        if trajectory_mode == "cartesian" and (
            workspace_min is None or workspace_max is None
        ):
            blockers.append(
                "pregrasp/final EEF settle verification requires commissioned "
                "Cartesian workspace bounds"
            )
        expected_mode = (
            "loaded_grasp"
            if command in ("grasp", "grasp-lift")
            else "air_grasp"
        )
        if (
            inspection.installed_audit is not None
            and inspection.installed_audit.artifact is not None
            and inspection.installed_audit.artifact["mode"] != expected_mode
        ):
            blockers.append(
                "installed-tool audit mode {} cannot execute {}".format(
                    inspection.installed_audit.artifact["mode"], command
                )
            )
        if inspection.plan is None:
            blockers.append("no staged grasp plan was built")
        else:
            thumb_stages = [
                stage
                for stage in inspection.plan.stages
                if stage.name == StageName.THUMB_PRESHAPE
            ]
            if len(thumb_stages) != 1:
                blockers.append("full grasp requires exactly one explicit q6 preshape stage")
            close_stages = [
                stage
                for stage in inspection.plan.stages
                if stage.name == StageName.INSPIRE_CLOSE
            ]
            if len(close_stages) == 1:
                targets = np.asarray(close_stages[0].inspire_angles, dtype=np.float64)
                if not np.all(targets == np.rint(targets)):
                    blockers.append("Inspire close targets must be integer registers")
                elif not q6_range[0] <= targets[5] <= q6_range[1]:
                    blockers.append(
                        "planned q6 target is outside the commissioned realtime range"
                    )
                if command == "air-grasp":
                    try:
                        validate_air_candidate_commissioning(config, targets)
                    except ValueError as exc:
                        message = str(exc)
                        if "no exact commissioned air-closure evidence" in message:
                            message = (
                                "planned six-axis air-closure target has no exact "
                                "commissioning evidence"
                            )
                        blockers.append(message)
                else:
                    commissioned_targets = (
                        config.get("loaded_lift", {}).get(
                            "commissioned_closed_hand_targets", []
                        )
                        if command == "grasp-lift"
                        else inspire.get("commissioned_air_closure_targets", [])
                    )
                    exact_targets = {
                        tuple(int(value) for value in target)
                        for target in commissioned_targets
                    }
                    if tuple(int(value) for value in targets) not in exact_targets:
                        blockers.append(
                            "planned six-axis loaded-closure target has no exact "
                            "commissioning evidence"
                        )
        material = str(config["tool"].get("material", "")).strip().lower()
        if command in ("grasp", "grasp-lift") and material == "pla":
            blockers.append("loaded grasp execution refuses the installed PLA adapter")
        # The sparse adapter point audit remains a diagnostic printout only;
        # it can neither replace nor override the installed-tool sidecar.
    return tuple(blockers)


def _confirmation_blockers(args: argparse.Namespace) -> Tuple[str, ...]:
    expected = [
        ("confirm_workspace_clear", WORKSPACE_CLEAR_TOKEN),
        ("confirm_immediate_stop", IMMEDIATE_STOP_TOKEN),
    ]
    if args.command == "default":
        expected.append(("confirm_pla_low_speed", PLA_COMMISSIONING_TOKEN))
    elif args.command == "pregrasp":
        expected.append(("confirm_pregrasp_only", PREGRASP_ONLY_TOKEN))
    elif args.command == "grasp-lift":
        expected.extend(
            [
                ("confirm_loaded_lift_round_trip", LOADED_LIFT_TOKEN),
                ("confirm_setdown_support", LOAD_SUPPORT_TOKEN),
                ("confirm_load_rated_tool", LOAD_RATED_TOKEN),
                ("confirm_q6_preshape", Q6_PRESHAPE_TOKEN),
                (
                    "confirm_installed_collision_model",
                    COLLISION_MODEL_TOKEN,
                ),
            ]
        )
    else:
        expected.append(
            (
                "confirm_full_grasp" if args.command == "grasp" else "confirm_air_grasp",
                FULL_GRASP_TOKEN if args.command == "grasp" else AIR_GRASP_TOKEN,
            )
        )
        expected.extend(
            [
                ("confirm_q6_preshape", Q6_PRESHAPE_TOKEN),
                ("confirm_installed_collision_model", COLLISION_MODEL_TOKEN),
            ]
        )
    blockers = []
    for attribute, token in expected:
        if getattr(args, attribute, None) != token:
            blockers.append(
                "{} must exactly equal {}".format(attribute.replace("_", "-"), token)
            )
    return tuple(blockers)


def _conditional_audit_blockers(
    args: argparse.Namespace, inspection: OfflineInspection
) -> Tuple[str, ...]:
    """Require the exact per-run environment authority token.

    The full air-grasp schema and the dedicated pregrasp schema use different
    decision keys.  Keep those meanings separate: an old scene timestamp is
    advisory for pregrasp, but the exact workspace token is never optional.
    """

    binding = (
        inspection.pregrasp_audit
        if args.command == "pregrasp"
        else inspection.installed_audit
    )
    if binding is None or binding.artifact is None:
        return ()
    decision = binding.artifact.get("decision", {})
    if args.command == "pregrasp":
        if decision.get("runtime_workspace_clear_required") is not True:
            return (
                "pregrasp-only audit must require a fresh runtime workspace-clear confirmation",
            )
    elif decision.get("runtime_operator_workspace_clear_required") is not True:
        return ()
    blockers = []
    if args.command == "pregrasp":
        if binding.artifact.get("execution_scope") != "open_hand_current_to_pregrasp_only":
            blockers.append("conditional workspace evidence has the wrong pregrasp scope")
    elif args.command != "air-grasp" or binding.artifact.get("mode") != "air_grasp":
        blockers.append(
            "conditional workspace evidence is consumable only by air-grasp"
        )
    if getattr(args, "confirm_workspace_clear", None) != WORKSPACE_CLEAR_TOKEN:
        blockers.append(
            "conditional workspace audit requires confirm-workspace-clear exactly {}".format(
                WORKSPACE_CLEAR_TOKEN
            )
        )
    return tuple(blockers)


def _load_hardware_types() -> Dict[str, Any]:
    """Lazy imports reached only after all offline gates have passed."""

    from anydex_pipeline.control_sequence import (
        AuditedJointSequencePlan,
        AuditedLoadedLiftSequencePlan,
        AuditedPregraspSequencePlan,
        DefaultSequencePlan,
        LoadedLiftPayload,
        SettleTolerances,
        StagedGraspSequence,
    )
    from anydex_pipeline.franka_sequence_driver import (
        FrankaMotionLimits,
        FrankaSequenceDriver,
    )
    from anydex_pipeline.inspire_sequence_driver import RH56SequenceDriver

    return {
        "DefaultSequencePlan": DefaultSequencePlan,
        "AuditedJointSequencePlan": AuditedJointSequencePlan,
        "AuditedLoadedLiftSequencePlan": AuditedLoadedLiftSequencePlan,
        "AuditedPregraspSequencePlan": AuditedPregraspSequencePlan,
        "LoadedLiftPayload": LoadedLiftPayload,
        "SettleTolerances": SettleTolerances,
        "StagedGraspSequence": StagedGraspSequence,
        "FrankaMotionLimits": FrankaMotionLimits,
        "FrankaSequenceDriver": FrankaSequenceDriver,
        "RH56SequenceDriver": RH56SequenceDriver,
    }


def _build_franka_limits(
    config: Mapping[str, Any],
    limits_type: Any,
    *,
    audited_tracking_tolerance_rad: Optional[float] = None,
) -> Any:
    franka = config["franka"]
    end_effector = franka["expected_end_effector"]
    workspace_min = franka.get("cartesian_workspace_min_m")
    workspace_max = franka.get("cartesian_workspace_max_m")
    return limits_type(
        expected_F_T_EE=np.asarray(franka["expected_F_T_EE"], dtype=np.float64),
        expected_m_ee_kg=float(end_effector["mass_kg"]),
        expected_F_x_Cee_m=np.asarray(
            end_effector["F_x_Cee_m"], dtype=np.float64
        ),
        expected_I_ee_kg_m2=np.asarray(
            end_effector["inertia_kg_m2"], dtype=np.float64
        ),
        joint_limits_rad=np.asarray(franka["joint_limits_rad"], dtype=np.float64),
        joint_limit_margin_rad=float(franka["joint_limit_margin_rad"]),
        max_joint_speed_rad_s=float(franka["default_max_joint_velocity_rad_s"]),
        max_joint_segment_rad=float(franka["default_max_joint_segment_rad"]),
        min_joint_duration_s=float(franka["default_min_duration_s"]),
        workspace_min_m=(
            None if workspace_min is None else np.asarray(workspace_min, dtype=np.float64)
        ),
        workspace_max_m=(
            None if workspace_max is None else np.asarray(workspace_max, dtype=np.float64)
        ),
        max_cartesian_speed_m_s=float(
            franka["cartesian_max_translation_velocity_m_s"]
        ),
        max_angular_speed_rad_s=float(
            franka["cartesian_max_rotation_velocity_rad_s"]
        ),
        max_segment_translation_m=float(
            franka["cartesian_max_segment_translation_m"]
        ),
        max_segment_rotation_rad=float(franka["cartesian_max_segment_rotation_rad"]),
        min_cartesian_duration_s=float(
            franka["cartesian_min_segment_duration_s"]
        ),
        joint_arrival_tolerance_rad=(
            float(franka["default_arrival_tolerance_rad"])
            if audited_tracking_tolerance_rad is None
            else min(
                float(franka["default_arrival_tolerance_rad"]),
                float(audited_tracking_tolerance_rad),
            )
        ),
        max_continuous_joint_tracking_error_rad=audited_tracking_tolerance_rad,
        translation_arrival_tolerance_m=float(
            franka["cartesian_arrival_position_tolerance_m"]
        ),
        rotation_arrival_tolerance_rad=float(
            franka["cartesian_arrival_rotation_tolerance_rad"]
        ),
        settle_time_s=float(franka["settle_time_s"]),
        settle_timeout_s=float(franka["settle_timeout_s"]),
        settle_poll_s=float(franka["settle_poll_s"]),
    )


def _connect_hardware(
    config: Mapping[str, Any], hardware_types: Mapping[str, Any], limits: Any
) -> Tuple[Any, Any]:
    """Connect hand, immediately disable it, then connect the arm."""

    inspire = config["inspire"]
    hand_type = hardware_types["RH56SequenceDriver"]
    arm_type = hardware_types["FrankaSequenceDriver"]
    hand = hand_type.connect(
        port=str(inspire["port"]),
        baud=int(inspire["baud"]),
        hand_id=int(inspire["hand_id"]),
        thumb_rotate_range=tuple(
            int(value) for value in inspire["thumb_rotate_validated_realtime_range"]
        ),
        motion_timeout_s=float(inspire["motion_timeout_s"]),
        angle_tolerance=int(inspire["arrival_tolerance_units"]),
    )
    try:
        # This is deliberately the first post-connect hardware action.  A
        # numeric ANGLE_SET left by an earlier process must not remain active
        # while Franka construction and read-only gates run.
        hand.adopt_disabled_state_and_verify()
        # Formal motion must fail before a control handle is created unless
        # libfranka can acquire the required real-time scheduling priority.
        arm = arm_type.connect(
            str(config["franka"]["ip"]),
            limits,
            enforce_realtime=True,
        )
    except BaseException as primary:
        try:
            hand.close()
        except BaseException as stop_error:
            raise RuntimeError(
                "hardware connection failed and RH56 stop is unconfirmed: {}; "
                "disable/close failed: {}".format(primary, stop_error)
            ) from primary
        raise
    return arm, hand


def _verify_installed_end_effector(
    arm: Any, config: Mapping[str, Any]
) -> Tuple[Any, int, int]:
    """Run the reviewed read-only arm gate and compare installed dynamics."""

    state = arm.robot.read_once()
    sampled_unix_ns = int(time.time_ns())
    sampled_monotonic_ns = int(time.monotonic_ns())
    validator = getattr(arm, "_validate_state", None)
    if not callable(validator):
        raise RuntimeError("Franka driver exposes no reviewed read-only state gate")
    # This checks Idle mode, errors, contacts/collisions, joint margins,
    # reported F_T_EE, and timing/success state before Inspire is allowed to
    # open.  No controller or robot configuration command is created.
    validator(state, require_idle=True, enforce_success=False)
    expected = config["franka"]["expected_end_effector"]
    expected_external_mass = float(
        getattr(arm, "expected_external_load_mass_kg", 0.0)
    )
    comparisons = (
        (
            "m_ee",
            np.asarray([float(getattr(state, "m_ee"))]),
            np.asarray([float(expected["mass_kg"])]),
            1.0e-4,
        ),
        (
            "F_x_Cee",
            np.asarray(getattr(state, "F_x_Cee"), dtype=np.float64),
            np.asarray(expected["F_x_Cee_m"], dtype=np.float64),
            1.0e-4,
        ),
        (
            "I_ee",
            np.asarray(getattr(state, "I_ee"), dtype=np.float64).reshape(-1),
            np.asarray(expected["inertia_kg_m2"], dtype=np.float64).reshape(-1),
            1.0e-5,
        ),
        (
            "m_load",
            np.asarray([float(getattr(state, "m_load"))]),
            np.asarray([expected_external_mass]),
            1.0e-5,
        ),
        (
            "m_total",
            np.asarray([float(getattr(state, "m_total"))]),
            np.asarray(
                [float(expected["mass_kg"]) + expected_external_mass]
            ),
            1.0e-4,
        ),
    )
    for name, actual, wanted, tolerance in comparisons:
        if actual.shape != wanted.shape or not np.all(np.isfinite(actual)):
            raise RuntimeError("Franka {} feedback is missing or malformed".format(name))
        if not np.allclose(actual, wanted, atol=tolerance, rtol=0.0):
            raise RuntimeError(
                "Franka installed {} differs from commissioned value: actual={} "
                "expected={}".format(name, actual.tolist(), wanted.tolist())
            )
    # Reuse this exact already-read, already-validated RobotState when the
    # opt-in native producer is installed.  This adds no FCI read and is also
    # exercised by the RH56 driver's existing external safety gate.
    publisher = getattr(arm, "publish_native_telemetry_state", None)
    if callable(publisher):
        publisher(state)
    return state, sampled_unix_ns, sampled_monotonic_ns


def _install_continuous_franka_gate(
    hand: Any, arm: Any, config: Mapping[str, Any]
) -> None:
    """Bind every subsequent RH56 feedback read to a fresh Franka safety gate."""

    installer = getattr(hand, "install_external_safety_check", None)
    if not callable(installer):
        raise RuntimeError("RH56 driver exposes no external safety-check installer")

    def check() -> None:
        _verify_installed_end_effector(arm, config)

    installer(check)


def _require_passing_installed_audit(
    inspection: OfflineInspection,
) -> Mapping[str, Any]:
    binding = inspection.installed_audit
    if binding is None or not binding.passed or binding.artifact is None:
        raise RuntimeError("passing installed-tool audit artifact is unavailable")
    return binding.artifact


def _require_passing_pregrasp_audit(
    inspection: OfflineInspection,
) -> Mapping[str, Any]:
    binding = inspection.pregrasp_audit
    if binding is None or not binding.passed or binding.artifact is None:
        raise RuntimeError("passing dedicated pregrasp-only audit is unavailable")
    return binding.artifact


def _require_passing_loaded_lift_audit(
    inspection: OfflineInspection,
) -> LoadedLiftAuditBinding:
    binding = inspection.loaded_lift_audit
    if binding is None or not binding.passed or binding.artifact is None:
        raise RuntimeError("passing independent loaded-lift audit is unavailable")
    if binding.base_artifact is None:
        raise RuntimeError("loaded-lift audit has no replayed base loaded-grasp audit")
    return binding


def _verify_live_q_matches_audit(arm: Any, artifact: Mapping[str, Any]) -> None:
    """Read-only final gate reached before either device is allowed to move."""

    state = arm.robot.read_once()
    validator = getattr(arm, "_validate_state", None)
    if not callable(validator):
        raise RuntimeError("Franka driver exposes no reviewed read-only state gate")
    validator(state, require_idle=True, enforce_success=False)
    actual = np.asarray(getattr(state, "q", None), dtype=np.float64)
    first = artifact["bindings"]["joint_path"]["waypoints"][0]
    if first["name"] != "current":
        raise RuntimeError("installed audit has no current joint waypoint")
    expected = np.asarray(first["q_rad"], dtype=np.float64)
    tolerance = float(artifact["policies"]["max_q_tracking_error_rad"])
    if actual.shape != (7,) or not np.all(np.isfinite(actual)):
        raise RuntimeError("Franka current q feedback is missing or malformed")
    error = float(np.max(np.abs(actual - expected)))
    if error > tolerance:
        raise RuntimeError(
            "Franka moved since collision audit: current-q error {:.6f}rad "
            "exceeds {:.6f}rad".format(error, tolerance)
        )
    now = time.time()
    scene = artifact["bindings"]["scene"]
    scene_age = now - float(scene["captured_at_s"])
    if scene_age < -0.05 or scene_age > float(
        artifact["policies"]["max_scene_age_s"]
    ):
        raise RuntimeError("audited scene expired before hardware preflight completed")


def _verify_live_q_matches_pregrasp_audit(
    arm: Any, artifact: Mapping[str, Any]
) -> None:
    """Bind the live stationary state to the dedicated prefix first waypoint."""

    state = arm.robot.read_once()
    validator = getattr(arm, "_validate_state", None)
    if not callable(validator):
        raise RuntimeError("Franka driver exposes no reviewed read-only state gate")
    validator(state, require_idle=True, enforce_success=False)
    actual = np.asarray(getattr(state, "q", None), dtype=np.float64)
    prefix = artifact["bindings"]["joint_prefix"]
    first = prefix["waypoints"][0]
    if first["name"] != "current":
        raise RuntimeError("pregrasp-only audit has no current joint waypoint")
    expected = np.asarray(first["q_rad"], dtype=np.float64)
    tolerance = float(prefix["max_q_tracking_error_rad"])
    if actual.shape != (7,) or not np.all(np.isfinite(actual)):
        raise RuntimeError("Franka current q feedback is missing or malformed")
    error = float(np.max(np.abs(actual - expected)))
    if error > tolerance:
        raise RuntimeError(
            "Franka moved since pregrasp-only audit: current-q error {:.6f}rad "
            "exceeds {:.6f}rad".format(error, tolerance)
        )
    # The bound scene return is advisory in this dedicated mode.  Environment
    # freshness comes from the exact per-run workspace-clear confirmation;
    # live-q remains an uncompromised machine-state gate above.
    captured_at = float(
        artifact["bindings"]["filtered_scene"]["captured_at_s"]
    )
    age_s = time.time() - captured_at
    capture_delta = float(
        artifact["bindings"]["filtered_scene"][
            "capture_to_prefix_current_linf_rad"
        ]
    )
    print(
        "[pregrasp-only/advisory] filtered-scene age={:.1f}s is not an "
        "execution gate; scene/prefix q delta={:.7f}rad is provenance only; "
        "current live-q and exact workspace-clear confirmation remain "
        "required".format(age_s, capture_delta)
    )


def _build_audited_joint_plan(
    inspection: OfflineInspection,
    plan_type: Any,
    settle: Any,
) -> Any:
    artifact = _require_passing_installed_audit(inspection)
    settle_binding_blockers = _audited_joint_settle_binding_blockers(inspection)
    if settle_binding_blockers:
        raise RuntimeError("; ".join(settle_binding_blockers))
    candidate = artifact["bindings"]["snapshot"]["selected_candidate"]
    execution = artifact["bindings"]["execution_plan"]
    waypoints = tuple(
        (item["name"], np.asarray(item["q_rad"], dtype=np.float64))
        for item in artifact["bindings"]["joint_path"]["waypoints"]
    )
    targets = tuple(int(round(value)) for value in candidate["hand_targets"])
    return plan_type(
        execution_eligible=True,
        mode=artifact["mode"],
        contact_and_lift_forbidden=bool(
            execution["contact_and_lift_forbidden"]
        ),
        audit_schema_version=int(artifact["schema_version"]),
        joint_pose_binding_verified=True,
        joint_waypoints=waypoints,
        pregrasp_pose=np.asarray(
            execution["pregrasp_pose_base_EE"], dtype=np.float64
        ),
        grasp_pose=np.asarray(execution["grasp_pose_base_EE"], dtype=np.float64),
        hand_target6=targets,
        max_q_tracking_error_rad=float(
            artifact["policies"]["max_q_tracking_error_rad"]
        ),
        settle_tolerances=settle,
        thumb_preshape_target=targets[5],
    )


def _build_loaded_lift_plan(
    inspection: OfflineInspection,
    hardware_types: Mapping[str, Any],
    base_settle: Any,
) -> Any:
    binding = _require_passing_loaded_lift_audit(inspection)
    artifact = binding.artifact
    base = binding.base_artifact
    assert artifact is not None and base is not None
    candidate = base["bindings"]["snapshot"]["selected_candidate"]
    execution = base["bindings"]["execution_plan"]
    targets = tuple(int(round(value)) for value in candidate["hand_targets"])
    grasp_plan = hardware_types["AuditedJointSequencePlan"](
        execution_eligible=True,
        mode="loaded_grasp",
        contact_and_lift_forbidden=False,
        audit_schema_version=int(base["schema_version"]),
        joint_pose_binding_verified=True,
        joint_waypoints=tuple(
            (item["name"], np.asarray(item["q_rad"], dtype=np.float64))
            for item in base["bindings"]["joint_path"]["waypoints"]
        ),
        pregrasp_pose=np.asarray(
            execution["pregrasp_pose_base_EE"], dtype=np.float64
        ),
        grasp_pose=np.asarray(
            execution["grasp_pose_base_EE"], dtype=np.float64
        ),
        hand_target6=targets,
        max_q_tracking_error_rad=float(
            base["policies"]["max_q_tracking_error_rad"]
        ),
        settle_tolerances=base_settle,
        thumb_preshape_target=targets[5],
    )
    payload = hardware_types["LoadedLiftPayload"](
        mass_kg=float(binding.payload_mass_kg),
        F_x_Cload_m=np.asarray(binding.payload_F_x_Cload_m, dtype=np.float64),
        I_load_kg_m2=np.asarray(
            binding.payload_inertia_kg_m2, dtype=np.float64
        ),
        binding_sha256=str(binding.payload_binding_sha256),
    )
    settle_values = binding.settle_tolerances
    if settle_values is None:
        raise RuntimeError("loaded-lift audit has no settle tolerance binding")
    lift_settle = hardware_types["SettleTolerances"](
        position_m=float(settle_values["position_m"]),
        orientation_rad=float(settle_values["orientation_rad"]),
        linear_speed_m_s=float(settle_values["linear_speed_m_s"]),
        angular_speed_rad_s=float(settle_values["angular_speed_rad_s"]),
        stable_seconds=float(settle_values["stable_seconds"]),
    )
    if (
        binding.artifact_sha256 is None
        or binding.lift_pose is None
        or binding.minimum_contact_axes is None
        or binding.max_q_tracking_error_rad is None
    ):
        raise RuntimeError("loaded-lift audit execution fields are incomplete")
    time_law = {
        "velocity": getattr(binding, "time_law_max_joint_velocity_rad_s", None),
        "acceleration": getattr(
            binding, "time_law_max_joint_acceleration_rad_s2", None
        ),
        "segment": getattr(binding, "time_law_max_dynamic_segment_rad", None),
        "duration": getattr(binding, "time_law_min_segment_duration_s", None),
    }
    approved_velocity = getattr(
        binding, "approved_max_joint_velocity_rad_s", None
    )
    approved_acceleration = getattr(
        binding, "approved_max_joint_acceleration_rad_s2", None
    )
    if any(value is None for value in time_law.values()) or (
        approved_velocity is None or approved_acceleration is None
    ):
        raise RuntimeError("loaded-lift audit has no complete execution time-law")
    time_law = {name: float(value) for name, value in time_law.items()}
    approved_velocity = float(approved_velocity)
    approved_acceleration = float(approved_acceleration)
    if time_law["velocity"] > approved_velocity + 1.0e-15 or (
        time_law["acceleration"] > approved_acceleration + 1.0e-15
    ):
        raise RuntimeError("loaded-lift time-law exceeds material approval")
    franka = inspection.config["franka"]
    if time_law["velocity"] > float(
        franka["default_max_joint_velocity_rad_s"]
    ) + 1.0e-15:
        raise RuntimeError("loaded-lift time-law exceeds runtime velocity limit")
    if time_law["segment"] > float(
        franka["default_max_joint_segment_rad"]
    ) + 1.0e-15:
        raise RuntimeError("loaded-lift time-law exceeds runtime segment limit")
    if time_law["duration"] + 1.0e-15 < float(
        franka["default_min_duration_s"]
    ):
        raise RuntimeError("loaded-lift time-law duration is below runtime minimum")
    return hardware_types["AuditedLoadedLiftSequencePlan"](
        execution_eligible=True,
        loaded_lift_audit_schema_version=int(artifact["schema_version"]),
        loaded_lift_binding_verified=True,
        loaded_lift_artifact_sha256=binding.artifact_sha256,
        grasp_plan=grasp_plan,
        lift_waypoints=binding.lift_waypoints,
        lift_pose=binding.lift_pose,
        payload=payload,
        minimum_contact_axes=binding.minimum_contact_axes,
        round_trip_setdown_required=True,
        max_q_tracking_error_rad=binding.max_q_tracking_error_rad,
        time_law_max_joint_velocity_rad_s=time_law["velocity"],
        time_law_max_joint_acceleration_rad_s2=time_law["acceleration"],
        time_law_max_dynamic_segment_rad=time_law["segment"],
        time_law_min_segment_duration_s=time_law["duration"],
        settle_tolerances=lift_settle,
    )


def _build_audited_pregrasp_plan(
    inspection: OfflineInspection,
    plan_type: Any,
    settle: Any,
) -> Any:
    artifact = _require_passing_pregrasp_audit(inspection)
    prefix = artifact["bindings"]["joint_prefix"]
    return plan_type(
        execution_eligible=True,
        audit_schema_version=int(artifact["schema_version"]),
        audit_artifact_sha256=str(artifact["artifact_sha256"]),
        prefix_contract_sha256=str(prefix["prefix_contract_sha256"]),
        joint_path_samples_sha256=str(prefix["samples_sha256"]),
        joint_waypoints=tuple(
            (item["name"], np.asarray(item["q_rad"], dtype=np.float64))
            for item in prefix["waypoints"]
        ),
        pregrasp_pose=np.asarray(
            prefix["pregrasp_pose_base_EE"], dtype=np.float64
        ),
        max_joint_step_rad=float(prefix["max_joint_step_rad"]),
        max_q_tracking_error_rad=float(prefix["max_q_tracking_error_rad"]),
        settle_tolerances=settle,
    )


def _state_pose_base_EE(state: Any) -> np.ndarray:
    """Decode libfranka's column-major O_T_EE at a non-realtime boundary."""

    raw = np.asarray(getattr(state, "O_T_EE"), dtype=np.float64)
    if raw.shape == (16,):
        pose = raw.reshape(4, 4, order="F")
    elif raw.shape == (4, 4):
        pose = raw.copy()
    else:
        raise RuntimeError("Franka boundary O_T_EE is not a finite 4x4 transform")
    if not np.all(np.isfinite(pose)):
        raise RuntimeError("Franka boundary O_T_EE is not a finite 4x4 transform")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-8, rtol=0.0):
        raise RuntimeError("Franka boundary O_T_EE has an invalid homogeneous row")
    rotation = pose[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6, rtol=0.0):
        raise RuntimeError("Franka boundary O_T_EE rotation is not in SO(3)")
    return pose


def _pose_tracking_error(
    actual: np.ndarray, target: np.ndarray
) -> Tuple[float, float]:
    expected = np.asarray(target, dtype=np.float64)
    if expected.shape != (4, 4) or not np.all(np.isfinite(expected)):
        raise RuntimeError("boundary target pose is not a finite 4x4 transform")
    position = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
    relative = actual[:3, :3].T @ expected[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return position, float(np.arccos(cosine))


def _atomic_write_pose_state(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one small JSON sample outside all realtime callbacks."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        ".{}.{}.tmp".format(destination.name, os.getpid())
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    try:
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _build_pose_state_observer(path: Path, arm: Any, hand: Any) -> Any:
    """Return a waypoint-rate publisher using the existing device owners.

    The observer is called only after a move/settle/hand method has returned;
    it never runs in libfranka's 1 kHz callback and never creates a second FCI
    or serial client.
    """

    destination = Path(path).expanduser().resolve()
    sequence_number = 0

    def observe(
        stage: str,
        target_q: Optional[np.ndarray],
        target_pose: Optional[np.ndarray],
        hand_targets: Optional[Tuple[int, ...]],
    ) -> None:
        nonlocal sequence_number
        state = arm.robot.read_once()
        validator = getattr(arm, "_validate_state", None)
        if not callable(validator):
            raise RuntimeError(
                "Franka driver exposes no reviewed boundary state validator"
            )
        validator(state, require_idle=True, enforce_success=False)
        actual_q = np.asarray(getattr(state, "q"), dtype=np.float64)
        if actual_q.shape != (7,) or not np.all(np.isfinite(actual_q)):
            raise RuntimeError("Franka boundary q is not a finite seven-vector")
        actual_pose = _state_pose_base_EE(state)

        hand_reader = getattr(hand, "read_state_snapshot", None)
        if not callable(hand_reader):
            raise RuntimeError("Inspire driver exposes no read-only state snapshot")
        hand_state = hand_reader()
        actual_hand = tuple(int(value) for value in hand_state.angles)
        actual_targets = tuple(int(value) for value in hand_state.angle_targets)
        if len(actual_hand) != 6 or len(actual_targets) != 6:
            raise RuntimeError("Inspire boundary state is not six-axis")

        sequence_number += 1
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "reference_frame": "robot_base",
            "timestamp_unix_s": float(time.time()),
            "T_reference_EE": actual_pose.tolist(),
            "stage": str(stage),
            "source": "execute_control_sequence_waypoint_boundary",
            "sequence": sequence_number,
            "q_rad": actual_q.tolist(),
            "hand": {
                "angle_targets": list(actual_targets),
                "angles": list(actual_hand),
                "positions": (
                    None
                    if hand_state.positions is None
                    else [int(value) for value in hand_state.positions]
                ),
                "forces": (
                    None
                    if hand_state.forces is None
                    else [int(value) for value in hand_state.forces]
                ),
                "currents_mA": [int(value) for value in hand_state.currents],
                "errors": [int(value) for value in hand_state.errors],
                "statuses": [int(value) for value in hand_state.statuses],
                "temperatures_C": [
                    int(value) for value in hand_state.temperatures
                ],
                "contact_axes": [str(value) for value in hand_state.contact_axes],
            },
            "target": {},
            "tracking_error": {},
        }
        if target_q is not None:
            expected_q = np.asarray(target_q, dtype=np.float64)
            if expected_q.shape != (7,) or not np.all(np.isfinite(expected_q)):
                raise RuntimeError("boundary target q is not a finite seven-vector")
            payload["target"]["q_rad"] = expected_q.tolist()
            payload["tracking_error"]["q_linf_rad"] = float(
                np.max(np.abs(actual_q - expected_q))
            )
        if target_pose is not None:
            expected_pose = np.asarray(target_pose, dtype=np.float64)
            position_error, rotation_error = _pose_tracking_error(
                actual_pose, expected_pose
            )
            payload["target"]["T_reference_EE"] = expected_pose.tolist()
            payload["tracking_error"]["position_m"] = position_error
            payload["tracking_error"]["rotation_rad"] = rotation_error
        if hand_targets is not None:
            expected_hand = tuple(int(value) for value in hand_targets)
            if len(expected_hand) != 6 or any(
                value < -1 or value > 1000 for value in expected_hand
            ):
                raise RuntimeError("boundary hand target is not a valid six-vector")
            axis_errors = [
                None if target < 0 else abs(actual - target)
                for actual, target in zip(actual_hand, expected_hand)
            ]
            numeric_errors = [value for value in axis_errors if value is not None]
            payload["target"]["hand_targets"] = list(expected_hand)
            payload["tracking_error"]["hand_abs_units"] = axis_errors
            payload["tracking_error"]["hand_max_abs_units"] = (
                None if not numeric_errors else max(numeric_errors)
            )

        _atomic_write_pose_state(destination, payload)
        print(
            "[pose-state] sequence={} stage={} path={}".format(
                sequence_number, stage, destination
            )
        )

    return observe


def _monitor_loaded_hold(
    sequence: Any,
    duration_s: float,
    *,
    poll_interval_s: float = LOADED_HOLD_POLL_SECONDS,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> None:
    """Refresh loaded arm/hand proofs at a bounded rate during lifted hold."""

    duration = float(duration_s)
    interval = float(poll_interval_s)
    if not np.isfinite(duration) or not 0.0 <= duration <= MAX_HOLD_SECONDS:
        raise ValueError("loaded hold duration is outside the reviewed bound")
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("loaded hold poll interval must be finite and positive")
    deadline = float(monotonic()) + duration
    while True:
        sequence.verify_loaded_lift_holding()
        remaining = deadline - float(monotonic())
        if remaining <= 0.0:
            return
        sleep(min(interval, remaining))


def _monitor_bounded_hold(
    sequence: Any,
    duration_s: float,
    *,
    poll_interval_s: float = BOUNDED_HOLD_POLL_SECONDS,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> None:
    """Refresh an ordinary grasp/air-grasp hold at a bounded low rate.

    The first verification is unconditional, including for a zero-duration
    hold.  Sleeping only spaces subsequent reads; elapsed time is always taken
    from the injected monotonic clock rather than inferred from sleep calls.
    """

    duration = float(duration_s)
    interval = float(poll_interval_s)
    if not np.isfinite(duration) or not 0.0 <= duration <= MAX_HOLD_SECONDS:
        raise ValueError("bounded hold duration is outside the reviewed bound")
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("bounded hold poll interval must be finite and positive")
    deadline = float(monotonic()) + duration
    while True:
        sequence.verify_bounded_holding()
        remaining = deadline - float(monotonic())
        if remaining <= 0.0:
            return
        sleep(min(interval, remaining))


def _load_continuous_telemetry_request(
    args: argparse.Namespace, inspection: OfflineInspection
) -> Optional[ExecutionTelemetryRequest]:
    """Validate an opt-in mapping/manifest pair without native imports."""

    mapping_path = getattr(args, "continuous_telemetry", None)
    manifest_path = getattr(args, "telemetry_session_manifest", None)
    if (mapping_path is None) != (manifest_path is None):
        raise ValueError(
            "--continuous-telemetry and --telemetry-session-manifest must be supplied together"
        )
    if mapping_path is None:
        return None
    python_dir = getattr(args, "continuous_telemetry_python_dir", None)
    if python_dir is None:
        raise ValueError(
            "continuous telemetry requires --continuous-telemetry-python-dir "
            "or ANYDEX_TELEMETRY_PYTHON_DIR"
        )
    if args.command == "inspect":
        return load_execution_telemetry_request(
            mapping_path=mapping_path,
            manifest_path=manifest_path,
            python_dir=python_dir,
        )
    if inspection.snapshot is None:
        raise ValueError(
            "continuous telemetry formal/dry-run execution requires a grasp snapshot"
        )
    return load_execution_telemetry_request(
        mapping_path=mapping_path,
        manifest_path=manifest_path,
        python_dir=python_dir,
        expected_command=str(args.command),
        expected_control_config_path=inspection.config_path,
        expected_snapshot_path=getattr(args, "snapshot", None),
        expected_installed_tool_audit_path=getattr(
            args, "installed_tool_audit", None
        ),
        expected_pregrasp_only_audit_path=getattr(
            args, "pregrasp_only_audit", None
        ),
        expected_loaded_lift_audit_path=getattr(
            args, "loaded_lift_audit", None
        ),
        expected_selected_index=int(inspection.snapshot.grasps.selected_index),
    )


def _start_continuous_telemetry(
    request: ExecutionTelemetryRequest,
    *,
    arm: Any,
    hand: Any,
    initial_validated_arm_state: Any,
    initial_arm_timestamp_unix_ns: int,
    initial_arm_timestamp_monotonic_ns: int,
    config: Mapping[str, Any],
) -> ContinuousTelemetryRuntime:
    """Start only after all existing live gates and final Desk check pass."""

    return ContinuousTelemetryRuntime.start(
        request,
        arm=arm,
        hand=hand,
        initial_validated_arm_state=initial_validated_arm_state,
        initial_arm_timestamp_unix_ns=initial_arm_timestamp_unix_ns,
        initial_arm_timestamp_monotonic_ns=initial_arm_timestamp_monotonic_ns,
        robot_id="fr3:{};rh56:{}".format(
            config["franka"]["ip"], config["inspire"]["hand_id"]
        ),
    )


def _execute_hardware(
    args: argparse.Namespace,
    inspection: OfflineInspection,
    telemetry_request: Optional[ExecutionTelemetryRequest] = None,
) -> int:
    """Import/connect only after the caller has completed all offline gates."""

    hardware_types = _load_hardware_types()
    artifact = (
        _require_passing_installed_audit(inspection)
        if args.command in ("grasp", "grasp-lift", "air-grasp")
        and args.trajectory_mode == "audited-joint"
        else None
    )
    pregrasp_artifact = (
        _require_passing_pregrasp_audit(inspection)
        if args.command == "pregrasp"
        else None
    )
    loaded_lift_binding = (
        _require_passing_loaded_lift_audit(inspection)
        if args.command == "grasp-lift"
        else None
    )
    audited_tracking_tolerance = None
    if artifact is not None:
        audited_tracking_tolerance = float(
            artifact["policies"]["max_q_tracking_error_rad"]
        )
        if loaded_lift_binding is not None:
            lift_tracking = loaded_lift_binding.max_q_tracking_error_rad
            if lift_tracking is None:
                raise RuntimeError("loaded-lift tracking tolerance is unavailable")
            audited_tracking_tolerance = min(
                audited_tracking_tolerance, float(lift_tracking)
            )
    elif pregrasp_artifact is not None:
        audited_tracking_tolerance = float(
            pregrasp_artifact["bindings"]["joint_prefix"][
                "max_q_tracking_error_rad"
            ]
        )
    limits = _build_franka_limits(
        inspection.config,
        hardware_types["FrankaMotionLimits"],
        audited_tracking_tolerance_rad=audited_tracking_tolerance,
    )
    settle = hardware_types["SettleTolerances"](
        position_m=float(
            inspection.config["franka"]["cartesian_arrival_position_tolerance_m"]
        ),
        orientation_rad=float(
            inspection.config["franka"]["cartesian_arrival_rotation_tolerance_rad"]
        ),
        stable_seconds=float(inspection.config["franka"]["settle_time_s"]),
    )

    arm = None
    hand = None
    sequence = None
    continuous_runtime: Optional[ContinuousTelemetryRuntime] = None
    initial_validated_arm_state = None
    initial_arm_timestamp_unix_ns = 0
    initial_arm_timestamp_monotonic_ns = 0
    primary_error: Optional[BaseException] = None
    interrupted = False
    manual_recovery_required = False
    cleanup_errors: List[str] = []
    telemetry_cleanup_errors: List[str] = []
    try:
        arm, hand = _connect_hardware(inspection.config, hardware_types, limits)
        validated_arm_sample = _verify_installed_end_effector(
            arm, inspection.config
        )
        if telemetry_request is None:
            # Legacy/fake drivers do not need to expose the telemetry timestamp
            # tuple when the feature is not requested.
            if isinstance(validated_arm_sample, tuple) and len(validated_arm_sample) == 3:
                (
                    initial_validated_arm_state,
                    initial_arm_timestamp_unix_ns,
                    initial_arm_timestamp_monotonic_ns,
                ) = validated_arm_sample
            else:
                initial_validated_arm_state = validated_arm_sample
        else:
            if not (
                isinstance(validated_arm_sample, tuple)
                and len(validated_arm_sample) == 3
            ):
                raise RuntimeError(
                    "continuous telemetry requires a timestamped validated arm sample"
                )
            (
                initial_validated_arm_state,
                initial_arm_timestamp_unix_ns,
                initial_arm_timestamp_monotonic_ns,
            ) = validated_arm_sample
        _install_continuous_franka_gate(hand, arm, inspection.config)
        if artifact is not None:
            binding = inspection.installed_audit
            if binding is None or binding.path is None:
                raise RuntimeError("installed-tool audit path disappeared")
            replayed = load_installed_tool_audit(
                binding.path, verify_files=True, require_pass=True
            )
            if replayed["artifact_sha256"] != artifact["artifact_sha256"]:
                raise RuntimeError("installed-tool audit changed before motion")
            artifact = replayed
            _verify_live_q_matches_audit(arm, artifact)
        if pregrasp_artifact is not None:
            binding = inspection.pregrasp_audit
            if binding is None or binding.path is None:
                raise RuntimeError("pregrasp-only audit path disappeared")
            replayed_prefix = load_pregrasp_only_audit(
                binding.path, verify_files=True, require_pass=True
            )
            if (
                replayed_prefix["artifact_sha256"]
                != pregrasp_artifact["artifact_sha256"]
            ):
                raise RuntimeError("pregrasp-only audit changed before motion")
            pregrasp_artifact = replayed_prefix
            _verify_live_q_matches_pregrasp_audit(arm, pregrasp_artifact)
        if loaded_lift_binding is not None:
            rebound = bind_loaded_lift_audit(
                loaded_lift_binding.path,
                base_loaded_grasp_audit_path=(
                    None
                    if inspection.installed_audit is None
                    else inspection.installed_audit.path
                ),
                config=inspection.config,
                config_path=inspection.config_path,
                expected_selected_candidate_index=(
                    None
                    if inspection.snapshot is None
                    else int(inspection.snapshot.grasps.selected_index)
                ),
                expected_grasp_q_rad=(
                    None
                    if loaded_lift_binding.base_artifact is None
                    else loaded_lift_binding.base_artifact["bindings"][
                        "joint_path"
                    ]["waypoints"][-1]["q_rad"]
                ),
                expected_grasp_pose_base_EE=(
                    None
                    if inspection.plan is None
                    else inspection.plan.T_reference_EE_grasp
                ),
                expected_closed_hand_targets=(
                    None
                    if loaded_lift_binding.base_artifact is None
                    else loaded_lift_binding.base_artifact["bindings"][
                        "snapshot"
                    ]["selected_candidate"]["hand_targets"]
                ),
                verify_files=True,
            )
            if (
                not rebound.passed
                or rebound.artifact_sha256
                != loaded_lift_binding.artifact_sha256
            ):
                raise RuntimeError(
                    "loaded-lift audit changed or failed replay before motion: "
                    + "; ".join(rebound.blockers)
                )
            loaded_lift_binding = rebound
            inspection = replace(inspection, loaded_lift_audit=rebound)
        # Close the short check-to-use gap created by hardware construction
        # and live-q/audit replay.  A Desk tab reopened after the initial
        # pre-import check must still fail before the first hand or arm motion.
        # The libfranka control connection itself does not use Desk's :443.
        require_uncontended_franka_https_link(
            str(inspection.config["franka"]["ip"])
        )
        sequence_kwargs: Dict[str, Any] = {"settle_tolerances": settle}
        if telemetry_request is not None:
            try:
                continuous_runtime = _start_continuous_telemetry(
                    telemetry_request,
                    arm=arm,
                    hand=hand,
                    initial_validated_arm_state=initial_validated_arm_state,
                    initial_arm_timestamp_unix_ns=initial_arm_timestamp_unix_ns,
                    initial_arm_timestamp_monotonic_ns=(
                        initial_arm_timestamp_monotonic_ns
                    ),
                    config=inspection.config,
                )
            except ContinuousTelemetryStartError as exc:
                # Preserve the runtime so final cleanup can retry an
                # identity-bound detach without ever closing a tap that a
                # driver still references.
                continuous_runtime = exc.runtime
                raise
            sequence_kwargs["transition_observer"] = (
                continuous_runtime.observe_transition
            )
            print(
                "[telemetry] native producer enabled path={} run_uuid={}".format(
                    telemetry_request.mapping_path,
                    telemetry_request.manifest.identity.run_uuid,
                )
            )
        pose_state_output = getattr(args, "pose_state_output", None)
        if pose_state_output is not None:
            sequence_kwargs["boundary_observer"] = _build_pose_state_observer(
                pose_state_output, arm, hand
            )
        sequence = hardware_types["StagedGraspSequence"](
            arm, hand, **sequence_kwargs
        )
        if args.command == "default":
            plan = hardware_types["DefaultSequencePlan"](
                default_execution_eligible=True,
                default_q=np.asarray(
                    inspection.config["franka"]["default_q_rad"], dtype=np.float64
                ),
                open_targets=tuple(
                    int(value)
                    for value in inspection.config["inspire"]["open_targets"]
                ),
            )
            state = sequence.run_to_default(plan)
            print("[sequence] reached {}".format(state.value))
        elif args.command == "pregrasp":
            prefix_plan = _build_audited_pregrasp_plan(
                inspection,
                hardware_types["AuditedPregraspSequencePlan"],
                settle,
            )
            state = sequence.run_to_pregrasp_joint_waypoints(prefix_plan)
            print(
                "[sequence] reached {}; no grasp/q6/bend stage exists in this mode".format(
                    state.value
                )
            )
        elif args.command == "grasp-lift":
            loaded_plan = _build_loaded_lift_plan(
                inspection, hardware_types, settle
            )
            state = sequence.run_loaded_lift_joint_waypoints(loaded_plan)
            print(
                "[sequence] reached {}; loaded hold remains active at lift".format(
                    state.value
                )
            )
            _monitor_loaded_hold(sequence, float(args.lift_hold_seconds))
            state = sequence.return_loaded_lift_to_setdown(loaded_plan)
            print(
                "[sequence] reached {}; load is supported, hand disabled, "
                "Franka external load cleared".format(state.value)
            )
        else:
            if inspection.plan is None:  # Defensive; checked before import/connect.
                raise RuntimeError("internal error: full-grasp plan is absent")
            if args.trajectory_mode != "audited-joint":
                raise RuntimeError(
                    "Cartesian full-grasp execution is intentionally disabled"
                )
            audited_plan = _build_audited_joint_plan(
                inspection,
                hardware_types["AuditedJointSequencePlan"],
                settle,
            )
            if artifact is not None and artifact["mode"] == "air_grasp":
                binder = getattr(hand, "bind_audited_no_contact_execution_path", None)
                if not callable(binder):
                    raise RuntimeError(
                        "RH56 driver cannot bind the audited hand execution path"
                    )
                binder(
                    artifact["bindings"]["hand_execution_path"],
                    feedback_envelope_policy=artifact["policies"][
                        "hand_feedback_envelope"
                    ],
                )
            if bool(getattr(args, "stop_after_default", False)):
                state = sequence.run_to_default_joint_waypoints(audited_plan)
                print(
                    "[sequence] reached {}; staged stop for fresh scene recapture".format(
                        state.value
                    )
                )
            else:
                state = sequence.run_full_joint_waypoints(audited_plan)
                print("[sequence] reached {}; bounded hold begins".format(state.value))
                _monitor_bounded_hold(sequence, float(args.hold_seconds))
                if artifact is not None and artifact["mode"] == "air_grasp":
                    state = sequence.return_air_hand_to_open()
                    print(
                        "[sequence] monitored reverse hand path reached {}".format(
                            state.value
                        )
                    )
        state = sequence.abort("bounded successful sequence cleanup")
        print("[sequence] cleanup reached {}".format(state.value))
    except KeyboardInterrupt as exc:
        primary_error = exc
        interrupted = True
    except BaseException as exc:
        primary_error = exc
    finally:
        # Sequence.abort is the preferred coordinated stop.  Direct fallbacks
        # cover connection/preflight failures and terminal sequencer faults.
        if sequence is not None and getattr(sequence.state, "value", "") not in (
            "stopped",
            "fault_latched",
        ):
            try:
                sequence.abort("final cleanup")
            except BaseException as exc:
                if bool(
                    getattr(sequence, "requires_manual_load_recovery", False)
                ):
                    manual_recovery_required = True
                    if primary_error is None:
                        primary_error = exc
                else:
                    cleanup_errors.append("coordinated abort: {}".format(exc))
        if sequence is not None and bool(
            getattr(sequence, "requires_manual_load_recovery", False)
        ):
            manual_recovery_required = True
        if arm is not None:
            timing = getattr(arm, "last_control_loop_telemetry", None)
            if timing is not None:
                timing_stream = (
                    sys.stderr
                    if int(timing.read_to_write_overruns) > 0
                    else sys.stdout
                )
                print(
                    "[Franka/FCI timing] kind={} samples={} "
                    "max_read_to_write={:.3f}us over_500us={}".format(
                        timing.kind,
                        int(timing.samples),
                        float(timing.max_read_to_write_us),
                        int(timing.read_to_write_overruns),
                    ),
                    file=timing_stream,
                )
            try:
                arm.stop()
            except BaseException as exc:
                cleanup_errors.append("Franka stop: {}".format(exc))
        if hand is not None and not manual_recovery_required:
            try:
                hand.close()
            except BaseException as exc:
                cleanup_errors.append("Inspire disable/close: {}".format(exc))
        # Device stop/disable (or loaded-hold preservation) is always attempted
        # before detaching display telemetry.  Teardown failures therefore do
        # not get mislabeled as an unconfirmed hardware stop.
        if continuous_runtime is not None:
            try:
                continuous_runtime.close()
            except BaseException as first:
                try:
                    continuous_runtime.close()
                except BaseException as second:
                    telemetry_cleanup_errors.append(
                        "first detach: {}; retry: {}".format(first, second)
                    )

    if manual_recovery_required:
        print(
            "[LOADED RECOVERY REQUIRED] Franka stop was requested, but Inspire "
            "numeric hold was intentionally NOT disabled because the payload may "
            "be suspended. Support/set down the object, keep 24 V available for "
            "the documented loaded recovery, and do not run another command.",
            file=sys.stderr,
        )

    if cleanup_errors:
        print(
            "[STOP UNCONFIRMED] {}".format("; ".join(cleanup_errors)),
            file=sys.stderr,
        )
        if primary_error is not None:
            print("[sequence] original error: {}".format(primary_error), file=sys.stderr)
        return 5
    if telemetry_cleanup_errors:
        print(
            "[telemetry cleanup] {}".format(
                "; ".join(telemetry_cleanup_errors)
            ),
            file=sys.stderr,
        )
        if primary_error is None:
            primary_error = RuntimeError("continuous telemetry cleanup failed")
    if manual_recovery_required:
        if primary_error is not None:
            print("[sequence] loaded fault: {}".format(primary_error), file=sys.stderr)
        return 6
    if interrupted:
        print("[sequence] interrupted; stop/disable cleanup completed", file=sys.stderr)
        return 130
    if primary_error is not None:
        print("[sequence] failed: {}".format(primary_error), file=sys.stderr)
        return 4
    print("[sequence] success; arm stopped and all-six hand output disabled")
    return 0


def _positive_samples(value: str) -> int:
    integer = int(value)
    if integer < 2:
        raise argparse.ArgumentTypeError("collision samples must be >= 2")
    return integer


def _nonnegative_margin(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("collision margin must be finite and >= 0")
    return number


def _bounded_hold(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= MAX_HOLD_SECONDS:
        raise argparse.ArgumentTypeError(
            "hold seconds must be finite in 0..{}".format(MAX_HOLD_SECONDS)
        )
    return number


def _add_offline_arguments(parser: argparse.ArgumentParser, *, snapshot: bool) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    if snapshot:
        parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--installed-tool-audit",
        type=Path,
        default=None,
        help="schema-v2 installed-tool collision audit sidecar",
    )
    parser.add_argument(
        "--pregrasp-only-audit",
        type=Path,
        default=None,
        help="dedicated current-to-pregrasp open-hand prefix audit sidecar",
    )
    parser.add_argument(
        "--loaded-lift-audit",
        type=Path,
        default=None,
        help=(
            "independent schema-v1 loaded close/lift/reverse-setdown audit; "
            "an ordinary loaded-grasp sidecar is never sufficient"
        ),
    )
    parser.add_argument(
        "--selected-index",
        type=int,
        default=None,
        help=(
            "in-memory official candidate selection; the source NPZ is never "
            "rewritten and a supplied audit must bind the same index"
        ),
    )
    parser.add_argument(
        "--collision-margin-m", type=_nonnegative_margin, default=0.005
    )
    parser.add_argument("--collision-samples", type=_positive_samples, default=21)
    parser.add_argument(
        "--continuous-telemetry",
        type=Path,
        default=None,
        metavar="MAP",
        help=(
            "opt-in native fixed-POD telemetry mapping to create during formal "
            "execution; requires --telemetry-session-manifest"
        ),
    )
    parser.add_argument(
        "--telemetry-session-manifest",
        type=Path,
        default=None,
        metavar="MANIFEST",
        help=(
            "strict session identity/artifact manifest paired with "
            "--continuous-telemetry"
        ),
    )
    parser.add_argument(
        "--continuous-telemetry-python-dir",
        type=Path,
        default=(
            None
            if not os.environ.get("ANYDEX_TELEMETRY_PYTHON_DIR")
            else Path(os.environ["ANYDEX_TELEMETRY_PYTHON_DIR"])
        ),
        metavar="DIR",
        help=(
            "directory containing the exact manifest-bound "
            "_anydex_franka_telemetry module; defaults to "
            "ANYDEX_TELEMETRY_PYTHON_DIR"
        ),
    )


def _add_base_confirmations(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confirm-workspace-clear",
        metavar=WORKSPACE_CLEAR_TOKEN,
        help="exact physical-workspace confirmation token",
    )
    parser.add_argument(
        "--confirm-immediate-stop",
        metavar=IMMEDIATE_STOP_TOKEN,
        help="exact immediate stop and 24 V cut readiness token",
    )
    parser.add_argument(
        "--pose-state-output",
        type=Path,
        default=None,
        help=(
            "atomically publish one robot/hand JSON sample at each verified "
            "waypoint or stage boundary; this is not a 1 kHz stream"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or execute the fail-closed open -> default -> pregrasp -> "
            "grasp -> settle -> q6 preshape -> bend-close sequence. With no "
            "subcommand, offline inspect is used."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser(
        "inspect", help="offline readiness, plan, and adapter-sweep report"
    )
    _add_offline_arguments(inspect_parser, snapshot=False)
    inspect_parser.add_argument("--snapshot", type=Path, default=None)

    default_parser = subparsers.add_parser(
        "default", help="open all six hand axes, then move Franka to default q"
    )
    _add_offline_arguments(default_parser, snapshot=False)
    _add_base_confirmations(default_parser)
    default_parser.add_argument(
        "--confirm-pla-low-speed",
        metavar=PLA_COMMISSIONING_TOKEN,
        help="exact token acknowledging low-speed unloaded PLA-only commissioning",
    )

    pregrasp_parser = subparsers.add_parser(
        "pregrasp",
        help=(
            "open/verify/disable RH56, then execute only a dedicated audited "
            "Franka joint prefix through pregrasp"
        ),
    )
    _add_offline_arguments(pregrasp_parser, snapshot=True)
    _add_base_confirmations(pregrasp_parser)
    pregrasp_parser.add_argument(
        "--confirm-pregrasp-only",
        metavar=PREGRASP_ONLY_TOKEN,
        help="exact confirmation that this run must stop at pregrasp",
    )
    pregrasp_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the bound prefix without importing or connecting hardware",
    )
    default_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print gates/stages without importing or connecting hardware",
    )

    grasp_parser = subparsers.add_parser(
        "grasp", help="execute one fully commissioned staged grasp"
    )
    _add_offline_arguments(grasp_parser, snapshot=True)
    _add_base_confirmations(grasp_parser)
    grasp_parser.add_argument(
        "--confirm-full-grasp", metavar=FULL_GRASP_TOKEN
    )
    grasp_parser.add_argument(
        "--trajectory-mode",
        choices=("cartesian", "audited-joint"),
        default="cartesian",
        help=(
            "audited-joint consumes the exact artifact waypoint polyline; "
            "cartesian remains intentionally locked"
        ),
    )
    grasp_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print gates/stages without importing or connecting hardware",
    )
    grasp_parser.add_argument(
        "--confirm-q6-preshape", metavar=Q6_PRESHAPE_TOKEN
    )
    grasp_parser.add_argument(
        "--confirm-installed-collision-model", metavar=COLLISION_MODEL_TOKEN
    )
    grasp_parser.add_argument(
        "--hold-seconds",
        type=_bounded_hold,
        default=2.0,
        help="bounded numeric grasp hold before stop/disable (0..10 s; default 2)",
    )
    grasp_parser.add_argument(
        "--stop-after-default",
        action="store_true",
        help=(
            "execute only the audited open/current/transit/default prefix, then "
            "stop and disable for a fresh scene recapture"
        ),
    )

    lift_parser = subparsers.add_parser(
        "grasp-lift",
        help=(
            "execute an independently audited loaded close/lift/reverse-setdown "
            "round trip; current PLA commissioning remains locked"
        ),
    )
    _add_offline_arguments(lift_parser, snapshot=True)
    _add_base_confirmations(lift_parser)
    lift_parser.add_argument(
        "--confirm-loaded-lift-round-trip", metavar=LOADED_LIFT_TOKEN
    )
    lift_parser.add_argument(
        "--confirm-setdown-support", metavar=LOAD_SUPPORT_TOKEN
    )
    lift_parser.add_argument(
        "--confirm-load-rated-tool", metavar=LOAD_RATED_TOKEN
    )
    lift_parser.add_argument(
        "--confirm-q6-preshape", metavar=Q6_PRESHAPE_TOKEN
    )
    lift_parser.add_argument(
        "--confirm-installed-collision-model", metavar=COLLISION_MODEL_TOKEN
    )
    lift_parser.add_argument(
        "--trajectory-mode",
        choices=("audited-joint",),
        default="audited-joint",
        help="loaded lift accepts only the exact independently audited joint path",
    )
    lift_parser.add_argument(
        "--lift-hold-seconds",
        type=_bounded_hold,
        default=1.0,
        help="bounded hold at audited lift before reverse setdown (0..10 s)",
    )
    lift_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print loaded-lift gates/stages without importing hardware",
    )

    air_parser = subparsers.add_parser(
        "air-grasp",
        help="execute a no-contact/no-lift closure at an audited retreated pose",
    )
    _add_offline_arguments(air_parser, snapshot=True)
    _add_base_confirmations(air_parser)
    air_parser.add_argument("--confirm-air-grasp", metavar=AIR_GRASP_TOKEN)
    air_parser.add_argument("--confirm-q6-preshape", metavar=Q6_PRESHAPE_TOKEN)
    air_parser.add_argument(
        "--confirm-installed-collision-model", metavar=COLLISION_MODEL_TOKEN
    )
    air_parser.add_argument(
        "--trajectory-mode",
        choices=("cartesian", "audited-joint"),
        default="cartesian",
        help=(
            "audited-joint consumes the exact artifact waypoint polyline; "
            "cartesian remains intentionally locked"
        ),
    )
    air_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print gates/stages without importing or connecting hardware",
    )
    air_parser.add_argument(
        "--hold-seconds",
        type=_bounded_hold,
        default=2.0,
        help="bounded no-contact closure hold before stop/disable (0..10 s)",
    )
    air_parser.add_argument(
        "--stop-after-default",
        action="store_true",
        help=(
            "execute only the audited open/current/transit/default prefix, then "
            "stop and disable for a fresh scene recapture"
        ),
    )
    return parser


def _arguments_with_default_inspect(argv: Optional[Sequence[str]]) -> List[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        return ["inspect"]
    if values[0] in ("-h", "--help"):
        return values
    if values[0] not in (
        "inspect",
        "default",
        "pregrasp",
        "grasp",
        "grasp-lift",
        "air-grasp",
    ):
        return ["inspect"] + values
    return values


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_arguments_with_default_inspect(argv))
    try:
        inspection = inspect_offline(
            args.config,
            getattr(args, "snapshot", None),
            collision_margin_m=float(args.collision_margin_m),
            collision_samples=int(args.collision_samples),
            installed_audit_path=getattr(args, "installed_tool_audit", None),
            pregrasp_audit_path=getattr(args, "pregrasp_only_audit", None),
            loaded_lift_audit_path=getattr(args, "loaded_lift_audit", None),
            selected_index=getattr(args, "selected_index", None),
        )
    except (OSError, TypeError, ValueError) as exc:
        print("[offline validation] failed: {}".format(exc), file=sys.stderr)
        return 2

    try:
        telemetry_request = _load_continuous_telemetry_request(args, inspection)
    except (OSError, TypeError, ValueError) as exc:
        print("[telemetry validation] failed: {}".format(exc), file=sys.stderr)
        print("[hardware] NOT IMPORTED; neither device was connected")
        return 2

    trajectory_mode = (
        "audited-joint"
        if args.command == "pregrasp"
        else getattr(args, "trajectory_mode", "cartesian")
    )
    # Display the readiness contract of the selected executor, not legacy
    # blockers that the exact audited-joint sidecar intentionally replaces.
    # Keep ``inspection`` itself unchanged for all subsequent evidence gates.
    display_default_blockers = inspection.readiness.default_motion_blockers
    display_full_blockers = inspection.readiness.full_grasp_blockers
    if args.command == "pregrasp":
        display_default_blockers = tuple(
            item
            for item in display_default_blockers
            if item
            not in (
                "open-hand path from the current pose to Franka default_q is not collision-verified",
                "installed adapter + hand collision model is unverified",
            )
        )
    if (
        args.command in ("grasp", "grasp-lift", "air-grasp")
        and trajectory_mode == "audited-joint"
    ):
        display_full_blockers = tuple(
            item
            for item in display_full_blockers
            if item
            != "Cartesian home/transit/grasp workspace is uncommissioned"
        )
    if args.command == "air-grasp":
        display_full_blockers = tuple(
            item
            for item in display_full_blockers
            if item
            != "installed PLA adapter is approved only for low-speed unloaded checks"
        )
    display_inspection = replace(
        inspection,
        readiness=ControlReadiness(
            display_default_blockers,
            display_full_blockers,
        ),
    )
    _print_inspection(display_inspection, command=args.command)
    if telemetry_request is not None:
        print(
            "[telemetry] manifest verified offline; run_uuid={} mapping={} "
            "motion_authorized=false; native producer NOT IMPORTED".format(
                telemetry_request.manifest.identity.run_uuid,
                telemetry_request.mapping_path,
            )
        )
    if args.command == "inspect":
        return 0

    if args.command == "default":
        readiness_blockers = inspection.readiness.default_motion_blockers
    elif args.command == "pregrasp":
        # The dedicated prefix audit replaces only the legacy global default
        # path boolean.  Mount/load/datum/model blockers remain mandatory;
        # closure/q6/full-grasp readiness is irrelevant to arm-only motion.
        readiness_blockers = tuple(
            item
            for item in inspection.readiness.default_motion_blockers
            if item
            not in (
                "open-hand path from the current pose to Franka default_q is not collision-verified",
                "installed adapter + hand collision model is unverified",
            )
        )
        if inspection.snapshot is None:
            readiness_blockers += ("no grasp snapshot was supplied",)
        if inspection.plan is None:
            readiness_blockers += ("no pregrasp pose plan was built",)
        if not bool(inspection.config["tool"].get("mount_transform_commissioned")):
            readiness_blockers += ("T_EE_hand is not commissioned",)
    else:
        readiness_blockers = inspection.readiness.full_grasp_blockers
    if (
        args.command in ("grasp", "grasp-lift", "air-grasp")
        and trajectory_mode == "audited-joint"
    ):
        # A passing schema-v2 sidecar binds the exact joint polyline and its
        # pregrasp/grasp FK poses.  Generic Cartesian workspace bounds apply
        # only to the intentionally locked Cartesian executor and must not
        # remain as a contradictory legacy readiness blocker here.
        readiness_blockers = tuple(
            item
            for item in readiness_blockers
            if item != "Cartesian home/transit/grasp workspace is uncommissioned"
        )
    if args.command == "air-grasp":
        readiness_blockers = tuple(
            item
            for item in readiness_blockers
            if item
            != "installed PLA adapter is approved only for low-speed unloaded checks"
        )
    blockers = list(readiness_blockers)
    blockers.extend(
        _runtime_config_blockers(
            inspection,
            command=args.command,
            trajectory_mode=trajectory_mode,
        )
    )
    if not bool(getattr(args, "dry_run", False)):
        blockers.extend(_confirmation_blockers(args))
        blockers.extend(_conditional_audit_blockers(args, inspection))
    if args.command in ("grasp", "grasp-lift", "air-grasp") and inspection.plan is not None:
        if not inspection.plan.execution_eligible:
            blockers.extend(inspection.plan.execution_blockers)
    if bool(getattr(args, "dry_run", False)):
        _print_staged_dry_run(
            inspection,
            command=args.command,
            trajectory_mode=trajectory_mode,
            stop_after_default=bool(
                getattr(args, "stop_after_default", False)
            ),
        )
    if blockers:
        _print_blockers("hardware execution gate", tuple(dict.fromkeys(blockers)))
        print("[hardware] NOT IMPORTED; neither device was connected")
        return 3

    if bool(getattr(args, "dry_run", False)):
        print("[hardware] NOT IMPORTED; dry-run completed")
        return 0

    # Desk/browser HTTPS shares the same dedicated Ethernet link as the FCI
    # 1 kHz UDP stream.  Inspect the kernel socket table only after every
    # offline/dry-run path has returned, but before importing a driver or
    # connecting either device.  The gate observes host state only: it never
    # sends a packet, closes a socket, or terminates the browser.
    try:
        require_uncontended_franka_https_link(
            str(inspection.config["franka"]["ip"])
        )
    except HostNetworkPreflightError as exc:
        print("[host network preflight] failed: {}".format(exc), file=sys.stderr)
        print("[hardware] NOT IMPORTED; neither device was connected")
        return 3

    # This line is intentionally the first path capable of importing drivers.
    return _execute_hardware(
        args, inspection, telemetry_request=telemetry_request
    )


if __name__ == "__main__":
    raise SystemExit(main())
