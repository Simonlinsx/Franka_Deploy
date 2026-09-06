#!/usr/bin/python3
"""Generate or replay a dedicated open-hand FR3 prefix collision audit.

This utility is offline.  It imports neither libfranka nor the Inspire serial
driver and never opens a camera.  The generated artifact covers only

``current -> default -> pregrasp``

in joint space.  FR3 self, V7/FR3, open-RH56/FR3, and the two exact bound
AnyDex object-cloud checks decide the result.  Full-scene checks are retained
and hashed as advisory observations; they are not silently removed.
Accordingly, the filtered-scene capture q remains exact provenance but need
not equal the independently joint-plan-bound prefix current q.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from anydex_pipeline.control_config import load_control_config  # noqa: E402
from anydex_pipeline.control_plan import is_official_snapshot  # noqa: E402
from anydex_pipeline.hppfcl_installed_tool_backend import (  # noqa: E402
    HppFclInstalledToolBackend,
    HppFclInstalledToolConfig,
    RH56_ADAPTER_MOUNT_EXCLUSIONS,
    T_EE_ADAPTER,
)
from anydex_pipeline.inspire_hand_model import InspireHandModel  # noqa: E402
from anydex_pipeline.inspire_open_configuration import (  # noqa: E402
    OFFICIAL_OPEN_JOINT_POSITIONS_RAD,
    official_open_configuration_provenance,
)
from anydex_pipeline.installed_tool_audit import (  # noqa: E402
    InstalledToolCollisionQuery,
    check_specs_for_mode,
)
from anydex_pipeline.pregrasp_only_audit import (  # noqa: E402
    REQUIRED_CHECK_IDS,
    build_pregrasp_joint_path,
    create_pregrasp_only_audit,
    load_pregrasp_only_audit,
)
from anydex_pipeline.snapshot import load_snapshot_npz  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
DEFAULT_SNAPSHOT = (
    ROOT / "runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz"
)
DEFAULT_FILTERED_SCENE = (
    ROOT / "runs/live_scene_installed_filtered_candidate51_cable_fixed_20260721.npz"
)
DEFAULT_JOINT_PLAN = (
    ROOT
    / "runs/candidate51_installed_air_joint_plan_current_fr3_limits_20260726.json"
)
DEFAULT_ADAPTER = ROOT / "assets/adapter/V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.stl"
DEFAULT_ANYDEX_ROOT = ROOT / "third_party/AnyDexGrasp"


def _advisory_scene_prefix_q_delta(
    scene_capture_q: Sequence[float], prefix_current_q: Sequence[float]
) -> float:
    """Return provenance-only q separation without imposing equality."""

    capture = np.asarray(scene_capture_q, dtype=np.float64)
    current = np.asarray(prefix_current_q, dtype=np.float64)
    if (
        capture.shape != (7,)
        or current.shape != (7,)
        or not np.all(np.isfinite(capture))
        or not np.all(np.isfinite(current))
    ):
        raise ValueError("scene capture q and prefix current q must be finite 7-vectors")
    return float(np.max(np.abs(capture - current)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_filtered_scene(
    path: Path,
) -> Tuple[Path, np.ndarray, np.ndarray, float, Path, Mapping[str, Any], Mapping[str, str]]:
    source = Path(path).expanduser().resolve()
    with np.load(str(source), allow_pickle=False) as archive:
        required = {
            "artifact_type",
            "scene_points",
            "capture_q_rad",
            "captured_at_unix_s",
            "calibration_id",
            "camera_serial",
            "filter_evidence_path",
            "filter_evidence_sha256",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError("filtered scene is missing {}".format(missing))
        if str(np.asarray(archive["artifact_type"]).item()) != (
            "installed_filtered_live_scene"
        ):
            raise ValueError("--filtered-scene has the wrong artifact type")
        points = np.asarray(archive["scene_points"], dtype=np.float64)
        capture_q = np.asarray(archive["capture_q_rad"], dtype=np.float64)
        captured = float(np.asarray(archive["captured_at_unix_s"]).item())
        metadata = {
            "calibration_id": str(np.asarray(archive["calibration_id"]).item()),
            "camera_serial": str(np.asarray(archive["camera_serial"]).item()),
        }
        evidence_path = Path(
            str(np.asarray(archive["filter_evidence_path"]).item())
        ).expanduser().resolve()
        evidence_sha = str(np.asarray(archive["filter_evidence_sha256"]).item())
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise ValueError("filtered scene points must be non-empty (N,3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("filtered scene contains NaN or infinity")
    if capture_q.shape != (7,) or not np.all(np.isfinite(capture_q)):
        raise ValueError("filtered scene capture_q_rad must be a finite 7-vector")
    if not evidence_path.is_file() or _sha256_file(evidence_path) != evidence_sha:
        raise ValueError("filtered-scene evidence path/hash replay failed")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("motion_authorized") is not False:
        raise ValueError("filter evidence must explicitly remain non-authorizing")
    if evidence.get("authoritative_for_unseen_camera_space") is not False:
        raise ValueError("filter evidence altered single-view authority semantics")
    return source, points, capture_q, captured, evidence_path, evidence, metadata


def _load_joint_plan(path: Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    artifact = json.loads(source.read_text(encoding="utf-8"))
    if (
        artifact.get("schema_version") != 1
        or artifact.get("artifact_type") != "installed_air_candidate_joint_plan"
        or artifact.get("motion_authorized") is not False
    ):
        raise ValueError("joint plan schema/type is invalid")
    integrity = artifact.get("integrity")
    unsigned = dict(artifact)
    unsigned.pop("integrity", None)
    if (
        not isinstance(integrity, dict)
        or integrity.get("algorithm") != "sha256-canonical-json-without-integrity"
        or integrity.get("payload_sha256") != _json_sha256(unsigned)
    ):
        raise ValueError("joint plan integrity check failed")
    return artifact


def _collision_request(
    *,
    scene: np.ndarray,
    object_points: np.ndarray,
    adapter_path: Path,
    T_EE_hand: np.ndarray,
    hand_model: InspireHandModel,
    max_q_tracking_error_rad: float,
) -> SimpleNamespace:
    # ``loaded_grasp`` avoids evaluating a post-pregrasp actuator path.  Only
    # the canonical full-path open checks are selected below; no closed-hand
    # observation enters the prefix artifact.
    return SimpleNamespace(
        mode="loaded_grasp",
        scene_points_base=scene,
        object_points_base=object_points,
        adapter_stl_path=adapter_path,
        T_EE_hand=np.asarray(T_EE_hand, dtype=np.float64),
        allowed_object_contact_links=("Link11", "Link22", "Link33", "Link44", "Link53"),
        max_q_tracking_error_rad=float(max_q_tracking_error_rad),
        hand_arrival_tolerance_units=25,
        scene_voxel_resolution_m=0.005,
        observed_scene_scope="calibrated_camera_frustum_voxel_grid",
        unknown_space_policy="occupied",
        hand_model=hand_model,
    )


def _observation_check(spec: Any, observation: Any, sample_count: int) -> Mapping[str, Any]:
    return {
        "check_id": spec.check_id,
        "scope": spec.scope,
        "expectation": spec.expectation,
        "coverage": spec.coverage,
        "authoritative": bool(observation.authoritative),
        "expected_sample_count": int(sample_count),
        "tested_sample_indices": list(observation.tested_sample_indices),
        "minimum_signed_distance_m": float(observation.minimum_signed_distance_m),
        "observed_pairs": list(observation.observed_pairs),
        "details": dict(observation.details or {}),
    }


def _open_hand_adapter_check(
    *,
    backend: HppFclInstalledToolBackend,
    q_path: np.ndarray,
    adapter_path: Path,
    T_EE_hand: np.ndarray,
    hand_mesh_paths: Mapping[str, Path],
    open_transforms: Mapping[str, np.ndarray],
    max_q_tracking_error_rad: float,
) -> Mapping[str, Any]:
    """Exact full-prefix V7/open-RH56 check in their common rigid frame.

    Arm interpolation and tracking uncertainty transform both bodies by the
    same rigid motion, so their relative clearance is invariant and its
    continuous motion bound is exactly zero.  We still evaluate every path
    sample and every non-mount hand link to make coverage explicit.
    """

    adapter_geometry = backend._load_mesh(  # pylint: disable=protected-access
        adapter_path, scale=backend.config.adapter_mesh_scale
    )
    hand_geometries = {
        name: backend._load_mesh(path)  # pylint: disable=protected-access
        for name, path in hand_mesh_paths.items()
    }
    exclusions = tuple(RH56_ADAPTER_MOUNT_EXCLUSIONS)
    if exclusions != ("Link111",):
        raise RuntimeError("unexpected RH56/V7 fixed-mount exclusion policy")
    checked_names = tuple(
        name for name in hand_geometries if name not in set(exclusions)
    )
    if len(checked_names) != 12:
        raise RuntimeError("open RH56/V7 audit must test twelve non-mount links")
    minimum = None
    observed_pairs = set()
    evaluated = 0
    for sample_index, q in enumerate(np.asarray(q_path, dtype=np.float64)):
        _, T_base_EE = backend._robot_state(q)  # pylint: disable=protected-access
        adapter = backend._object(  # pylint: disable=protected-access
            adapter_geometry, T_base_EE @ T_EE_ADAPTER
        )
        hand_objects = backend._hand_objects(  # pylint: disable=protected-access
            hand_geometries,
            open_transforms,
            T_base_EE @ np.asarray(T_EE_hand, dtype=np.float64),
        )
        for name in checked_names:
            pair = "adapter / {}".format(name)
            item = backend._mesh_pair_distance(  # pylint: disable=protected-access
                adapter,
                hand_objects[name],
                pair=pair,
                sample_index=sample_index,
            )
            evaluated += 1
            if minimum is None or item.distance_m < minimum.distance_m:
                minimum = item
            if item.intersecting:
                observed_pairs.add(pair)
    if minimum is None:
        raise RuntimeError("open RH56/V7 audit produced no mesh evidence")
    return {
        "check_id": "rh56_open_adapter_path",
        "scope": "open_rh56_vs_adapter",
        "expectation": "clear",
        "coverage": "full_path",
        "authoritative": True,
        "expected_sample_count": int(len(q_path)),
        "tested_sample_indices": list(range(len(q_path))),
        "minimum_signed_distance_m": float(minimum.distance_m),
        "observed_pairs": sorted(observed_pairs),
        "details": {
            "engine": "Pinocchio + HPP-FCL",
            "geometry_model": "exact triangle meshes",
            "evaluated_pair_count": int(evaluated),
            "minimum_pair": minimum.pair,
            "minimum_sample_index": int(minimum.sample_index),
            "nearest_point_a_base_m": (
                None
                if minimum.nearest_point_a is None
                else list(minimum.nearest_point_a)
            ),
            "nearest_point_b_base_m": (
                None
                if minimum.nearest_point_b is None
                else list(minimum.nearest_point_b)
            ),
            "intersecting": bool(minimum.intersecting),
            "penetration_depth_trustworthy": bool(
                minimum.penetration_depth_trustworthy
            ),
            "path_semantics": "piecewise linear in joint space over the bound q samples",
            "authoritative_for": "the bound joint-waypoint prefix only",
            "not_authoritative_for": "any altered mount or hand configuration",
            "fixed_mount_pair_exclusions": list(exclusions),
            "checked_non_mount_link_count": len(checked_names),
            "rigid_relative_transform_invariant": True,
            "relative_transform": "inv(T_base_EE@T_EE_adapter)@(T_base_EE@T_EE_hand)",
            "max_q_tracking_error_rad": float(max_q_tracking_error_rad),
            "joint_tracking_uncertainty_applied": True,
            "continuous_segment_envelope_verified": True,
            "conservative_motion_bound_m": 0.0,
            "minimum_distance_is_after_motion_bound": True,
            "nominal_minimum_signed_distance_m": float(minimum.distance_m),
            "motion_bound_method": (
                "zero: V7 and open RH56 share the same rigid EE transform, "
                "including interpolation and tracking error"
            ),
        },
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError("output exists; pass --overwrite: {}".format(output))
    temporary = output.with_name(".{}.{}.tmp".format(output.name, os.getpid()))
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(output))


def _generate(args: argparse.Namespace) -> int:
    config, config_path = load_control_config(args.config.expanduser().resolve())
    snapshot_path = args.snapshot.expanduser().resolve()
    snapshot = load_snapshot_npz(snapshot_path)
    if not is_official_snapshot(snapshot):
        raise SystemExit("snapshot is not provenance-complete official AnyDex output")
    index = int(args.candidate_index)
    if index < 0 or index >= snapshot.grasps.count:
        raise SystemExit("candidate index is outside the snapshot")
    if snapshot.grasps.hand_angles is None or snapshot.grasps.widths_m is None:
        raise SystemExit("snapshot lacks Inspire targets or grasp widths")

    (
        scene_path,
        scene,
        capture_q,
        captured,
        filter_evidence_path,
        _filter_evidence,
        scene_metadata,
    ) = _load_filtered_scene(args.filtered_scene)
    if snapshot.calibration_id != scene_metadata["calibration_id"]:
        raise SystemExit("snapshot/scene calibration_id mismatch")
    if snapshot.camera_serial != scene_metadata["camera_serial"]:
        raise SystemExit("snapshot/scene camera_serial mismatch")

    joint_plan_path = args.joint_plan.expanduser().resolve()
    plan_artifact = _load_joint_plan(joint_plan_path)
    if plan_artifact["inputs"]["config"]["sha256"] != _sha256_file(config_path):
        raise SystemExit("joint plan is not bound to the current control profile")
    if plan_artifact["inputs"]["snapshot"]["sha256"] != _sha256_file(snapshot_path):
        raise SystemExit("joint plan is not bound to the selected snapshot")
    if int(plan_artifact["candidate"]["index"]) != index:
        raise SystemExit("joint plan candidate differs from --candidate-index")
    plan = plan_artifact["joint_plan"]
    q_current = np.asarray(plan["q_start_rad"], dtype=np.float64)
    q_default = np.asarray(plan["q_default_rad"], dtype=np.float64)
    q_pregrasp = np.asarray(plan["q_pregrasp_rad"], dtype=np.float64)
    scene_prefix_q_linf = _advisory_scene_prefix_q_delta(
        capture_q, q_current
    )
    print(
        "[pregrasp-audit/advisory] scene capture q is provenance only; "
        "capture-to-prefix-current linf={:.7f}rad, prefix current remains "
        "hard-bound to the joint plan and runtime live-q gate".format(
            scene_prefix_q_linf
        )
    )
    if not np.array_equal(
        q_default, np.asarray(config["franka"]["default_q_rad"], dtype=np.float64)
    ):
        raise SystemExit("joint plan default q differs from the control profile")
    pregrasp_pose = np.asarray(
        plan_artifact["air_geometry"]["pregrasp_pose_base_EE"], dtype=np.float64
    )
    max_joint_step = float(plan["requested_max_joint_step_rad"])
    if not 0.001 <= max_joint_step <= 0.02:
        raise SystemExit("joint-plan max step is outside [0.001,0.02]")
    waypoints = (
        ("current", q_current),
        ("default", q_default),
        ("pregrasp", q_pregrasp),
    )
    q_path, segments = build_pregrasp_joint_path(waypoints, max_joint_step)

    hand = InspireHandModel.from_anydex_root(
        args.anydex_root.expanduser().resolve(), mesh_resolution="full"
    )
    type_id = int(np.asarray(snapshot.grasps.type_ids)[index])
    closure = hand.configuration(
        type_id, float(np.asarray(snapshot.grasps.widths_m)[index])
    )
    selected_targets = np.asarray(snapshot.grasps.hand_angles[index], dtype=np.float64)
    if not np.array_equal(np.rint(closure.actuator_registers), selected_targets):
        raise SystemExit("official hand mapping differs from snapshot targets")
    open_transforms = hand.link_mesh_transforms(
        np.eye(4), OFFICIAL_OPEN_JOINT_POSITIONS_RAD
    )
    closed_transforms = hand.link_mesh_transforms(
        np.eye(4), closure.joint_positions_rad
    )
    mesh_dir = hand.urdf_path.parent.parent / "meshes"
    mesh_paths = {
        item.name: (mesh_dir / item.mesh_filename).resolve() for item in hand.links
    }
    tracking = float(args.max_q_tracking_error_rad)
    if not 0.0 < tracking <= 0.01:
        raise SystemExit("--max-q-tracking-error-rad must be in (0,0.01]")
    backend = HppFclInstalledToolBackend(
        HppFclInstalledToolConfig(max_q_tracking_error_rad=tracking)
    )
    T_EE_hand = np.asarray(config["tool"]["T_EE_hand"], dtype=np.float64)
    request = _collision_request(
        scene=scene,
        object_points=snapshot.object_points,
        adapter_path=args.adapter.expanduser().resolve(),
        T_EE_hand=T_EE_hand,
        hand_model=hand,
        max_q_tracking_error_rad=tracking,
    )
    query = InstalledToolCollisionQuery(
        request=request,
        q_path_rad=q_path,
        path_segments=segments,
        adapter_sha256=_sha256_file(request.adapter_stl_path),
        hand_link_mesh_paths=mesh_paths,
        T_hand_open_link_visual=open_transforms,
        T_hand_closed_link_visual=closed_transforms,
        T_hand_waypoint_link_visual=(),
        hand_dense_interval_q12_rad=(),
        hand_interval_feedback_tube_q12_rad=(),
        hand_interval_feedback_q12_lower_rad=(),
        hand_interval_feedback_q12_upper_rad=(),
    )
    started = time.time()
    observations = backend.evaluate(query)
    elapsed = time.time() - started
    specs = {item.check_id: item for item in check_specs_for_mode("loaded_grasp")}
    open_adapter_check = _open_hand_adapter_check(
        backend=backend,
        q_path=q_path,
        adapter_path=request.adapter_stl_path,
        T_EE_hand=T_EE_hand,
        hand_mesh_paths=mesh_paths,
        open_transforms=open_transforms,
        max_q_tracking_error_rad=tracking,
    )
    checks = []
    for check_id in REQUIRED_CHECK_IDS:
        if check_id == "rh56_open_adapter_path":
            checks.append(open_adapter_check)
        else:
            checks.append(
                _observation_check(
                    specs[check_id], observations[check_id], len(q_path)
                )
            )

    configured_age = float(config["grasp"]["air_audit_max_scene_age_s"])
    max_scene_age = (
        configured_age if args.max_scene_age_s is None else float(args.max_scene_age_s)
    )
    if not 0.0 < max_scene_age <= 3600.0:
        raise SystemExit("--max-scene-age-s must be in (0,3600]")
    artifact = create_pregrasp_only_audit(
        config_path=config_path,
        snapshot_path=snapshot_path,
        filtered_scene_path=scene_path,
        filter_evidence_path=filter_evidence_path,
        adapter_path=request.adapter_stl_path,
        joint_plan_path=joint_plan_path,
        candidate_index=index,
        T_EE_hand=T_EE_hand,
        scene_points=scene,
        object_points=snapshot.object_points,
        scene_capture_q=capture_q,
        scene_captured_at_s=captured,
        waypoints=waypoints,
        pregrasp_pose=pregrasp_pose,
        max_joint_step_rad=max_joint_step,
        max_q_tracking_error_rad=tracking,
        scene_clearance_margin_m=float(args.scene_clearance_margin_m),
        robot_clearance_margin_m=float(args.robot_clearance_margin_m),
        object_clearance_margin_m=float(args.object_clearance_margin_m),
        max_scene_age_s=max_scene_age,
        q_path=q_path,
        path_segments=segments,
        checks=checks,
        collision_backend={
            "name": backend.identity.name,
            "version": backend.identity.version,
            "implementation_sha256": backend.identity.implementation_sha256,
            "configuration_sha256": backend.identity.configuration_sha256,
            "evaluation_mode": "loaded_grasp_observations_prefix_subset_only",
            "elapsed_s": elapsed,
        },
        open_hand_configuration=official_open_configuration_provenance(
            args.anydex_root.expanduser().resolve()
        ),
        created_at_s=time.time(),
    )
    _atomic_write_json(args.output, artifact, bool(args.overwrite))
    print("[pregrasp-audit] output={}".format(args.output.expanduser().resolve()))
    print("[pregrasp-audit] artifact_sha256={}".format(artifact["artifact_sha256"]))
    print(
        "[pregrasp-audit] passed={} samples={} elapsed={:.2f}s motion_authorized=false".format(
            artifact["decision"]["passed"], len(q_path), elapsed
        )
    )
    scene_binding = artifact["bindings"]["filtered_scene"]
    if scene_binding["age_exceeded_recommended_max"]:
        print(
            "[pregrasp-audit][advisory] scene age at creation={:.1f}s exceeds "
            "recommended {:.1f}s; runtime still requires a fresh exact "
            "workspace-clear token".format(
                float(scene_binding["age_at_artifact_creation_s"]),
                float(scene_binding["configured_recommended_max_scene_age_s"]),
            )
        )
    for item in artifact["checks"]:
        role = "advisory" if item["advisory_only"] else "required"
        print(
            "  {} ({}) clearance={:.6f}m passed={}".format(
                item["check_id"],
                role,
                float(item["minimum_signed_distance_m"]),
                item["required_geometry_passed"],
            )
        )
    return 0 if artifact["decision"]["passed"] else 2


def _verify(args: argparse.Namespace) -> int:
    artifact = load_pregrasp_only_audit(
        args.artifact,
        verify_files=bool(args.verify_files),
        require_pass=bool(args.require_pass),
    )
    print("[pregrasp-audit] replay valid")
    print("[pregrasp-audit] artifact_sha256={}".format(artifact["artifact_sha256"]))
    print("[pregrasp-audit] passed={}".format(artifact["decision"]["passed"]))
    print("[pregrasp-audit] motion_authorized=false")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate/replay offline FR3 + open RH56 pregrasp-prefix evidence"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="run offline collision backend")
    generate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    generate.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    generate.add_argument("--filtered-scene", type=Path, default=DEFAULT_FILTERED_SCENE)
    generate.add_argument("--joint-plan", type=Path, default=DEFAULT_JOINT_PLAN)
    generate.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    generate.add_argument("--anydex-root", type=Path, default=DEFAULT_ANYDEX_ROOT)
    generate.add_argument("--candidate-index", type=int, default=51)
    generate.add_argument("--max-q-tracking-error-rad", type=float, default=0.002)
    generate.add_argument("--scene-clearance-margin-m", type=float, default=0.005)
    generate.add_argument("--robot-clearance-margin-m", type=float, default=0.002)
    generate.add_argument("--object-clearance-margin-m", type=float, default=0.002)
    generate.add_argument(
        "--max-scene-age-s",
        type=float,
        default=None,
        help=(
            "Recommended advisory-scene age recorded as provenance; default "
            "uses the profile value. Exceeding it prints a warning but never "
            "replaces or weakens prefix live-q/object/path/operator gates."
        ),
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--overwrite", action="store_true")

    verify = commands.add_parser("verify", help="replay artifact integrity")
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--verify-files", action="store_true")
    verify.add_argument("--require-pass", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return _generate(args)
    return _verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
