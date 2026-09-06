#!/usr/bin/python3
"""Replay one FR3/V7/RH56 joint path through the native collision backend.

This program is deliberately offline.  It imports neither libfranka nor the
Inspire serial driver, and its JSON output is diagnostic evidence rather than
an execution audit or a motion command.
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
from typing import Any, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from anydex_pipeline.control_plan import is_official_snapshot  # noqa: E402
from anydex_pipeline.hppfcl_installed_tool_backend import (  # noqa: E402
    HppFclInstalledToolBackend,
    HppFclInstalledToolConfig,
)
from anydex_pipeline.inspire_hand_model import InspireHandModel  # noqa: E402
from anydex_pipeline.inspire_open_configuration import (  # noqa: E402
    OFFICIAL_OPEN_JOINT_POSITIONS_RAD,
    official_open_configuration_provenance,
)
from anydex_pipeline.rh56_actuator_mapping import (  # noqa: E402
    OfficialRH56ActuatorMapper,
)
from anydex_pipeline.rh56_hand_path import (  # noqa: E402
    Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS,
    build_rh56_no_contact_execution_path,
    rh56_hand_interval_feedback_q12_envelopes,
    rh56_hand_interval_q12_paths_and_feedback_tubes,
)
from anydex_pipeline.installed_tool_audit import (  # noqa: E402
    InstalledToolCollisionQuery,
    V7_T_EE_HAND,
    build_joint_path,
    check_specs_for_mode,
)
from anydex_pipeline.snapshot import load_snapshot_npz  # noqa: E402


DEFAULT_ADAPTER = ROOT / "assets/adapter/V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.stl"
DEFAULT_ANYDEX_ROOT = ROOT / "third_party/AnyDexGrasp"
DEFAULT_Q = (0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, 0.0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(
        ("dtype=<f8;shape={};".format(",".join(map(str, array.shape)))).encode(
            "ascii"
        )
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_filtered_scene(path: Path):
    source = path.expanduser().resolve()
    with np.load(str(source), allow_pickle=False) as archive:
        required = {"artifact_type", "scene_points", "capture_q_rad"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError("filtered scene is missing {}".format(missing))
        if str(np.asarray(archive["artifact_type"]).item()) != (
            "installed_filtered_live_scene"
        ):
            raise ValueError("--filtered-scene is not an installed-filter artifact")
        points = np.asarray(archive["scene_points"], dtype=np.float64)
        capture_q = np.asarray(archive["capture_q_rad"], dtype=np.float64)
        metadata = {
            key: np.asarray(archive[key]).item()
            for key in (
                "captured_at_unix_s",
                "calibration_id",
                "camera_serial",
                "filter_evidence_path",
                "source_live_scene_sha256",
            )
            if key in archive and np.asarray(archive[key]).shape == ()
        }
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise ValueError("filtered scene_points must be non-empty (N,3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("filtered scene contains NaN or infinity")
    if capture_q.shape != (7,) or not np.all(np.isfinite(capture_q)):
        raise ValueError("filtered scene capture_q_rad must be a finite 7-vector")
    evidence_path = Path(str(metadata.get("filter_evidence_path", "")))
    if not evidence_path.is_file():
        raise ValueError("filtered scene has no readable filter evidence")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("motion_authorized") is not False:
        raise ValueError("filter evidence must explicitly remain non-authorizing")
    return source, points, capture_q, metadata, evidence_path.resolve(), evidence


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError("{} must contain seven finite radians".format(name))
    return result


def _diagnostic_collision_request(
    *,
    mode: str,
    scene: np.ndarray,
    object_points: np.ndarray,
    adapter_stl_path: Path,
    max_q_tracking_error_rad: float,
    hand_arrival_tolerance_units: int,
    hand_model: InspireHandModel,
) -> SimpleNamespace:
    """Build the complete backend request contract used by offline diagnostics."""

    return SimpleNamespace(
        mode=mode,
        scene_points_base=scene,
        object_points_base=object_points,
        adapter_stl_path=adapter_stl_path,
        T_EE_hand=V7_T_EE_HAND,
        allowed_object_contact_links=("Link11", "Link22", "Link33", "Link44", "Link53"),
        max_q_tracking_error_rad=float(max_q_tracking_error_rad),
        robot_clearance_margin_m=0.002,
        hand_self_clearance_margin_m=0.0,
        hand_arrival_tolerance_units=int(hand_arrival_tolerance_units),
        q6_reverse_hysteresis_tolerance_units=(
            Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS
        ),
        scene_voxel_resolution_m=0.005,
        observed_scene_scope="calibrated_camera_frustum_voxel_grid",
        unknown_space_policy="occupied",
        hand_model=hand_model,
    )


def _policy_result(spec: Any, observation: Any) -> Mapping[str, Any]:
    margins = {
        "robot": 0.002,
        "hand_self": 0.0,
        "scene": 0.005,
        "object_approach": 0.002,
    }
    distance = float(observation.minimum_signed_distance_m)
    failures = []
    if not observation.authoritative:
        failures.append("observation is non-authoritative")
    if spec.expectation == "clear":
        margin = margins[spec.margin_policy]
        if distance <= margin:
            failures.append(
                "distance {:.6f}m does not exceed {:.6f}m".format(distance, margin)
            )
        if observation.observed_pairs:
            failures.append("collision pair observed")
    else:
        if distance > 0.002:
            failures.append("hand/object distance exceeds 0.002m contact threshold")
        if distance < -0.003:
            failures.append("hand/object penetration exceeds 0.003m")
    return {
        "check_id": spec.check_id,
        "scope": spec.scope,
        "expectation": spec.expectation,
        "coverage": spec.coverage,
        "authoritative": bool(observation.authoritative),
        "minimum_signed_distance_m": distance,
        "observed_pairs": list(observation.observed_pairs),
        "details": dict(observation.details or {}),
        "diagnostic_policy_passed": not failures,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Pinocchio/HPP-FCL replay of exact FR3, V7 and all 13 RH56 "
            "meshes against a filtered captured scene. Never commands hardware."
        )
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--filtered-scene", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--mode", choices=("air_grasp", "loaded_grasp"), default="air_grasp")
    parser.add_argument(
        "--check-id",
        action="append",
        default=[],
        choices=sorted(
            {
                spec.check_id
                for mode in ("air_grasp", "loaded_grasp")
                for spec in check_specs_for_mode(mode)
            }
        ),
        help=(
            "evaluate only the named offline diagnostic check; repeat for a subset. "
            "Omit to evaluate the normal complete set"
        ),
    )
    parser.add_argument("--current-q", type=float, nargs=7, default=None)
    parser.add_argument(
        "--default-transit-q", type=float, nargs=7, action="append", default=[]
    )
    parser.add_argument(
        "--approach-transit-q", type=float, nargs=7, action="append", default=[]
    )
    parser.add_argument("--default-q", type=float, nargs=7, default=DEFAULT_Q)
    parser.add_argument("--pregrasp-q", type=float, nargs=7, required=True)
    parser.add_argument("--grasp-q", type=float, nargs=7, required=True)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.02)
    parser.add_argument("--max-q-tracking-error-rad", type=float, default=0.002)
    parser.add_argument(
        "--hand-arrival-tolerance-units",
        type=int,
        default=25,
        help="All-six RH56 feedback uncertainty used by the air-path collision tube",
    )
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--anydex-root", type=Path, default=DEFAULT_ANYDEX_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    normal_specs = check_specs_for_mode(args.mode)
    normal_ids = {spec.check_id for spec in normal_specs}
    unknown_for_mode = sorted(set(args.check_id) - normal_ids)
    if unknown_for_mode:
        raise SystemExit(
            "--check-id is not part of {}: {}".format(
                args.mode, ", ".join(unknown_for_mode)
            )
        )
    selected_specs = tuple(
        spec
        for spec in normal_specs
        if not args.check_id or spec.check_id in set(args.check_id)
    )
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite: {}".format(output))
    snapshot_path = args.snapshot.expanduser().resolve()
    snapshot = load_snapshot_npz(snapshot_path)
    if not is_official_snapshot(snapshot):
        raise SystemExit("snapshot is not provenance-complete official AnyDexGrasp output")
    index = int(args.candidate_index)
    if index < 0 or index >= snapshot.grasps.count:
        raise SystemExit("candidate index is outside the snapshot")
    scene_path, scene, capture_q, scene_meta, filter_evidence_path, filter_evidence = (
        _load_filtered_scene(args.filtered_scene)
    )
    if snapshot.calibration_id != str(scene_meta.get("calibration_id", "")):
        raise SystemExit("snapshot/filtered-scene calibration_id mismatch")
    if snapshot.camera_serial != str(scene_meta.get("camera_serial", "")):
        raise SystemExit("snapshot/filtered-scene camera_serial mismatch")

    hand = InspireHandModel.from_anydex_root(args.anydex_root, mesh_resolution="full")
    type_id = int(np.asarray(snapshot.grasps.type_ids)[index])
    widths = snapshot.grasps.widths_m
    if widths is None:
        raise SystemExit("snapshot has no grasp widths")
    closure = hand.configuration(type_id, float(np.asarray(widths)[index]))
    if snapshot.grasps.hand_angles is None:
        raise SystemExit("snapshot has no Inspire actuator targets")
    selected_targets = np.asarray(snapshot.grasps.hand_angles[index], dtype=np.float64)
    if not np.array_equal(np.rint(closure.actuator_registers), selected_targets):
        raise SystemExit("official hand mapping differs from snapshot actuator targets")

    current_q = capture_q if args.current_q is None else _vector(args.current_q, "current-q")
    default_q = _vector(args.default_q, "default-q")
    pregrasp_q = _vector(args.pregrasp_q, "pregrasp-q")
    grasp_q = _vector(args.grasp_q, "grasp-q")
    transits = tuple(
        _vector(item, "default-transit-q[{}]".format(i))
        for i, item in enumerate(args.default_transit_q)
    )
    approach_transits = tuple(
        _vector(item, "approach-transit-q[{}]".format(i))
        for i, item in enumerate(args.approach_transit_q)
    )
    q_path, segments = build_joint_path(
        current_q,
        default_q,
        pregrasp_q,
        grasp_q,
        max_joint_step_rad=float(args.max_joint_step_rad),
        default_transit_q_rad=transits,
        approach_transit_q_rad=approach_transits,
    )
    open_transforms = hand.link_mesh_transforms(
        np.eye(4), OFFICIAL_OPEN_JOINT_POSITIONS_RAD
    )
    closed_transforms = hand.link_mesh_transforms(
        np.eye(4), closure.joint_positions_rad
    )
    if not 0 <= int(args.hand_arrival_tolerance_units) <= 100:
        raise SystemExit("--hand-arrival-tolerance-units must be in 0..100")
    if args.mode == "air_grasp":
        hand_path = build_rh56_no_contact_execution_path(
            tuple(int(value) for value in selected_targets), step_units=25
        )
        mapper = OfficialRH56ActuatorMapper(
            hand.mapping_path.parent / "driver_routine_to_angle.xls"
        )
        hand_waypoint_transforms = tuple(
            hand.link_mesh_transforms(
                np.eye(4), mapper.to_joint_positions_rad(item.actuator_configuration)
            )
            for item in hand_path.waypoints
        )
        hand_dense_intervals, hand_feedback_tubes = (
            rh56_hand_interval_q12_paths_and_feedback_tubes(
                hand_path,
                mapper,
                int(args.hand_arrival_tolerance_units),
            )
        )
        hand_feedback_lower, hand_feedback_upper = (
            rh56_hand_interval_feedback_q12_envelopes(
                hand_path,
                mapper,
                int(args.hand_arrival_tolerance_units),
            )
        )
    else:
        hand_waypoint_transforms = ()
        hand_dense_intervals = ()
        hand_feedback_tubes = ()
        hand_feedback_lower = ()
        hand_feedback_upper = ()
    mesh_dir = hand.urdf_path.parent.parent / "meshes"
    mesh_paths = {
        item.name: (mesh_dir / item.mesh_filename).resolve() for item in hand.links
    }
    backend = HppFclInstalledToolBackend(
        HppFclInstalledToolConfig(
            max_q_tracking_error_rad=float(args.max_q_tracking_error_rad)
        )
    )
    request = _diagnostic_collision_request(
        mode=args.mode,
        scene=scene,
        object_points=snapshot.object_points,
        adapter_stl_path=args.adapter.expanduser().resolve(),
        max_q_tracking_error_rad=float(args.max_q_tracking_error_rad),
        hand_arrival_tolerance_units=int(args.hand_arrival_tolerance_units),
        hand_model=hand,
    )
    query = InstalledToolCollisionQuery(
        request=request,
        q_path_rad=q_path,
        path_segments=segments,
        adapter_sha256=_sha256_file(request.adapter_stl_path),
        hand_link_mesh_paths=mesh_paths,
        T_hand_open_link_visual=open_transforms,
        T_hand_closed_link_visual=closed_transforms,
        T_hand_waypoint_link_visual=hand_waypoint_transforms,
        hand_dense_interval_q12_rad=hand_dense_intervals,
        hand_interval_feedback_tube_q12_rad=hand_feedback_tubes,
        hand_interval_feedback_q12_lower_rad=hand_feedback_lower,
        hand_interval_feedback_q12_upper_rad=hand_feedback_upper,
    )
    started = time.time()
    observations = backend.evaluate_checks(
        query, tuple(spec.check_id for spec in selected_specs)
    )
    elapsed = time.time() - started
    checks = [
        _policy_result(spec, observations[spec.check_id])
        for spec in selected_specs
    ]
    source_capture_time = float(scene_meta.get("captured_at_unix_s", 0.0))
    result = {
        "artifact_type": "fr3_v7_rh56_offline_collision_diagnostic",
        "schema_version": 1,
        "created_at_unix_s": time.time(),
        "meaning": "offline replay evidence only; never a motion command",
        "diagnostic_subset": [spec.check_id for spec in selected_specs],
        "motion_authorized": False,
        "mode": args.mode,
        "backend": {
            "name": backend.identity.name,
            "version": backend.identity.version,
            "implementation_sha256": backend.identity.implementation_sha256,
            "configuration_sha256": backend.identity.configuration_sha256,
            "elapsed_s": elapsed,
        },
        "sources": {
            "snapshot_path": str(snapshot_path),
            "snapshot_sha256": _sha256_file(snapshot_path),
            "filtered_scene_path": str(scene_path),
            "filtered_scene_sha256": _sha256_file(scene_path),
            "filter_evidence_path": str(filter_evidence_path),
            "filter_evidence_sha256": _sha256_file(filter_evidence_path),
            "adapter_path": str(request.adapter_stl_path),
            "adapter_sha256": _sha256_file(request.adapter_stl_path),
        },
        "scene": {
            "point_count": int(len(scene)),
            "points_sha256": _sha256_array(scene),
            "captured_at_unix_s": source_capture_time,
            "age_at_diagnostic_s": (
                time.time() - source_capture_time if source_capture_time > 0 else None
            ),
            "filter_motion_authorized": filter_evidence.get("motion_authorized"),
            "unknown_camera_space_verified": False,
        },
        "candidate": {
            "index": index,
            "type_id": type_id,
            "width_m": float(closure.width_m),
            "actuator_targets": selected_targets.tolist(),
            "closure_joint_positions_rad": closure.joint_positions_rad.tolist(),
        },
        "open_hand_configuration": dict(
            official_open_configuration_provenance(args.anydex_root)
        ),
        "joint_path": {
            "semantics": "piecewise linear in joint space",
            "sample_count": int(len(q_path)),
            "samples_sha256": _sha256_array(q_path),
            "samples_rad": q_path.tolist(),
            "segments": [dict(item) for item in segments],
            "maximum_joint_step_rad": float(
                np.max(np.abs(np.diff(q_path, axis=0)))
            ),
            "max_q_tracking_error_rad": float(args.max_q_tracking_error_rad),
        },
        "checks": checks,
        "summary": {
            "all_geometric_and_authority_policies_passed": all(
                item["diagnostic_policy_passed"] for item in checks
            ),
            "authoritative_check_count": sum(
                bool(item["authoritative"]) for item in checks
            ),
            "required_check_count": len(checks),
            "blocking_note": (
                "captured point checks remain non-authoritative for occluded space"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(".{}.{}.tmp".format(output.name, os.getpid()))
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(output))
    print(
        "[collision] checks={} authoritative={} all_passed={} elapsed={:.2f}s".format(
            len(checks),
            result["summary"]["authoritative_check_count"],
            result["summary"]["all_geometric_and_authority_policies_passed"],
            elapsed,
        )
    )
    print("[collision] output={}".format(output))
    print("[collision] motion_authorized=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
