#!/usr/bin/python3
"""Generate a strict schema-v2 installed-tool audit for an unloaded air grasp.

The program is offline: it never imports a robot or RH56 driver.  Captured
point evidence remains non-authoritative and is carried as an explicit
runtime workspace-clear condition.  Loaded/contact audit generation is not
exposed by this entry point.  Joint waypoints come only from a checksummed,
deterministic installed-air joint-plan manifest; copied CLI joint vectors are
not accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from anydex_pipeline.control_config import (  # noqa: E402
    load_control_config,
    verify_adapter_assets,
)
from anydex_pipeline.control_plan import (  # noqa: E402
    is_official_snapshot,
    validate_rigid_transform,
)
from anydex_pipeline.air_target_poses import derive_air_target_poses  # noqa: E402
from anydex_pipeline.air_audit_readiness import (  # noqa: E402
    validate_air_candidate_commissioning,
)
from anydex_pipeline.hppfcl_installed_tool_backend import (  # noqa: E402
    HppFclInstalledToolBackend,
    HppFclInstalledToolConfig,
)
from anydex_pipeline.inspire_hand_model import InspireHandModel  # noqa: E402
from anydex_pipeline.inspire_open_configuration import (  # noqa: E402
    OFFICIAL_OPEN_ACTUATOR_TARGETS,
    OFFICIAL_OPEN_JOINT_POSITIONS_RAD,
)
from anydex_pipeline.joint_path_sampling import (  # noqa: E402
    CANONICAL_JOINT_SAMPLING_ALGORITHM,
    canonical_joint_path_samples,
)
from anydex_pipeline.rh56_hand_path import (  # noqa: E402
    validate_rh56_hand_execution_path,
)
from anydex_pipeline.rh56_commissioning import json_sha256  # noqa: E402
from anydex_pipeline.installed_tool_audit import (  # noqa: E402
    InstalledToolAuditRequest,
    RUNTIME_WORKSPACE_CLEAR_TOKEN,
    V7_T_EE_HAND,
    build_installed_tool_collision_query,
    run_installed_tool_audit,
    write_installed_tool_audit,
)
from anydex_pipeline.installed_tool_static_cache import (  # noqa: E402
    StaticCacheCombinedBackend,
    build_static_cache_artifact,
    write_static_cache_artifact,
)
from anydex_pipeline.snapshot import load_snapshot_npz  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
DEFAULT_ANYDEX_ROOT = ROOT / "third_party/AnyDexGrasp"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(archive: Any, key: str) -> Any:
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError("{} must be a scalar".format(key))
    return value.item()


def _load_filtered_scene(path: Path):
    source = path.expanduser().resolve()
    with np.load(str(source), allow_pickle=False) as archive:
        required = {
            "artifact_type",
            "filtered_scene_points",
            "reference_frame",
            "scene_excludes_object",
            "capture_q_rad",
            "capture_q_source",
            "captured_at_unix_s",
            "calibration_id",
            "camera_serial",
            "filter_evidence_path",
            "filter_evidence_sha256",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError("filtered scene is missing {}".format(missing))
        if _scalar(archive, "artifact_type") != "installed_filtered_live_scene":
            raise ValueError("filtered scene artifact_type is invalid")
        if _scalar(archive, "reference_frame") != "robot_base":
            raise ValueError("filtered scene reference_frame must be robot_base")
        if bool(_scalar(archive, "scene_excludes_object")) is not True:
            raise ValueError("filtered scene must explicitly exclude the object")
        points = np.asarray(archive["filtered_scene_points"], dtype=np.float64)
        capture_q = np.asarray(archive["capture_q_rad"], dtype=np.float64)
        capture_q_source = str(_scalar(archive, "capture_q_source"))
        captured_at = float(_scalar(archive, "captured_at_unix_s"))
        calibration_id = str(_scalar(archive, "calibration_id"))
        camera_serial = str(_scalar(archive, "camera_serial"))
        evidence_path = Path(str(_scalar(archive, "filter_evidence_path"))).resolve()
        expected_evidence_hash = str(_scalar(archive, "filter_evidence_sha256"))
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise ValueError("filtered scene points must be non-empty (N,3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("filtered scene points contain NaN or infinity")
    if capture_q.shape != (7,) or not np.all(np.isfinite(capture_q)):
        raise ValueError("filtered scene capture q must be a finite 7-vector")
    if capture_q_source != "cli_asserted":
        raise ValueError(
            "filtered scene capture_q_source must be cli_asserted; this field "
            "cannot claim an automatic Franka measurement"
        )
    if not np.isfinite(captured_at) or captured_at <= 0.0:
        raise ValueError("filtered scene capture timestamp is invalid")
    if not evidence_path.is_file():
        raise ValueError("filtered scene evidence sidecar is missing")
    if _sha256_file(evidence_path) != expected_evidence_hash:
        raise ValueError("filtered scene evidence hash differs from its NPZ binding")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("motion_authorized") is not False
        or evidence.get("authoritative_for_unseen_camera_space") is not False
        or evidence.get("unknown_space_policy_applied") is not False
    ):
        raise ValueError("filtered scene evidence illegally promotes point authority")
    return (
        source,
        points,
        capture_q,
        capture_q_source,
        captured_at,
        calibration_id,
        camera_serial,
        evidence_path,
    )


def _q(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError("{} must contain seven finite radians".format(name))
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(
        ("dtype=<f8;shape={};".format(",".join(map(str, array.shape)))).encode(
            "ascii"
        )
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _manifest_joint_path(
    waypoints: Sequence[np.ndarray], max_step: float
) -> tuple[np.ndarray, tuple[int, ...]]:
    return canonical_joint_path_samples(waypoints, max_step)


def _load_joint_plan(
    path: Path,
    *,
    config_path: Path,
    snapshot_path: Path,
    urdf_path: Path,
    candidate_index: int,
    canonical_pose: np.ndarray,
    hand_pose: np.ndarray,
    hand_targets: np.ndarray,
    capture_q: np.ndarray,
    default_q: np.ndarray,
    joint_limits: np.ndarray,
    joint_limit_margin: float,
    retreat_distance_m: float,
    pregrasp_distance_m: float,
    pregrasp_pose: np.ndarray,
    final_pose: np.ndarray,
) -> tuple[Path, np.ndarray, np.ndarray, float, Mapping[str, Any]]:
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    required_root = {
        "air_geometry", "artifact_type", "cable_constraint", "candidate",
        "collision_check", "fk_residual", "inputs", "integrity",
        "hand_execution_path", "joint_plan", "motion_authorized", "planner", "schema_version",
    }
    if not isinstance(payload, dict) or set(payload) != required_root:
        raise ValueError("joint-plan manifest root schema is invalid")
    integrity = payload["integrity"]
    unsealed = dict(payload)
    del unsealed["integrity"]
    if (
        not isinstance(integrity, dict)
        or integrity.get("algorithm")
        != "sha256-canonical-json-without-integrity"
        or integrity.get("payload_sha256") != json_sha256(unsealed)
    ):
        raise ValueError("joint-plan manifest self checksum is invalid")
    if (
        payload["schema_version"] != 1
        or payload["artifact_type"] != "installed_air_candidate_joint_plan"
        or payload["motion_authorized"] is not False
    ):
        raise ValueError("joint-plan manifest type/authority is invalid")
    hand_path = validate_rh56_hand_execution_path(payload["hand_execution_path"])
    if hand_path.target != tuple(int(value) for value in hand_targets):
        raise ValueError("joint-plan hand execution path target differs from candidate")

    expected_inputs = {
        "config": config_path,
        "snapshot": snapshot_path,
        "fr3_urdf": urdf_path,
    }
    inputs = payload["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != set(expected_inputs):
        raise ValueError("joint-plan input bindings are invalid")
    for name, expected_path in expected_inputs.items():
        binding = inputs[name]
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or Path(str(binding["path"])).resolve() != expected_path.resolve()
            or binding["sha256"] != _sha256_file(expected_path)
        ):
            raise ValueError("joint-plan {} binding differs".format(name))

    candidate = payload["candidate"]
    if (
        not isinstance(candidate, dict)
        or candidate.get("index") != candidate_index
        or not np.array_equal(
            np.asarray(candidate.get("hand_targets"), dtype=np.float64), hand_targets
        )
        or not np.allclose(
            np.asarray(candidate.get("canonical_pose_base"), dtype=np.float64),
            canonical_pose,
            atol=1e-12,
            rtol=0.0,
        )
        or not np.allclose(
            np.asarray(candidate.get("hand_pose_base"), dtype=np.float64),
            hand_pose,
            atol=1e-12,
            rtol=0.0,
        )
    ):
        raise ValueError("joint-plan selected candidate differs from the audit")

    geometry = payload["air_geometry"]
    if (
        not isinstance(geometry, dict)
        or not np.isclose(
            float(geometry.get("retreat_distance_m", -1.0)),
            retreat_distance_m,
            atol=1e-15,
            rtol=0.0,
        )
        or not np.isclose(
            float(geometry.get("pregrasp_extra_distance_m", -1.0)),
            pregrasp_distance_m,
            atol=1e-15,
            rtol=0.0,
        )
        or not np.allclose(
            np.asarray(geometry.get("pregrasp_pose_base_EE"), dtype=np.float64),
            pregrasp_pose,
            atol=1e-10,
            rtol=0.0,
        )
        or not np.allclose(
            np.asarray(geometry.get("final_air_pose_base_EE"), dtype=np.float64),
            final_pose,
            atol=1e-10,
            rtol=0.0,
        )
    ):
        raise ValueError("joint-plan air geometry differs from the audit")

    planner = payload["planner"]
    ik = planner.get("ik", {}) if isinstance(planner, dict) else {}
    if (
        planner.get("name") != "plan_installed_air_candidate"
        or planner.get("version") != 1
        or planner.get("deterministic") is not True
        or ik.get("method") != "damped_least_squares_fixed_q7"
        or ik.get("random_starts") != 0
        or float(ik.get("position_tolerance_m", 1.0)) > 1e-9
        or float(ik.get("rotation_tolerance_rad", 1.0)) > 1e-9
    ):
        raise ValueError("joint-plan deterministic IK contract is invalid")
    residual = payload["fk_residual"]
    if (
        not isinstance(residual, dict)
        or residual.get("passed") is not True
        or float(residual.get("pregrasp_position_m", 1.0))
        > float(ik["position_tolerance_m"])
        or float(residual.get("final_air_position_m", 1.0))
        > float(ik["position_tolerance_m"])
        or float(residual.get("pregrasp_rotation_rad", 1.0))
        > float(ik["rotation_tolerance_rad"])
        or float(residual.get("final_air_rotation_rad", 1.0))
        > float(ik["rotation_tolerance_rad"])
    ):
        raise ValueError("joint-plan FK residual evidence is invalid")

    plan = payload["joint_plan"]
    q_start = _q(plan.get("q_start_rad"), "joint-plan q_start")
    q_default = _q(plan.get("q_default_rad"), "joint-plan q_default")
    q_pre = _q(plan.get("q_pregrasp_rad"), "joint-plan q_pregrasp")
    q_final = _q(plan.get("q_final_air_rad"), "joint-plan q_final_air")
    maximum = float(plan.get("requested_max_joint_step_rad", 0.0))
    if (
        plan.get("trajectory_contract") != "piecewise_linear_joint_space_v1"
        or plan.get(
            "sampling_algorithm", CANONICAL_JOINT_SAMPLING_ALGORITHM
        )
        != CANONICAL_JOINT_SAMPLING_ALGORITHM
        or plan.get("waypoint_order")
        != ["start", "default", "pregrasp", "final_air"]
        or plan.get("joint_limits_with_margin_passed") is not True
        or not np.array_equal(q_start, capture_q)
        or not np.array_equal(q_default, default_q)
        or not np.array_equal(
            np.asarray(plan.get("configured_joint_limits_rad"), dtype=np.float64),
            joint_limits,
        )
        or not np.isclose(
            float(plan.get("joint_limit_margin_rad", -1.0)),
            joint_limit_margin,
            atol=1e-15,
            rtol=0.0,
        )
        or not 0.001 <= maximum <= 0.02
    ):
        raise ValueError(
            "joint-plan path/config contract is invalid; q_start must equal the "
            "fresh capture and q_default must equal the control profile"
        )
    lower = joint_limits[:, 0] + joint_limit_margin
    upper = joint_limits[:, 1] - joint_limit_margin
    if any(
        np.any(q < lower) or np.any(q > upper)
        for q in (q_start, q_default, q_pre, q_final)
    ):
        raise ValueError("joint-plan waypoint violates configured joint margins")
    path_samples, intervals = _manifest_joint_path(
        (q_start, q_default, q_pre, q_final), maximum
    )
    if (
        plan.get("segment_interval_counts") != list(intervals)
        or plan.get("sample_count") != len(path_samples)
        or plan.get("q_path_sha256") != _array_sha256(path_samples)
        or float(plan.get("actual_max_joint_step_rad", 1.0)) > maximum + 1e-12
    ):
        raise ValueError("joint-plan sampled path checksum/step contract is invalid")
    collision = payload["collision_check"]
    if (
        collision.get("bare_fr3_self_collision_free") is not True
        or collision.get("scene_collision_checked") is not False
        or collision.get("adapter_collision_checked") is not False
        or collision.get("rh56_collision_checked") is not False
        or collision.get("installed_tool_audit_required") is not True
    ):
        raise ValueError("joint-plan collision scope is invalid")
    cable = payload["cable_constraint"]
    if (
        cable.get("q7_zero_assumed") is not False
        or cable.get("manual_cable_slack_confirmation_required") is not True
    ):
        raise ValueError("joint-plan cable constraint is invalid")
    return source, q_pre, q_final, maximum, hand_path.as_dict()


def _validate_candidate_q6_commissioning(
    config: Mapping[str, Any], selected_targets: np.ndarray
) -> None:
    if config["inspire"].get("six_axis_coupled_closure_commissioned") is not True:
        raise ValueError(
            "inspire.six_axis_coupled_closure_commissioned is not true"
        )
    validate_air_candidate_commissioning(config, selected_targets)


def _rotation_error_rad(actual: np.ndarray, expected: np.ndarray) -> float:
    relative = np.asarray(expected[:3, :3]).T @ np.asarray(actual[:3, :3])
    cosine = float(np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0))
    return float(np.arccos(cosine))


def _verify_ik(
    backend: HppFclInstalledToolBackend,
    q: np.ndarray,
    expected_pose: np.ndarray,
    *,
    name: str,
    max_position_error_m: float,
    max_rotation_error_rad: float,
) -> None:
    _, actual = backend._robot_state(q)  # Hardware-free Pinocchio FK.
    position_error = float(np.linalg.norm(actual[:3, 3] - expected_pose[:3, 3]))
    rotation_error = _rotation_error_rad(actual, expected_pose)
    if position_error > max_position_error_m or rotation_error > max_rotation_error_rad:
        raise ValueError(
            "{} IK/FK mismatch: position={:.6f}m rotation={:.6f}rad, limits="
            "{:.6f}m/{:.6f}rad".format(
                name,
                position_error,
                rotation_error,
                max_position_error_m,
                max_rotation_error_rad,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an offline schema-v2 conditional air-grasp audit. "
            "Loaded/contact generation is intentionally unavailable."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--filtered-scene", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--joint-plan", type=Path, required=True)
    parser.add_argument("--max-q-tracking-error-rad", type=float, default=0.002)
    parser.add_argument("--max-ik-position-error-m", type=float, default=0.0005)
    parser.add_argument("--max-ik-rotation-error-rad", type=float, default=0.001)
    parser.add_argument("--anydex-root", type=Path, default=DEFAULT_ANYDEX_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--build-static-cache",
        action="store_true",
        help=(
            "write only the scene-independent FR3/V7/RH56 collision cache to "
            "--output; never writes a schema-v2 audit"
        ),
    )
    parser.add_argument(
        "--static-cache",
        type=Path,
        help=(
            "tamper-evident static cache; when supplied, only fresh scene/object "
            "checks run before normal schema-v2 composition"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite: {}".format(output))
    if args.build_static_cache and args.static_cache is not None:
        raise SystemExit("--build-static-cache and --static-cache are mutually exclusive")
    if args.build_static_cache and output.exists():
        raise SystemExit("static cache evidence is immutable and cannot be overwritten")
    if not 0.0005 <= float(args.max_q_tracking_error_rad) <= 0.005:
        raise SystemExit("--max-q-tracking-error-rad must be in [0.0005,0.005]")
    if not 0.0005 <= float(args.max_ik_position_error_m) <= 0.005:
        raise SystemExit("--max-ik-position-error-m must be in [0.0005,0.005]")
    if not 0.001 <= float(args.max_ik_rotation_error_rad) <= 0.02:
        raise SystemExit("--max-ik-rotation-error-rad must be in [0.001,0.02]")

    config, config_path = load_control_config(args.config)
    adapter = verify_adapter_assets(config, config_path)
    if str(config["tool"].get("material", "")).lower() != "pla":
        raise SystemExit("this entry point is reviewed for the installed PLA air test")
    T_EE_hand = validate_rigid_transform(
        np.asarray(config["tool"]["T_EE_hand"], dtype=np.float64),
        "tool.T_EE_hand",
    )
    if not np.allclose(T_EE_hand, V7_T_EE_HAND, atol=1e-12, rtol=0.0):
        raise SystemExit("control profile T_EE_hand is not the commissioned V7 value")
    # Canonicalize the JSON decimal spelling to the exact commissioned array
    # whose byte hash is required by the strict audit schema.
    T_EE_hand = V7_T_EE_HAND.copy()
    retreat = float(config["grasp"]["air_retreat_distance_m"])
    pregrasp_distance = float(config["grasp"]["air_pregrasp_distance_m"])
    observed_scene_margin = float(
        config["grasp"]["air_audit_observed_scene_margin_m"]
    )
    hand_self_clearance_margin = float(
        config["grasp"].get("air_audit_rh56_self_clearance_margin_m", 0.0)
    )
    max_scene_age = float(config["grasp"]["air_audit_max_scene_age_s"])
    if not 0.0 < max_scene_age <= 120.0:
        raise SystemExit("profile air audit freshness window must be in (0,120] s")

    snapshot_path = args.snapshot.expanduser().resolve()
    snapshot = load_snapshot_npz(snapshot_path)
    if not is_official_snapshot(snapshot):
        raise SystemExit("snapshot is not provenance-complete official AnyDexGrasp output")
    index = int(args.candidate_index)
    if index < 0 or index >= snapshot.grasps.count:
        raise SystemExit("candidate index is outside the snapshot")
    if snapshot.grasps.hand_poses is None or snapshot.grasps.hand_angles is None:
        raise SystemExit("snapshot candidate has no RH56 pose/targets")
    if snapshot.grasps.widths_m is None:
        raise SystemExit("snapshot candidate has no grasp width")
    selected_targets = np.asarray(
        snapshot.grasps.hand_angles[index], dtype=np.float64
    )
    try:
        _validate_candidate_q6_commissioning(config, selected_targets)
    except ValueError as exc:
        raise SystemExit(
            "candidate is not commissioned for six-axis closure: {}".format(exc)
        )

    (
        scene_path,
        scene_points,
        capture_q,
        capture_q_source,
        captured_at,
        calibration_id,
        camera_serial,
        filter_evidence_path,
    ) = _load_filtered_scene(args.filtered_scene)
    print(
        "[air-audit][q-provenance] capture_q_source={}; operator assertion only; "
        "the artifact binds it but formal execution must independently read and "
        "compare live Franka q".format(capture_q_source)
    )
    if calibration_id != snapshot.calibration_id or camera_serial != snapshot.camera_serial:
        raise SystemExit("snapshot/filtered-scene calibration or camera differs")
    audit_started = time.time()
    scene_age = audit_started - captured_at
    if scene_age < -0.05 or scene_age > max_scene_age:
        if args.build_static_cache:
            print(
                "[static-cache] bootstrap scene age {:.2f}s is outside the "
                "final freshness window; scene/object/timestamps are excluded "
                "from this cache and a second capture remains mandatory".format(
                    scene_age
                )
            )
        else:
            print(
                "[air-audit][stale] filtered scene age {:.2f}s is outside the "
                "profile window {:.2f}s; writing a failed/expired artifact".format(
                    scene_age, max_scene_age
                )
            )

    hand = InspireHandModel.from_anydex_root(args.anydex_root, mesh_resolution="full")
    type_id = int(np.asarray(snapshot.grasps.type_ids)[index])
    closure = hand.configuration(
        type_id, float(np.asarray(snapshot.grasps.widths_m)[index])
    )
    if not np.array_equal(np.rint(closure.actuator_registers), selected_targets):
        raise SystemExit("official RH56 mapping differs from candidate targets")

    canonical_pose = validate_rigid_transform(
        np.asarray(snapshot.grasps.canonical_poses[index], dtype=np.float64),
        "candidate canonical pose",
    )
    selected_hand_pose = validate_rigid_transform(
        np.asarray(snapshot.grasps.hand_poses[index], dtype=np.float64),
        "candidate hand pose",
    )
    air_targets = derive_air_target_poses(
        canonical_pose,
        selected_hand_pose,
        np.asarray(snapshot.grasps.approach_axis_local, dtype=np.float64),
        T_EE_hand,
        retreat_distance_m=retreat,
        pregrasp_extra_distance_m=pregrasp_distance,
    )
    approach = air_targets.approach_reference
    planned_hand_pose = air_targets.T_reference_hand_final_air
    grasp_pose_eef = air_targets.T_reference_EE_final_air
    pregrasp_pose_eef = air_targets.T_reference_EE_pregrasp

    backend_config = HppFclInstalledToolConfig(
        max_q_tracking_error_rad=float(args.max_q_tracking_error_rad)
    )
    try:
        (
            joint_plan_path,
            pregrasp_q,
            grasp_q,
            max_joint_step_rad,
            hand_execution_path,
        ) = _load_joint_plan(
            args.joint_plan,
            config_path=config_path,
            snapshot_path=snapshot_path,
            urdf_path=backend_config.fr3_urdf_path,
            candidate_index=index,
            canonical_pose=canonical_pose,
            hand_pose=selected_hand_pose,
            hand_targets=selected_targets,
            capture_q=capture_q,
            default_q=np.asarray(config["franka"]["default_q_rad"], dtype=np.float64),
            joint_limits=np.asarray(
                config["franka"]["joint_limits_rad"], dtype=np.float64
            ),
            joint_limit_margin=float(config["franka"]["joint_limit_margin_rad"]),
            retreat_distance_m=retreat,
            pregrasp_distance_m=pregrasp_distance,
            pregrasp_pose=pregrasp_pose_eef,
            final_pose=grasp_pose_eef,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit("joint-plan manifest rejected: {}".format(exc))
    backend = HppFclInstalledToolBackend(
        backend_config,
        progress_callback=lambda message: print(
            "[collision-progress] {}".format(message), flush=True
        ),
    )
    _verify_ik(
        backend,
        pregrasp_q,
        pregrasp_pose_eef,
        name="pregrasp",
        max_position_error_m=float(args.max_ik_position_error_m),
        max_rotation_error_rad=float(args.max_ik_rotation_error_rad),
    )
    _verify_ik(
        backend,
        grasp_q,
        grasp_pose_eef,
        name="air final",
        max_position_error_m=float(args.max_ik_position_error_m),
        max_rotation_error_rad=float(args.max_ik_rotation_error_rad),
    )

    request = InstalledToolAuditRequest(
        mode="air_grasp",
        adapter_stl_path=adapter.mesh_path,
        T_EE_hand=T_EE_hand,
        control_config_path=config_path,
        snapshot_source_path=snapshot_path,
        snapshot_schema_version=2,
        snapshot_reference_frame=snapshot.reference_frame,
        snapshot_frame_id=int(snapshot.frame_id),
        snapshot_timestamp_s=float(snapshot.timestamp_s),
        snapshot_calibration_id=snapshot.calibration_id,
        snapshot_camera_serial=snapshot.camera_serial,
        snapshot_model_name=snapshot.model_name,
        snapshot_representation_checkpoint_sha256=(
            snapshot.representation_checkpoint_sha256
        ),
        snapshot_decision_checkpoint_sha256s=tuple(
            snapshot.decision_checkpoint_sha256s
        ),
        snapshot_official_source_commit=snapshot.official_source_commit,
        selected_candidate_index=index,
        selected_canonical_pose_base=canonical_pose,
        selected_hand_pose_base=selected_hand_pose,
        selected_hand_targets=selected_targets,
        plan_hand_pose_base=planned_hand_pose,
        plan_pregrasp_pose_base_EE=pregrasp_pose_eef,
        plan_grasp_pose_base_EE=grasp_pose_eef,
        air_retreat_distance_m=retreat,
        pregrasp_distance_m=pregrasp_distance,
        scene_source_path=scene_path,
        scene_points_base=scene_points,
        scene_capture_q_rad=capture_q,
        scene_captured_at_s=captured_at,
        current_q_rad=capture_q,
        current_q_captured_at_s=captured_at,
        default_q_rad=np.asarray(config["franka"]["default_q_rad"], dtype=np.float64),
        default_transit_q_rad=(),
        approach_transit_q_rad=(),
        joint_plan_source_path=joint_plan_path,
        pregrasp_q_rad=pregrasp_q,
        grasp_q_rad=grasp_q,
        object_source_path=snapshot_path,
        object_points_base=snapshot.object_points,
        hand_model=hand,
        open_hand_joint_positions_rad=OFFICIAL_OPEN_JOINT_POSITIONS_RAD,
        open_actuator_targets=OFFICIAL_OPEN_ACTUATOR_TARGETS,
        closure_hand_joint_positions_rad=closure.joint_positions_rad,
        closure_actuator_targets=selected_targets,
        hand_execution_path=hand_execution_path,
        hand_arrival_tolerance_units=int(
            config["inspire"]["arrival_tolerance_units"]
        ),
        audit_started_at_s=audit_started,
        scene_filter_evidence_path=filter_evidence_path,
        max_scene_age_s=max_scene_age,
        max_current_q_age_s=max_scene_age,
        max_joint_step_rad=float(max_joint_step_rad),
        max_q_tracking_error_rad=float(args.max_q_tracking_error_rad),
        scene_clearance_margin_m=observed_scene_margin,
        hand_self_clearance_margin_m=hand_self_clearance_margin,
    )
    if args.build_static_cache:
        static_started = time.monotonic()
        print(
            "[collision-progress] prepare_query "
            "(hand FK + official 1-register actuator envelopes)",
            flush=True,
        )
        query, _ = build_installed_tool_collision_query(request)
        cache = build_static_cache_artifact(query, backend)
        written = write_static_cache_artifact(output, cache)
        print(
            "[static-cache] EVIDENCE-PASS checks={} output={}".format(
                len(cache["static_check_ids"]), written
            )
        )
        print(
            "[static-cache] scene/object points excluded; motion_authorized=false"
        )
        print(
            "[static-cache] wall_s={:.3f} (outside final scene freshness window)".format(
                time.monotonic() - static_started
            )
        )
        return 0

    audit_backend = (
        StaticCacheCombinedBackend(backend, args.static_cache)
        if args.static_cache is not None
        else backend
    )
    if args.static_cache is not None:
        print(
            "[air-audit] static cache supplied; evaluating only fresh "
            "scene/object checks before normal schema-v2 composition"
        )
    print(
        "[collision-progress] prepare_query "
        "(hand FK + official 1-register actuator envelopes)",
        flush=True,
    )
    fresh_phase_started = time.monotonic()
    artifact = run_installed_tool_audit(request, audit_backend)
    written = write_installed_tool_audit(output, artifact)
    print(
        "[air-audit] geometry_plus_schema_wall_s={:.3f}".format(
            time.monotonic() - fresh_phase_started
        )
    )
    print(
        "[air-audit] passed={} pass_kind={} samples={} runtime_condition={}".format(
            artifact["decision"]["passed"],
            artifact["decision"]["pass_kind"],
            artifact["bindings"]["joint_path"]["sample_count"],
            artifact["decision"]["runtime_operator_workspace_clear_required"],
        )
    )
    print("[air-audit] output={}".format(written))
    print("[air-audit] motion_authorized=false")
    if artifact["decision"]["runtime_operator_workspace_clear_required"]:
        print(
            "[air-audit] executor requires --confirm-workspace-clear {}".format(
                RUNTIME_WORKSPACE_CLEAR_TOKEN
            )
        )
    if not artifact["decision"]["passed"]:
        for reason in artifact["decision"]["reasons"]:
            print("[air-audit][blocker] {}".format(reason))
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
