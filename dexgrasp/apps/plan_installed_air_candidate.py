#!/usr/bin/env python3
"""Deterministic, hardware-free IK plan for one installed-tool air candidate.

The planner binds an official AnyDex snapshot, the commissioned V7 profile,
and the official FR3 URDF.  It deliberately checks only FR3 joint limits and
bare-arm self collision.  Scene, adapter, and RH56 collision evidence belongs
to the separate installed-tool audit; this manifest can never authorize
motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from anydex_pipeline.control_config import (
    load_control_config,
    validate_rigid_transform,
)
from anydex_pipeline.control_plan import is_official_snapshot
from anydex_pipeline.air_target_poses import derive_air_target_poses
from anydex_pipeline.hppfcl_installed_tool_backend import (
    DEFAULT_FR3_URDF,
    DEFAULT_FRANKA_DESCRIPTION_SHARE,
    OFFICIAL_FRANKA_DISABLED_SELF_PAIRS,
)
from anydex_pipeline.joint_path_sampling import (
    CANONICAL_JOINT_SAMPLING_ALGORITHM,
    canonical_joint_path_samples,
)
from anydex_pipeline.rh56_hand_path import build_rh56_no_contact_execution_path
from anydex_pipeline.rh56_commissioning import (
    atomic_write_evidence,
    seal_evidence,
)
from anydex_pipeline.snapshot import load_snapshot_npz


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "fr3_rh56_v7_commissioning.json"
)
DEFAULT_CANDIDATE_INDEX = 51
DEFAULT_Q7_RAD = 1.3525
DEFAULT_MAX_JOINT_STEP_RAD = 0.005
DETERMINISTIC_IK_SEED_Q = np.asarray(
    [0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, 0.0],
    dtype=np.float64,
)
IK_POSITION_TOLERANCE_M = 1e-9
IK_ROTATION_TOLERANCE_RAD = 1e-9
IK_MAX_ITERATIONS = 1000
IK_DAMPING = 1e-12
IK_STEP = 0.2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(
        "dtype=<f8;shape={};".format(
            ",".join(str(int(item)) for item in array.shape)
        ).encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _pose_residual(pin: Any, model: Any, frame_id: int, q: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    desired = pin.SE3(target[:3, :3], target[:3, 3])
    error = pin.log6(data.oMf[frame_id].actInv(desired)).vector
    return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))


def _solve_fixed_q7_ik(
    pin: Any,
    model: Any,
    frame_id: int,
    target: np.ndarray,
    initial_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    q7_rad: float,
) -> Tuple[np.ndarray, float, float, int]:
    desired = pin.SE3(target[:3, :3], target[:3, 3])
    q = np.clip(np.asarray(initial_q, dtype=np.float64), lower, upper)
    q[6] = float(q7_rad)
    data = model.createData()
    best: Optional[Tuple[float, float, float, np.ndarray, int]] = None

    for iteration in range(IK_MAX_ITERATIONS + 1):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current = data.oMf[frame_id]
        current_to_desired = current.actInv(desired)
        error = pin.log6(current_to_desired).vector
        position_error = float(np.linalg.norm(error[:3]))
        rotation_error = float(np.linalg.norm(error[3:]))
        metric = position_error + rotation_error
        if best is None or metric < best[0]:
            best = (metric, position_error, rotation_error, q.copy(), iteration)
        if (
            position_error <= IK_POSITION_TOLERANCE_M
            and rotation_error <= IK_ROTATION_TOLERANCE_RAD
        ):
            return q.copy(), position_error, rotation_error, iteration
        if iteration == IK_MAX_ITERATIONS:
            break

        jacobian = pin.computeFrameJacobian(
            model, data, q, frame_id, pin.ReferenceFrame.LOCAL
        )
        jacobian = -pin.Jlog6(current_to_desired.inverse()) @ jacobian
        reduced = jacobian[:, :6]
        velocity6 = -reduced.T @ np.linalg.solve(
            reduced @ reduced.T + IK_DAMPING * np.eye(6), error
        )
        velocity_norm = float(np.linalg.norm(velocity6))
        if velocity_norm > 1.0:
            velocity6 /= velocity_norm
        velocity = np.zeros(7, dtype=np.float64)
        velocity[:6] = velocity6
        q = pin.integrate(model, q, velocity * IK_STEP)
        q = np.clip(q, lower, upper)
        q[6] = float(q7_rad)

    assert best is not None
    raise RuntimeError(
        "fixed-q7 IK did not converge; best position={:.9g}m "
        "rotation={:.9g}rad at iteration={} q={}".format(
            best[1], best[2], best[4], np.round(best[3], 9).tolist()
        )
    )


def build_joint_path(
    waypoints: Sequence[np.ndarray], max_joint_step_rad: float
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Sample a piecewise-linear joint path with an L-infinity step bound."""

    return canonical_joint_path_samples(waypoints, max_joint_step_rad)


def _configure_bare_self_pairs(pin: Any, model: Any, collision_model: Any) -> Tuple[Tuple[str, str], ...]:
    disabled = {
        tuple(sorted(pair)) for pair in OFFICIAL_FRANKA_DISABLED_SELF_PAIRS
    }
    collision_model.removeAllCollisionPairs()
    tested = []
    for first in range(len(collision_model.geometryObjects)):
        for second in range(first + 1, len(collision_model.geometryObjects)):
            first_object = collision_model.geometryObjects[first]
            second_object = collision_model.geometryObjects[second]
            names = (
                model.frames[first_object.parentFrame].name,
                model.frames[second_object.parentFrame].name,
            )
            if tuple(sorted(names)) in disabled:
                continue
            collision_model.addCollisionPair(pin.CollisionPair(first, second))
            tested.append(names)
    if not tested:
        raise RuntimeError("official SRDF exclusions removed every self pair")
    return tuple(tested)


def _check_bare_self_collision(
    pin: Any, model: Any, collision_model: Any, q_path: np.ndarray
) -> Tuple[Tuple[str, str], ...]:
    tested = _configure_bare_self_pairs(pin, model, collision_model)
    data = model.createData()
    geometry_data = pin.GeometryData(collision_model)
    for sample_index, q in enumerate(q_path):
        if not pin.computeCollisions(
            model, data, collision_model, geometry_data, q, False
        ):
            continue
        colliding = []
        for pair_index, result in enumerate(geometry_data.collisionResults):
            if not result.isCollision():
                continue
            pair = collision_model.collisionPairs[pair_index]
            colliding.append(
                (
                    model.frames[
                        collision_model.geometryObjects[pair.first].parentFrame
                    ].name,
                    model.frames[
                        collision_model.geometryObjects[pair.second].parentFrame
                    ].name,
                )
            )
        raise RuntimeError(
            "bare FR3 self collision at path sample {}: {}".format(
                sample_index, colliding
            )
        )
    return tested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline deterministic installed-air candidate IK planner"
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-index", type=int, default=DEFAULT_CANDIDATE_INDEX)
    parser.add_argument("--q7-rad", type=float, default=DEFAULT_Q7_RAD)
    parser.add_argument(
        "--start-q-rad",
        type=float,
        nargs=7,
        default=None,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
        help=(
            "Measured installed starting state. If omitted, start equals the "
            "profile's cable-friendly franka.default_q_rad; never assume q7=0."
        ),
    )
    parser.add_argument(
        "--max-joint-step-rad", type=float, default=DEFAULT_MAX_JOINT_STEP_RAD
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_FR3_URDF)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    snapshot_path = args.snapshot.expanduser().resolve()
    config_path_requested = args.config.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise SystemExit("output already exists; refusing to overwrite: {}".format(output_path))
    if not snapshot_path.is_file() or not urdf_path.is_file():
        raise SystemExit("snapshot or URDF does not exist")
    if not 0.001 <= float(args.max_joint_step_rad) <= 0.02:
        raise SystemExit("--max-joint-step-rad must be in [0.001, 0.02]")

    before_hashes = {
        "snapshot": _sha256_file(snapshot_path),
        "config": _sha256_file(config_path_requested),
        "urdf": _sha256_file(urdf_path),
    }
    config, config_path = load_control_config(config_path_requested)
    snapshot = load_snapshot_npz(snapshot_path)
    if not is_official_snapshot(snapshot):
        raise SystemExit("snapshot is not provenance-complete official AnyDexGrasp output")
    index = int(args.candidate_index)
    if index < 0 or index >= snapshot.grasps.count:
        raise SystemExit("--candidate-index is outside the snapshot")
    if snapshot.grasps.hand_poses is None:
        raise SystemExit("selected candidate has no hand pose")

    franka = config["franka"]
    tool = config["tool"]
    grasp = config["grasp"]
    q_default = np.asarray(franka["default_q_rad"], dtype=np.float64)
    limits = np.asarray(franka["joint_limits_rad"], dtype=np.float64)
    margin = float(franka["joint_limit_margin_rad"])
    lower = limits[:, 0] + margin
    upper = limits[:, 1] - margin
    q7 = float(args.q7_rad)
    if not np.isfinite(q7) or not lower[6] <= q7 <= upper[6]:
        raise SystemExit("--q7-rad violates the configured joint-limit margin")
    if args.start_q_rad is None:
        q_start = q_default.copy()
        q_start_source = "profile:franka.default_q_rad"
    else:
        q_start = np.asarray(args.start_q_rad, dtype=np.float64)
        q_start_source = "cli:--start-q-rad"
    if q_start.shape != (7,) or not np.all(np.isfinite(q_start)):
        raise SystemExit("--start-q-rad must contain seven finite radians")
    if np.any(q_start < lower) or np.any(q_start > upper):
        raise SystemExit("--start-q-rad violates configured joint-limit margins")

    canonical = np.asarray(snapshot.grasps.canonical_poses[index], dtype=np.float64)
    hand_pose = np.asarray(snapshot.grasps.hand_poses[index], dtype=np.float64)
    air_targets = derive_air_target_poses(
        canonical,
        hand_pose,
        np.asarray(snapshot.grasps.approach_axis_local, dtype=np.float64),
        np.asarray(tool["T_EE_hand"], dtype=np.float64),
        retreat_distance_m=float(grasp["air_retreat_distance_m"]),
        pregrasp_extra_distance_m=float(grasp["air_pregrasp_distance_m"]),
    )
    approach = air_targets.approach_reference
    nominal_pose = air_targets.T_reference_EE_nominal
    pregrasp_pose = air_targets.T_reference_EE_pregrasp
    final_pose = air_targets.T_reference_EE_final_air

    try:
        import pinocchio as pin
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(
            "Pinocchio is unavailable; use scripts/plan_installed_air_candidate.sh"
        ) from exc

    model, collision_model, _ = pin.buildModelsFromUrdf(
        str(urdf_path), [str(DEFAULT_FRANKA_DESCRIPTION_SHARE.parent)]
    )
    if int(model.nq) != 7 or len(collision_model.geometryObjects) != 8:
        raise RuntimeError(
            "unexpected FR3 topology: nq={}, collision_meshes={}".format(
                model.nq, len(collision_model.geometryObjects)
            )
        )
    frame_id = int(model.getFrameId("link8"))
    if frame_id >= len(model.frames):
        raise RuntimeError("FR3 URDF has no link8 frame")

    q_pre, pre_position, pre_rotation, pre_iterations = _solve_fixed_q7_ik(
        pin,
        model,
        frame_id,
        pregrasp_pose,
        DETERMINISTIC_IK_SEED_Q,
        lower,
        upper,
        q7,
    )
    q_final, final_position, final_rotation, final_iterations = _solve_fixed_q7_ik(
        pin, model, frame_id, final_pose, q_pre, lower, upper, q7
    )
    for name, q in (
        ("configured default", q_default),
        ("installed start", q_start),
        ("pregrasp", q_pre),
        ("final", q_final),
    ):
        if np.any(q < lower) or np.any(q > upper):
            raise RuntimeError("{} q violates configured joint-limit margins".format(name))

    # Recompute residuals independently from the solver termination state.
    pre_position, pre_rotation = _pose_residual(
        pin, model, frame_id, q_pre, pregrasp_pose
    )
    final_position, final_rotation = _pose_residual(
        pin, model, frame_id, q_final, final_pose
    )
    if (
        pre_position > IK_POSITION_TOLERANCE_M
        or pre_rotation > IK_ROTATION_TOLERANCE_RAD
        or final_position > IK_POSITION_TOLERANCE_M
        or final_rotation > IK_ROTATION_TOLERANCE_RAD
    ):
        raise RuntimeError("independent FK residual check failed")

    q_path, segment_intervals = build_joint_path(
        (q_start, q_default, q_pre, q_final), float(args.max_joint_step_rad)
    )
    tested_pairs = _check_bare_self_collision(
        pin, model, collision_model, q_path
    )
    actual_max_step = float(np.max(np.abs(np.diff(q_path, axis=0))))

    after_hashes = {
        "snapshot": _sha256_file(snapshot_path),
        "config": _sha256_file(config_path),
        "urdf": _sha256_file(urdf_path),
    }
    if after_hashes != before_hashes:
        raise RuntimeError("an input changed during planning; discard the result")

    source_indices = snapshot.grasps.source_indices
    hand_angles = snapshot.grasps.hand_angles
    widths = snapshot.grasps.widths_m
    depths = snapshot.grasps.depths_m
    canonical_hand_path = build_rh56_no_contact_execution_path(
        np.rint(hand_angles[index]).astype(int).tolist(), step_units=25
    )
    artifact: Mapping[str, Any] = seal_evidence(
        {
            "schema_version": 1,
            "artifact_type": "installed_air_candidate_joint_plan",
            "planner": {
                "name": "plan_installed_air_candidate",
                "version": 1,
                "deterministic": True,
                "pinocchio_version": str(getattr(pin, "__version__", "unknown")),
                "ik": {
                    "method": "damped_least_squares_fixed_q7",
                    "pregrasp_numerical_seed_rad": DETERMINISTIC_IK_SEED_Q.tolist(),
                    "numerical_seed_is_not_a_motion_waypoint": True,
                    "q7_rad": q7,
                    "position_tolerance_m": IK_POSITION_TOLERANCE_M,
                    "rotation_tolerance_rad": IK_ROTATION_TOLERANCE_RAD,
                    "max_iterations": IK_MAX_ITERATIONS,
                    "damping": IK_DAMPING,
                    "step": IK_STEP,
                    "random_starts": 0,
                },
            },
            "inputs": {
                "snapshot": {"path": str(snapshot_path), "sha256": before_hashes["snapshot"]},
                "config": {"path": str(config_path), "sha256": before_hashes["config"]},
                "fr3_urdf": {"path": str(urdf_path), "sha256": before_hashes["urdf"]},
            },
            "candidate": {
                "index": index,
                "snapshot_selected_index": int(snapshot.grasps.selected_index),
                "source_index": int(source_indices[index]) if source_indices is not None else None,
                "score": float(snapshot.grasps.scores[index]),
                "type_id": int(snapshot.grasps.type_ids[index]),
                "width_m": float(widths[index]) if widths is not None else None,
                "depth_m": float(depths[index]) if depths is not None else None,
                "hand_targets": (
                    np.rint(hand_angles[index]).astype(int).tolist()
                    if hand_angles is not None
                    else None
                ),
                "snapshot_collision_checked": bool(snapshot.grasps.collision_checked[index]),
                "snapshot_collision_free": bool(snapshot.grasps.collision_free[index]),
                "canonical_pose_base": canonical.tolist(),
                "hand_pose_base": hand_pose.tolist(),
            },
            "air_geometry": {
                "approach_axis_base": approach.tolist(),
                "retreat_distance_m": float(grasp["air_retreat_distance_m"]),
                "pregrasp_extra_distance_m": float(grasp["air_pregrasp_distance_m"]),
                "nominal_contact_pose_base_EE": nominal_pose.tolist(),
                "pregrasp_pose_base_EE": pregrasp_pose.tolist(),
                "final_air_pose_base_EE": final_pose.tolist(),
            },
            "joint_plan": {
                "trajectory_contract": "piecewise_linear_joint_space_v1",
                "sampling_algorithm": CANONICAL_JOINT_SAMPLING_ALGORITHM,
                "waypoint_order": ["start", "default", "pregrasp", "final_air"],
                "q_start_rad": q_start.tolist(),
                "q_start_source": q_start_source,
                "q_default_rad": q_default.tolist(),
                "q_default_provenance": dict(
                    franka.get("default_q_provenance", {})
                ),
                "q_pregrasp_rad": q_pre.tolist(),
                "q_final_air_rad": q_final.tolist(),
                "configured_joint_limits_rad": limits.tolist(),
                "joint_limit_margin_rad": margin,
                "joint_limits_with_margin_passed": True,
                "requested_max_joint_step_rad": float(args.max_joint_step_rad),
                "actual_max_joint_step_rad": actual_max_step,
                "segment_interval_counts": list(segment_intervals),
                "sample_count": int(len(q_path)),
                "q_path_sha256": _array_sha256(q_path),
                "start_to_default_delta_rad": (q_default - q_start).tolist(),
                "start_to_default_linf_rad": float(
                    np.max(np.abs(q_default - q_start))
                ),
                "default_to_pregrasp_delta_rad": (q_pre - q_default).tolist(),
                "default_to_pregrasp_linf_rad": float(
                    np.max(np.abs(q_pre - q_default))
                ),
                "start_to_pregrasp_delta_rad": (q_pre - q_start).tolist(),
                "start_to_pregrasp_linf_rad": float(
                    np.max(np.abs(q_pre - q_start))
                ),
                "q7_start_rad": float(q_start[6]),
                "q7_default_rad": float(q_default[6]),
                "q7_target_rad": float(q_pre[6]),
                "q7_start_to_default_rotation_rad": float(
                    q_default[6] - q_start[6]
                ),
                "q7_default_to_target_rotation_rad": float(
                    q_pre[6] - q_default[6]
                ),
                "q7_rotation_rad": float(q_pre[6] - q_start[6]),
                "q7_absolute_rotation_rad": float(abs(q_pre[6] - q_start[6])),
            },
            "hand_execution_path": canonical_hand_path.as_dict(),
            "fk_residual": {
                "pregrasp_position_m": pre_position,
                "pregrasp_rotation_rad": pre_rotation,
                "pregrasp_iterations": pre_iterations,
                "final_air_position_m": final_position,
                "final_air_rotation_rad": final_rotation,
                "final_air_iterations": final_iterations,
                "passed": True,
            },
            "collision_check": {
                "bare_fr3_self_collision_free": True,
                "sample_count": int(len(q_path)),
                "official_srdf_disabled_pairs": [
                    list(pair) for pair in OFFICIAL_FRANKA_DISABLED_SELF_PAIRS
                ],
                "tested_pairs": [list(pair) for pair in tested_pairs],
                "scene_collision_checked": False,
                "adapter_collision_checked": False,
                "rh56_collision_checked": False,
                "installed_tool_audit_required": True,
            },
            "cable_constraint": {
                "q7_zero_assumed": False,
                "cable_geometry_or_slack_verified": False,
                "manual_cable_slack_confirmation_required": True,
            },
            "motion_authorized": False,
        }
    )
    try:
        output_sha256 = atomic_write_evidence(output_path, artifact)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc
    print("[planner] wrote {}".format(output_path))
    print("[planner] sha256={}".format(output_sha256))
    print("[planner] q_pregrasp_rad={}".format(q_pre.tolist()))
    print("[planner] q_final_air_rad={}".format(q_final.tolist()))
    print(
        "[planner] FK exact, joint margins pass, bare FR3 self collision clear; "
        "scene/tool audit NOT run; motion_authorized=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
