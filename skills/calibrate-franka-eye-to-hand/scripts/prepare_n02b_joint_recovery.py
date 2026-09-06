#!/usr/bin/env python3
"""Prepare the second immutable N02 J4 recovery spec or plan offline."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml


ROOT = Path("/home/qiaoguanren/code/franka")
RUNS = ROOT / "beta/dynamic_object_pcd/calibration_runs"
SOURCE_SPEC = RUNS / "fr3-d435-342222071785-20260813-depth-span-N02-singularity-JR01-kinematics-spec.yaml"
SOURCE_PLAN = RUNS / "fr3-d435-342222071785-20260813-depth-span-N02-singularity-JR01-plan.yaml"
SPEC = RUNS / "fr3-d435-342222071785-20260813-depth-span-N02B-singularity-JR02-kinematics-spec.yaml"
AUDIT = RUNS / "fr3-d435-342222071785-20260813-depth-span-N02B-singularity-JR02-kinematics-audit.json"
PLAN = RUNS / "fr3-d435-342222071785-20260813-depth-span-N02B-singularity-JR02-plan.yaml"
Q = np.asarray(
    [0.3832743167877197, 0.05938331037759781, 0.2649497389793396,
     -1.0317878723144531, -0.20852909982204437, 1.4712927341461182,
     -0.1089610606431961], dtype=np.float64
)
DELTA = np.asarray([0.0, 0.0, 0.0, -0.03490658503988659, 0.0, 0.0, 0.0])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    encoded = yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def fk_metrics(q: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    urdf = ROOT.parent / "libfranka/test/fr3.urdf"
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    frame_id = model.getFrameId("link8")

    def fk(value: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(model, data, value)
        pin.updateFramePlacements(model, data)
        placement = data.oMf[frame_id]
        transform = np.eye(4)
        transform[:3, :3] = placement.rotation
        transform[:3, 3] = placement.translation
        return transform

    start_T = fk(q)
    target_T = fk(target)
    xyz = []
    minimum_sigma = float("inf")
    maximum_condition = 0.0
    maximum_displacement = 0.0
    maximum_rotation = 0.0
    maximum_board = 0.0
    for alpha in np.linspace(0.0, 1.0, 1001):
        value = q + alpha * (target - q)
        transform = fk(value)
        xyz.append(transform[:3, 3].copy())
        jacobian = pin.computeFrameJacobian(model, data, value, frame_id, pin.ReferenceFrame.LOCAL)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        minimum_sigma = min(minimum_sigma, float(singular[-1]))
        maximum_condition = max(maximum_condition, float(singular[0] / singular[-1]))
        displacement = float(np.linalg.norm(transform[:3, 3] - start_T[:3, 3]))
        cosine = float(np.clip((np.trace(start_T[:3, :3].T @ transform[:3, :3]) - 1) * 0.5, -1, 1))
        rotation = float(np.arccos(cosine))
        maximum_displacement = max(maximum_displacement, displacement)
        maximum_rotation = max(maximum_rotation, rotation)
        maximum_board = max(maximum_board, displacement + 0.6 * np.sin(rotation * 0.5))
    xyz_array = np.asarray(xyz)
    return start_T, target_T, {
        "minimum_z": float(np.min(xyz_array[:, 2])),
        "maximum_displacement": maximum_displacement,
        "maximum_rotation_deg": float(np.degrees(maximum_rotation)),
        "maximum_board": maximum_board,
        "minimum_sigma": minimum_sigma,
        "maximum_condition": maximum_condition,
    }


def make_spec() -> None:
    target = Q + DELTA
    start_T, target_T, _metrics = fk_metrics(Q, target)
    value = yaml.safe_load(SOURCE_SPEC.read_text(encoding="utf-8"))
    value["recovery"]["expected_start_q_rad"] = Q.tolist()
    value["recovery"]["target_q_rad"] = target.tolist()
    value["recovery"]["expected_start_T_base_ee"] = start_T.tolist()
    value["recovery"]["expected_target_T_base_ee"] = target_T.tolist()
    publish(SPEC, value)
    print(f"{SPEC} sha256={digest(SPEC)}")


def make_plan() -> None:
    if not AUDIT.is_file():
        raise FileNotFoundError(f"missing audit {AUDIT}")
    target = Q + DELTA
    start_T, target_T, metrics = fk_metrics(Q, target)
    value = yaml.safe_load(SOURCE_PLAN.read_text(encoding="utf-8"))
    value["session_slug"] = "fr3-d435-342222071785-20260813-depth-span-N02B-singularity-JR02"
    value["provenance"]["reason"] = "escape_N02B_cartesian_start_singularity_before_rotated_near_extension"
    value["provenance"]["kinematics_spec"] = str(SPEC)
    value["provenance"]["kinematics_spec_sha256"] = digest(SPEC)
    value["recovery"]["id"] = "JR02"
    value["recovery"]["expected_start_q_rad"] = Q.tolist()
    value["recovery"]["commanded_delta_rad"] = DELTA.tolist()
    value["recovery"]["target_q_rad"] = target.tolist()
    value["recovery"]["expected_start_T_base_ee"] = start_T.tolist()
    value["recovery"]["expected_target_T_base_ee"] = target_T.tolist()
    provenance = value["recovery"]["kinematics_provenance"]
    provenance["audit_artifact_path"] = str(AUDIT)
    provenance["audit_artifact_sha256"] = digest(AUDIT)
    provenance["minimum_interpolation_eef_z_m"] = metrics["minimum_z"]
    provenance["maximum_interpolation_eef_center_displacement_m"] = metrics["maximum_displacement"]
    provenance["maximum_interpolation_eef_rotation_deg"] = metrics["maximum_rotation_deg"]
    value["safety"]["workspace_bounds_base_m"] = {
        "x": [0.426, 0.451], "y": [0.280, 0.305], "z": [0.805, 0.840]
    }
    value["safety"]["minimum_eef_z_m"] = 0.805
    value["safety"]["conservative_board_sweep_displacement_m"] = 0.027
    value["motion_authorization"]["recovery_id"] = "JR02"
    value["motion_authorization"]["interpretation"] = "one_bounded_fail_closed_J4_minus_2deg_recovery_from_N02B"
    value["motion_authorization"]["consumed"] = False
    publish(PLAN, value)
    print(f"{PLAN} sha256={digest(PLAN)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("spec", "plan"))
    args = parser.parse_args()
    make_spec() if args.phase == "spec" else make_plan()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
