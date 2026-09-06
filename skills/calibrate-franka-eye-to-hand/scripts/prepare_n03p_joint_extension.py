#!/usr/bin/env python3
"""Prepare two audited bounded joint moves from N03P to the rotated N04Q pose.

This is an offline artifact generator.  It never imports pylibfranka and never
opens either the robot or camera.  The two stages keep every per-stage joint
delta below the commissioning wrapper's 5 degree cap and bind the official
FR3 URDF forward-kinematics audit before a runnable plan can be published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml


ROOT = Path("/home/qiaoguanren/code/franka")
RUNS = ROOT / "beta/dynamic_object_pcd/calibration_runs"
SOURCE_SPEC = RUNS / "fr3-d435-342222071785-20260813-depth-span-N02B-singularity-JR02-kinematics-spec.yaml"
SOURCE_PLAN = RUNS / "fr3-d435-342222071785-20260813-depth-span-N02B-singularity-JR02-plan.yaml"
AUDITOR = ROOT / "skills/calibrate-franka-eye-to-hand/scripts/audit_joint_recovery_kinematics.py"

STEM = "fr3-d435-342222071785-20260813-depth-span-N03P-to-rotated-N04Q"
SPEC_A = RUNS / f"{STEM}-A-kinematics-spec.yaml"
SPEC_B = RUNS / f"{STEM}-B-kinematics-spec.yaml"
AUDIT_A = RUNS / f"{STEM}-A-kinematics-audit.json"
AUDIT_B = RUNS / f"{STEM}-B-kinematics-audit.json"
PLAN_A = RUNS / f"{STEM}-A-plan.yaml"
PLAN_B = RUNS / f"{STEM}-B-plan.yaml"
RETURN_STEM = "fr3-d435-342222071785-20260813-depth-span-rotated-N04Q-to-N03P"
RETURN_SPEC_A = RUNS / f"{RETURN_STEM}-A-kinematics-spec.yaml"
RETURN_SPEC_B = RUNS / f"{RETURN_STEM}-B-kinematics-spec.yaml"
RETURN_AUDIT_A = RUNS / f"{RETURN_STEM}-A-kinematics-audit.json"
RETURN_AUDIT_B = RUNS / f"{RETURN_STEM}-B-kinematics-audit.json"
RETURN_PLAN_A = RUNS / f"{RETURN_STEM}-A-plan.yaml"
RETURN_PLAN_B = RUNS / f"{RETURN_STEM}-B-plan.yaml"

# Exact read-only FCI sample at the stationary N03P intermediate pose.
Q_START = np.asarray(
    [
        0.4453023374080658,
        0.15120969712734222,
        0.2649933993816376,
        -0.9028395414352417,
        -0.2541167140007019,
        1.3900994062423706,
        -0.08512729406356812,
    ],
    dtype=np.float64,
)

# The closest locally regular IK branch for the N04 position with an 8 degree
# EEF-local +Y rotation.  Offline solution residual: 0.000313 mm / 0.003702 deg.
Q_TARGET = np.asarray(
    [
        0.4840293221563047,
        0.1918711267697205,
        0.28514261604615687,
        -0.8750059997046896,
        -0.2808388730568696,
        1.5308830436589551,
        -0.039528824500299556,
    ],
    dtype=np.float64,
)
Q_MIDDLE = 0.5 * (Q_START + Q_TARGET)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_yaml(path: Path, value: object) -> None:
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


def model_and_fk():
    urdf = ROOT.parent / "libfranka/test/fr3.urdf"
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    frame_id = model.getFrameId("link8")

    def fk(q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        placement = data.oMf[frame_id]
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = placement.rotation
        result[:3, 3] = placement.translation
        return result

    return model, data, frame_id, fk


def make_specs() -> None:
    _, _, _, fk = model_and_fk()
    template = yaml.safe_load(SOURCE_SPEC.read_text(encoding="utf-8"))
    for suffix, start, target, output in (
        ("A", Q_START, Q_MIDDLE, SPEC_A),
        ("B", Q_MIDDLE, Q_TARGET, SPEC_B),
    ):
        value = json.loads(json.dumps(template))
        recovery = value["recovery"]
        recovery["expected_start_q_rad"] = start.tolist()
        recovery["target_q_rad"] = target.tolist()
        recovery["expected_start_T_base_ee"] = fk(start).tolist()
        recovery["expected_target_T_base_ee"] = fk(target).tolist()
        path = value["path"]
        path["workspace_min_m"] = [0.425, 0.300, 0.805]
        path["workspace_max_m"] = [0.460, 0.350, 0.850]
        path["minimum_eef_z_m"] = 0.805
        path["maximum_board_sweep_m"] = 0.070
        publish_yaml(output, value)
        print(f"SPEC_{suffix}={output} sha256={sha256(output)}")


def make_return_specs() -> None:
    _, _, _, fk = model_and_fk()
    template = yaml.safe_load(SOURCE_SPEC.read_text(encoding="utf-8"))
    for suffix, start, target, output in (
        ("A", Q_TARGET, Q_MIDDLE, RETURN_SPEC_A),
        ("B", Q_MIDDLE, Q_START, RETURN_SPEC_B),
    ):
        value = json.loads(json.dumps(template))
        recovery = value["recovery"]
        recovery["expected_start_q_rad"] = start.tolist()
        recovery["target_q_rad"] = target.tolist()
        recovery["expected_start_T_base_ee"] = fk(start).tolist()
        recovery["expected_target_T_base_ee"] = fk(target).tolist()
        path = value["path"]
        path["workspace_min_m"] = [0.425, 0.300, 0.805]
        path["workspace_max_m"] = [0.460, 0.350, 0.850]
        path["minimum_eef_z_m"] = 0.805
        path["maximum_board_sweep_m"] = 0.070
        publish_yaml(output, value)
        print(f"RETURN_SPEC_{suffix}={output} sha256={sha256(output)}")


def make_plans() -> None:
    _, _, _, fk = model_and_fk()
    template = yaml.safe_load(SOURCE_PLAN.read_text(encoding="utf-8"))
    for suffix, recovery_id, start, target, spec, audit, output in (
        ("A", "N04QA", Q_START, Q_MIDDLE, SPEC_A, AUDIT_A, PLAN_A),
        ("B", "N04QB", Q_MIDDLE, Q_TARGET, SPEC_B, AUDIT_B, PLAN_B),
    ):
        if not audit.is_file():
            raise FileNotFoundError(f"missing required audit {audit}")
        audit_value = json.loads(audit.read_text(encoding="utf-8"))
        if audit_value.get("status") != "pass":
            raise ValueError(f"audit did not pass: {audit}")
        value = json.loads(json.dumps(template))
        value["session_slug"] = f"{STEM}-{suffix}"
        value["provenance"]["reason"] = (
            "reach_near_depth_station_with_audited_rotated_nonsingular_IK_branch"
        )
        value["provenance"]["kinematics_spec"] = str(spec)
        value["provenance"]["kinematics_spec_sha256"] = sha256(spec)
        value["provenance"]["motion_wrapper_sha256"] = sha256(
            ROOT / "skills/calibrate-franka-eye-to-hand/scripts/move_calibration_pose.py"
        )
        safety = value["safety"]
        safety["workspace_bounds_base_m"] = {
            "x": [0.425, 0.460],
            "y": [0.300, 0.350],
            "z": [0.805, 0.850],
        }
        safety["minimum_eef_z_m"] = 0.805
        safety["conservative_board_sweep_displacement_m"] = min(
            0.070,
            float(audit_value["maximum_conservative_board_displacement_m"]) + 0.001,
        )
        recovery = value["recovery"]
        recovery["id"] = recovery_id
        recovery["expected_start_q_rad"] = start.tolist()
        recovery["commanded_delta_rad"] = (target - start).tolist()
        recovery["target_q_rad"] = target.tolist()
        recovery["expected_start_T_base_ee"] = fk(start).tolist()
        recovery["expected_target_T_base_ee"] = fk(target).tolist()
        provenance = recovery["kinematics_provenance"]
        provenance["audit_artifact_path"] = str(audit)
        provenance["audit_artifact_sha256"] = sha256(audit)
        provenance["minimum_interpolation_eef_z_m"] = float(
            audit_value["minimum_eef_z_m"]
        )
        provenance["maximum_interpolation_eef_center_displacement_m"] = float(
            audit_value["maximum_eef_center_displacement_m"]
        )
        provenance["maximum_interpolation_eef_rotation_deg"] = float(
            audit_value["maximum_eef_rotation_deg"]
        )
        authorization = value["motion_authorization"]
        authorization["recovery_id"] = recovery_id
        authorization["source_text"] = (
            "我固定好了，你自己调整位置，并且复验；不需要管周围情况，现在是安全的，别考虑那么多"
        )
        authorization["interpretation"] = (
            f"bounded_stage_{suffix}_of_rotated_near_depth_extension"
        )
        authorization["consumed"] = False
        publish_yaml(output, value)
        print(f"PLAN_{suffix}={output} sha256={sha256(output)}")


def make_return_plans() -> None:
    _, _, _, fk = model_and_fk()
    template = yaml.safe_load(SOURCE_PLAN.read_text(encoding="utf-8"))
    for suffix, recovery_id, start, target, spec, audit, output in (
        ("A", "N04QRA", Q_TARGET, Q_MIDDLE, RETURN_SPEC_A, RETURN_AUDIT_A, RETURN_PLAN_A),
        ("B", "N04QRB", Q_MIDDLE, Q_START, RETURN_SPEC_B, RETURN_AUDIT_B, RETURN_PLAN_B),
    ):
        if not audit.is_file():
            raise FileNotFoundError(f"missing required audit {audit}")
        audit_value = json.loads(audit.read_text(encoding="utf-8"))
        if audit_value.get("status") != "pass":
            raise ValueError(f"audit did not pass: {audit}")
        value = json.loads(json.dumps(template))
        value["session_slug"] = f"{RETURN_STEM}-{suffix}"
        value["provenance"]["reason"] = "return_from_rotated_near_depth_station"
        value["provenance"]["kinematics_spec"] = str(spec)
        value["provenance"]["kinematics_spec_sha256"] = sha256(spec)
        value["provenance"]["motion_wrapper_sha256"] = sha256(
            ROOT / "skills/calibrate-franka-eye-to-hand/scripts/move_calibration_pose.py"
        )
        safety = value["safety"]
        safety["workspace_bounds_base_m"] = {
            "x": [0.425, 0.460],
            "y": [0.300, 0.350],
            "z": [0.805, 0.850],
        }
        safety["minimum_eef_z_m"] = 0.805
        safety["conservative_board_sweep_displacement_m"] = min(
            0.070,
            float(audit_value["maximum_conservative_board_displacement_m"]) + 0.001,
        )
        recovery = value["recovery"]
        recovery["id"] = recovery_id
        recovery["expected_start_q_rad"] = start.tolist()
        recovery["commanded_delta_rad"] = (target - start).tolist()
        recovery["target_q_rad"] = target.tolist()
        recovery["expected_start_T_base_ee"] = fk(start).tolist()
        recovery["expected_target_T_base_ee"] = fk(target).tolist()
        provenance = recovery["kinematics_provenance"]
        provenance["audit_artifact_path"] = str(audit)
        provenance["audit_artifact_sha256"] = sha256(audit)
        provenance["minimum_interpolation_eef_z_m"] = float(audit_value["minimum_eef_z_m"])
        provenance["maximum_interpolation_eef_center_displacement_m"] = float(
            audit_value["maximum_eef_center_displacement_m"]
        )
        provenance["maximum_interpolation_eef_rotation_deg"] = float(
            audit_value["maximum_eef_rotation_deg"]
        )
        authorization = value["motion_authorization"]
        authorization["recovery_id"] = recovery_id
        authorization["source_text"] = (
            "我固定好了，你自己调整位置，并且复验；不需要管周围情况，现在是安全的，别考虑那么多"
        )
        authorization["interpretation"] = f"bounded_stage_{suffix}_return_from_rotated_near_depth_station"
        authorization["consumed"] = False
        publish_yaml(output, value)
        print(f"RETURN_PLAN_{suffix}={output} sha256={sha256(output)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("specs", "plans", "return-specs", "return-plans")
    )
    args = parser.parse_args()
    if args.phase == "specs":
        make_specs()
    elif args.phase == "plans":
        make_plans()
    elif args.phase == "return-specs":
        make_return_specs()
    else:
        make_return_plans()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
