#!/usr/bin/env python3
"""Create six audited, immutable joint-recovery plans from N03P to S01.

This is an offline artifact generator.  It never imports pylibfranka and never
opens the robot, camera, or RH56.  The path is split so every segment remains
inside the commissioning wrapper's 5 degree per-joint and 9 degree vector
limits.  Runnable plans are emitted only after the corresponding official-URDF
kinematics audits exist and report ``status=pass``.
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
SOURCE_SPEC = RUNS / (
    "fr3-d435-342222071785-20260813-depth-span-N02B-singularity-"
    "JR02-kinematics-spec.yaml"
)
SOURCE_PLAN = RUNS / (
    "fr3-d435-342222071785-20260813-depth-span-N02B-singularity-"
    "JR02-plan.yaml"
)
MOTION_WRAPPER = (
    ROOT
    / "skills/calibrate-franka-eye-to-hand/scripts/move_calibration_pose.py"
)
STEM = "fr3-d435-342222071785-20260813-N03P-to-S01-bounded-return-v1"
SEGMENTS = 6

# Fresh stationary FCI sample at N03P.  The live start gate still requires the
# actual robot joints to agree within 0.001 rad before the first segment runs.
Q_N03P = np.asarray(
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

# Closest regular official-URDF IK branch for the previously observed S01
# reference transform.  Offline residual: <5e-9 mm and <5e-11 deg.
Q_S01 = np.asarray(
    [
        0.3650974712392323,
        -0.1530336768699866,
        0.10486191228524824,
        -1.3121584393214798,
        -0.10982516201028351,
        1.564822423392783,
        -0.24389721687534036,
    ],
    dtype=np.float64,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_yaml(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    encoded = yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode(
        "utf-8"
    )
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

    return fk


def waypoints() -> list[np.ndarray]:
    return [
        Q_N03P + (Q_S01 - Q_N03P) * (index / SEGMENTS)
        for index in range(SEGMENTS + 1)
    ]


def paths(index: int) -> tuple[Path, Path, Path]:
    prefix = RUNS / f"{STEM}-{index:02d}"
    return (
        Path(f"{prefix}-kinematics-spec.yaml"),
        Path(f"{prefix}-kinematics-audit.json"),
        Path(f"{prefix}-plan.yaml"),
    )


def make_specs() -> None:
    fk = model_and_fk()
    template = yaml.safe_load(SOURCE_SPEC.read_text(encoding="utf-8"))
    points = waypoints()
    for index, (start, target) in enumerate(zip(points[:-1], points[1:]), 1):
        spec_path, _, _ = paths(index)
        value = json.loads(json.dumps(template))
        recovery = value["recovery"]
        recovery["expected_start_q_rad"] = start.tolist()
        recovery["target_q_rad"] = target.tolist()
        recovery["expected_start_T_base_ee"] = fk(start).tolist()
        recovery["expected_target_T_base_ee"] = fk(target).tolist()
        path = value["path"]
        path["workspace_min_m"] = [0.415, 0.200, 0.815]
        path["workspace_max_m"] = [0.450, 0.320, 0.840]
        path["minimum_eef_z_m"] = 0.815
        path["maximum_board_sweep_m"] = 0.030
        publish_yaml(spec_path, value)
        print(f"SPEC_{index:02d}={spec_path} sha256={sha256(spec_path)}")


def make_plans() -> None:
    fk = model_and_fk()
    template = yaml.safe_load(SOURCE_PLAN.read_text(encoding="utf-8"))
    points = waypoints()
    for index, (start, target) in enumerate(zip(points[:-1], points[1:]), 1):
        spec_path, audit_path, plan_path = paths(index)
        if not audit_path.is_file():
            raise FileNotFoundError(f"missing required audit {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "pass":
            raise ValueError(f"audit did not pass: {audit_path}")
        value = json.loads(json.dumps(template))
        recovery_id = f"N03S01R{index}"
        value["session_slug"] = f"{STEM}-{index:02d}"
        provenance = value["provenance"]
        provenance["reason"] = (
            f"bounded_stage_{index}_of_{SEGMENTS}_return_from_N03P_to_S01"
        )
        provenance["kinematics_spec"] = str(spec_path)
        provenance["kinematics_spec_sha256"] = sha256(spec_path)
        provenance["motion_wrapper_sha256"] = sha256(MOTION_WRAPPER)
        safety = value["safety"]
        safety["workspace_bounds_base_m"] = {
            "x": [0.415, 0.450],
            "y": [0.200, 0.320],
            "z": [0.815, 0.840],
        }
        safety["minimum_eef_z_m"] = 0.815
        safety["conservative_board_sweep_displacement_m"] = min(
            0.030,
            float(audit["maximum_conservative_board_displacement_m"]) + 0.001,
        )
        recovery = value["recovery"]
        recovery["id"] = recovery_id
        recovery["expected_start_q_rad"] = start.tolist()
        recovery["commanded_delta_rad"] = (target - start).tolist()
        recovery["target_q_rad"] = target.tolist()
        recovery["expected_start_T_base_ee"] = fk(start).tolist()
        recovery["expected_target_T_base_ee"] = fk(target).tolist()
        kinematics = recovery["kinematics_provenance"]
        kinematics["audit_artifact_path"] = str(audit_path)
        kinematics["audit_artifact_sha256"] = sha256(audit_path)
        kinematics["minimum_interpolation_eef_z_m"] = float(
            audit["minimum_eef_z_m"]
        )
        kinematics["maximum_interpolation_eef_center_displacement_m"] = float(
            audit["maximum_eef_center_displacement_m"]
        )
        kinematics["maximum_interpolation_eef_rotation_deg"] = float(
            audit["maximum_eef_rotation_deg"]
        )
        authorization = value["motion_authorization"]
        authorization["recovery_id"] = recovery_id
        authorization["source_text"] = (
            "我固定好了，你自己调整位置，并且复验；"
            "不需要管周围情况，现在是安全的，别考虑那么多"
        )
        authorization["interpretation"] = (
            f"bounded_stage_{index}_of_{SEGMENTS}_return_N03P_to_S01"
        )
        authorization["consumed"] = False
        publish_yaml(plan_path, value)
        print(f"PLAN_{index:02d}={plan_path} sha256={sha256(plan_path)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("specs", "plans"))
    args = parser.parse_args()
    if args.phase == "specs":
        make_specs()
    else:
        make_plans()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
