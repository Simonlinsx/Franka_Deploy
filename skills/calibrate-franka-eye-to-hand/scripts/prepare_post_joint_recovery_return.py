#!/usr/bin/env python3
"""Prepare the single Cartesian N02-recovery -> N01 return edge offline."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import yaml


ROOT = Path("/home/qiaoguanren/code/franka")
SOURCE = ROOT / "beta/dynamic_object_pcd/calibration_runs/fr3-d435-342222071785-20260813-424x240-60hz-depth-span-plans/05-N02-to-N01.plan.yaml"
OUTPUT = ROOT / "beta/dynamic_object_pcd/calibration_runs/fr3-d435-342222071785-20260813-depth-span-post-JR01-to-N01.plan.yaml"

START = np.asarray(
    [
        [0.7035400691796166, 0.6007585043169796, 0.3796321779691042, 0.4409271616724197],
        [0.633895690143444, -0.771994222933142, 0.04691666841774735, 0.29435204482644134],
        [0.32125943577785643, 0.20763944531008502, -0.9239470957122023, 0.8152379114883049],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
N01 = np.asarray(
    [
        [0.6944427490234375, 0.594502329826355, 0.4053594172000885, 0.4310245618323879],
        [0.6263642311096191, -0.7767239212989807, 0.06608971953392029, 0.25084822053079954],
        [0.35414284467697144, 0.2080071121454239, -0.9117652773857117, 0.8270284082346597],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def so3_log(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cosine)
    if theta < 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    vector = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    return vector * (theta / (2.0 * math.sin(theta)))


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    plan = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    offset = N01[:3, 3] - START[:3, 3]
    rotation_vector_deg = np.degrees(so3_log(START[:3, :3].T @ N01[:3, :3]))
    plan["session_slug"] = "fr3-d435-342222071785-20260813-depth-span-post-JR01-to-N01"
    plan["provenance"]["purpose"] = "return_from_audited_N02_J4_singularity_escape_to_N01"
    plan["reference_pose"] = {
        "source": "official_URDF_FK_of_audited_and_executed_JR01_endpoint",
        "xyz_base_m": START[:3, 3].tolist(),
        "T_base_ee": START.tolist(),
    }
    plan["training_poses"] = [
        {
            "id": "JR01",
            "xyz_offset_base_m": [0.0, 0.0, 0.0],
            "rotation_vector_eef_deg": [0.0, 0.0, 0.0],
        }
    ]
    plan["holdout_poses"] = [
        {
            "id": "N01",
            "xyz_offset_base_m": offset.tolist(),
            "rotation_vector_eef_deg": rotation_vector_deg.tolist(),
        }
    ]
    plan["planned_sequence"] = {
        "single_edge_only": True,
        "start_pose_id": "JR01",
        "target_pose_id": "N01",
        "capture_at_target": False,
    }
    plan["motion_history"] = []
    plan["motion_authorization"] = {
        "explicit_user_authorization_recorded": True,
        "scope": "single_pose",
        "start_pose_id": "JR01",
        "pose_id": "N01",
        "source_text": "我固定好了，你自己调整位置，并且复验",
        "authorization_interpretation": "return_from_the_audited_bounded_singularity_escape_without_repeated_prompts",
        "consumed": False,
    }
    encoded = yaml.safe_dump(plan, sort_keys=False, allow_unicode=True).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(OUTPUT, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        f"{OUTPUT} sha256={hashlib.sha256(encoded).hexdigest()} "
        f"translation_m={np.linalg.norm(offset):.9f} "
        f"rotation_deg={np.linalg.norm(rotation_vector_deg):.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
