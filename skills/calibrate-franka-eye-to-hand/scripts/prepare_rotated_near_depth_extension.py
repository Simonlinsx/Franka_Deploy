#!/usr/bin/env python3
"""Prepare immutable repeat and rotated near-depth extension edges offline."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import yaml


ROOT = Path("/home/qiaoguanren/code/franka")
SOURCE_DIR = ROOT / "beta/dynamic_object_pcd/calibration_runs/fr3-d435-342222071785-20260813-424x240-60hz-depth-span-plans"
OUT = ROOT / "beta/dynamic_object_pcd/calibration_runs/fr3-d435-342222071785-20260813-rotated-near-depth-extension-plans"
S01 = np.asarray(
    [
        [0.6944427490234375, 0.594502329826355, 0.4053594172000885, 0.4260011911392212],
        [0.6263642311096191, -0.7767239212989807, 0.06608971953392029, 0.21128949522972107],
        [0.35414284467697144, 0.2080071121454239, -0.9117652773857117, 0.8238862752914429],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
STEP40 = np.asarray(
    [0.005023370693166713, 0.03955872530107847, 0.0031421329432168304],
    dtype=np.float64,
)
POST_JR = np.asarray(
    [
        [0.7035400691796166, 0.6007585043169796, 0.3796321779691042, 0.4409271616724197],
        [0.633895690143444, -0.771994222933142, 0.04691666841774735, 0.29435204482644134],
        [0.32125943577785643, 0.20763944531008502, -0.9239470957122023, 0.8152379114883049],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def pose(level: float, rotation: np.ndarray) -> np.ndarray:
    result = rotation.copy()
    result[:3, 3] = S01[:3, 3] + level * STEP40
    return result


def publish(path: Path, value: object) -> str:
    encoded = yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(encoded).hexdigest()


def make_plan(template: dict, start_id: str, target_id: str, start: np.ndarray, target: np.ndarray) -> dict:
    result = copy.deepcopy(template)
    result["session_slug"] = f"fr3-d435-342222071785-20260813-near-extension-{start_id}-to-{target_id}"
    result["provenance"]["purpose"] = "native_424x240_60hz_rotated_near_depth_extension"
    result["reference_pose"] = {
        "source": f"audited_exact_{start_id}",
        "xyz_base_m": start[:3, 3].tolist(),
        "T_base_ee": start.tolist(),
    }
    result["training_poses"] = [{
        "id": start_id,
        "xyz_offset_base_m": [0.0, 0.0, 0.0],
        "rotation_vector_eef_deg": [0.0, 0.0, 0.0],
    }]
    relative = start[:3, :3].T @ target[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cosine)
    if theta < 1.0e-12:
        rotation_vector = np.zeros(3)
    else:
        skew = np.asarray([
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ])
        rotation_vector = skew * (theta / (2.0 * math.sin(theta)))
    result["holdout_poses"] = [{
        "id": target_id,
        "xyz_offset_base_m": (target[:3, 3] - start[:3, 3]).tolist(),
        "rotation_vector_eef_deg": np.degrees(rotation_vector).tolist(),
    }]
    result["planned_sequence"] = {
        "single_edge_only": True,
        "start_pose_id": start_id,
        "target_pose_id": target_id,
        "capture_at_target": target_id == "N04R",
    }
    result["motion_history"] = []
    result["motion_authorization"] = {
        "explicit_user_authorization_recorded": True,
        "scope": "single_pose",
        "start_pose_id": start_id,
        "pose_id": target_id,
        "source_text": "我固定好了，你自己调整位置，并且复验",
        "authorization_interpretation": "execute_reviewed_rotated_near_depth_extension_without_repeated_prompts",
        "consumed": False,
    }
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    template = yaml.safe_load((SOURCE_DIR / "01-S01-to-N01.plan.yaml").read_text(encoding="utf-8"))
    n01 = pose(1.0, S01)
    n02 = pose(2.0, S01)
    # A 55 mm additional camera-axis displacement is 1.375 original steps.
    n03r = pose(2.6875, POST_JR)
    n04r = pose(3.3750, POST_JR)
    edges = [
        ("S01", "N01B", S01, n01),
        ("N01B", "N02B", n01, n02),
        ("JR02", "N03R", POST_JR, n03r),
        ("N03R", "N04R", n03r, n04r),
        ("N04R", "N03R2", n04r, n03r),
        ("N03R2", "JR02R", n03r, POST_JR),
    ]
    records = []
    for ordinal, (start_id, target_id, start, target) in enumerate(edges, 1):
        plan = make_plan(template, start_id, target_id, start, target)
        path = OUT / f"{ordinal:02d}-{start_id}-to-{target_id}.plan.yaml"
        digest = publish(path, plan)
        records.append({
            "ordinal": ordinal,
            "start": start_id,
            "target": target_id,
            "plan": str(path),
            "sha256": digest,
            "translation_m": float(np.linalg.norm(target[:3, 3] - start[:3, 3])),
        })
    manifest = {
        "schema_version": 1,
        "kind": "rotated_near_depth_extension_plan_manifest",
        "hardware_opened": False,
        "robot_motion_commanded": False,
        "edges": records,
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(OUT / "manifest.json", flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"manifest": str(OUT / "manifest.json"), "edges": records}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
