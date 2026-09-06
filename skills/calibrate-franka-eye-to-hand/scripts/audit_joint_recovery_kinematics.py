#!/usr/bin/env python3
"""Offline FK/path audit for one bounded FR3 joint-recovery artifact.

This script imports Pinocchio but never imports pylibfranka, opens FCI, or
commands hardware.  Run it with the system Python that owns the ROS Pinocchio
installation (currently ``/usr/bin/python3`` on this workstation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pinocchio as pin
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite {size}-vector")
    return vector


def _transform(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
        raise ValueError(f"{name} has invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise ValueError(f"{name} rotation determinant is not +1")
    return matrix


def _rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


def _pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def audit(spec_path: Path) -> Dict[str, Any]:
    spec_path = spec_path.expanduser().resolve(strict=True)
    payload = _mapping(yaml.safe_load(spec_path.read_text(encoding="utf-8")), "spec")
    if payload.get("schema_version") != 1:
        raise ValueError("schema_version must be exactly 1")
    if payload.get("kind") != "franka_joint_recovery_kinematics_spec":
        raise ValueError("unexpected spec kind")

    urdf_record = _mapping(payload.get("urdf"), "urdf")
    urdf_path = Path(str(urdf_record.get("path"))).expanduser().resolve(strict=True)
    expected_urdf_sha = str(urdf_record.get("sha256") or "")
    actual_urdf_sha = _sha256(urdf_path)
    if actual_urdf_sha != expected_urdf_sha:
        raise ValueError(
            f"URDF SHA mismatch: expected={expected_urdf_sha} actual={actual_urdf_sha}"
        )
    frame_name = str(urdf_record.get("frame") or "")

    recovery = _mapping(payload.get("recovery"), "recovery")
    start_q = _vector(recovery.get("expected_start_q_rad"), 7, "expected_start_q_rad")
    target_q = _vector(recovery.get("target_q_rad"), 7, "target_q_rad")
    expected_start = _transform(
        recovery.get("expected_start_T_base_ee"), "expected_start_T_base_ee"
    )
    expected_target = _transform(
        recovery.get("expected_target_T_base_ee"), "expected_target_T_base_ee"
    )

    path = _mapping(payload.get("path"), "path")
    sample_count = path.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 21:
        raise ValueError("path.sample_count must be an integer >= 21")
    workspace_min = _vector(path.get("workspace_min_m"), 3, "workspace_min_m")
    workspace_max = _vector(path.get("workspace_max_m"), 3, "workspace_max_m")
    minimum_eef_z = float(path.get("minimum_eef_z_m"))
    board_extent = float(path.get("board_max_extent_from_eef_m"))
    maximum_board_sweep = float(path.get("maximum_board_sweep_m"))
    joint_limits = np.asarray(path.get("joint_limits_rad"), dtype=np.float64)
    if joint_limits.shape != (7, 2) or not np.all(np.isfinite(joint_limits)):
        raise ValueError("joint_limits_rad must be a finite 7x2 matrix")
    joint_margin = float(path.get("joint_limit_margin_rad"))
    if (
        np.any(workspace_min >= workspace_max)
        or not np.isfinite(minimum_eef_z)
        or not np.isfinite(board_extent)
        or board_extent <= 0.0
        or not np.isfinite(maximum_board_sweep)
        or maximum_board_sweep <= 0.0
        or not np.isfinite(joint_margin)
        or joint_margin <= 0.0
    ):
        raise ValueError("path limits are invalid")

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    frame_id = model.getFrameId(frame_name)
    if frame_id >= len(model.frames):
        raise ValueError(f"URDF frame {frame_name!r} not found")

    def fk(q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        placement = data.oMf[frame_id]
        return _pose(placement.rotation.copy(), placement.translation.copy())

    def jacobian_metrics(q: np.ndarray) -> Dict[str, Any]:
        jacobian = pin.computeFrameJacobian(
            model, data, q, frame_id, pin.ReferenceFrame.LOCAL
        )
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        return {
            "singular_values": singular_values.tolist(),
            "sigma_min": float(singular_values[-1]),
            "condition_number": float(singular_values[0] / singular_values[-1]),
        }

    start_fk = fk(start_q)
    target_fk = fk(target_q)
    fk_translation_tolerance = float(recovery.get("fk_translation_tolerance_m", 1.0e-6))
    fk_rotation_tolerance = math.radians(
        float(recovery.get("fk_rotation_tolerance_deg", 1.0e-4))
    )
    start_fk_translation = float(
        np.linalg.norm(start_fk[:3, 3] - expected_start[:3, 3])
    )
    start_fk_rotation = _rotation_error(start_fk[:3, :3], expected_start[:3, :3])
    target_fk_translation = float(
        np.linalg.norm(target_fk[:3, 3] - expected_target[:3, 3])
    )
    target_fk_rotation = _rotation_error(target_fk[:3, :3], expected_target[:3, :3])
    if (
        start_fk_translation > fk_translation_tolerance
        or start_fk_rotation > fk_rotation_tolerance
        or target_fk_translation > fk_translation_tolerance
        or target_fk_rotation > fk_rotation_tolerance
    ):
        raise ValueError("planned poses do not match official-URDF FK")

    xyz_min = np.full(3, np.inf)
    xyz_max = np.full(3, -np.inf)
    maximum_center_displacement = 0.0
    maximum_rotation = 0.0
    maximum_board_displacement = 0.0
    minimum_joint_limit_margin = np.inf
    minimum_sigma = np.inf
    maximum_condition = 0.0
    for alpha in np.linspace(0.0, 1.0, sample_count):
        q = start_q + alpha * (target_q - start_q)
        transform = fk(q)
        xyz = transform[:3, 3]
        xyz_min = np.minimum(xyz_min, xyz)
        xyz_max = np.maximum(xyz_max, xyz)
        if np.any(xyz < workspace_min) or np.any(xyz > workspace_max):
            raise ValueError(f"interpolation sample {alpha:.9f} leaves workspace")
        if xyz[2] < minimum_eef_z:
            raise ValueError(f"interpolation sample {alpha:.9f} violates minimum EEF z")
        center = float(np.linalg.norm(xyz - start_fk[:3, 3]))
        rotation = _rotation_error(start_fk[:3, :3], transform[:3, :3])
        board = center + 2.0 * board_extent * math.sin(0.5 * rotation)
        maximum_center_displacement = max(maximum_center_displacement, center)
        maximum_rotation = max(maximum_rotation, rotation)
        maximum_board_displacement = max(maximum_board_displacement, board)
        lower_margin = q - joint_limits[:, 0]
        upper_margin = joint_limits[:, 1] - q
        minimum_joint_limit_margin = min(
            minimum_joint_limit_margin,
            float(np.min(np.minimum(lower_margin, upper_margin))),
        )
        metrics = jacobian_metrics(q)
        minimum_sigma = min(minimum_sigma, metrics["sigma_min"])
        maximum_condition = max(maximum_condition, metrics["condition_number"])

    if minimum_joint_limit_margin < joint_margin:
        raise ValueError("interpolation violates the configured joint-limit margin")
    if maximum_board_displacement > maximum_board_sweep:
        raise ValueError("interpolation exceeds the configured board-sweep bound")

    return {
        "schema_version": 1,
        "kind": "franka_joint_recovery_kinematics_audit",
        "status": "pass",
        "hardware_connected": False,
        "motion_performed": False,
        "spec_path": str(spec_path),
        "spec_sha256": _sha256(spec_path),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "pinocchio_version": str(pin.__version__),
        "urdf_path": str(urdf_path),
        "urdf_sha256": actual_urdf_sha,
        "frame": frame_name,
        "sample_count": sample_count,
        "expected_start_fk_error": {
            "translation_m": start_fk_translation,
            "rotation_deg": math.degrees(start_fk_rotation),
        },
        "expected_target_fk_error": {
            "translation_m": target_fk_translation,
            "rotation_deg": math.degrees(target_fk_rotation),
        },
        "path_xyz_min_m": xyz_min.tolist(),
        "path_xyz_max_m": xyz_max.tolist(),
        "minimum_eef_z_m": float(xyz_min[2]),
        "maximum_eef_center_displacement_m": maximum_center_displacement,
        "maximum_eef_rotation_deg": math.degrees(maximum_rotation),
        "maximum_conservative_board_displacement_m": maximum_board_displacement,
        "minimum_joint_limit_margin_rad": minimum_joint_limit_margin,
        "minimum_jacobian_sigma": minimum_sigma,
        "maximum_jacobian_condition_number": maximum_condition,
        "start_jacobian": jacobian_metrics(start_q),
        "target_jacobian": jacobian_metrics(target_q),
        "start_fk_T_base_ee": start_fk.tolist(),
        "target_fk_T_base_ee": target_fk.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    try:
        result = audit(args.spec)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"[joint recovery kinematics audit] FAIL: {exc}")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
