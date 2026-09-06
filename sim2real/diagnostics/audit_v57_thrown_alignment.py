"""Offline, hardware-free V57 alpha=0.5 camera/task alignment audit.

This audit proves only that the pinned simulation task, reset pose, camera
calibration and runtime depth envelope are mutually consistent.  It never
authorizes robot execution; physical RGB-D/point-cloud holdouts remain a
separate commissioning gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from sim2real.deployment.bundle import DeployBundle
from sim2real.deployment.runner import DEFAULT_BUNDLE
from sim2real.observation.camera_profile import (
    load_resolved_task_config,
    resolve_runtime_camera_contract,
    resolve_runtime_task_contract,
)
from sim2real.tasks.thrown_contract import (
    assess_v57_camera_visibility,
    resolve_v57_thrown_task_contract,
    v57_task_summary,
)
from sim2real.contracts.v94 import V94Contract


class V57AlignmentAuditError(RuntimeError):
    """The offline alignment audit cannot produce accepted evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_alignment_audit(
    *,
    resolved_config: Path,
    bundle_path: Path = DEFAULT_BUNDLE,
) -> dict[str, Any]:
    config_path = Path(resolved_config).expanduser().resolve(strict=True)
    bundle_source = Path(bundle_path).expanduser().resolve(strict=True)
    config, profile = load_resolved_task_config(config_path)
    if str(profile.get("name", "")) != "thrown_object":
        raise V57AlignmentAuditError("resolved config is not the thrown_object task")
    if str(profile.get("simulation_curriculum", "")) != "alpha_0_5":
        raise V57AlignmentAuditError("only V57 alpha_0_5 is commissioned")

    bundle = DeployBundle(bundle_source)
    bundle.verify()
    base_contract = V94Contract.from_bundle(bundle).with_runtime_policy_rate_hz(20.0)
    camera_contract = resolve_runtime_camera_contract(base_contract, config_path)
    task_contract = resolve_v57_thrown_task_contract(config_path)
    if task_contract is None:
        raise V57AlignmentAuditError("resolved config has no V57 task contract")
    resolved_contract = resolve_runtime_task_contract(base_contract, config_path)

    selected = assess_v57_camera_visibility(task_contract, camera_contract)
    alpha_one = assess_v57_camera_visibility(
        task_contract, camera_contract, curriculum="alpha_1_0"
    )
    checks = {
        "selected_curriculum_is_alpha_0_5": (
            task_contract.selected_curriculum.name == "alpha_0_5"
        ),
        "policy_rate_is_20_hz": task_contract.control_hz == 20.0,
        "rh56_speed_set_is_600_all_axes": bool(
            np.array_equal(
                task_contract.rh56_speed_register_order,
                np.full(6, 600, dtype=np.int32),
            )
        ),
        "rh56_reset_is_open_all_axes": bool(
            np.array_equal(
                task_contract.rh56_open_register_order,
                np.full(6, 1000, dtype=np.int32),
            )
        ),
        "alpha_0_5_target_center_box_fully_visible": (
            selected.target_center_box_fully_visible
        ),
        "alpha_0_5_object_support_depth_covered": (
            selected.object_support_depth_covered
        ),
        "resolved_reset_matches_v57": bool(
            np.array_equal(
                resolved_contract.q_home_rad,
                task_contract.franka_q_home_rad,
            )
        ),
        "camera_is_424x240_at_60_hz": bool(
            camera_contract.camera_width == 424
            and camera_contract.camera_height == 240
            and camera_contract.camera_rate_hz == 60.0
        ),
        "camera_serial_is_342222071785": (
            camera_contract.camera_serial == "342222071785"
        ),
        "robot_execution_latch_matches_commissioning_status": bool(
            (
                str(profile.get("commissioning_status", "")) == "accepted"
                and profile.get("robot_execution_enabled") is True
            )
            or (
                str(profile.get("commissioning_status", "")) != "accepted"
                and profile.get("robot_execution_enabled") is False
            )
        ),
    }
    accepted = all(checks.values())
    calibration_path = Path(str(config["extrinsics"]["calibration_file"])).resolve()
    result: dict[str, Any] = {
        "schema": "v57_alpha_0_5_static_alignment_audit_v1",
        "static_alignment_accepted": accepted,
        "robot_execution_authorized": False,
        "checks": checks,
        "resolved_config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "bundle": {
            "path": str(bundle_source),
            "sha256": _sha256(bundle_source),
        },
        "calibration": {
            "path": str(calibration_path),
            "sha256": _sha256(calibration_path),
            "calibration_id": camera_contract.calibration_id,
            "camera_serial": camera_contract.camera_serial,
        },
        "v57": v57_task_summary(config_path, camera_contract),
        "alpha_1_0_out_of_scope": {
            "commissioned": False,
            "camera_visibility": alpha_one.as_dict(),
            "reason": (
                "alpha_1_0 is not selected; its target-center box has only "
                f"{alpha_one.target_center_vertices_inside}/"
                f"{alpha_one.target_center_vertex_count} projected corners in view"
            ),
        },
        "remaining_physical_gates": [
            "recorded 424x240@60 RGB-D holdout with the fixed camera",
            "text grounding and current-frame mask acceptance",
            "fresh full 128-point robot_base cloud at 20 Hz",
            "known robot_base point closure in the selected alpha_0_5 volume",
            "task-specific checkpoint/profile admission before robot execution",
        ],
    }
    if not accepted:
        failed = [name for name, value in checks.items() if not value]
        raise V57AlignmentAuditError(
            "static V57 alignment checks failed: " + ", ".join(failed)
        )
    return result


def _write_exclusive_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"audit refuses to overwrite: {destination}")
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = build_alignment_audit(
        resolved_config=args.resolved_config,
        bundle_path=args.bundle,
    )
    _write_exclusive_atomic(args.output, report)
    print(f"[V57 alpha=0.5 static alignment PASS] {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V57AlignmentAuditError",
    "build_alignment_audit",
]
