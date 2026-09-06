#!/usr/bin/env python3
"""Move only Franka to the SHA-bound V61 six-expert reset pose.

The reviewed segmented, low-speed Franka-only reset core is reused.  RH56,
camera, perception and policy modules are never opened.  Hardware motion
requires ``--execute`` plus five exact operator confirmations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT / "src", WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anydex_pipeline.control_config import (  # noqa: E402
    load_control_config,
    verify_adapter_assets,
)
from dexgrasp.apps import reset_franka_v94_training as reviewed_reset  # noqa: E402
from sim2real.tasks.launcher import materialize_task_config  # noqa: E402
from sim2real.tasks.thrown_contract import (  # noqa: E402
    resolve_v57_thrown_task_contract,
)


DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
FR3_SYSTEM_LIMITS = np.asarray(
    [
        [-2.9007, 2.9007],
        [-1.8361, 1.8361],
        [-2.9007, 2.9007],
        [-3.0770, -0.1169],
        [-2.8763, 2.8763],
        [0.4398, 4.6216],
        [-3.0508, 3.0508],
    ],
    dtype=np.float64,
)
JOINT_LIMIT_MARGIN_RAD = 0.02

INSTALLED_TOKEN = "RH56_INSTALLED_ON_FR3"
SWEEP_CLEAR_TOKEN = "FR3_RH56_CURRENT_TO_V61_SIXEXPERT_RESET_SWEEP_CLEAR"
STOP_TOKEN = "FR3_STOP_READY"
PLA_TOKEN = "PLA_LOW_SPEED_UNLOADED_ONLY"
TARGET_TOKEN = "V61_SIXEXPERT_Q_HOME_SHA_VERIFIED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task-config", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-installed", metavar=INSTALLED_TOKEN)
    parser.add_argument("--confirm-workspace-clear", metavar=SWEEP_CLEAR_TOKEN)
    parser.add_argument("--confirm-stop-ready", metavar=STOP_TOKEN)
    parser.add_argument("--confirm-pla-low-speed", metavar=PLA_TOKEN)
    parser.add_argument("--confirm-v61-target", metavar=TARGET_TOKEN)
    return parser


def _require_confirmations(args: argparse.Namespace) -> None:
    required = (
        ("--confirm-installed", args.confirm_installed, INSTALLED_TOKEN),
        ("--confirm-workspace-clear", args.confirm_workspace_clear, SWEEP_CLEAR_TOKEN),
        ("--confirm-stop-ready", args.confirm_stop_ready, STOP_TOKEN),
        ("--confirm-pla-low-speed", args.confirm_pla_low_speed, PLA_TOKEN),
        ("--confirm-v61-target", args.confirm_v61_target, TARGET_TOKEN),
    )
    missing = [
        f"{flag} {token}"
        for flag, actual, token in required
        if actual != token
    ]
    if missing:
        raise ValueError(
            "exact confirmations required before hardware import: "
            + "; ".join(missing)
        )


def _target(task_config: Optional[Path]) -> tuple[np.ndarray, str, str, Path]:
    if task_config is None:
        resolved, _metadata = materialize_task_config("thrown_object_v61")
    else:
        resolved = task_config.expanduser().resolve()
    task = resolve_v57_thrown_task_contract(resolved)
    if (
        task is None
        or task.selected_curriculum.name != "alpha_0_5"
        or "V61NaturalPalmCatchTilt50" not in task.teacher
    ):
        raise ValueError("task config is not the pinned V61 alpha=0.5 candidate")
    target = np.asarray(task.franka_q_home_rad, dtype=np.float64)
    if target.shape != (7,) or not np.all(np.isfinite(target)):
        raise ValueError("V61 q_home is malformed")
    digest = hashlib.sha256(
        np.ascontiguousarray(target.astype("<f8")).tobytes()
    ).hexdigest()
    return target, digest, task.source_sha256, resolved


def _reset_config(config: dict, target: np.ndarray) -> dict:
    result = copy.deepcopy(config)
    franka = result["franka"]
    franka["default_q_rad"] = target.tolist()
    franka["joint_limits_rad"] = FR3_SYSTEM_LIMITS.tolist()
    franka["joint_limit_margin_rad"] = JOINT_LIMIT_MARGIN_RAD
    franka["default_max_joint_velocity_rad_s"] = 0.03
    franka["default_max_joint_segment_rad"] = 0.10
    franka["default_min_duration_s"] = 8.0
    reviewed_reset.DEFAULT_RESET._validate_static_request(result)
    lower = FR3_SYSTEM_LIMITS[:, 0] + JOINT_LIMIT_MARGIN_RAD
    upper = FR3_SYSTEM_LIMITS[:, 1] - JOINT_LIMIT_MARGIN_RAD
    if np.any(target <= lower) or np.any(target >= upper):
        raise ValueError("V61 q_home violates the pinned FR3 reset margin")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, config_path = load_control_config(args.config)
        assets = verify_adapter_assets(config, config_path)
        target, target_sha256, task_sha256, task_config = _target(args.task_config)
        runtime_config = _reset_config(config, target)
        minimum_margin = float(
            np.min(
                np.minimum(
                    target - (FR3_SYSTEM_LIMITS[:, 0] + JOINT_LIMIT_MARGIN_RAD),
                    (FR3_SYSTEM_LIMITS[:, 1] - JOINT_LIMIT_MARGIN_RAD) - target,
                )
            )
        )
        print(
            f"[config] {config_path}\n"
            f"[adapter] verified={assets.mesh_path} sha256={assets.mesh_sha256}"
        )
        print(
            "[V61 reset] q_home={} q_home_sha256_f64_le={} "
            "task_contract_sha256={} task_config={}".format(
                np.round(target, 9).tolist(),
                target_sha256,
                task_sha256,
                task_config,
            )
        )
        print(
            "[motion envelope] Franka-only, <=0.03rad/s, <=0.10rad/joint "
            "segments; RH56/policy/perception unopened; minimum joint-limit "
            f"clearance={minimum_margin:.6f}rad"
        )
        if not args.execute:
            print("[DRY RUN] no hardware driver was imported")
            return 0
        _require_confirmations(args)
        print(
            "[path authority] current-to-V61 reset sweep and physical stop "
            "readiness confirmed",
            flush=True,
        )
        proof = reviewed_reset.DEFAULT_RESET.run_franka_only_reset(runtime_config)
        if not np.array_equal(np.asarray(proof.target_q_rad), target):
            raise RuntimeError("Franka reset proof target differs from V61 contract")
        print(
            "[PASS] Franka reached V61 six-expert q_home; "
            f"max_error={proof.final_linf_error_rad:.7f}rad; RH56 not accessed"
        )
        return 0
    except KeyboardInterrupt:
        print("[interrupted] Ctrl+C received; Franka stop was requested", file=sys.stderr)
        return 130
    except reviewed_reset.DEFAULT_RESET.DefaultStopUnconfirmed as exc:
        print(f"[{exc}]", file=sys.stderr)
        return 5
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[failed] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
