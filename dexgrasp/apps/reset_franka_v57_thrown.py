#!/usr/bin/env python3
"""Move only Franka to the SHA-bound V57 alpha=0.5 reset pose.

The command reuses the reviewed low-speed FrankaSequenceDriver reset path.
RH56 is neither imported nor accessed.  This is a reset-only command: it does
not start perception, load a policy, or authorize thrown-object rollout.
"""

from __future__ import annotations

import argparse
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
from sim2real.tasks.thrown_contract import (  # noqa: E402
    resolve_v57_thrown_task_contract,
)


# Reuse the reset-only installed-tool profile: unlike the policy profile its
# default-q evidence field is intentionally null, so overriding q with the
# independently SHA-bound V57 target does not misrepresent historical V94
# evidence as authorization for this task pose.
DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
DEFAULT_TASK_CONFIG = (
    WORKSPACE
    / "dexgrasp/runs/task_configs/thrown_object-36e9d1b13bab3cec.yaml"
)

INSTALLED_TOKEN = "RH56_INSTALLED_ON_FR3"
SWEEP_CLEAR_TOKEN = "FR3_RH56_CURRENT_TO_V57_ALPHA0P5_RESET_SWEEP_CLEAR"
STOP_TOKEN = "FR3_STOP_READY"
PLA_TOKEN = "PLA_LOW_SPEED_UNLOADED_ONLY"
TARGET_TOKEN = "V57_ALPHA0P5_Q_HOME_SHA_VERIFIED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-installed", metavar=INSTALLED_TOKEN)
    parser.add_argument("--confirm-workspace-clear", metavar=SWEEP_CLEAR_TOKEN)
    parser.add_argument("--confirm-stop-ready", metavar=STOP_TOKEN)
    parser.add_argument("--confirm-pla-low-speed", metavar=PLA_TOKEN)
    parser.add_argument("--confirm-v57-target", metavar=TARGET_TOKEN)
    return parser


def _require_confirmations(args: argparse.Namespace) -> None:
    required = (
        ("--confirm-installed", args.confirm_installed, INSTALLED_TOKEN),
        ("--confirm-workspace-clear", args.confirm_workspace_clear, SWEEP_CLEAR_TOKEN),
        ("--confirm-stop-ready", args.confirm_stop_ready, STOP_TOKEN),
        ("--confirm-pla-low-speed", args.confirm_pla_low_speed, PLA_TOKEN),
        ("--confirm-v57-target", args.confirm_v57_target, TARGET_TOKEN),
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


def _target(task_config: Path) -> tuple[np.ndarray, str, str]:
    task = resolve_v57_thrown_task_contract(task_config)
    if task is None or task.selected_curriculum.name != "alpha_0_5":
        raise ValueError("task config is not the commissioned V57 alpha_0_5 stage")
    target = np.asarray(task.franka_q_home_rad, dtype=np.float64)
    if target.shape != (7,) or not np.all(np.isfinite(target)):
        raise ValueError("V57 alpha_0_5 q_home is malformed")
    digest = hashlib.sha256(
        np.ascontiguousarray(target.astype("<f8")).tobytes()
    ).hexdigest()
    return target, digest, task.source_sha256


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, config_path = load_control_config(args.config)
        assets = verify_adapter_assets(config, config_path)
        target, target_sha256, task_sha256 = _target(args.task_config)
        runtime_config = reviewed_reset._reset_config(config, target)
        print(
            f"[config] {config_path}\n"
            f"[adapter] verified={assets.mesh_path} sha256={assets.mesh_sha256}"
        )
        print(
            "[V57 alpha=0.5 reset] q_home={} q_home_sha256_f64_le={} "
            "task_contract_sha256={}".format(
                np.round(target, 9).tolist(), target_sha256, task_sha256
            )
        )
        print(
            "[motion envelope] Franka-only, <=0.05rad/s, <=0.20rad/joint "
            "segments; RH56/policy/perception unopened"
        )
        if args.dry_run:
            print("[DRY RUN] no hardware driver was imported")
            return 0
        _require_confirmations(args)
        print(
            "[path authority] current-to-V57-alpha0.5 reset sweep and physical "
            "stop readiness confirmed",
            flush=True,
        )
        proof = reviewed_reset.DEFAULT_RESET.run_franka_only_reset(runtime_config)
        if not np.array_equal(np.asarray(proof.target_q_rad), target):
            raise RuntimeError("Franka reset proof target differs from V57 contract")
        print(
            "[PASS] Franka reached V57 alpha=0.5 q_home; "
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
