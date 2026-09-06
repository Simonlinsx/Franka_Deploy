#!/usr/bin/env python3
"""Return Franka with the installed RH56 to the V94 training q_home.

This is a dedicated low-speed, unloaded reset entry point.  It reads and
verifies q_home from the transferred deployment bundle, then reuses the
reviewed FrankaSequenceDriver motion implementation from
reset_franka_default.py.  RH56 is neither imported nor accessed; its current
pose must be included in the operator's swept-workspace confirmation.  This
does not enable policy execution or modify any commissioning flag.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

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
from sim2real.deployment.bundle import DeployBundle  # noqa: E402
from sim2real.contracts.v94 import V94Contract  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
DEFAULT_BUNDLE = WORKSPACE / "data/test_fixtures/sim2real/deploy.zip"

INSTALLED_TOKEN = "RH56_INSTALLED_ON_FR3"
SWEEP_CLEAR_TOKEN = "FR3_RH56_CURRENT_TO_V94_TRAINING_RESET_SWEEP_CLEAR"
STOP_TOKEN = "FR3_STOP_READY"
PLA_TOKEN = "PLA_LOW_SPEED_UNLOADED_ONLY"
TARGET_TOKEN = "V94_TRAINING_Q_HOME_VERIFIED"


def _load_default_reset_module() -> Any:
    source = Path(__file__).with_name("reset_franka_default.py")
    spec = importlib.util.spec_from_file_location(
        "dexgrasp_reset_franka_default_reused", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reviewed Franka reset implementation: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DEFAULT_RESET = _load_default_reset_module()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Move only Franka at <=0.05rad/s to the hash-bound V94 training "
            "q_home without opening or accessing RH56"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-installed", metavar=INSTALLED_TOKEN)
    parser.add_argument("--confirm-workspace-clear", metavar=SWEEP_CLEAR_TOKEN)
    parser.add_argument("--confirm-stop-ready", metavar=STOP_TOKEN)
    parser.add_argument("--confirm-pla-low-speed", metavar=PLA_TOKEN)
    parser.add_argument("--confirm-training-target", metavar=TARGET_TOKEN)
    return parser


def _require_confirmations(args: argparse.Namespace) -> None:
    required = (
        ("--confirm-installed", args.confirm_installed, INSTALLED_TOKEN),
        ("--confirm-workspace-clear", args.confirm_workspace_clear, SWEEP_CLEAR_TOKEN),
        ("--confirm-stop-ready", args.confirm_stop_ready, STOP_TOKEN),
        ("--confirm-pla-low-speed", args.confirm_pla_low_speed, PLA_TOKEN),
        ("--confirm-training-target", args.confirm_training_target, TARGET_TOKEN),
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


def _training_target(bundle_path: Path) -> tuple[np.ndarray, Mapping[str, Any]]:
    bundle = DeployBundle(bundle_path)
    verification = bundle.verify()
    contract = V94Contract.from_bundle(bundle)
    if contract.bundle_status != "nominal_dynamic_accepted_dr_stress_pending":
        raise ValueError(f"unexpected V94 bundle status {contract.bundle_status!r}")
    target = np.asarray(contract.q_home_rad, dtype=np.float64)
    if target.shape != (7,) or not np.all(np.isfinite(target)):
        raise ValueError("V94 q_home is malformed")
    if np.any(target < contract.joint_limits_rad[:, 0]) or np.any(
        target > contract.joint_limits_rad[:, 1]
    ):
        raise ValueError("V94 q_home is outside bundle joint limits")
    target_sha256 = hashlib.sha256(
        np.ascontiguousarray(target.astype("<f8")).tobytes()
    ).hexdigest()
    metadata = {
        "bundle_contract": verification.bundle_contract,
        "checked_files": int(verification.checked_files),
        "checkpoint_sha256": verification.primary_checkpoint_sha256,
        "q_home_sha256_f64_le": target_sha256,
    }
    return target, metadata


def _reset_config(
    config: Mapping[str, Any], target_q_rad: np.ndarray
) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    result["franka"]["default_q_rad"] = [float(value) for value in target_q_rad]
    DEFAULT_RESET._validate_static_request(result)
    limits = np.asarray(result["franka"]["joint_limits_rad"], dtype=np.float64)
    margin = float(result["franka"]["joint_limit_margin_rad"])
    if limits.shape != (7, 2) or np.any(
        target_q_rad <= limits[:, 0] + margin
    ) or np.any(target_q_rad >= limits[:, 1] - margin):
        raise ValueError("V94 q_home violates commissioned joint-limit margin")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, config_path = load_control_config(args.config)
        assets = verify_adapter_assets(config, config_path)
        target_q, metadata = _training_target(args.bundle)
        runtime_config = _reset_config(config, target_q)
        print(
            "[config] {}\n[adapter] verified={} sha256={}".format(
                config_path, assets.mesh_path, assets.mesh_sha256
            )
        )
        print(
            "[V94 bundle] contract={} checked_files={} checkpoint_sha256={}".format(
                metadata["bundle_contract"],
                metadata["checked_files"],
                metadata["checkpoint_sha256"],
            )
        )
        print(
            "[V94 training reset] q_home={} q_home_sha256_f64_le={}".format(
                np.round(target_q, 9).tolist(),
                metadata["q_home_sha256_f64_le"],
            )
        )
        if args.dry_run:
            print(
                "[DRY RUN] Franka would move only to the V94 q_home at "
                "<=0.05rad/s; RH56 would not be imported or accessed. "
                "No hardware driver was imported."
            )
            return 0
        _require_confirmations(args)
        print(
            "[path authority] exact per-run current-to-V94-training-reset "
            "swept-workspace confirmation accepted; policy execution remains locked"
        )
        DEFAULT_RESET.run_franka_only_reset(runtime_config)
        print(
            "[PASS] Franka reached the hash-bound V94 training q_home; RH56 "
            "was not accessed; policy execution is still locked"
        )
        return 0
    except KeyboardInterrupt:
        print(
            "[interrupted] Ctrl+C received; Franka stop was requested",
            file=sys.stderr,
        )
        return 130
    except DEFAULT_RESET.DefaultStopUnconfirmed as exc:
        print(f"[{exc}]", file=sys.stderr)
        return 5
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[failed] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
