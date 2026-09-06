"""Fail-closed audit for any future V94 hardware execution entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .bundle import DeployBundle
from sim2real.contracts.v94 import V94Contract
from .verify import verify_v94_bundle

DEFAULT_DEPLOY_CONFIG = (
    Path(__file__).resolve().parents[1] / "v94_deploy_config.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON number in {path}: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _resolve(source: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


@dataclass(frozen=True)
class HardwareReadiness:
    ready: bool
    blockers: Tuple[str, ...]
    offline_alignment_passed: bool
    simulation_home_delta_from_recorded_real_rad: np.ndarray
    policy_nominal_max_velocity_rad_s: float
    commissioned_max_velocity_rad_s: float


def audit_hardware_readiness(
    config_path: str | Path = DEFAULT_DEPLOY_CONFIG,
) -> HardwareReadiness:
    source = Path(config_path).expanduser().resolve()
    config = _load_json(source)
    if config.get("schema_version") != 1:
        raise ValueError("V94 deploy config schema_version must be 1")
    bundle_path = _resolve(source, config.get("bundle"), "bundle")
    profile_path = _resolve(
        source, config.get("commissioning_profile"), "commissioning_profile"
    )
    profile = _load_json(profile_path)
    execution = _mapping(config.get("execution"), "execution")
    bundle = DeployBundle(bundle_path)
    contract = V94Contract.from_bundle(bundle)
    verify_v94_bundle(bundle_path)

    blockers = []
    if contract.bundle_status != "nominal_dynamic_accepted_dr_stress_pending":
        blockers.append(f"unexpected bundle status {contract.bundle_status!r}")
    required_flags = (
        "hardware_writes_enabled",
        "full_dr_stress_accepted",
        "fr3_hardware_revision_verified",
        "policy_reset_path_collision_verified",
        "policy_workspace_collision_verified",
        "installed_payload_dynamics_measured",
        "rh56_fingertip_fk_commissioned",
        "rh56_six_axis_policy_motion_commissioned",
        "camera_control_latency_measured",
        "camera_robot_time_alignment_verified",
        "external_object_mask_depth_cleaning_verified",
        "policy_reset_scene_height_verified",
        "live_observation_action_replay_verified",
        "policy_rate_tracking_verified",
        "external_hold_gate_commissioned",
        "external_emergency_stop_verified",
    )
    for name in required_flags:
        if execution.get(name) is not True:
            blockers.append(f"execution.{name} is not commissioned")

    franka = _mapping(profile.get("franka"), "commissioning.franka")
    inspire = _mapping(profile.get("inspire"), "commissioning.inspire")
    tool = _mapping(profile.get("tool"), "commissioning.tool")
    provenance = _mapping(
        franka.get("default_q_provenance"),
        "commissioning.franka.default_q_provenance",
    )
    if provenance.get("motion_authorized") is not True:
        blockers.append("commissioning default-q motion is not authorized")
    if franka.get("default_path_collision_verified") is not True:
        blockers.append("commissioning Franka path collision is unverified")
    if tool.get("installed_collision_model_verified") is not True:
        blockers.append("installed FR3/adapter/RH56 collision model is unverified")
    if tool.get("low_speed_unloaded_commissioning_only") is not False:
        blockers.append("installed tool remains restricted to low-speed unloaded tests")
    if inspire.get("six_axis_coupled_closure_commissioned") is not True:
        blockers.append("RH56 six-axis coupled closure is not commissioned")

    real_default = np.asarray(franka.get("default_q_rad"), dtype=np.float64)
    if real_default.shape != (7,) or not np.all(np.isfinite(real_default)):
        raise ValueError("commissioning Franka default_q_rad is invalid")
    home_delta = contract.q_home_rad.astype(np.float64) - real_default
    if float(np.max(np.abs(home_delta))) > 0.05:
        blockers.append(
            "simulation reset differs materially from the recorded real default pose"
        )
    commissioned_velocity = float(franka.get("default_max_joint_velocity_rad_s"))
    if contract.maximum_nominal_arm_velocity_rad_s > commissioned_velocity + 1.0e-12:
        blockers.append(
            "policy nominal arm target rate exceeds the commissioned Franka rate"
        )
    return HardwareReadiness(
        ready=not blockers,
        blockers=tuple(blockers),
        offline_alignment_passed=True,
        simulation_home_delta_from_recorded_real_rad=home_delta,
        policy_nominal_max_velocity_rad_s=contract.maximum_nominal_arm_velocity_rad_s,
        commissioned_max_velocity_rad_s=commissioned_velocity,
    )


__all__ = [
    "DEFAULT_DEPLOY_CONFIG",
    "HardwareReadiness",
    "audit_hardware_readiness",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit V94 hardware readiness without opening any device."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_DEPLOY_CONFIG)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_hardware_readiness(args.config)
        payload = {
            "ready": report.ready,
            "offline_alignment_passed": report.offline_alignment_passed,
            "blockers": list(report.blockers),
            "simulation_home_delta_from_recorded_real_rad": (
                report.simulation_home_delta_from_recorded_real_rad.tolist()
            ),
            "policy_nominal_max_velocity_rad_s": (
                report.policy_nominal_max_velocity_rad_s
            ),
            "commissioned_max_velocity_rad_s": report.commissioned_max_velocity_rad_s,
            "hardware_writes": False,
        }
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print("V94 hardware readiness: " + ("READY" if report.ready else "LOCKED"))
            for blocker in report.blockers:
                print(f"  - {blocker}")
            print("  hardware: untouched (offline audit only)")
        return 0 if report.ready else 2
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"V94 hardware readiness audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
