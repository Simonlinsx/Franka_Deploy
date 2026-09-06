"""Strict portability proof for RH56 evidence collected at the V94 reset.

Installed-hand commissioning evidence is bound to the generic V7 profile.
During the final coupled run Franka was read-only and continuously observed at
the V94 q_home.  This hardware-free bridge permits only the three
evidence-derived RH56 fields to move onto an otherwise compatible V94 base
profile; every other common field must remain byte-for-byte equivalent as
parsed JSON.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from .rh56_commissioning import (
    derived_config_proposal,
    load_evidence,
    verify_evidence,
)


V94_PROFILE_ID = "fr3_rh56_v94_reset_locked_v1"
V94_POLICY_CONTRACT = "v94_inspire_semantic_13d"
MAXIMUM_V94_Q_HOME_ERROR_RAD = 0.001
MAXIMUM_V94_IDLE_SPEED_RAD_S = 0.01


@dataclass(frozen=True)
class V94RH56BridgeVerification:
    blockers: Tuple[str, ...]
    derived_profile: Optional[dict[str, Any]]
    proposal: Optional[dict[str, Any]]

    @property
    def passed(self) -> bool:
        return not self.blockers


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {label} {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _common_hardware_contract(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the documented V94 reset/dynamics-only differences."""

    common = copy.deepcopy(dict(profile))
    common.pop("profile_id", None)
    common.pop("policy_contract", None)
    franka = common.get("franka")
    if not isinstance(franka, dict):
        raise ValueError("profile has no franka object")
    franka.pop("default_q_rad", None)
    franka.pop("default_q_provenance", None)
    franka.pop("online_max_joint_acceleration_rad_s2", None)
    franka.pop("online_max_joint_jerk_rad_s3", None)
    franka.pop("online_max_tracking_error_rad", None)
    return common


def _verify_v94_franka_read_only_alignment(
    evidence: Mapping[str, Any],
    v94_profile: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    record = evidence.get("franka_read_only")
    if not isinstance(record, Mapping) or record.get("verified") is not True:
        return ["commissioning evidence lacks verified read-only Franka state"]
    if record.get("connection") != "read_once_only_no_controller_no_robot_write":
        blockers.append("Franka evidence does not prove the read-only connection contract")
    continuous = record.get("continuous_gate")
    if (
        not isinstance(continuous, Mapping)
        or continuous.get("failure") is not None
        or not isinstance(continuous.get("check_count"), int)
        or int(continuous["check_count"]) <= 0
    ):
        blockers.append("Franka continuous read-only gate is absent or failed")

    franka = v94_profile.get("franka")
    q_home = franka.get("default_q_rad") if isinstance(franka, Mapping) else None
    if (
        not isinstance(q_home, list)
        or len(q_home) != 7
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in q_home
        )
    ):
        return blockers + ["V94 base profile default_q_rad is malformed"]

    states = [
        ("initial", record.get("initial")),
        ("final", record.get("final")),
        (
            "continuous_gate.last_success",
            continuous.get("last_success") if isinstance(continuous, Mapping) else None,
        ),
    ]
    for name, state in states:
        if not isinstance(state, Mapping):
            blockers.append(f"Franka {name} read-only state is missing")
            continue
        q = state.get("q_rad")
        dq = state.get("dq_rad_s")
        if (
            state.get("robot_mode") != "RobotMode.Idle"
            or not isinstance(q, list)
            or len(q) != 7
            or not isinstance(dq, list)
            or len(dq) != 7
        ):
            blockers.append(f"Franka {name} state is not a complete Idle sample")
            continue
        try:
            q_values = [float(item) for item in q]
            dq_values = [float(item) for item in dq]
        except (TypeError, ValueError):
            blockers.append(f"Franka {name} q/dq state is non-numeric")
            continue
        if not all(math.isfinite(item) for item in q_values + dq_values):
            blockers.append(f"Franka {name} q/dq state is non-finite")
            continue
        error = max(abs(actual - target) for actual, target in zip(q_values, q_home))
        if error > MAXIMUM_V94_Q_HOME_ERROR_RAD:
            blockers.append(
                f"Franka {name} differs from V94 q_home by {error:.9f}rad"
            )
        speed = max(abs(item) for item in dq_values)
        if speed > MAXIMUM_V94_IDLE_SPEED_RAD_S:
            blockers.append(
                f"Franka {name} speed {speed:.9f}rad/s exceeds the bridge limit"
            )
    return blockers


def verify_v94_rh56_profile_bridge(
    evidence_path: Path,
    *,
    source_config_path: Path,
    v94_base_profile_path: Path,
) -> V94RH56BridgeVerification:
    blockers: list[str] = []
    source_verification = verify_evidence(
        evidence_path,
        config_path=source_config_path,
        require_coupled=True,
    )
    blockers.extend(source_verification.blockers)
    try:
        evidence, _ = load_evidence(evidence_path)
        source = _load_json_object(source_config_path, "source control profile")
        v94 = _load_json_object(v94_base_profile_path, "V94 base profile")
    except ValueError as exc:
        return V94RH56BridgeVerification((str(exc),), None, None)

    bound_snapshot = evidence.get("control_profile", {}).get("snapshot")
    if source != bound_snapshot:
        blockers.append("source profile differs from the verified evidence snapshot")
    if v94.get("profile_id") != V94_PROFILE_ID:
        blockers.append("V94 base profile_id is not the reviewed reset-locked identity")
    if v94.get("policy_contract") != V94_POLICY_CONTRACT:
        blockers.append("V94 base profile policy_contract is not reviewed")
    try:
        if _common_hardware_contract(source) != _common_hardware_contract(v94):
            blockers.append(
                "V7 evidence profile and V94 base differ outside allowed reset-only fields"
            )
    except ValueError as exc:
        blockers.append(str(exc))
    blockers.extend(_verify_v94_franka_read_only_alignment(evidence, v94))

    proposal = None
    if not blockers:
        proposal = derived_config_proposal(evidence)
        updated = copy.deepcopy(v94)
        inspire = updated.get("inspire")
        if not isinstance(inspire, dict):
            blockers.append("V94 base profile has no mutable inspire object")
        else:
            inspire["thumb_rotate_validated_realtime_range"] = proposal[
                "inspire.thumb_rotate_validated_realtime_range"
            ]
            inspire["six_axis_coupled_closure_commissioned"] = proposal[
                "inspire.six_axis_coupled_closure_commissioned"
            ]
            inspire["commissioned_air_closure_targets"] = proposal[
                "inspire.commissioned_air_closure_targets"
            ]
    else:
        updated = None
    return V94RH56BridgeVerification(
        tuple(dict.fromkeys(blockers)),
        None if blockers else updated,
        None if blockers else proposal,
    )


def verify_materialized_v94_rh56_profile(
    evidence_path: Path,
    profile_path: Path,
    *,
    source_config_path: Path,
    v94_base_profile_path: Path,
) -> V94RH56BridgeVerification:
    bridge = verify_v94_rh56_profile_bridge(
        evidence_path,
        source_config_path=source_config_path,
        v94_base_profile_path=v94_base_profile_path,
    )
    if not bridge.passed:
        return bridge
    try:
        actual = _load_json_object(profile_path, "materialized V94 RH56 profile")
    except ValueError as exc:
        return V94RH56BridgeVerification((str(exc),), None, bridge.proposal)
    if actual != bridge.derived_profile:
        return V94RH56BridgeVerification(
            ("materialized V94 profile is not the exact bridge-derived update",),
            bridge.derived_profile,
            bridge.proposal,
        )
    return bridge
