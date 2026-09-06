#!/usr/bin/env python3
"""Offline-only, fail-closed C2 commissioning preflight for V94.

This module is an evidence verifier, not a control entry point.  It imports no
hardware adapter, opens no device, creates no motion authorization and exposes
no execute/force option.  A PASS means only that the exact bundle, deployment
configuration, shadow capture and named commissioning artifacts are eligible
to be presented to the operator-authorization boundary described in
``docs/sim2real/CLOSED_LOOP_CONTROL_DESIGN.md``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    workspace = Path(__file__).resolve().parents[2]
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from sim2real.deployment.shadow_readiness import assess_v94_shadow_readiness
    from sim2real.closed_loop_core import AUTHORIZATION_SCOPE, SafetyState
    from sim2real.deployment.bundle import DeployBundle
    from sim2real.deployment.safety import DEFAULT_DEPLOY_CONFIG
    from sim2real.contracts.v94 import V94Contract
    from sim2real.deployment.verify import verify_v94_bundle
else:
    from sim2real.deployment.shadow_readiness import assess_v94_shadow_readiness
    from sim2real.closed_loop_core import AUTHORIZATION_SCOPE, SafetyState
    from .bundle import DeployBundle
    from .safety import DEFAULT_DEPLOY_CONFIG
    from sim2real.contracts.v94 import V94Contract
    from .verify import verify_v94_bundle


PREFLIGHT_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_KIND = "v94_c2_commissioning_evidence"
READINESS_LEVEL = "C2_BOUNDED_CLOSED_LOOP"
EXPECTED_BUNDLE_STATUS = "nominal_dynamic_accepted_dr_stress_pending"
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_HASHED_FILE_BYTES = 1024 * 1024 * 1024
MIN_RH56_COMMAND_RATE_HZ = 60.0
MIN_DUAL_ACK_RATE_HZ = 60.0
MAX_DUAL_ACK_GAP_S = 1.0 / 30.0
MAX_POLICY_WATCHDOG_TIMEOUT_S = 0.050
MIN_FRANKA_LOOP_RATE_HZ = 1000.0


# Every item is physical or hardware-in-the-loop evidence which cannot be
# inferred from a config boolean or from read-only shadow inference.
REQUIRED_EVIDENCE_ITEMS: tuple[tuple[str, str], ...] = (
    ("fr3_hardware_revision", "installed_hardware"),
    ("installed_payload_dynamics", "installed_hardware"),
    ("installed_fr3_adapter_rh56_collision_model", "collision"),
    ("policy_reset_path_collision", "collision"),
    ("policy_workspace_collision", "collision"),
    ("physical_deadman_acceptance", "interlocks"),
    ("physical_emergency_stop_acceptance", "interlocks"),
    ("franka_persistent_fci_1khz_session", "franka_control"),
    ("franka_single_owner_state_source", "franka_control"),
    ("franka_online_velocity_acceleration_jerk_limits", "franka_control"),
    ("franka_command_watchdog_and_fault_stop", "franka_control"),
    ("franka_tracking_and_collision_monitoring", "franka_control"),
    ("rh56_single_owner_serial_session", "rh56_control"),
    ("rh56_full_six_axis_and_fingertip_fk", "rh56_control"),
    ("rh56_full_thumb_rotation_range", "rh56_control"),
    ("rh56_60hz_write_readback", "rh56_control"),
    ("rh56_fault_disable_and_verified_stop", "rh56_control"),
    ("dual_device_same_sequence_ack_transaction", "transaction"),
    ("dual_device_rate_and_max_gap", "transaction"),
    ("camera_control_latency", "observation"),
    ("camera_robot_time_alignment", "observation"),
    ("external_object_mask_depth_cleaning", "observation"),
    ("policy_reset_scene_height", "observation"),
)


REQUIRED_EXECUTION_FLAGS: tuple[str, ...] = (
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


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        info = path.stat()
    except OSError as exc:
        raise ValueError(f"cannot stat {label} JSON {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ValueError(f"{label} JSON must be a non-empty regular file: {path}")
    if info.st_size > MAX_JSON_BYTES:
        raise ValueError(f"{label} JSON is too large: {info.st_size} bytes")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON number is forbidden: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must contain an object: {path}")
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


def _sha256_regular_file(path: Path) -> str:
    try:
        info = path.stat()
    except OSError as exc:
        raise ValueError(f"cannot stat evidence-bound file {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ValueError(f"evidence-bound path is not a non-empty regular file: {path}")
    if info.st_size > MAX_HASHED_FILE_BYTES:
        raise ValueError(f"evidence-bound file is too large: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot hash evidence-bound file {path}: {exc}") from exc
    return digest.hexdigest()


def _check(
    checks: list[dict[str, Any]],
    code: str,
    category: str,
    passed: bool,
    failure: str,
    **details: Any,
) -> None:
    checks.append(
        {
            "code": code,
            "category": category,
            "passed": bool(passed),
            "failure": None if passed else failure,
            **details,
        }
    )


def _finite_positive_vector(value: Any, size: int) -> Optional[list[float]]:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        return None
    if np.any(result <= 0.0):
        return None
    return result.tolist()


def _finite_metric(metrics: Mapping[str, Any], name: str) -> Optional[float]:
    value = metrics.get(name)
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _shadow_checks(
    checks: list[dict[str, Any]], shadow_path: Path, bundle_path: Path
) -> dict[str, Any]:
    try:
        report = assess_v94_shadow_readiness(shadow_path, bundle_path=bundle_path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        report = {
            "result": "FAIL",
            "checks": [],
            "failed_checks": ["shadow_assessor_error"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    indexed = {
        item.get("name"): item
        for item in report.get("checks", [])
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    overall = report.get("result") == "PASS"
    _check(
        checks,
        "observation_shadow_readiness",
        "observation",
        overall,
        "strict read-only observation shadow assessment did not pass",
        failed_shadow_checks=list(report.get("failed_checks", [])),
        error=report.get("error"),
    )
    replay = indexed.get("offline_audit_and_policy_replay_pass", {})
    _check(
        checks,
        "shadow_action_replay",
        "action_replay",
        replay.get("passed") is True,
        "shadow action replay/audit evidence did not pass",
        shadow_check=dict(replay),
    )
    reset = indexed.get("recorded_reset_aligned", {})
    _check(
        checks,
        "shadow_reset_alignment",
        "reset",
        reset.get("passed") is True,
        "object and Franka/RH56 shadow reset alignment did not pass",
        shadow_check=dict(reset),
    )
    write_checks = (
        indexed.get("hardware_writes_false", {}).get("passed") is True
        and indexed.get("robot_command_writes_false", {}).get("passed") is True
    )
    _check(
        checks,
        "shadow_contains_no_robot_writes",
        "read_only_safety",
        write_checks,
        "shadow artifact does not prove both hardware and robot command writes were false",
    )
    return report


def _static_checks(
    checks: list[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    contract: V94Contract,
    bundle_verification: Any,
) -> None:
    _check(
        checks,
        "packaged_bundle_golden_replay",
        "action_replay",
        getattr(bundle_verification, "hardware_writes", None) is False,
        "packaged bundle hash/checkpoint/golden replay verification did not pass read-only",
        verification=asdict(bundle_verification),
    )
    _check(
        checks,
        "bundle_status_eligible_for_c2_only",
        "bundle",
        contract.bundle_status == EXPECTED_BUNDLE_STATUS,
        "deployment bundle has an unexpected status",
        actual=contract.bundle_status,
        expected=EXPECTED_BUNDLE_STATUS,
        c3_distribution_randomization_claimed=False,
    )

    execution = _mapping(config.get("execution"), "execution")
    _check(
        checks,
        "offline_preflight_hardware_writes_disabled",
        "read_only_safety",
        execution.get("hardware_writes_enabled") is False,
        "offline commissioning preflight requires execution.hardware_writes_enabled=false",
        actual=execution.get("hardware_writes_enabled"),
    )
    for name in REQUIRED_EXECUTION_FLAGS:
        _check(
            checks,
            f"execution_{name}",
            "commissioning_config",
            execution.get(name) is True,
            f"execution.{name} is not commissioned",
            actual=execution.get(name),
        )

    _check(
        checks,
        "profile_mode_c2_commissioned",
        "commissioning_config",
        profile.get("mode") == "c2_bounded_closed_loop_commissioned",
        "commissioning profile is not explicitly in C2 bounded closed-loop mode",
        actual=profile.get("mode"),
        expected="c2_bounded_closed_loop_commissioned",
    )

    franka = _mapping(profile.get("franka"), "commissioning.franka")
    inspire = _mapping(profile.get("inspire"), "commissioning.inspire")
    tool = _mapping(profile.get("tool"), "commissioning.tool")
    default_q = np.asarray(franka.get("default_q_rad"), dtype=np.float64)
    q_valid = default_q.shape == (7,) and np.all(np.isfinite(default_q))
    home_error = (
        float(np.max(np.abs(default_q - contract.q_home_rad)))
        if q_valid
        else None
    )
    _check(
        checks,
        "commissioning_profile_reset_matches_training",
        "reset",
        q_valid and home_error is not None and home_error <= 0.05,
        "commissioning profile default pose differs materially from V94 training reset",
        maximum_joint_error_rad=home_error,
        maximum_allowed_error_rad=0.05,
    )

    commissioned_rate = _finite_metric(
        franka, "default_max_joint_velocity_rad_s"
    )
    rate_passed = bool(
        commissioned_rate is not None
        and commissioned_rate + 1.0e-12
        >= contract.maximum_nominal_arm_velocity_rad_s
    )
    _check(
        checks,
        "policy_rate_within_commissioned_franka_envelope",
        "velocity_safety",
        rate_passed,
        "V94 nominal arm target rate exceeds commissioned Franka rate",
        policy_nominal_rate_rad_s=contract.maximum_nominal_arm_velocity_rad_s,
        commissioned_rate_rad_s=commissioned_rate,
    )

    for field in (
        "online_max_joint_acceleration_rad_s2",
        "online_max_joint_jerk_rad_s3",
        "online_max_tracking_error_rad",
    ):
        values = _finite_positive_vector(franka.get(field), 7)
        _check(
            checks,
            f"franka_{field}",
            "velocity_safety",
            values is not None,
            f"commissioning.franka.{field} must contain seven positive finite values",
            configured=values,
        )

    _check(
        checks,
        "franka_reset_path_collision_profile",
        "collision",
        franka.get("default_path_collision_verified") is True,
        "commissioning Franka reset path collision is unverified",
    )
    _check(
        checks,
        "installed_collision_model_profile",
        "collision",
        tool.get("installed_collision_model_verified") is True,
        "installed FR3/adapter/RH56 collision model is unverified",
    )
    _check(
        checks,
        "installed_tool_not_low_speed_unloaded_only",
        "collision",
        tool.get("low_speed_unloaded_commissioning_only") is False,
        "installed tool remains restricted to low-speed unloaded commissioning",
    )
    _check(
        checks,
        "rh56_six_axis_profile",
        "rh56_control",
        inspire.get("six_axis_coupled_closure_commissioned") is True,
        "RH56 six-axis coupled policy motion is uncommissioned",
    )
    thumb_range = np.asarray(
        inspire.get("thumb_rotate_validated_realtime_range", []), dtype=np.float64
    )
    thumb_passed = bool(
        thumb_range.shape == (2,)
        and np.all(np.isfinite(thumb_range))
        and thumb_range[0] <= 0.0
        and thumb_range[1] >= 1000.0
    )
    _check(
        checks,
        "rh56_full_thumb_rotation_profile",
        "rh56_control",
        thumb_passed,
        "RH56 commissioned realtime thumb rotation does not cover register range 0..1000",
        actual=thumb_range.tolist() if thumb_range.shape == (2,) else None,
        required=[0, 1000],
    )


def _evidence_checks(
    checks: list[dict[str, Any]],
    *,
    evidence_path: Optional[Path],
    expected_bindings: Mapping[str, str],
) -> Optional[dict[str, Any]]:
    evidence: Optional[dict[str, Any]] = None
    evidence_error: Optional[str] = None
    if evidence_path is None:
        evidence_error = "--evidence was not provided"
    else:
        try:
            evidence = _load_json(evidence_path, label="commissioning evidence")
        except ValueError as exc:
            evidence_error = str(exc)

    _check(
        checks,
        "commissioning_evidence_manifest_present",
        "commissioning_evidence",
        evidence is not None,
        "a valid commissioning evidence manifest is required",
        error=evidence_error,
    )
    if evidence is None:
        for name, category in REQUIRED_EVIDENCE_ITEMS:
            _check(
                checks,
                f"evidence_{name}",
                category,
                False,
                f"commissioning evidence item {name!r} is missing",
                reason="commissioning evidence manifest unavailable",
            )
        for name in expected_bindings:
            _check(
                checks,
                f"evidence_binding_{name}",
                "artifact_binding",
                False,
                f"commissioning evidence is not bound to the exact {name}",
            )
        for name in (
            "franka_control_loop_rate_hz",
            "rh56_sustained_command_rate_hz",
            "dual_device_sustained_ack_rate_hz",
            "dual_device_max_interaction_gap_s",
            "policy_command_watchdog_timeout_s",
        ):
            _check(
                checks,
                f"evidence_metric_{name}",
                "commissioning_metrics",
                False,
                f"required commissioning metric {name!r} is missing",
            )
        return None

    _check(
        checks,
        "commissioning_evidence_schema",
        "commissioning_evidence",
        evidence.get("schema_version") == EVIDENCE_SCHEMA_VERSION
        and evidence.get("kind") == EVIDENCE_KIND,
        "commissioning evidence schema/kind is invalid",
        actual_schema_version=evidence.get("schema_version"),
        actual_kind=evidence.get("kind"),
    )
    run_id = evidence.get("run_id")
    _check(
        checks,
        "commissioning_evidence_run_id",
        "commissioning_evidence",
        isinstance(run_id, str) and bool(run_id.strip()),
        "commissioning evidence run_id must be non-empty",
        actual=run_id,
    )

    bindings_value = evidence.get("bindings")
    bindings = bindings_value if isinstance(bindings_value, Mapping) else {}
    for name, expected in expected_bindings.items():
        actual = bindings.get(name)
        _check(
            checks,
            f"evidence_binding_{name}",
            "artifact_binding",
            isinstance(actual, str)
            and len(actual) == 64
            and actual.lower() == expected,
            f"commissioning evidence is not bound to the exact {name}",
            expected_sha256=expected,
            actual_sha256=actual,
        )

    items_value = evidence.get("items")
    items = items_value if isinstance(items_value, Mapping) else {}
    artifact_hash_cache: dict[Path, str] = {}
    manifest_parent = evidence_path.parent
    for name, category in REQUIRED_EVIDENCE_ITEMS:
        raw = items.get(name)
        passed = False
        detail: dict[str, Any] = {}
        if isinstance(raw, Mapping):
            artifact_value = raw.get("artifact")
            expected_hash = raw.get("sha256")
            reviewed_by = raw.get("reviewed_by")
            detail.update(
                artifact=artifact_value,
                expected_sha256=expected_hash,
                reviewed_by=reviewed_by,
            )
            if isinstance(artifact_value, str) and artifact_value.strip():
                artifact = Path(artifact_value).expanduser()
                if not artifact.is_absolute():
                    artifact = manifest_parent / artifact
                artifact = artifact.resolve()
                try:
                    actual_hash = artifact_hash_cache.get(artifact)
                    if actual_hash is None:
                        actual_hash = _sha256_regular_file(artifact)
                        artifact_hash_cache[artifact] = actual_hash
                    detail["actual_sha256"] = actual_hash
                    passed = bool(
                        raw.get("result") == "PASS"
                        and isinstance(expected_hash, str)
                        and len(expected_hash) == 64
                        and expected_hash.lower() == actual_hash
                        and isinstance(reviewed_by, str)
                        and bool(reviewed_by.strip())
                    )
                except ValueError as exc:
                    detail["artifact_error"] = str(exc)
        _check(
            checks,
            f"evidence_{name}",
            category,
            passed,
            f"commissioning evidence item {name!r} is missing, unreviewed, failed, or hash-invalid",
            **detail,
        )

    metrics_value = evidence.get("metrics")
    metrics = metrics_value if isinstance(metrics_value, Mapping) else {}
    metric_rules = (
        (
            "franka_control_loop_rate_hz",
            MIN_FRANKA_LOOP_RATE_HZ,
            None,
            "persistent Franka control loop was not verified at 1 kHz",
        ),
        (
            "rh56_sustained_command_rate_hz",
            MIN_RH56_COMMAND_RATE_HZ,
            None,
            "RH56 verified write/readback did not sustain 60 Hz",
        ),
        (
            "dual_device_sustained_ack_rate_hz",
            MIN_DUAL_ACK_RATE_HZ,
            None,
            "dual-device exact-sequence acknowledgement rate is too low",
        ),
        (
            "dual_device_max_interaction_gap_s",
            0.0,
            MAX_DUAL_ACK_GAP_S,
            "dual-device maximum inter-action gap is too large",
        ),
        (
            "policy_command_watchdog_timeout_s",
            0.0,
            MAX_POLICY_WATCHDOG_TIMEOUT_S,
            "policy command watchdog timeout is absent or too large",
        ),
    )
    for name, minimum, maximum, failure in metric_rules:
        actual = _finite_metric(metrics, name)
        passed = bool(
            actual is not None
            and (actual >= minimum if maximum is None else actual > minimum)
            and (maximum is None or actual <= maximum)
        )
        _check(
            checks,
            f"evidence_metric_{name}",
            "commissioning_metrics",
            passed,
            failure,
            actual=actual,
            minimum=minimum,
            minimum_inclusive=maximum is None,
            inclusive_maximum=maximum,
        )
    return evidence


def assess_v94_commissioning_preflight(
    shadow_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_DEPLOY_CONFIG,
    evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Assess offline evidence for C2 eligibility without authorizing motion."""

    shadow = Path(shadow_path).expanduser().resolve()
    config_source = Path(config_path).expanduser().resolve()
    evidence_source = (
        None if evidence_path is None else Path(evidence_path).expanduser().resolve()
    )
    config_sha256 = _sha256_regular_file(config_source)
    config = _load_json(config_source, label="deployment config")
    if _sha256_regular_file(config_source) != config_sha256:
        raise ValueError("deployment config changed while preflight was reading it")
    if config.get("schema_version") != 1:
        raise ValueError("V94 deploy config schema_version must be 1")
    bundle_path = _resolve(config_source, config.get("bundle"), "bundle")
    profile_path = _resolve(
        config_source, config.get("commissioning_profile"), "commissioning_profile"
    )
    initial_bindings = {
        "shadow_npz_sha256": _sha256_regular_file(shadow),
        "bundle_sha256": _sha256_regular_file(bundle_path),
        "deploy_config_sha256": config_sha256,
        "commissioning_profile_sha256": _sha256_regular_file(profile_path),
    }
    profile = _load_json(profile_path, label="commissioning profile")
    if profile.get("schema_version") != 1:
        raise ValueError("commissioning profile schema_version must be 1")

    # These operations verify only regular files and deterministic policy replay.
    # Neither module imports a hardware adapter or has a command path.
    bundle_verification = verify_v94_bundle(bundle_path)
    contract = V94Contract.from_bundle(DeployBundle(bundle_path))
    checks: list[dict[str, Any]] = []
    shadow_report = _shadow_checks(checks, shadow, bundle_path)
    _static_checks(
        checks,
        config=config,
        profile=profile,
        contract=contract,
        bundle_verification=bundle_verification,
    )

    final_bindings = {
        "shadow_npz_sha256": _sha256_regular_file(shadow),
        "bundle_sha256": _sha256_regular_file(bundle_path),
        "deploy_config_sha256": _sha256_regular_file(config_source),
        "commissioning_profile_sha256": _sha256_regular_file(profile_path),
    }
    if final_bindings != initial_bindings:
        changed = sorted(
            name
            for name in initial_bindings
            if final_bindings[name] != initial_bindings[name]
        )
        raise ValueError(
            "preflight input changed during verification: " + ", ".join(changed)
        )
    expected_bindings = initial_bindings
    evidence = _evidence_checks(
        checks,
        evidence_path=evidence_source,
        expected_bindings=expected_bindings,
    )
    failed = [item for item in checks if not item["passed"]]
    result = "PASS" if not failed else "FAIL"
    return {
        "result": result,
        "preflight_schema_version": PREFLIGHT_SCHEMA_VERSION,
        "readiness_level": READINESS_LEVEL,
        "offline_only": True,
        "device_access": False,
        "hardware_writes": False,
        "robot_command_writes": False,
        "arming_state": SafetyState.DISARMED.value.upper(),
        "motion_authorization_created": False,
        "physical_motion_authorized": False,
        "future_authorization_scope_required": AUTHORIZATION_SCOPE,
        "eligible_for_operator_authorization": result == "PASS",
        "c3_task_or_production_readiness_claimed": False,
        "run_id": None if evidence is None else evidence.get("run_id"),
        "inputs": {
            "shadow_npz": str(shadow),
            "deploy_config": str(config_source),
            "bundle": str(bundle_path),
            "commissioning_profile": str(profile_path),
            "evidence_manifest": (
                None if evidence_source is None else str(evidence_source)
            ),
            "sha256": expected_bindings,
        },
        "checks": checks,
        "failed_checks": [item["code"] for item in failed],
        "blockers": [
            {
                "code": item["code"],
                "category": item["category"],
                "message": item["failure"],
            }
            for item in failed
        ],
        "shadow_result": shadow_report.get("result"),
        "note": (
            "PASS is only offline C2 preflight eligibility. This tool remains "
            "disarmed and cannot authorize or execute Franka/RH56 motion."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-only, fail-closed V94 C2 commissioning evidence preflight; "
            "contains no device or motion path."
        )
    )
    parser.add_argument("shadow_npz", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_DEPLOY_CONFIG)
    parser.add_argument(
        "--evidence",
        type=Path,
        help=(
            "schema-v1 C2 commissioning evidence manifest; omission is an "
            "explicit FAIL, never an override"
        ),
    )
    return parser


def _fatal_report(exc: BaseException) -> dict[str, Any]:
    return {
        "result": "FAIL",
        "preflight_schema_version": PREFLIGHT_SCHEMA_VERSION,
        "readiness_level": READINESS_LEVEL,
        "offline_only": True,
        "device_access": False,
        "hardware_writes": False,
        "robot_command_writes": False,
        "arming_state": SafetyState.DISARMED.value.upper(),
        "motion_authorization_created": False,
        "physical_motion_authorized": False,
        "future_authorization_scope_required": AUTHORIZATION_SCOPE,
        "eligible_for_operator_authorization": False,
        "c3_task_or_production_readiness_claimed": False,
        "failed_checks": ["preflight_input_or_verification_error"],
        "blockers": [
            {
                "code": "preflight_input_or_verification_error",
                "category": "offline_verification",
                "message": f"{type(exc).__name__}: {exc}",
            }
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = assess_v94_commissioning_preflight(
            args.shadow_npz,
            config_path=args.config,
            evidence_path=args.evidence,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        report = _fatal_report(exc)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["result"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
