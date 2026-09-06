#!/usr/bin/env python3
"""Strict, offline-only readiness gate for a V94 shadow-preview capture.

This assessor deliberately separates a deterministic replay audit from the
timing and read-only conditions needed for a useful commissioning shadow run.
Passing it does not establish semantic action correctness, safety of motion,
or authorization to command either robot.

Only a local NPZ and the packaged deployment bundle are opened.  This module
does not import hardware adapters and has no robot command path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    workspace = Path(__file__).resolve().parents[2]
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from sim2real.diagnostics.audit_v94_preview import (
        DEFAULT_BUNDLE,
        DEFAULT_REPLAY_ATOL,
        audit_v94_preview,
    )
else:
    from sim2real.diagnostics.audit_v94_preview import (
        DEFAULT_BUNDLE,
        DEFAULT_REPLAY_ATOL,
        audit_v94_preview,
    )


ASSESSOR_SCHEMA_VERSION = 2
MAX_INPUT_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_CAMERA_AGE_S = 0.100
DEFAULT_MIN_ACTION_RATE_HZ = 57.0
MAX_POLICY_USES_PER_CAMERA_FRAME = 2
POLICY_REUSE_HOLD_SCHEMA_VERSION = 1
POLICY_REUSE_HOLD_REASON = "camera_frame_policy_reuse_limit"
POLICY_REUSE_GUARD_SEMANTICS = (
    "third_eligible_tick_holds_without_policy_state_or_action_evidence"
)


def _scalar(values: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in values:
        raise ValueError(f"missing required shadow-readiness field: {name}")
    value = np.asarray(values[name])
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return value


def _scalar_bool(values: Mapping[str, np.ndarray], name: str) -> bool:
    value = _scalar(values, name)
    if value.dtype.kind != "b":
        raise ValueError(f"{name} must be a scalar boolean")
    return bool(value)


def _scalar_float(values: Mapping[str, np.ndarray], name: str) -> float:
    value = _scalar(values, name)
    if value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a scalar number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _scalar_integer(values: Mapping[str, np.ndarray], name: str) -> int:
    value = _scalar(values, name)
    if value.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a scalar integer")
    return int(value)


def _read_no_pickle_npz(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"shadow NPZ not found: {source}")
    size = source.stat().st_size
    if size <= 0 or size > MAX_INPUT_BYTES:
        raise ValueError(f"shadow NPZ has an unsafe file size: {size} bytes")
    try:
        with np.load(source, allow_pickle=False) as archive:
            values = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        # Loading an object array with allow_pickle=False reaches this path.
        raise ValueError(f"invalid no-pickle shadow NPZ {source}: {exc}") from exc
    object_fields = [
        name for name, value in values.items() if np.asarray(value).dtype.kind == "O"
    ]
    if object_fields:
        raise ValueError(f"forbidden object dtype fields: {sorted(object_fields)}")
    return values


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **details}


def _camera_frame_reuse(
    values: Mapping[str, np.ndarray], steps: int
) -> dict[str, Any]:
    if "pointcloud_frame_id" not in values:
        raise ValueError("missing required shadow-readiness field: pointcloud_frame_id")
    frame_ids = np.asarray(values["pointcloud_frame_id"])
    if frame_ids.shape != (steps,) or frame_ids.dtype.kind not in "iu":
        raise ValueError(
            f"pointcloud_frame_id must be an integer [{steps}] array"
        )
    if steps > 1 and np.any(np.diff(frame_ids.astype(np.int64)) < 0):
        raise ValueError("pointcloud_frame_id must be nondecreasing")
    _, counts = np.unique(frame_ids, return_counts=True)
    unique_frames = int(counts.size)
    reused_frames = int(np.count_nonzero(counts > 1))
    repeated_policy_uses = int(np.sum(counts - 1))
    uses_per_unique = float(steps / unique_frames)
    maximum_uses = int(np.max(counts))
    return {
        "policy_steps": steps,
        "unique_camera_frames": unique_frames,
        "reused_camera_frames": reused_frames,
        "reused_camera_frame_fraction": float(reused_frames / unique_frames),
        "repeated_policy_use_fraction": float(repeated_policy_uses / steps),
        "policy_uses_per_unique_camera_frame": uses_per_unique,
        "maximum_policy_uses_for_one_camera_frame": maximum_uses,
        "maximum_allowed_policy_uses_per_camera_frame": (
            MAX_POLICY_USES_PER_CAMERA_FRAME
        ),
        "passed": bool(
            uses_per_unique <= MAX_POLICY_USES_PER_CAMERA_FRAME
            and maximum_uses <= MAX_POLICY_USES_PER_CAMERA_FRAME
        ),
    }


def _policy_reuse_guard_evidence(
    values: Mapping[str, np.ndarray], steps: int
) -> dict[str, Any]:
    """Cross-check action rows and separately recorded third-tick holds."""

    required_base_fields = {
        "pointcloud_frame_id",
        "policy_action_use_index_for_camera_frame",
        "maximum_policy_actions_per_camera_frame",
        "policy_action_evidence_count",
        "policy_reuse_guard_hold_count",
        "policy_action_or_reuse_hold_opportunity_count",
        "policy_reuse_guard_semantics",
        "policy_hold_schema_version",
        "policy_hold_count",
    }
    missing = sorted(required_base_fields.difference(values))
    if missing:
        return {
            "passed": False,
            "available": False,
            "missing_fields": missing,
            "expected_maximum_policy_actions_per_camera_frame": (
                MAX_POLICY_USES_PER_CAMERA_FRAME
            ),
        }

    maximum = _scalar_integer(values, "maximum_policy_actions_per_camera_frame")
    action_count = _scalar_integer(values, "policy_action_evidence_count")
    guard_hold_count = _scalar_integer(values, "policy_reuse_guard_hold_count")
    opportunity_count = _scalar_integer(
        values, "policy_action_or_reuse_hold_opportunity_count"
    )
    hold_schema = _scalar_integer(values, "policy_hold_schema_version")
    hold_count = _scalar_integer(values, "policy_hold_count")
    semantics_value = _scalar(values, "policy_reuse_guard_semantics")
    if semantics_value.dtype.kind not in "US":
        raise ValueError("policy_reuse_guard_semantics must be a scalar string")
    semantics = str(semantics_value)
    if guard_hold_count < 0 or hold_count < 0 or opportunity_count < 0:
        raise ValueError("policy reuse guard counts must be non-negative")

    frame_ids = np.asarray(values["pointcloud_frame_id"])
    use_indices = np.asarray(values["policy_action_use_index_for_camera_frame"])
    if frame_ids.shape != (steps,) or frame_ids.dtype.kind not in "iu":
        raise ValueError(f"pointcloud_frame_id must be an integer [{steps}] array")
    if use_indices.shape != (steps,) or use_indices.dtype.kind not in "iu":
        raise ValueError(
            "policy_action_use_index_for_camera_frame must be an integer "
            f"[{steps}] array"
        )
    expected_use_indices = np.empty(steps, dtype=np.int64)
    previous_frame: Optional[int] = None
    running_uses = 0
    for index, raw_frame_id in enumerate(frame_ids.astype(np.int64)):
        frame_id = int(raw_frame_id)
        if previous_frame is not None and frame_id < previous_frame:
            raise ValueError("pointcloud_frame_id must be nondecreasing")
        if frame_id != previous_frame:
            running_uses = 0
            previous_frame = frame_id
        running_uses += 1
        expected_use_indices[index] = running_uses
    action_use_indices_match = bool(
        np.array_equal(use_indices.astype(np.int64), expected_use_indices)
    )

    hold_required_fields = {
        "policy_hold_diagnostic_index",
        "policy_hold_reason",
        "policy_hold_camera_frame_id",
        "policy_hold_pointcloud_frame_id",
        "policy_hold_action_uses_before_hold",
        "policy_hold_maximum_policy_actions_per_camera_frame",
        "policy_hold_new_policy_action_evidence",
        "policy_hold_policy_state_mutated",
        "policy_hold_write",
        "policy_hold_waits_for_new_camera_frame",
        "policy_hold_control_tick_started_monotonic_s",
        "policy_hold_control_tick_finished_monotonic_s",
        "policy_hold_control_tick_interval_from_previous_action_s",
    }
    missing_hold_fields = (
        sorted(hold_required_fields.difference(values)) if hold_count > 0 else []
    )
    hold_evidence_passed = not missing_hold_fields
    hold_action_counts_match = hold_count == 0
    hold_flags_passed = hold_count == 0
    hold_timings_passed = hold_count == 0
    hold_frames: list[int] = []
    if hold_count > 0 and not missing_hold_fields:
        def hold_array(name: str, dtype_kinds: str) -> np.ndarray:
            array = np.asarray(values[name])
            if array.shape != (hold_count,) or array.dtype.kind not in dtype_kinds:
                raise ValueError(
                    f"{name} must be a {dtype_kinds!r} [{hold_count}] array"
                )
            return array

        diagnostic_indices = hold_array("policy_hold_diagnostic_index", "iu")
        reasons = hold_array("policy_hold_reason", "US")
        hold_array("policy_hold_camera_frame_id", "iu")
        hold_frame_ids = hold_array("policy_hold_pointcloud_frame_id", "iu")
        uses_before_hold = hold_array(
            "policy_hold_action_uses_before_hold", "iu"
        )
        hold_maximum = hold_array(
            "policy_hold_maximum_policy_actions_per_camera_frame", "iu"
        )
        new_action = hold_array(
            "policy_hold_new_policy_action_evidence", "b"
        )
        state_mutated = hold_array("policy_hold_policy_state_mutated", "b")
        writes = hold_array("policy_hold_write", "b")
        waits = hold_array("policy_hold_waits_for_new_camera_frame", "b")
        started = hold_array(
            "policy_hold_control_tick_started_monotonic_s", "fiu"
        ).astype(np.float64)
        finished = hold_array(
            "policy_hold_control_tick_finished_monotonic_s", "fiu"
        ).astype(np.float64)
        intervals = hold_array(
            "policy_hold_control_tick_interval_from_previous_action_s", "fiu"
        ).astype(np.float64)

        _, action_frame_counts = np.unique(frame_ids.astype(np.int64), return_counts=True)
        action_frame_values = np.unique(frame_ids.astype(np.int64))
        count_by_frame = {
            int(frame): int(count)
            for frame, count in zip(action_frame_values, action_frame_counts)
        }
        hold_frames = hold_frame_ids.astype(np.int64).tolist()
        hold_action_counts_match = all(
            count_by_frame.get(int(frame), 0) == MAX_POLICY_USES_PER_CAMERA_FRAME
            for frame in hold_frames
        )
        hold_flags_passed = bool(
            np.array_equal(
                diagnostic_indices.astype(np.int64),
                np.arange(hold_count, dtype=np.int64),
            )
            and np.all(reasons.astype(str) == POLICY_REUSE_HOLD_REASON)
            and np.all(uses_before_hold == MAX_POLICY_USES_PER_CAMERA_FRAME)
            and np.all(hold_maximum == MAX_POLICY_USES_PER_CAMERA_FRAME)
            and not np.any(new_action)
            and not np.any(state_mutated)
            and not np.any(writes)
            and np.all(waits)
        )
        hold_timings_passed = bool(
            np.all(np.isfinite(started))
            and np.all(np.isfinite(finished))
            and np.all(np.isfinite(intervals))
            and np.all(finished >= started)
            and np.all(intervals >= 0.0)
        )
        hold_evidence_passed = bool(
            hold_action_counts_match and hold_flags_passed and hold_timings_passed
        )

    count_conservation_passed = bool(
        action_count == steps
        and guard_hold_count == hold_count
        and opportunity_count == steps + hold_count
    )
    configuration_passed = bool(
        maximum == MAX_POLICY_USES_PER_CAMERA_FRAME
        and hold_schema == POLICY_REUSE_HOLD_SCHEMA_VERSION
        and semantics == POLICY_REUSE_GUARD_SEMANTICS
    )
    passed = bool(
        configuration_passed
        and count_conservation_passed
        and action_use_indices_match
        and hold_evidence_passed
    )
    return {
        "passed": passed,
        "available": True,
        "maximum_policy_actions_per_camera_frame": maximum,
        "policy_action_evidence_count": action_count,
        "policy_reuse_guard_hold_count": guard_hold_count,
        "policy_action_or_reuse_hold_opportunity_count": opportunity_count,
        "policy_hold_schema_version": hold_schema,
        "policy_hold_count": hold_count,
        "policy_hold_pointcloud_frame_ids": hold_frames,
        "configuration_passed": configuration_passed,
        "count_conservation_passed": count_conservation_passed,
        "action_use_indices_match": action_use_indices_match,
        "hold_action_counts_match": hold_action_counts_match,
        "hold_flags_passed": hold_flags_passed,
        "hold_timings_passed": hold_timings_passed,
        "hold_evidence_passed": hold_evidence_passed,
        "missing_hold_fields": missing_hold_fields,
    }


def assess_v94_shadow_readiness(
    shadow_path: str | Path,
    *,
    bundle_path: str | Path = DEFAULT_BUNDLE,
    replay_atol: float = DEFAULT_REPLAY_ATOL,
    maximum_camera_age_s: float = DEFAULT_MAX_CAMERA_AGE_S,
    minimum_action_rate_hz: float = DEFAULT_MIN_ACTION_RATE_HZ,
) -> dict[str, Any]:
    """Assess one already-recorded, read-only live preview NPZ."""

    maximum_age = float(maximum_camera_age_s)
    minimum_rate = float(minimum_action_rate_hz)
    if not np.isfinite(maximum_age) or maximum_age <= 0.0:
        raise ValueError("maximum_camera_age_s must be finite and positive")
    if not np.isfinite(minimum_rate) or minimum_rate <= 0.0:
        raise ValueError("minimum_action_rate_hz must be finite and positive")

    values = _read_no_pickle_npz(shadow_path)
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "no_object_dtype",
            True,
            inspected_fields=len(values),
            note="NPZ was loaded with allow_pickle=False and every dtype was inspected.",
        )
    )

    hardware_writes = _scalar_bool(values, "hardware_writes")
    robot_command_writes = _scalar_bool(values, "robot_command_writes")
    checks.append(
        _check("hardware_writes_false", not hardware_writes, actual=hardware_writes)
    )
    checks.append(
        _check(
            "robot_command_writes_false",
            not robot_command_writes,
            actual=robot_command_writes,
        )
    )

    configured_age = _scalar_float(values, "configured_max_camera_age_s")
    effective_age_gate = min(maximum_age, configured_age)
    configured_age_passed = bool(0.0 < configured_age <= maximum_age)
    checks.append(
        _check(
            "configured_camera_age_within_gate",
            configured_age_passed,
            configured_max_camera_age_s=configured_age,
            maximum_allowed_camera_age_s=maximum_age,
        )
    )

    action_ages = np.asarray(values.get("pointcloud_age_at_action_s"))
    if action_ages.ndim != 1 or action_ages.dtype.kind not in "fiu":
        raise ValueError("pointcloud_age_at_action_s must be a one-dimensional number array")
    if action_ages.size < 1 or not np.all(np.isfinite(action_ages)):
        raise ValueError("pointcloud_age_at_action_s must be non-empty and finite")
    steps = int(action_ages.size)
    action_age_max = float(np.max(action_ages))
    action_age_passed = bool(
        configured_age_passed
        and np.all(action_ages >= 0.0)
        and np.all(action_ages <= effective_age_gate)
    )
    checks.append(
        _check(
            "every_pointcloud_age_at_action_within_gate",
            action_age_passed,
            steps=steps,
            maximum_observed_age_s=action_age_max,
            effective_age_gate_s=effective_age_gate,
        )
    )

    rebootstrap_count = _scalar_integer(values, "policy_rebootstrap_count")
    checks.append(
        _check(
            "no_policy_rebootstrap",
            rebootstrap_count == 0,
            policy_rebootstrap_count=rebootstrap_count,
        )
    )

    requested_duration = _scalar_float(values, "requested_capture_duration_s")
    if requested_duration <= 0.0:
        raise ValueError("requested_capture_duration_s must be positive")
    effective_action_rate = float(steps / requested_duration)
    checks.append(
        _check(
            "commissioning_effective_action_rate",
            effective_action_rate >= minimum_rate,
            steps=steps,
            requested_capture_duration_s=requested_duration,
            effective_action_rate_hz=effective_action_rate,
            minimum_action_rate_hz=minimum_rate,
            threshold_semantics=(
                "engineering commissioning threshold; not a deployment-bundle "
                "policy semantic"
            ),
        )
    )

    reuse = _camera_frame_reuse(values, steps)
    checks.append(_check("camera_30hz_to_policy_60hz_reuse_contract", **reuse))
    reuse_guard = _policy_reuse_guard_evidence(values, steps)
    checks.append(_check("camera_policy_reuse_guard_evidence", **reuse_guard))

    audit_error: Optional[str] = None
    audit_report: Optional[dict[str, Any]] = None
    try:
        audit_report = audit_v94_preview(
            shadow_path,
            bundle_path=bundle_path,
            replay_atol=replay_atol,
        )
        audit_passed = bool(
            audit_report.get("result") == "PASS"
            and audit_report.get("policy_replay", {}).get("passed") is True
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        audit_passed = False
        audit_error = f"{type(exc).__name__}: {exc}"
    policy_replay = None if audit_report is None else audit_report["policy_replay"]
    checks.append(
        _check(
            "offline_audit_and_policy_replay_pass",
            audit_passed,
            audit_result=None if audit_report is None else audit_report.get("result"),
            policy_replay=policy_replay,
            error=audit_error,
        )
    )

    reset_reference_aligned = bool(
        audit_report is not None
        and audit_report.get("reset_reference_aligned") is True
    )
    robot_reset = (
        {} if audit_report is None else audit_report.get("robot_reset_alignment", {})
    )
    robot_reset_aligned = bool(
        isinstance(robot_reset, Mapping)
        and robot_reset.get("available") is True
        and robot_reset.get("aligned") is True
    )
    checks.append(
        _check(
            "recorded_reset_aligned",
            reset_reference_aligned and robot_reset_aligned,
            object_reset_reference_aligned=reset_reference_aligned,
            robot_reset_alignment_available=bool(robot_reset.get("available", False)),
            franka_rh56_reset_aligned=robot_reset_aligned,
        )
    )

    failures = [item["name"] for item in checks if not item["passed"]]
    return {
        "result": "PASS" if not failures else "FAIL",
        "assessor_schema_version": ASSESSOR_SCHEMA_VERSION,
        "offline_only": True,
        "input_npz": str(Path(shadow_path).expanduser().resolve()),
        "checks": checks,
        "failed_checks": failures,
        "summary": {
            "steps": steps,
            "requested_capture_duration_s": requested_duration,
            "effective_action_rate_hz": effective_action_rate,
            "configured_max_camera_age_s": configured_age,
            "maximum_pointcloud_age_at_action_s": action_age_max,
            "policy_rebootstrap_count": rebootstrap_count,
            "camera_frame_reuse": reuse,
            "camera_policy_reuse_guard": reuse_guard,
        },
        "minimum_action_rate_threshold_semantics": (
            "engineering commissioning threshold; not a deployment-bundle semantic"
        ),
        "semantic_action_correctness_claimed": False,
        "physical_motion_authorized": False,
        "note": (
            "PASS means only that this read-only shadow capture met the strict "
            "offline commissioning gates. It does not authorize hardware control."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict offline readiness gate for a live_v94_preview NPZ."
    )
    parser.add_argument("shadow_npz", type=Path)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--replay-atol", type=float, default=DEFAULT_REPLAY_ATOL)
    parser.add_argument(
        "--maximum-camera-age-s", type=float, default=DEFAULT_MAX_CAMERA_AGE_S
    )
    parser.add_argument(
        "--minimum-action-rate-hz",
        type=float,
        default=DEFAULT_MIN_ACTION_RATE_HZ,
        help=(
            "engineering commissioning threshold only (default: 57 Hz); "
            "not a deployment-bundle semantic"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = assess_v94_shadow_readiness(
            args.shadow_npz,
            bundle_path=args.bundle,
            replay_atol=args.replay_atol,
            maximum_camera_age_s=args.maximum_camera_age_s,
            minimum_action_rate_hz=args.minimum_action_rate_hz,
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if report["result"] == "PASS" else 2
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        report = {
            "result": "FAIL",
            "assessor_schema_version": ASSESSOR_SCHEMA_VERSION,
            "offline_only": True,
            "error": f"{type(exc).__name__}: {exc}",
            "semantic_action_correctness_claimed": False,
            "physical_motion_authorized": False,
        }
        print(json.dumps(report, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
