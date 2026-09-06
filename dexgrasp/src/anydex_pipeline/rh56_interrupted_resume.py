"""Resume an interrupted RH56 open recovery from its sealed failure record.

The first recovery may stop between command waypoints because RH56 bend-axis
hysteresis can leave ``ANGLE_ACT`` beyond a command-centred arrival band.  A
resume therefore anchors each new interval at the freshly verified *actual*
endpoint and derives the next ascending target from that endpoint.  This keeps
the motion monotonic while making the endpoint acceptance and feedback envelope
use the same overshoot allowance.

Importing this module performs no device discovery and opens no hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from .inspire_sequence_driver import (
    DISABLED_TARGETS,
    RH56MotionStopped,
    RH56SequenceDriverError,
)
from .rh56_commissioning import json_sha256, load_evidence, sha256_file
from .rh56_interrupted_recovery import (
    InterruptedRecoveryDriver,
    InterruptedRecoveryPlan,
    RECOVERY_EVIDENCE_KIND,
    build_interrupted_recovery_plan,
)


RESUME_EVIDENCE_KIND = "installed_rh56_interrupted_open_resume_v1"
BEND_REVERSE_OVERSHOOT_TOLERANCE_UNITS = 50
BEND_REVERSE_BOOTSTRAP_UNITS = 50


def _six_json(value: Any, name: str, *, allow_disabled: bool = False) -> Tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 6
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must be a six-integer JSON array")
    lower = -1 if allow_disabled else 0
    result = tuple(int(item) for item in value)
    if any(item < lower or item > 1000 for item in result):
        raise ValueError(f"{name} is outside {lower}..1000")
    return result


def _validate_source_bindings(value: Any) -> list[str]:
    required = {
        "recovery_cli",
        "recovery_module",
        "rh56_sequence_driver",
        "rh56_hand_path",
        "rh56_register_api",
        "franka_sequence_driver",
        "adapter_mesh",
        "adapter_provenance",
    }
    if not isinstance(value, list):
        return ["recovery source_bindings must be an array"]
    blockers = []
    names = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            blockers.append(f"source_bindings[{index}] is not an object")
            continue
        name = str(item.get("name", ""))
        names.add(name)
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            blockers.append(f"source_bindings[{index}] has malformed SHA-256")
        elif not path.is_file():
            blockers.append(f"bound recovery source is missing: {path}")
        elif sha256_file(path) != digest:
            blockers.append(f"bound recovery source changed: {path}")
    missing = sorted(required - names)
    if missing:
        blockers.append("required recovery sources are missing: " + ", ".join(missing))
    return blockers


def _telemetry_phase_index(phase: str, prefix: str) -> int:
    if not str(phase).startswith(prefix):
        raise ValueError(f"telemetry phase {phase!r} does not start with {prefix!r}")
    suffix = str(phase)[len(prefix):]
    if len(suffix) != 4 or not suffix.isdigit():
        raise ValueError(f"telemetry phase {phase!r} has no four-digit index")
    return int(suffix)


@dataclass(frozen=True)
class InterruptedResumePlan:
    recovery_evidence_path: Path
    recovery_file_sha256: str
    recovery_payload_sha256: str
    recovery_run_id: str
    upstream: InterruptedRecoveryPlan
    last_completed_bend_index: int
    last_attempted_bend_index: int
    last_completed_command: Tuple[int, ...]
    last_attempted_command: Tuple[int, ...]
    permitted_live_pinky_range: Tuple[int, int]

    @property
    def target_q6(self) -> int:
        return int(self.upstream.target_q6)

    @property
    def step_units(self) -> int:
        return int(self.upstream.step_units)

    @property
    def q6_return_waypoints(self) -> Tuple[int, ...]:
        return self.upstream.q6_return_waypoints

    @property
    def original_speeds(self) -> Tuple[int, ...]:
        return self.upstream.original_speeds

    @property
    def original_forces(self) -> Tuple[int, ...]:
        return self.upstream.original_forces

    def as_binding(self) -> dict[str, Any]:
        return {
            "kind": "interrupted_installed_rh56_open_recovery_failure_v1",
            "path": str(self.recovery_evidence_path),
            "file_sha256": self.recovery_file_sha256,
            "payload_sha256": self.recovery_payload_sha256,
            "run_id": self.recovery_run_id,
            "upstream_failed_commissioning": self.upstream.as_binding(),
            "last_completed_bend_index": self.last_completed_bend_index,
            "last_attempted_bend_index": self.last_attempted_bend_index,
            "last_completed_command": list(self.last_completed_command),
            "last_attempted_command": list(self.last_attempted_command),
            "permitted_live_pinky_range": list(self.permitted_live_pinky_range),
            "target_q6": self.target_q6,
            "step_units": self.step_units,
            "q6_return_waypoints": list(self.q6_return_waypoints),
            "original_speeds": list(self.original_speeds),
            "original_forces": list(self.original_forces),
        }


def build_interrupted_resume_plan(
    evidence_path: Path,
    *,
    expected_config: Mapping[str, Any],
    expected_config_path: Path,
) -> InterruptedResumePlan:
    evidence, resolved = load_evidence(evidence_path)
    blockers = []
    if evidence.get("schema_version") != 1:
        blockers.append("recovery schema_version must be 1")
    if evidence.get("kind") != RECOVERY_EVIDENCE_KIND:
        blockers.append(f"recovery kind must be {RECOVERY_EVIDENCE_KIND}")
    integrity = evidence.get("integrity")
    if not isinstance(integrity, Mapping):
        blockers.append("recovery integrity object is missing")
    else:
        unsigned = {key: value for key, value in evidence.items() if key != "integrity"}
        if integrity.get("payload_sha256") != json_sha256(unsigned):
            blockers.append("recovery canonical payload SHA-256 mismatch")
    blockers.extend(_validate_source_bindings(evidence.get("source_bindings")))
    if evidence.get("motion_authorized") is not False:
        blockers.append("recovery motion_authorized must remain false")
    if evidence.get("commissioning_unlock_claimed") is not False:
        blockers.append("recovery must not claim a commissioning unlock")

    profile = evidence.get("control_profile")
    expected_path = Path(expected_config_path).expanduser().resolve()
    if not isinstance(profile, Mapping):
        blockers.append("recovery control_profile binding is missing")
    else:
        snapshot = profile.get("snapshot")
        if not isinstance(snapshot, Mapping) or dict(snapshot) != dict(expected_config):
            blockers.append("recovery belongs to a different control profile")
        elif profile.get("parsed_sha256") != json_sha256(snapshot):
            blockers.append("recovery profile snapshot hash mismatch")
        if Path(str(profile.get("path", ""))).expanduser().resolve() != expected_path:
            blockers.append("recovery belongs to a different profile path")
        if profile.get("file_sha256") != sha256_file(expected_path):
            blockers.append("control profile changed after the failed recovery")

    upstream_binding = evidence.get("failed_commissioning_binding")
    if not isinstance(upstream_binding, Mapping):
        blockers.append("recovery lacks its upstream failed-commissioning binding")
        upstream = None
    else:
        try:
            upstream = build_interrupted_recovery_plan(
                Path(str(upstream_binding.get("path", ""))),
                expected_config=expected_config,
                expected_config_path=expected_path,
            )
        except ValueError as exc:
            blockers.append(str(exc))
            upstream = None
        else:
            if dict(upstream_binding) != upstream.as_binding():
                blockers.append("upstream failed-commissioning binding changed")

    result = evidence.get("result")
    final = evidence.get("final")
    request = evidence.get("request")
    telemetry = evidence.get("telemetry")
    if not isinstance(result, Mapping):
        blockers.append("recovery result object is missing")
    else:
        error = result.get("operation_error")
        if result.get("status") != "fail":
            blockers.append("resume requires a failed recovery record")
        if result.get("adopted_disabled_verified") is not True:
            blockers.append("failed recovery did not verify initial disable")
        if result.get("recovered_open_verified") is not False:
            blockers.append("failed recovery already claims open success")
        if not isinstance(error, str) or "coupled_air_return_step_" not in error or "ANGLE_ACT escaped" not in error:
            blockers.append("failed recovery is not the reviewed bend-envelope stop")
    if not isinstance(final, Mapping):
        blockers.append("failed recovery final object is missing")
    else:
        if final.get("angle_targets") != list(DISABLED_TARGETS):
            blockers.append("failed recovery did not read back all-six disabled targets")
        if final.get("errors") != [0] * 6:
            blockers.append("failed recovery ended with a device error")
    if not isinstance(request, Mapping):
        blockers.append("failed recovery request object is missing")
    if not isinstance(telemetry, list) or not telemetry:
        blockers.append("failed recovery telemetry is missing")
    if upstream is not None and isinstance(request, Mapping):
        if request.get("interrupted_targets") != list(upstream.interrupted_targets):
            blockers.append("failed recovery start differs from its upstream evidence")
        if request.get("bend_return_waypoints") != [list(item) for item in upstream.bend_return_waypoints]:
            blockers.append("failed recovery bend path differs from its upstream evidence")
        if request.get("q6_return_waypoints") != list(upstream.q6_return_waypoints):
            blockers.append("failed recovery q6 path differs from its upstream evidence")

    if blockers:
        raise ValueError("interrupted resume is LOCKED: " + "; ".join(dict.fromkeys(blockers)))
    assert upstream is not None
    assert isinstance(telemetry, list)
    phase_prefix = "coupled_air_return_step_"
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for sample in telemetry:
        if not isinstance(sample, Mapping):
            raise ValueError("interrupted resume is LOCKED: telemetry sample is not an object")
        phase = str(sample.get("phase", ""))
        q6_suffix = phase[len("q6_return_"):] if phase.startswith("q6_return_") else ""
        if len(q6_suffix) == 4 and q6_suffix.isdigit():
            raise ValueError("interrupted resume is LOCKED: failed recovery already started q6 return")
        if not phase.startswith(phase_prefix):
            continue
        index = _telemetry_phase_index(phase, phase_prefix)
        if not 0 <= index < len(upstream.bend_return_waypoints):
            raise ValueError("interrupted resume is LOCKED: bend phase index is outside the bound path")
        expected = upstream.bend_return_waypoints[index]
        if _six_json(sample.get("angle_targets"), "telemetry.angle_targets", allow_disabled=True) != expected:
            raise ValueError("interrupted resume is LOCKED: bend telemetry target changed")
        if _six_json(sample.get("errors"), "telemetry.errors") != (0,) * 6:
            raise ValueError("interrupted resume is LOCKED: bend telemetry contains a device error")
        grouped.setdefault(index, []).append(sample)
    if not grouped:
        raise ValueError("interrupted resume is LOCKED: no bend recovery waypoint was attempted")
    indices = sorted(grouped)
    if indices != list(range(indices[-1] + 1)):
        raise ValueError("interrupted resume is LOCKED: bend telemetry is not a contiguous prefix")
    last_attempted = indices[-1]
    tolerance = int(request["step_units"])
    stable_required = int(request["endpoint_stable_samples"])
    completed = []
    for index in indices:
        expected = upstream.bend_return_waypoints[index]
        samples = grouped[index]
        tail = samples[-stable_required:]
        if len(tail) == stable_required and all(
            all(int(status) == 2 for status in sample.get("statuses", ()))
            and abs(int(sample["angles"][0]) - expected[0]) <= tolerance
            for sample in tail
        ):
            completed.append(index)
    if not completed or completed != list(range(completed[-1] + 1)):
        raise ValueError("interrupted resume is LOCKED: completed bend waypoints are not a prefix")
    last_completed = completed[-1]
    if last_completed >= last_attempted:
        raise ValueError("interrupted resume is LOCKED: no incomplete bend waypoint remains")

    live_candidates = [
        int(sample["angles"][0])
        for sample in grouped[last_attempted]
        if isinstance(sample.get("angles"), list) and len(sample["angles"]) == 6
    ]
    live_candidates.extend(
        int(sample["angles"][0])
        for sample in grouped[last_completed][-stable_required:]
    )
    live_candidates.append(int(final["angles"][0]))
    live_range = (min(live_candidates), max(live_candidates))
    if not 0 <= live_range[0] <= live_range[1] < 1000:
        raise ValueError("interrupted resume is LOCKED: derived live pinky range is invalid")
    return InterruptedResumePlan(
        recovery_evidence_path=resolved,
        recovery_file_sha256=sha256_file(resolved),
        recovery_payload_sha256=str(integrity["payload_sha256"]),
        recovery_run_id=str(evidence.get("run_id", "")),
        upstream=upstream,
        last_completed_bend_index=last_completed,
        last_attempted_bend_index=last_attempted,
        last_completed_command=upstream.bend_return_waypoints[last_completed],
        last_attempted_command=upstream.bend_return_waypoints[last_attempted],
        permitted_live_pinky_range=live_range,
    )


class InterruptedResumeDriver(InterruptedRecoveryDriver):
    """Continue bend opening from fresh actual feedback, then return q6."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_resume_bend_waypoints: Tuple[Tuple[int, ...], ...] = ()
        self.last_resume_bend_actual_endpoints: Tuple[int, ...] = ()

    def resume_interrupted_open_to_completion(
        self,
        plan: InterruptedResumePlan,
        *,
        max_axis_current_ma: int = 400,
        endpoint_stable_samples: int = 3,
        max_inactive_drift_units: int = 8,
    ) -> Tuple[int, ...]:
        if not isinstance(plan, InterruptedResumePlan):
            raise TypeError("plan must be an InterruptedResumePlan")
        if plan.step_units != self.thumb_preshape_step_units:
            raise ValueError("resume plan step size differs from the driver")
        if not 2 <= int(endpoint_stable_samples) <= 10:
            raise ValueError("endpoint_stable_samples must be in 2..10")
        if not 1 <= int(max_inactive_drift_units) <= 20:
            raise ValueError("max_inactive_drift_units must be in 1..20")

        with self._operation_lock:
            try:
                self._ensure_motion_allowed()
                if not self._disabled_verified or self._numeric_hold_targets is not None:
                    raise RH56SequenceDriverError(
                        "resume requires a fresh adopt_disabled_state_and_verify"
                    )
                started = self._monotonic()
                preflight = self._read_feedback("interrupted_resume_preflight", started)
                self._check_fault_feedback(preflight, "interrupted resume preflight")
                if preflight.angle_targets != DISABLED_TARGETS:
                    raise RH56SequenceDriverError("resume preflight requires all-six ANGLE_SET=-1")
                if not all(status in (2, 0xFF) for status in preflight.statuses):
                    raise RH56SequenceDriverError("resume preflight requires every actuator idle")
                if any(abs(int(current)) > self.stop_max_axis_current_ma for current in preflight.currents):
                    raise RH56SequenceDriverError("resume preflight current exceeds the idle bound")
                if max(preflight.temperatures) >= 50:
                    raise RH56SequenceDriverError("resume preflight requires every actuator below 50C")
                low, high = plan.permitted_live_pinky_range
                pinky = int(preflight.angles[0])
                if not low <= pinky <= high:
                    raise RH56SequenceDriverError(
                        f"live pinky ANGLE_ACT={pinky} is outside evidence-bound range {low}..{high}"
                    )
                if any(int(value) < self.open_min_angle for value in preflight.angles[1:5]):
                    raise RH56SequenceDriverError("resume requires the other four bend axes open")
                if abs(int(preflight.angles[5]) - plan.target_q6) > min(self.angle_tolerance, 20):
                    raise RH56SequenceDriverError("resume q6 differs from the evidence-bound endpoint")

                caps = self._commissioning_current_caps(max_axis_current_ma)
                self._check_commissioning_currents(preflight, caps, "interrupted resume preflight")
                self._original_speeds = plan.original_speeds
                self._original_forces = plan.original_forces
                self._configure_commissioning_settings()
                inactive_reference = tuple(int(value) for value in preflight.angles[1:5])
                previous_actual = pinky
                target = min(1000, previous_actual + BEND_REVERSE_BOOTSTRAP_UNITS)
                dynamic_waypoints = []
                actual_endpoints = []
                while True:
                    waypoint = (target, 1000, 1000, 1000, 1000, plan.target_q6)
                    dynamic_waypoints.append(waypoint)
                    self.last_resume_bend_waypoints = tuple(dynamic_waypoints)
                    phase = f"interrupted_resume_bend_{len(dynamic_waypoints) - 1:04d}"
                    self._write_six_verified(
                        self._constant("REG_ANGLE_SET"), waypoint, numeric_motion=True
                    )
                    deadline = self._monotonic() + self.motion_timeout_s
                    stable = 0
                    accepted_actual = None
                    lower = max(0, previous_actual - max(4, self.angle_tolerance // 2))
                    upper = min(1000, target + BEND_REVERSE_OVERSHOOT_TOLERANCE_UNITS)
                    while True:
                        if self._stop_requested.is_set():
                            raise RH56MotionStopped("interrupted bend resume stopped by request")
                        feedback = self._read_feedback(phase, started)
                        self._check_fault_feedback(feedback, phase)
                        self._check_commissioning_currents(feedback, caps, phase)
                        if feedback.angle_targets != waypoint:
                            raise RH56SequenceDriverError(f"{phase}: ANGLE_SET changed")
                        if any(status == 3 for status in feedback.statuses):
                            raise RH56SequenceDriverError(f"{phase}: contact during free-air resume")
                        if any(status not in (0, 1, 2) for status in feedback.statuses):
                            raise RH56SequenceDriverError(f"{phase}: unsupported actuator status")
                        actual = int(feedback.angles[0])
                        if not lower <= actual <= upper:
                            raise RH56SequenceDriverError(
                                f"{phase}: pinky ANGLE_ACT={actual} escaped actual-anchored interval {lower}..{upper}"
                            )
                        for offset, (reference, actual_inactive) in enumerate(
                            zip(inactive_reference, feedback.angles[1:5]), start=1
                        ):
                            if (
                                int(actual_inactive) < self.open_min_angle
                                or abs(int(actual_inactive) - reference) > int(max_inactive_drift_units)
                            ):
                                raise RH56SequenceDriverError(
                                    f"{phase}: inactive bend axis {offset} moved"
                                )
                        if abs(int(feedback.angles[5]) - plan.target_q6) > min(self.angle_tolerance, 20):
                            raise RH56SequenceDriverError(f"{phase}: q6 drifted during bend resume")
                        endpoint_ok = (
                            actual >= self.open_min_angle
                            if target == 1000
                            else target - self.angle_tolerance <= actual <= upper
                        )
                        progress_ok = target == 1000 or actual - previous_actual >= 3
                        current_ok = all(
                            abs(int(current)) <= self.stop_max_axis_current_ma
                            for current in feedback.currents
                        )
                        reached = (
                            endpoint_ok
                            and progress_ok
                            and current_ok
                            and all(status == 2 for status in feedback.statuses)
                        )
                        stable = stable + 1 if reached else 0
                        if stable >= int(endpoint_stable_samples):
                            accepted_actual = actual
                            break
                        if self._monotonic() >= deadline:
                            raise RH56SequenceDriverError(
                                f"{phase}: timed out at ANGLE_ACT={actual}, target={target}"
                            )
                        self._sleep(self.poll_interval_s)
                    assert accepted_actual is not None
                    actual_endpoints.append(accepted_actual)
                    self.last_resume_bend_actual_endpoints = tuple(actual_endpoints)
                    previous_actual = accepted_actual
                    if target == 1000:
                        break
                    target = min(1000, previous_actual + plan.step_units)
                    if target <= previous_actual:
                        raise RH56SequenceDriverError("resume could not derive an ascending target")

                self.last_resume_bend_waypoints = tuple(dynamic_waypoints)
                self.last_resume_bend_actual_endpoints = tuple(actual_endpoints)
                # The bend is now open.  Reuse the reviewed q6-only reverse
                # state machine; it starts by disabling all six outputs and
                # never sends a direct q6=1000 jump.
                self._preshaped_q6 = plan.target_q6
                self._commissioned_q6_waypoints = plan.upstream.q6_forward_waypoints
                self._commissioned_bend_waypoints = None
                self._numeric_hold_targets = (
                    1000, 1000, 1000, 1000, 1000, plan.target_q6
                )
                self._disabled_verified = False
            except BaseException as exc:
                self._fail_and_latch(exc)
                raise

        returned = self.return_commissioned_thumb_to_open(
            max_axis_current_ma=max_axis_current_ma,
            endpoint_stable_samples=endpoint_stable_samples,
            max_inactive_drift_units=max_inactive_drift_units,
            direct_q6_return_to_open=False,
        )
        if tuple(returned) != plan.q6_return_waypoints:
            error = RH56SequenceDriverError("runtime q6 path differs from the bound resume plan")
            self._fail_and_latch(error)
            raise error
        return tuple(returned)


__all__ = [
    "BEND_REVERSE_BOOTSTRAP_UNITS",
    "BEND_REVERSE_OVERSHOOT_TOLERANCE_UNITS",
    "InterruptedResumeDriver",
    "InterruptedResumePlan",
    "RESUME_EVIDENCE_KIND",
    "build_interrupted_resume_plan",
]
