#!/usr/bin/env python3
"""Fail-closed admission boundary for a future bounded V94 C2 run.

The command-line interface in this module is deliberately offline-only.  It
can assess a requested number of policy steps, but it contains no production
interlock, authorization, Franka backend, or RH56 transport wiring and can
therefore never start motion.

The reusable :class:`BoundedC2Orchestrator` is the narrow boundary a future
reviewed hardware entry point may call.  A runtime factory is invoked only
after all of the following are true:

* the exact C2 report is PASS and all report-bound artifacts are unchanged;
* both the Franka and RH56 sealed preflight objects were derived from it;
* the exact commissioning profile supplies every persistent-session limit;
* a live, run-scoped :class:`MotionAuthorization` is still active; and
* the shared gate has already verified physical interlocks, Franka rest, and
  RH56 disable.

SIGINT requests the same verified stop path as normal completion.  It is not a
safety substitute: admission still requires the independently commissioned
deadman/E-stop, command watchdogs, and dual-device stop evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import signal
import threading
import time
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

if __package__ in (None, ""):
    import sys

    workspace = Path(__file__).resolve().parents[2]
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from sim2real.closed_loop_core import (  # type: ignore[no-redef]
        AUTHORIZATION_SCOPE,
        ClosedLoopSafetyGate,
        MotionAuthorization,
        SafetyState,
    )
    from sim2real.deployment.preflight import (  # type: ignore[no-redef]
        assess_v94_commissioning_preflight,
    )
    from sim2real.deployment.safety import (  # type: ignore[no-redef]
        DEFAULT_DEPLOY_CONFIG,
    )
    from sim2real.deployment.bundle import DeployBundle  # type: ignore[no-redef]
    from robot_control.franka.session import (  # type: ignore[no-redef]
        CommissionedFrankaEnvelope,
        FrankaSessionMode,
        VerifiedFrankaPreflightToken,
        load_commissioned_franka_envelope,
        verified_franka_preflight_token_from_report,
    )
    from robot_control.rh56.actuator import (  # type: ignore[no-redef]
        RH56ActuationPreflight,
    )
    from sim2real.contracts.v94 import V94Contract  # type: ignore[no-redef]
else:
    from sim2real.closed_loop_core import (
        AUTHORIZATION_SCOPE,
        ClosedLoopSafetyGate,
        MotionAuthorization,
        SafetyState,
    )
    from sim2real.deployment.preflight import assess_v94_commissioning_preflight
    from sim2real.deployment.safety import DEFAULT_DEPLOY_CONFIG
    from sim2real.deployment.bundle import DeployBundle
    from robot_control.franka.session import (
        CommissionedFrankaEnvelope,
        FrankaSessionMode,
        VerifiedFrankaPreflightToken,
        load_commissioned_franka_envelope,
        verified_franka_preflight_token_from_report,
    )
    from robot_control.rh56.actuator import RH56ActuationPreflight
    from sim2real.contracts.v94 import V94Contract


CLI_STATIC_BLOCKERS: Tuple[str, ...] = (
    "trusted_run_scoped_authorization_boundary_not_wired",
    "physical_deadman_estop_source_not_wired",
    "franka_persistent_runtime_factory_not_wired",
    "rh56_transactional_transport_not_wired",
    "bounded_v94_policy_runtime_not_wired",
)


class BoundedC2AdmissionError(RuntimeError):
    """A failure before any motion runtime may be constructed."""


class BoundedC2RunError(RuntimeError):
    """A terminal bounded-run or verified-stop failure."""


class BoundedC2State(str, Enum):
    DISARMED = "disarmed"
    RUNNING = "running"
    STOPPED_VERIFIED = "stopped_verified"
    FAULT_LATCHED = "fault_latched"


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 0.0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(numeric)


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sha256_regular_file(path: Path) -> str:
    if not path.is_file():
        raise BoundedC2AdmissionError(f"bound artifact is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_report_sha256(report: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BoundedC2AdmissionError(
            "preflight report is not canonical JSON data"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _report_artifact_bindings(
    report: Mapping[str, Any],
) -> Tuple[Tuple[str, str, str], ...]:
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping):
        raise BoundedC2AdmissionError("preflight report inputs are missing")
    hashes = inputs.get("sha256")
    if not isinstance(hashes, Mapping):
        raise BoundedC2AdmissionError("preflight artifact hashes are missing")
    names = (
        ("shadow_npz", "shadow_npz_sha256"),
        ("bundle", "bundle_sha256"),
        ("deploy_config", "deploy_config_sha256"),
        ("commissioning_profile", "commissioning_profile_sha256"),
    )
    result = []
    for path_name, hash_name in names:
        raw_path = inputs.get(path_name)
        expected = hashes.get(hash_name)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise BoundedC2AdmissionError(
                f"preflight input path {path_name!r} is missing"
            )
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)
        ):
            raise BoundedC2AdmissionError(
                f"preflight hash {hash_name!r} is invalid"
            )
        path = Path(raw_path).expanduser().resolve()
        actual = _sha256_regular_file(path)
        if actual != expected:
            raise BoundedC2AdmissionError(
                f"preflight-bound artifact changed: {path_name}"
            )
        result.append((path_name, str(path), expected))
    return tuple(result)


_ADMISSION_SEAL = object()


@dataclass(frozen=True, init=False)
class BoundedC2Admission:
    """Sealed, expiring admission for one exact bounded execution run."""

    run_id: str
    requested_policy_steps: int
    policy_rate_hz: float
    requested_nominal_duration_s: float
    hard_deadline_monotonic_s: float
    authorization: MotionAuthorization
    franka_preflight_token: VerifiedFrankaPreflightToken
    rh56_preflight: RH56ActuationPreflight
    franka_envelope: CommissionedFrankaEnvelope
    safety_gate: ClosedLoopSafetyGate
    preflight_report_sha256: str
    artifact_bindings: Tuple[Tuple[str, str, str], ...]

    def __init__(self, *, _seal: object, **values: Any) -> None:
        if _seal is not _ADMISSION_SEAL:
            raise TypeError("BoundedC2Admission must come from seal_c2_admission")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def require_active(self, *, now_monotonic_s: object) -> None:
        now = _finite_scalar(now_monotonic_s, "now_monotonic_s")
        if not self.authorization.issued_monotonic_s <= now < self.authorization.expires_monotonic_s:
            raise BoundedC2AdmissionError("run-scoped motion authorization expired")
        if now >= self.hard_deadline_monotonic_s:
            raise BoundedC2AdmissionError("bounded C2 admission deadline expired")
        self.franka_preflight_token.require_active(
            run_id=self.run_id,
            stage=FrankaSessionMode.C2_V94_POLICY,
            envelope=self.franka_envelope,
            now_monotonic_s=now,
        )
        if self.safety_gate.state is not SafetyState.ARMED:
            raise BoundedC2AdmissionError("closed-loop safety gate is not armed")
        if self.safety_gate.authorization_id != self.authorization.authorization_id:
            raise BoundedC2AdmissionError("safety-gate authorization changed")
        self.safety_gate.require_motion(
            run_id=self.run_id,
            now_monotonic_s=now,
        )
        for path_name, raw_path, expected in self.artifact_bindings:
            if _sha256_regular_file(Path(raw_path)) != expected:
                self.safety_gate.latch_fault(
                    f"preflight-bound artifact changed before runtime: {path_name}"
                )
                raise BoundedC2AdmissionError(
                    f"preflight-bound artifact changed before runtime: {path_name}"
                )


def seal_c2_admission(
    report: Mapping[str, Any],
    *,
    run_id: str,
    requested_policy_steps: object,
    authorization: MotionAuthorization,
    safety_gate: ClosedLoopSafetyGate,
    now_monotonic_s: object,
) -> BoundedC2Admission:
    """Validate and seal every no-device C2 boundary for one run.

    This function never constructs a hardware backend.  The gate must already
    have been armed by a trusted live boundary after checking external
    interlocks, Franka rest, and RH56 disable.
    """

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    if not isinstance(authorization, MotionAuthorization):
        raise TypeError("authorization must be MotionAuthorization")
    if not isinstance(safety_gate, ClosedLoopSafetyGate):
        raise TypeError("safety_gate must be ClosedLoopSafetyGate")
    active_run_id = str(run_id).strip()
    if not active_run_id:
        raise ValueError("run_id must be non-empty")
    if authorization.run_id != active_run_id:
        raise BoundedC2AdmissionError("motion authorization run ID mismatch")
    if authorization.scope != AUTHORIZATION_SCOPE:
        raise BoundedC2AdmissionError("motion authorization scope mismatch")
    steps = _positive_integer(requested_policy_steps, "requested_policy_steps")
    now = _finite_scalar(now_monotonic_s, "now_monotonic_s")

    # Both constructors reject any non-PASS, incomplete, unbound report.
    rh56_preflight = RH56ActuationPreflight.from_report(report)
    bindings = _report_artifact_bindings(report)
    bound_paths = {name: Path(path) for name, path, _digest in bindings}
    envelope = load_commissioned_franka_envelope(
        bound_paths["commissioning_profile"]
    )
    if rh56_preflight.commissioning_profile_sha256 != envelope.profile_sha256:
        raise BoundedC2AdmissionError(
            "Franka and RH56 preflight profile bindings differ"
        )
    bundle = DeployBundle(bound_paths["bundle"])
    contract = V94Contract.from_bundle(bundle)
    policy_rate_hz = _finite_scalar(contract.policy_rate_hz, "policy_rate_hz")
    if policy_rate_hz <= 0.0:
        raise BoundedC2AdmissionError("checkpoint policy rate is invalid")
    nominal_duration = steps / policy_rate_hz
    if nominal_duration > envelope.maximum_session_duration_s + 1.0e-12:
        raise BoundedC2AdmissionError(
            "requested policy steps exceed the commissioned session duration"
        )
    if not authorization.issued_monotonic_s <= now < authorization.expires_monotonic_s:
        raise BoundedC2AdmissionError("run-scoped motion authorization is not active")
    if now + nominal_duration > authorization.expires_monotonic_s + 1.0e-12:
        raise BoundedC2AdmissionError(
            "motion authorization does not cover the requested nominal k-step duration"
        )
    if safety_gate.state is not SafetyState.ARMED:
        raise BoundedC2AdmissionError("closed-loop safety gate is not armed")
    if safety_gate.authorization_id != authorization.authorization_id:
        raise BoundedC2AdmissionError("safety-gate authorization identity mismatch")
    safety_gate.require_motion(run_id=active_run_id, now_monotonic_s=now)

    token_expiry = min(
        authorization.expires_monotonic_s,
        now + envelope.maximum_session_duration_s,
    )
    franka_token = verified_franka_preflight_token_from_report(
        report,
        envelope=envelope,
        run_id=active_run_id,
        stage=FrankaSessionMode.C2_V94_POLICY,
        issued_monotonic_s=now,
        expires_monotonic_s=token_expiry,
    )
    admission = BoundedC2Admission(
        _seal=_ADMISSION_SEAL,
        run_id=active_run_id,
        requested_policy_steps=steps,
        policy_rate_hz=policy_rate_hz,
        requested_nominal_duration_s=nominal_duration,
        hard_deadline_monotonic_s=token_expiry,
        authorization=authorization,
        franka_preflight_token=franka_token,
        rh56_preflight=rh56_preflight,
        franka_envelope=envelope,
        safety_gate=safety_gate,
        preflight_report_sha256=_canonical_report_sha256(report),
        artifact_bindings=bindings,
    )
    admission.require_active(now_monotonic_s=now)
    return admission


@dataclass(frozen=True)
class BoundedC2StopProof:
    franka_stop_verified: bool
    rh56_disabled_verified: bool

    def __post_init__(self) -> None:
        if not isinstance(self.franka_stop_verified, (bool, np.bool_)):
            raise TypeError("franka_stop_verified must be boolean")
        if not isinstance(self.rh56_disabled_verified, (bool, np.bool_)):
            raise TypeError("rh56_disabled_verified must be boolean")


@runtime_checkable
class BoundedC2Runtime(Protocol):
    """Injected combined runtime; no production implementation lives here."""

    physical_interlocks_configured: bool
    independent_command_watchdogs_configured: bool
    verified_dual_device_stop_supported: bool

    def run(
        self,
        *,
        maximum_policy_steps: int,
        hard_deadline_monotonic_s: float,
        stop_requested: threading.Event,
    ) -> Mapping[str, Any]: ...

    def request_stop(self, reason: str) -> None: ...

    def stop_and_verify(self) -> BoundedC2StopProof: ...


class BoundedC2Orchestrator:
    """Single-use wrapper around one post-admission injected runtime factory."""

    def __init__(
        self,
        admission: BoundedC2Admission,
        *,
        runtime_factory: Callable[[BoundedC2Admission], BoundedC2Runtime],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(admission, BoundedC2Admission):
            raise TypeError("admission must be a sealed BoundedC2Admission")
        if not callable(runtime_factory):
            raise TypeError("runtime_factory must be callable")
        if not callable(monotonic):
            raise TypeError("monotonic must be callable")
        self.admission = admission
        self._runtime_factory = runtime_factory
        self._monotonic = monotonic
        self._state = BoundedC2State.DISARMED
        self._runtime: Optional[BoundedC2Runtime] = None
        self._stop_requested = threading.Event()
        self._used = False

    @property
    def state(self) -> BoundedC2State:
        return self._state

    def request_stop(self, reason: str = "external stop request") -> None:
        self._stop_requested.set()
        runtime = self._runtime
        if runtime is not None:
            runtime.request_stop(str(reason).strip() or "external stop request")

    @staticmethod
    def _require_runtime_safety_contract(runtime: BoundedC2Runtime) -> None:
        for name in (
            "physical_interlocks_configured",
            "independent_command_watchdogs_configured",
            "verified_dual_device_stop_supported",
        ):
            if getattr(runtime, name, None) is not True:
                raise BoundedC2RunError(
                    f"runtime safety contract is not active: {name}"
                )

    def run(self) -> Mapping[str, Any]:
        if self._used:
            raise BoundedC2RunError("bounded C2 orchestrator is single-use")
        self._used = True
        now = _finite_scalar(self._monotonic(), "monotonic clock")
        # This is the final gate before the factory may create/open a backend.
        self.admission.require_active(now_monotonic_s=now)

        runtime: Optional[BoundedC2Runtime] = None
        old_sigint: Any = None
        signal_installed = False
        failure: Optional[BaseException] = None
        result: Optional[Mapping[str, Any]] = None
        stop_proof: Optional[BoundedC2StopProof] = None
        try:
            runtime = self._runtime_factory(self.admission)
            if not isinstance(runtime, BoundedC2Runtime):
                raise BoundedC2RunError(
                    "runtime factory did not return BoundedC2Runtime"
                )
            self._runtime = runtime
            self._require_runtime_safety_contract(runtime)
            self.admission.require_active(
                now_monotonic_s=_finite_scalar(
                    self._monotonic(), "monotonic clock"
                )
            )

            if threading.current_thread() is threading.main_thread():
                old_sigint = signal.getsignal(signal.SIGINT)

                def on_sigint(_signum: int, _frame: Any) -> None:
                    self.request_stop("SIGINT")

                signal.signal(signal.SIGINT, on_sigint)
                signal_installed = True

            self._state = BoundedC2State.RUNNING
            value = runtime.run(
                maximum_policy_steps=self.admission.requested_policy_steps,
                hard_deadline_monotonic_s=self.admission.hard_deadline_monotonic_s,
                stop_requested=self._stop_requested,
            )
            if not isinstance(value, Mapping):
                raise BoundedC2RunError("runtime result must be a mapping")
            completed = _nonnegative_integer(
                value.get("completed_policy_steps"), "completed_policy_steps"
            )
            if completed > self.admission.requested_policy_steps:
                raise BoundedC2RunError(
                    "runtime exceeded the admitted policy-step bound"
                )
            result = dict(value)
        except BaseException as exc:
            failure = exc
            self.admission.safety_gate.latch_fault(
                f"bounded C2 runtime failed: {type(exc).__name__}: {exc}"
            )
        finally:
            if signal_installed:
                signal.signal(signal.SIGINT, old_sigint)
            self._stop_requested.set()
            if runtime is not None:
                try:
                    runtime.request_stop(
                        "normal k-step completion"
                        if failure is None
                        else "terminal runtime failure"
                    )
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                        self.admission.safety_gate.latch_fault(
                            f"runtime stop request failed: {exc}"
                        )
                try:
                    stop_proof = runtime.stop_and_verify()
                    if not isinstance(stop_proof, BoundedC2StopProof):
                        raise BoundedC2RunError(
                            "runtime returned no typed dual-device stop proof"
                        )
                    if not (
                        stop_proof.franka_stop_verified
                        and stop_proof.rh56_disabled_verified
                    ):
                        raise BoundedC2RunError(
                            "dual-device stop was not verified"
                        )
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                    self.admission.safety_gate.latch_fault(
                        f"bounded C2 stop verification failed: {exc}"
                    )

        if failure is None:
            assert stop_proof is not None
            try:
                self.admission.safety_gate.disarm_after_verified_stop(
                    run_id=self.admission.run_id,
                    franka_stop_verified=stop_proof.franka_stop_verified,
                    rh56_disabled_verified=stop_proof.rh56_disabled_verified,
                )
            except BaseException as exc:
                failure = exc
                self.admission.safety_gate.latch_fault(
                    f"post-stop disarm failed: {exc}"
                )
        if failure is not None:
            self._state = BoundedC2State.FAULT_LATCHED
            if isinstance(failure, (KeyboardInterrupt, SystemExit)):
                raise failure
            raise BoundedC2RunError(f"bounded C2 run failed: {failure}") from failure
        self._state = BoundedC2State.STOPPED_VERIFIED
        assert result is not None
        return result


def build_disarmed_cli_report(
    *,
    requested_policy_steps: object,
    preflight_report: Mapping[str, Any],
    envelope_error: Optional[str],
) -> dict[str, Any]:
    """Build the offline CLI result; this function has no admission side effects."""

    steps = _positive_integer(requested_policy_steps, "requested_policy_steps")
    failed = preflight_report.get("failed_checks")
    failed_codes = (
        [str(item) for item in failed]
        if isinstance(failed, Sequence) and not isinstance(failed, (str, bytes))
        else ["preflight_failed_checks_invalid"]
    )
    blockers = [f"preflight:{code}" for code in failed_codes]
    if preflight_report.get("result") != "PASS" and not blockers:
        blockers.append("preflight:result_not_pass")
    if envelope_error:
        blockers.append(f"commissioned_franka_envelope_unloadable:{envelope_error}")
    blockers.extend(CLI_STATIC_BLOCKERS)
    return {
        "result": "BLOCKED",
        "arming_state": SafetyState.DISARMED.value.upper(),
        "requested_policy_steps": steps,
        "preflight_result": preflight_report.get("result"),
        "shadow_result": preflight_report.get("shadow_result"),
        "eligible_for_operator_authorization": False,
        "motion_command_available": False,
        "device_access": False,
        "hardware_writes": False,
        "robot_command_writes": False,
        "backend_factory_called": False,
        "sigint_is_only_safety_mechanism": False,
        "blockers": blockers,
        "note": (
            "Offline admission only. No production runtime or transport is wired; "
            "the CLI cannot create MotionAuthorization or start hardware."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-only admission audit for a requested bounded V94 C2 run; "
            "contains no hardware execution option."
        )
    )
    parser.add_argument("shadow_npz", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_DEPLOY_CONFIG)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        preflight = assess_v94_commissioning_preflight(
            args.shadow_npz,
            config_path=args.config,
            evidence_path=args.evidence,
        )
        envelope_error: Optional[str] = None
        try:
            inputs = preflight.get("inputs")
            if not isinstance(inputs, Mapping):
                raise ValueError("preflight inputs are unavailable")
            profile = inputs.get("commissioning_profile")
            if not isinstance(profile, str) or not profile:
                raise ValueError("commissioning profile path is unavailable")
            load_commissioned_franka_envelope(profile)
        except (OSError, TypeError, ValueError) as exc:
            envelope_error = f"{type(exc).__name__}: {exc}"
        report = build_disarmed_cli_report(
            requested_policy_steps=args.steps,
            preflight_report=preflight,
            envelope_error=envelope_error,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        report = build_disarmed_cli_report(
            requested_policy_steps=args.steps,
            preflight_report={
                "result": "FAIL",
                "failed_checks": ["preflight_input_or_verification_error"],
                "shadow_result": None,
            },
            envelope_error=f"{type(exc).__name__}: {exc}",
        )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "V94 bounded k-step admission: "
            f"{report['result']} / {report['arming_state']}"
        )
        print(f"requested_policy_steps={report['requested_policy_steps']}")
        for blocker in report["blockers"]:
            print(f"  - {blocker}")
        print("hardware untouched; no motion command exists in this CLI")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BoundedC2Admission",
    "BoundedC2AdmissionError",
    "BoundedC2Orchestrator",
    "BoundedC2RunError",
    "BoundedC2Runtime",
    "BoundedC2State",
    "BoundedC2StopProof",
    "CLI_STATIC_BLOCKERS",
    "build_disarmed_cli_report",
    "seal_c2_admission",
]
