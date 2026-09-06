"""Reusable, fail-closed wiring for a bounded V94 C2 hardware runtime.

This module intentionally provides no executable entry point.  It joins the
existing persistent Franka owner, transactional RH56 actuator, exact dual-ACK
ledger, and :class:`~sim2real.runtime.bounded_c2_orchestrator.BoundedC2Orchestrator`
runtime protocol behind dependency-injected factories.

The RH56 transport constructor must be inert.  The returned transport is
opened only inside :meth:`BoundedV94C2Runtime.run`, after the sealed admission,
live physical-safety supervisor, and run-scoped authorization are rechecked.
The provided :class:`LinuxRH56TransportFactory` satisfies that boundary by
constructing (but not opening) :class:`LinuxRH56TransactionalTransport`.

No loop iteration is reported as a completed policy step merely because a
model command was produced or one device accepted it.  Completion is derived
exclusively from :class:`~sim2real.closed_loop_core.ExecutedActionLedger` after
both devices acknowledge the same sequence.  A partial acknowledgement,
deadline, interlock loss, source failure, or worker failure latches the shared
gate and drives both verified stop paths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import importlib
import math
import threading
import time
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np

from .bounded_c2_orchestrator import (
    BoundedC2Admission,
    BoundedC2AdmissionError,
    BoundedC2Runtime,
    BoundedC2StopProof,
)
from sim2real.closed_loop_core import (
    ClosedLoopCommand,
    ClosedLoopProtocolError,
    ExecutedActionLedger,
)
from robot_control.franka.session import (
    ExperimentalSupervisedFrankaEnvelope,
    FrankaJointTarget,
    FrankaPersistentSession,
    FrankaPersistentTelemetry,
    FrankaSessionMode,
    FrankaTargetSampleHold,
    FrankaTargetSource,
    PersistentFrankaBackend,
    SupervisedFrankaPreflightToken,
)
from robot_control.rh56.linux_transport import LinuxRH56TransactionalTransport
from robot_control.rh56.actuator import (
    RH56ActuatorState,
    SupervisedRH56Preflight,
    RH56TransactionalActuator,
    RH56TransactionalTransport,
)
from sim2real.closed_loop_core import ClosedLoopSafetyGate, MotionAuthorization
from sim2real.contracts.actions import (
    LEGACY_FRANKA_ACTION_CONTRACT_ID,
    QD_G015_FRANKA_ACTION_CONTRACT_ID,
    normalize_franka_action_contract_id,
)
from sim2real.contracts.v94 import INITIAL_PREVIOUS_ACTION13


class BoundedC2HardwareRuntimeError(RuntimeError):
    """A terminal dependency, transaction, worker, or stop-coordination fault."""


_RUNTIME_SEAL = object()
_SUPERVISED_ADMISSION_SEAL = object()
_NO_STAGE_HOLD_WATCHDOG_FRACTION = 0.90


def _plain_tracking_snapshot(snapshot: object) -> Mapping[str, Any]:
    """Convert optional owner tracking evidence to an audit-safe mapping.

    Physical tracking was added after the original watchdog-owner protocol.
    Keeping this adapter structural lets older test doubles remain usable
    while production owners persist their complete immutable dataclass.
    """

    if snapshot is None:
        return {
            "available": False,
            "verdict": "not_exercised",
            "challenged_axes": (),
            "verified_axes": (),
        }
    if is_dataclass(snapshot):
        raw: object = asdict(snapshot)
    elif isinstance(snapshot, Mapping):
        raw = dict(snapshot)
    else:
        names = (
            "verdict",
            "challenged_axes",
            "verified_axes",
            "failed_axes",
            "active_challenge_axes",
            "latest_feedback_fresh",
            "latest_feedback_age_s",
            "latest_target",
            "latest_angles",
            "latest_positions",
            "tracking_significant_gap_units",
            "tracking_min_progress_units",
            "tracking_timeout_s",
            "failure_reason",
        )
        raw = {
            name: getattr(snapshot, name)
            for name in names
            if hasattr(snapshot, name)
        }
    if not isinstance(raw, Mapping):
        raise BoundedC2HardwareRuntimeError(
            "RH56 physical tracking snapshot is not a mapping"
        )
    result = dict(raw)
    result.setdefault("available", True)
    result.setdefault("verdict", "not_exercised")
    result.setdefault("challenged_axes", ())
    result.setdefault("verified_axes", ())
    return result


@dataclass(frozen=True)
class BoundedPolicyTickHold:
    """One explicit policy opportunity that stages no actuator command.

    This is used for transient observation conditions such as the third use of
    one 30 Hz camera frame.  It is deliberately not a ``ClosedLoopCommand``:
    no ledger sequence, mapper state, previous-action state, or actuator
    sample-hold may advance because of it.
    """

    sequence: int
    observed_monotonic_s: float
    retry_not_before_monotonic_s: float
    camera_frame_id: int
    reason: str
    policy_state_mutated: bool = False
    action_staged: bool = False

    def __post_init__(self) -> None:
        sequence = _positive_integer(self.sequence, "hold sequence")
        observed = _finite_float(
            self.observed_monotonic_s, "hold observed_monotonic_s"
        )
        retry = _finite_float(
            self.retry_not_before_monotonic_s,
            "hold retry_not_before_monotonic_s",
        )
        try:
            numeric_frame = float(self.camera_frame_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("hold camera_frame_id must be a non-negative integer") from exc
        if (
            isinstance(self.camera_frame_id, (bool, np.bool_))
            or not math.isfinite(numeric_frame)
            or not numeric_frame.is_integer()
            or numeric_frame < 0.0
        ):
            raise ValueError("hold camera_frame_id must be a non-negative integer")
        reason = str(self.reason).strip()
        if not reason:
            raise ValueError("hold reason must be non-empty")
        if retry < observed:
            raise ValueError("hold retry time cannot precede its observation")
        if self.policy_state_mutated is not False:
            raise ValueError("a no-stage hold cannot mutate policy state")
        if self.action_staged is not False:
            raise ValueError("a no-stage hold cannot stage an actuator action")
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "observed_monotonic_s", observed)
        object.__setattr__(self, "retry_not_before_monotonic_s", retry)
        object.__setattr__(self, "camera_frame_id", int(numeric_frame))
        object.__setattr__(self, "reason", reason)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _effective_rh56_inter_command_watchdog_s(
    admission: BoundedC2Admission | SupervisedV94Admission,
) -> float:
    """Resolve the watchdog that a no-command observation hold must precede."""

    preflight = admission.rh56_preflight
    value = getattr(
        preflight,
        "verified_inter_command_watchdog_timeout_s",
        None,
    )
    if value is None:
        value = getattr(
            preflight,
            "verified_command_watchdog_timeout_s",
            None,
        )
    return _positive_float(value, "effective RH56 inter-command watchdog")


def _validate_no_stage_hold_precedes_watchdog(
    maximum_hold_s: float,
    admission: BoundedC2Admission | SupervisedV94Admission,
) -> float:
    """Require a diagnostic stop edge with margin ahead of RH56's watchdog."""

    maximum_hold = _positive_float(
        maximum_hold_s,
        "maximum_consecutive_no_stage_hold_s",
    )
    watchdog = _effective_rh56_inter_command_watchdog_s(admission)
    policy_period_s = 1.0 / _positive_float(
        admission.policy_rate_hz,
        "policy_rate_hz",
    )
    maximum_safe_hold = (
        watchdog * _NO_STAGE_HOLD_WATCHDOG_FRACTION - policy_period_s
    )
    if maximum_safe_hold <= 0.0:
        raise ValueError(
            "effective RH56 inter-command watchdog leaves no bounded "
            "observation-hold interval after one policy period and 10% margin: "
            f"policy_period={policy_period_s:.6f}s "
            f"watchdog={watchdog:.6f}s"
        )
    if maximum_hold > maximum_safe_hold:
        raise ValueError(
            "maximum_consecutive_no_stage_hold_s plus one policy period must "
            "leave at least 10% margin before the effective RH56 "
            "inter-command watchdog: "
            f"hold={maximum_hold:.6f}s "
            f"policy_period={policy_period_s:.6f}s "
            f"watchdog={watchdog:.6f}s "
            f"maximum={maximum_safe_hold:.6f}s"
        )
    return watchdog


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


@dataclass(frozen=True, init=False)
class SupervisedV94Admission:
    """Sealed <=720-step execution permit, explicitly not a C2 admission."""

    run_id: str
    requested_policy_steps: int
    policy_rate_hz: float
    hard_deadline_monotonic_s: float
    authorization: MotionAuthorization
    franka_preflight_token: SupervisedFrankaPreflightToken
    rh56_preflight: SupervisedRH56Preflight
    franka_envelope: ExperimentalSupervisedFrankaEnvelope
    safety_gate: ClosedLoopSafetyGate
    franka_reference_q_rad: np.ndarray
    maximum_start_error_rad: float
    maximum_tick_target_delta_rad: float
    maximum_episode_delta_rad: Optional[float]
    franka_static_provenance_prevalidated: bool
    franka_action_contract_id: str
    initial_previous_action13: np.ndarray
    classification: str = "experimental_operator_supervised_non_c2"

    def __init__(self, *, _seal: object, **values: Any) -> None:
        if _seal is not _SUPERVISED_ADMISSION_SEAL:
            raise TypeError(
                "SupervisedV94Admission must come from the interactive runner"
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def require_active(self, *, now_monotonic_s: object) -> None:
        now = _finite_float(now_monotonic_s, "now_monotonic_s")
        if not (
            self.authorization.issued_monotonic_s
            <= now
            < self.authorization.expires_monotonic_s
        ):
            raise BoundedC2HardwareRuntimeError(
                "supervised run-scoped authorization expired"
            )
        if now >= self.hard_deadline_monotonic_s:
            raise BoundedC2HardwareRuntimeError("supervised run deadline expired")


def _seal_supervised_v94_admission(
    *,
    run_id: str,
    requested_policy_steps: object,
    policy_rate_hz: object,
    hard_deadline_monotonic_s: object,
    authorization: MotionAuthorization,
    franka_preflight_token: SupervisedFrankaPreflightToken,
    rh56_preflight: SupervisedRH56Preflight,
    franka_envelope: ExperimentalSupervisedFrankaEnvelope,
    safety_gate: ClosedLoopSafetyGate,
    franka_reference_q_rad: object,
    maximum_start_error_rad: object,
    maximum_tick_target_delta_rad: object,
    maximum_episode_delta_rad: object,
    franka_static_provenance_prevalidated: object,
    franka_action_contract_id: object = LEGACY_FRANKA_ACTION_CONTRACT_ID,
    initial_previous_action13: object = INITIAL_PREVIOUS_ACTION13,
) -> SupervisedV94Admission:
    steps = _positive_integer(requested_policy_steps, "requested_policy_steps")
    if steps > 720:
        raise ValueError("supervised policy-step bound exceeds 720")
    rate = _positive_float(policy_rate_hz, "policy_rate_hz")
    deadline = _finite_float(hard_deadline_monotonic_s, "hard deadline")
    reference = np.asarray(franka_reference_q_rad, dtype=np.float64)
    if reference.shape != (7,) or not np.all(np.isfinite(reference)):
        raise ValueError("Franka reference q must contain seven finite radians")
    reference = reference.copy()
    reference.setflags(write=False)
    if not isinstance(authorization, MotionAuthorization):
        raise TypeError("authorization must be MotionAuthorization")
    if not isinstance(franka_preflight_token, SupervisedFrankaPreflightToken):
        raise TypeError("supervised Franka token is required")
    if not isinstance(rh56_preflight, SupervisedRH56Preflight):
        raise TypeError("supervised RH56 token is required")
    if rh56_preflight.authorizes_c2 is not False:
        raise ValueError("supervised RH56 token cannot authorize C2")
    if not isinstance(franka_envelope, ExperimentalSupervisedFrankaEnvelope):
        raise TypeError("experimental supervised Franka envelope is required")
    if not isinstance(safety_gate, ClosedLoopSafetyGate):
        raise TypeError("safety_gate must be ClosedLoopSafetyGate")
    active_run = str(run_id).strip()
    if authorization.run_id != active_run:
        raise ValueError("authorization run ID differs from supervised run")
    if franka_static_provenance_prevalidated is not True:
        raise ValueError(
            "supervised admission requires fresh read-only Franka static provenance"
        )
    action_contract = normalize_franka_action_contract_id(
        franka_action_contract_id
    )
    initial_previous = np.asarray(initial_previous_action13, dtype=np.float32)
    if initial_previous.shape != (13,) or not np.all(np.isfinite(initial_previous)):
        raise ValueError(
            "supervised initial_previous_action13 must contain 13 finite values"
        )
    if np.any(initial_previous < -1.0) or np.any(initial_previous > 1.0):
        raise ValueError(
            "supervised initial_previous_action13 must lie in [-1,1]"
        )
    initial_previous = initial_previous.copy()
    initial_previous.setflags(write=False)
    if (
        action_contract == QD_G015_FRANKA_ACTION_CONTRACT_ID
        and np.any(initial_previous != 0.0)
    ):
        raise ValueError("q_d g015 reset requires zero previous action")
    episode_limit = (
        None
        if action_contract == QD_G015_FRANKA_ACTION_CONTRACT_ID
        else _positive_float(
            maximum_episode_delta_rad, "maximum_episode_delta_rad"
        )
    )
    if (
        action_contract == QD_G015_FRANKA_ACTION_CONTRACT_ID
        and maximum_episode_delta_rad is not None
    ):
        raise ValueError(
            "q_d g015 admission must disable the home-centered episode limit"
        )
    return SupervisedV94Admission(
        _seal=_SUPERVISED_ADMISSION_SEAL,
        run_id=active_run,
        requested_policy_steps=steps,
        policy_rate_hz=rate,
        hard_deadline_monotonic_s=deadline,
        authorization=authorization,
        franka_preflight_token=franka_preflight_token,
        rh56_preflight=rh56_preflight,
        franka_envelope=franka_envelope,
        safety_gate=safety_gate,
        franka_reference_q_rad=reference,
        maximum_start_error_rad=_positive_float(
            maximum_start_error_rad, "maximum_start_error_rad"
        ),
        maximum_tick_target_delta_rad=_positive_float(
            maximum_tick_target_delta_rad, "maximum_tick_target_delta_rad"
        ),
        maximum_episode_delta_rad=episode_limit,
        franka_static_provenance_prevalidated=True,
        franka_action_contract_id=action_contract,
        initial_previous_action13=initial_previous,
        classification="experimental_operator_supervised_non_c2",
    )


@runtime_checkable
class ManagedRH56TransactionalTransport(RH56TransactionalTransport, Protocol):
    """Transactional transport with an explicit, non-writing lifecycle."""

    @property
    def is_open(self) -> bool: ...

    def open(self) -> "ManagedRH56TransactionalTransport": ...

    def close(self) -> None: ...


@runtime_checkable
class BoundedC2LiveSafetySupervisor(Protocol):
    """Trusted live boundary omitted from the offline admission CLI.

    A production implementation must read the independently commissioned
    physical deadman/E-stop and update/latch the shared gate.  Merely returning
    successfully without that hardware-backed check is not sufficient for a
    real deployment.
    """

    physical_interlocks_configured: bool
    independent_command_watchdogs_configured: bool
    verified_dual_device_stop_supported: bool

    def require_motion(
        self,
        admission: BoundedC2Admission,
        *,
        now_monotonic_s: float,
    ) -> None: ...


@runtime_checkable
class BoundedV94PolicyTickSource(Protocol):
    """Prepare and transactionally finalize exact V94 policy commands.

    ``prepare`` must derive the observation's previous-action field from the
    supplied dual-committed vector.  ``commit`` is called only after the ledger
    proves the same sequence was acknowledged by both devices.  ``abort`` is
    terminal notification; it is never permission to advance mapper state.
    """

    def prepare(
        self,
        *,
        sequence: int,
        previous_executed_action13: np.ndarray,
        now_monotonic_s: float,
        hard_deadline_monotonic_s: float,
    ) -> Union[ClosedLoopCommand, BoundedPolicyTickHold]: ...

    def commit(
        self,
        command: ClosedLoopCommand,
        *,
        action_ledger: ExecutedActionLedger,
    ) -> None: ...

    def abort(self, command: ClosedLoopCommand, *, reason: str) -> None: ...


@runtime_checkable
class ManagedRH56WatchdogOwner(Protocol):
    """Dedicated serial owner whose worker enforces the RH56 heartbeat.

    The runtime thread may submit work and read immutable feedback only.  It
    must never claim, execute, poll, disable, or close the actuator/transport
    directly on this production path.
    """

    independent_watchdog_active: bool
    fault_callback_configured: bool
    first_command_readiness_verified: bool

    def start(self) -> Any: ...

    def submit(self, command: ClosedLoopCommand) -> Any: ...

    def execute(
        self,
        command: ClosedLoopCommand,
        *,
        timeout_s: Optional[float] = None,
    ) -> Any: ...

    def raise_if_faulted(self) -> None: ...

    def feedback_history_snapshot(self, *, maximum_age_s: float) -> Any: ...

    def require_fresh_feedback_history(
        self,
        *,
        minimum_samples: int,
        maximum_age_s: float,
    ) -> Any: ...

    def request_stop(self) -> None: ...

    def stop_and_close(self, timeout_s: Optional[float] = None) -> Any: ...


@runtime_checkable
class ManagedV94PolicyTickSourceFactory(Protocol):
    """Camera-first, late-bound construction for production observations."""

    construction_is_inert: bool

    def open_and_warm_camera(
        self,
        admission: BoundedC2Admission,
        *,
        hard_deadline_monotonic_s: float,
    ) -> None: ...

    def wait_for_camera_pose_alignment(
        self,
        *,
        franka_session: FrankaPersistentSession,
        hard_deadline_monotonic_s: float,
    ) -> None: ...

    def __call__(
        self,
        *,
        franka_session: FrankaPersistentSession,
        rh56_feedback_source: ManagedRH56WatchdogOwner,
        hard_deadline_monotonic_s: float,
    ) -> BoundedV94PolicyTickSource: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class LinuxRH56TransportFactory:
    """Inert constructor for the reviewed Linux RH56 serial adapter.

    Calling this factory still does not import the RH56 API, discover a port,
    or open a descriptor.  ``BoundedV94C2Runtime`` owns the later ``open`` and
    ``close`` calls.
    """

    port: Optional[str]
    baud_rate: int = 115200
    hand_id: int = 1
    debug: bool = False
    api_module_name: str = "examples.inspire_rh56_test"
    module_loader: Callable[[str], Any] = importlib.import_module
    monotonic: Callable[[], float] = time.monotonic

    construction_is_inert: bool = True

    def __call__(self) -> LinuxRH56TransactionalTransport:
        return LinuxRH56TransactionalTransport(
            self.port,
            baud_rate=self.baud_rate,
            hand_id=self.hand_id,
            debug=self.debug,
            api_module_name=self.api_module_name,
            module_loader=self.module_loader,
            monotonic=self.monotonic,
        )


class BoundedV94C2RuntimeFactory:
    """Build one hardware-inert combined runtime from reviewed dependencies.

    ``rh56_actuator_factory`` is deliberately required: current/temperature,
    stop-drift, stop-current and feedback-age bounds are not present in the
    current V94 commissioning profile, so this module must not invent them or
    silently use generic actuator defaults for real hardware.
    """

    def __init__(
        self,
        *,
        franka_backend_factory: Optional[
            Callable[[], PersistentFrankaBackend]
        ] = None,
        supervised_franka_session_factory: Optional[Callable[..., Any]] = None,
        rh56_transport_factory: Optional[
            Callable[[], ManagedRH56TransactionalTransport]
        ] = None,
        rh56_actuator_factory: Optional[Callable[
            [ManagedRH56TransactionalTransport, BoundedC2Admission],
            RH56TransactionalActuator,
        ]] = None,
        policy_tick_source: Optional[BoundedV94PolicyTickSource] = None,
        rh56_watchdog_owner_factory: Optional[Callable[..., ManagedRH56WatchdogOwner]] = None,
        policy_tick_source_factory: Optional[ManagedV94PolicyTickSourceFactory] = None,
        live_safety_supervisor: BoundedC2LiveSafetySupervisor,
        franka_stop_join_timeout_s: object,
        commit_poll_interval_s: object,
        maximum_consecutive_no_stage_hold_s: object = 0.025,
        monotonic: Callable[[], float] = time.monotonic,
        realtime: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        for value, name in (
            (monotonic, "monotonic"),
            (realtime, "realtime"),
            (sleep, "sleep"),
        ):
            if not callable(value):
                raise TypeError(f"{name} must be callable")
        if (franka_backend_factory is None) == (
            supervised_franka_session_factory is None
        ):
            raise ValueError(
                "configure exactly one Franka backend or supervised session factory"
            )
        if franka_backend_factory is not None and not callable(
            franka_backend_factory
        ):
            raise TypeError("franka_backend_factory must be callable")
        if supervised_franka_session_factory is not None and not callable(
            supervised_franka_session_factory
        ):
            raise TypeError("supervised_franka_session_factory must be callable")
        legacy_path = (
            rh56_transport_factory is not None
            or rh56_actuator_factory is not None
            or policy_tick_source is not None
        )
        owner_path = (
            rh56_watchdog_owner_factory is not None
            or policy_tick_source_factory is not None
        )
        if legacy_path == owner_path:
            raise ValueError(
                "configure exactly one complete RH56/policy runtime path"
            )
        if legacy_path:
            for value, name in (
                (rh56_transport_factory, "rh56_transport_factory"),
                (rh56_actuator_factory, "rh56_actuator_factory"),
            ):
                if not callable(value):
                    raise TypeError(f"{name} must be callable")
            if not isinstance(policy_tick_source, BoundedV94PolicyTickSource):
                raise TypeError("policy_tick_source does not implement its protocol")
        else:
            if not callable(rh56_watchdog_owner_factory):
                raise TypeError("rh56_watchdog_owner_factory must be callable")
            if not isinstance(
                policy_tick_source_factory, ManagedV94PolicyTickSourceFactory
            ):
                raise TypeError(
                    "policy_tick_source_factory does not implement its protocol"
                )
            if policy_tick_source_factory.construction_is_inert is not True:
                raise ValueError("policy source factory construction must be inert")
        if not isinstance(live_safety_supervisor, BoundedC2LiveSafetySupervisor):
            raise TypeError(
                "live_safety_supervisor does not implement its protocol"
            )
        join_timeout = _positive_float(
            franka_stop_join_timeout_s, "franka_stop_join_timeout_s"
        )
        poll_interval = _positive_float(
            commit_poll_interval_s, "commit_poll_interval_s"
        )
        maximum_hold = _positive_float(
            maximum_consecutive_no_stage_hold_s,
            "maximum_consecutive_no_stage_hold_s",
        )

        self._franka_backend_factory = franka_backend_factory
        self._supervised_franka_session_factory = (
            supervised_franka_session_factory
        )
        self._rh56_transport_factory = rh56_transport_factory
        self._rh56_actuator_factory = rh56_actuator_factory
        self._policy_tick_source = policy_tick_source
        self._rh56_watchdog_owner_factory = rh56_watchdog_owner_factory
        self._policy_tick_source_factory = policy_tick_source_factory
        self._live_safety_supervisor = live_safety_supervisor
        self._franka_stop_join_timeout_s = join_timeout
        self._commit_poll_interval_s = poll_interval
        self._maximum_consecutive_no_stage_hold_s = maximum_hold
        self._monotonic = monotonic
        self._realtime = realtime
        self._sleep = sleep

    def __call__(self, admission: BoundedC2Admission) -> "BoundedV94C2Runtime":
        if not isinstance(admission, BoundedC2Admission):
            raise TypeError("runtime factory requires a sealed BoundedC2Admission")
        now = _finite_float(self._monotonic(), "monotonic clock")
        admission.require_active(now_monotonic_s=now)
        _validate_no_stage_hold_precedes_watchdog(
            self._maximum_consecutive_no_stage_hold_s,
            admission,
        )
        if self._franka_backend_factory is None:
            raise ValueError(
                "formal bounded C2 runtime requires a Franka backend factory"
            )
        ledger = ExecutedActionLedger(
            maximum_commit_latency_s=(
                admission.rh56_preflight.verified_command_watchdog_timeout_s
            )
        )
        target_hold = FrankaTargetSampleHold()
        franka_session = FrankaPersistentSession(
            run_id=admission.run_id,
            mode=FrankaSessionMode.C2_V94_POLICY,
            envelope=admission.franka_envelope,
            target_hold=target_hold,
            safety_gate=admission.safety_gate,
            backend_factory=self._franka_backend_factory,
            action_ledger=ledger,
            enable_c2_measured_hold_bootstrap=(
                self._policy_tick_source_factory is not None
            ),
            monotonic=self._monotonic,
            realtime=self._realtime,
        )
        return BoundedV94C2Runtime(
            _seal=_RUNTIME_SEAL,
            admission=admission,
            action_ledger=ledger,
            target_hold=target_hold,
            franka_session=franka_session,
            rh56_transport_factory=self._rh56_transport_factory,
            rh56_actuator_factory=self._rh56_actuator_factory,
            policy_tick_source=self._policy_tick_source,
            rh56_watchdog_owner_factory=self._rh56_watchdog_owner_factory,
            policy_tick_source_factory=self._policy_tick_source_factory,
            live_safety_supervisor=self._live_safety_supervisor,
            franka_stop_join_timeout_s=self._franka_stop_join_timeout_s,
            commit_poll_interval_s=self._commit_poll_interval_s,
            maximum_consecutive_no_stage_hold_s=(
                self._maximum_consecutive_no_stage_hold_s
            ),
            monotonic=self._monotonic,
            sleep=self._sleep,
        )


class SupervisedV94RuntimeFactory(BoundedV94C2RuntimeFactory):
    """Build the same dual-owner machinery under a distinct non-C2 permit."""

    def __call__(
        self, admission: SupervisedV94Admission
    ) -> "BoundedV94C2Runtime":
        if not isinstance(admission, SupervisedV94Admission):
            raise TypeError("supervised runtime requires SupervisedV94Admission")
        now = _finite_float(self._monotonic(), "monotonic clock")
        admission.require_active(now_monotonic_s=now)
        _validate_no_stage_hold_precedes_watchdog(
            self._maximum_consecutive_no_stage_hold_s,
            admission,
        )
        ledger = ExecutedActionLedger(
            maximum_commit_latency_s=(
                admission.rh56_preflight.verified_command_watchdog_timeout_s
            ),
            initial_previous_action13=admission.initial_previous_action13,
        )
        native_factory = self._supervised_franka_session_factory
        if native_factory is None:
            assert self._franka_backend_factory is not None
            target_hold = FrankaTargetSampleHold()
            franka_session = FrankaPersistentSession(
                run_id=admission.run_id,
                mode=FrankaSessionMode.SUPERVISED_V94,
                envelope=admission.franka_envelope,
                target_hold=target_hold,
                safety_gate=admission.safety_gate,
                backend_factory=self._franka_backend_factory,
                action_ledger=ledger,
                enable_c2_measured_hold_bootstrap=(
                    self._policy_tick_source_factory is not None
                ),
                supervised_reference_q_rad=admission.franka_reference_q_rad,
                supervised_maximum_start_error_rad=(
                    admission.maximum_start_error_rad
                ),
                supervised_maximum_tick_target_delta_rad=(
                    admission.maximum_tick_target_delta_rad
                ),
                supervised_maximum_episode_delta_rad=(
                    admission.maximum_episode_delta_rad
                ),
                franka_action_contract_id=admission.franka_action_contract_id,
                supervised_static_provenance_prevalidated=(
                    admission.franka_static_provenance_prevalidated
                ),
                monotonic=self._monotonic,
                realtime=self._realtime,
            )
        else:
            franka_session = native_factory(
                admission=admission,
                action_ledger=ledger,
            )
            target_hold = getattr(franka_session, "target_hold", None)
            if not isinstance(target_hold, FrankaTargetSampleHold):
                raise TypeError(
                    "supervised Franka session factory returned no valid target_hold"
                )
            for name in (
                "run",
                "request_clean_stop",
                "pose_ring",
                "c2_bootstrap_ready",
                "last_telemetry",
                "envelope",
            ):
                if not hasattr(franka_session, name):
                    raise TypeError(
                        "supervised Franka session factory returned an invalid "
                        f"session: missing {name}"
                    )
        return BoundedV94C2Runtime(
            _seal=_RUNTIME_SEAL,
            admission=admission,
            action_ledger=ledger,
            target_hold=target_hold,
            franka_session=franka_session,
            rh56_transport_factory=self._rh56_transport_factory,
            rh56_actuator_factory=self._rh56_actuator_factory,
            policy_tick_source=self._policy_tick_source,
            rh56_watchdog_owner_factory=self._rh56_watchdog_owner_factory,
            policy_tick_source_factory=self._policy_tick_source_factory,
            live_safety_supervisor=self._live_safety_supervisor,
            franka_stop_join_timeout_s=self._franka_stop_join_timeout_s,
            commit_poll_interval_s=self._commit_poll_interval_s,
            maximum_consecutive_no_stage_hold_s=(
                self._maximum_consecutive_no_stage_hold_s
            ),
            monotonic=self._monotonic,
            sleep=self._sleep,
            supervised_non_c2=True,
        )


class BoundedV94C2Runtime:
    """Single-use combined runtime implementing :class:`BoundedC2Runtime`."""

    def __init__(
        self,
        *,
        _seal: object,
        admission: BoundedC2Admission | SupervisedV94Admission,
        action_ledger: ExecutedActionLedger,
        target_hold: FrankaTargetSampleHold,
        franka_session: FrankaPersistentSession,
        rh56_transport_factory: Optional[
            Callable[[], ManagedRH56TransactionalTransport]
        ],
        rh56_actuator_factory: Optional[Callable[
            [ManagedRH56TransactionalTransport, BoundedC2Admission],
            RH56TransactionalActuator,
        ]],
        policy_tick_source: Optional[BoundedV94PolicyTickSource],
        rh56_watchdog_owner_factory: Optional[
            Callable[..., ManagedRH56WatchdogOwner]
        ],
        policy_tick_source_factory: Optional[ManagedV94PolicyTickSourceFactory],
        live_safety_supervisor: BoundedC2LiveSafetySupervisor,
        franka_stop_join_timeout_s: float,
        commit_poll_interval_s: float,
        maximum_consecutive_no_stage_hold_s: float,
        monotonic: Callable[[], float],
        sleep: Callable[[float], None],
        supervised_non_c2: bool = False,
    ) -> None:
        if _seal is not _RUNTIME_SEAL:
            raise TypeError(
                "BoundedV94C2Runtime must come from BoundedV94C2RuntimeFactory"
            )
        self.admission = admission
        self.action_ledger = action_ledger
        self.target_hold = target_hold
        self.franka_session = franka_session
        self._rh56_transport_factory = rh56_transport_factory
        self._rh56_actuator_factory = rh56_actuator_factory
        self._policy_tick_source = policy_tick_source
        self._rh56_watchdog_owner_factory = rh56_watchdog_owner_factory
        self._policy_tick_source_factory = policy_tick_source_factory
        self._live_safety_supervisor = live_safety_supervisor
        self._franka_stop_join_timeout_s = franka_stop_join_timeout_s
        self._commit_poll_interval_s = commit_poll_interval_s
        self._maximum_consecutive_no_stage_hold_s = _positive_float(
            maximum_consecutive_no_stage_hold_s,
            "maximum_consecutive_no_stage_hold_s",
        )
        self._no_stage_hold_effective_rh56_watchdog_s = (
            _validate_no_stage_hold_precedes_watchdog(
                self._maximum_consecutive_no_stage_hold_s,
                admission,
            )
        )
        self._monotonic = monotonic
        self._sleep = sleep
        self._supervised_non_c2 = bool(supervised_non_c2)

        self.physical_interlocks_configured = (
            live_safety_supervisor.physical_interlocks_configured is True
        )
        self.independent_command_watchdogs_configured = (
            live_safety_supervisor.independent_command_watchdogs_configured is True
        )
        self.verified_dual_device_stop_supported = (
            live_safety_supervisor.verified_dual_device_stop_supported is True
        )
        self.operator_supervision_confirmed = bool(
            getattr(live_safety_supervisor, "operator_supervision_confirmed", False)
        )

        self._state_lock = threading.Lock()
        self._run_started = False
        self._stop_requested = threading.Event()
        self._franka_thread: Optional[threading.Thread] = None
        self._franka_done = threading.Event()
        self._franka_error: Optional[BaseException] = None
        self._franka_telemetry: Optional[FrankaPersistentTelemetry] = None
        self._rh56_transport: Optional[ManagedRH56TransactionalTransport] = None
        self._rh56_actuator: Optional[RH56TransactionalActuator] = None
        self._rh56_watchdog_owner: Optional[ManagedRH56WatchdogOwner] = None
        self._rh56_callback_fault: Optional[object] = None
        self._pending_command: Optional[ClosedLoopCommand] = None
        self._failure_pending_command: Optional[ClosedLoopCommand] = None
        self._stop_proof: Optional[BoundedC2StopProof] = None
        self._stop_errors: Tuple[str, ...] = ()
        self._committed_commands: list[Mapping[str, Any]] = []
        # Small, bounded diagnostics retained independently of run()'s return
        # value.  A terminal actuator fault may raise before the normal result
        # mapping is built, so the outer supervised audit reads this snapshot
        # directly from the runtime during cleanup.
        self._no_stage_policy_hold_count = 0
        self._no_stage_policy_hold_counts_by_reason: dict[str, int] = {}
        self._last_no_stage_policy_hold: Optional[Mapping[str, Any]] = None
        self._current_no_stage_hold_started_monotonic_s: Optional[float] = None
        self._current_no_stage_hold_effective_end_monotonic_s: Optional[float] = None
        self._current_no_stage_hold_count = 0
        self._current_no_stage_hold_first_camera_frame_id: Optional[int] = None
        self._current_no_stage_hold_last_camera_frame_id: Optional[int] = None
        self._current_no_stage_hold_frame_changes = 0
        self._maximum_consecutive_no_stage_hold_count = 0
        self._maximum_consecutive_no_stage_hold_duration_s = 0.0
        self._policy_schedule_skipped_slot_count = 0
        self._policy_schedule_max_release_lateness_s = 0.0
        self._policy_schedule_mode = "legacy_minimum_period"
        self._startup_non_actuated_completed_steps = 0
        self._rollout_trigger_to_first_action_s: Optional[float] = None

    @property
    def stop_errors(self) -> Tuple[str, ...]:
        with self._state_lock:
            return self._stop_errors

    @property
    def franka_telemetry(self) -> Optional[FrankaPersistentTelemetry]:
        with self._state_lock:
            return self._franka_telemetry

    @property
    def failure_pending_command(self) -> Optional[ClosedLoopCommand]:
        """Immutable command that was still awaiting dual ACK at failure."""

        with self._state_lock:
            return self._failure_pending_command

    @property
    def committed_commands_snapshot(self) -> Tuple[Mapping[str, Any], ...]:
        """Audit evidence for every command that crossed the dual-ACK barrier.

        This snapshot is deliberately available even when :meth:`run` raises
        after the hardware transaction completed (for example during policy
        source commit or the final one-tick hold).  A failed run must never
        make an already executed command disappear from its audit artifact.
        """

        with self._state_lock:
            return tuple(dict(item) for item in self._committed_commands)

    @property
    def runtime_diagnostics_snapshot(self) -> Mapping[str, Any]:
        """Bounded timing/hold evidence, available on success and failure."""

        with self._state_lock:
            last_hold = (
                None
                if self._last_no_stage_policy_hold is None
                else dict(self._last_no_stage_policy_hold)
            )
            source = self._policy_tick_source
            result: dict[str, Any] = {
                "no_stage_policy_hold_count": self._no_stage_policy_hold_count,
                "no_stage_policy_hold_counts_by_reason": dict(
                    self._no_stage_policy_hold_counts_by_reason
                ),
                "last_no_stage_policy_hold": last_hold,
                "maximum_consecutive_no_stage_hold_count": (
                    self._maximum_consecutive_no_stage_hold_count
                ),
                "maximum_consecutive_no_stage_hold_duration_s": (
                    self._maximum_consecutive_no_stage_hold_duration_s
                ),
                "maximum_consecutive_no_stage_hold_s": (
                    self._maximum_consecutive_no_stage_hold_s
                ),
                "no_stage_hold_effective_rh56_watchdog_s": (
                    self._no_stage_hold_effective_rh56_watchdog_s
                ),
                "policy_schedule": {
                    "mode": self._policy_schedule_mode,
                    "skipped_slot_count": int(
                        self._policy_schedule_skipped_slot_count
                    ),
                    "max_release_lateness_s": float(
                        self._policy_schedule_max_release_lateness_s
                    ),
                },
                "startup_non_actuated_completed_steps": int(
                    self._startup_non_actuated_completed_steps
                ),
            }
            if self._rollout_trigger_to_first_action_s is not None:
                result["rollout_trigger_to_first_action_s"] = float(
                    self._rollout_trigger_to_first_action_s
                )
        if source is not None:
            diagnostics = getattr(source, "diagnostics_snapshot", None)
            if diagnostics is not None:
                try:
                    result["observation_source"] = (
                        dict(diagnostics)
                        if not isinstance(diagnostics, dict)
                        else dict(diagnostics)
                    )
                except BaseException as exc:
                    result["observation_source"] = {
                        "diagnostics_error": f"{type(exc).__name__}: {exc}"
                    }
        source_factory = self._policy_tick_source_factory
        if source_factory is not None:
            trigger_diagnostics = getattr(
                source_factory, "rollout_trigger_diagnostics", None
            )
            if trigger_diagnostics is not None:
                try:
                    result["rollout_trigger"] = dict(trigger_diagnostics)
                except BaseException as exc:
                    result["rollout_trigger"] = {
                        "diagnostics_error": f"{type(exc).__name__}: {exc}"
                    }
        return result

    def _record_no_stage_policy_hold(
        self, hold: BoundedPolicyTickHold
    ) -> None:
        """Update constant-space diagnostics for one side-effect-free hold."""

        observed = float(hold.observed_monotonic_s)
        # The liveness bound is anchored to the first actual observation.
        # A source-proposed retry time is scheduling advice only and must
        # never make the streak appear older or renew its terminal deadline.
        effective_end = observed
        with self._state_lock:
            if self._current_no_stage_hold_started_monotonic_s is None:
                self._current_no_stage_hold_started_monotonic_s = observed
                self._current_no_stage_hold_effective_end_monotonic_s = (
                    effective_end
                )
                self._current_no_stage_hold_count = 0
                self._current_no_stage_hold_first_camera_frame_id = int(
                    hold.camera_frame_id
                )
                self._current_no_stage_hold_last_camera_frame_id = int(
                    hold.camera_frame_id
                )
                self._current_no_stage_hold_frame_changes = 0
            else:
                current_end = (
                    self._current_no_stage_hold_effective_end_monotonic_s
                )
                self._current_no_stage_hold_effective_end_monotonic_s = max(
                    effective_end,
                    observed if current_end is None else current_end,
                )
                if (
                    self._current_no_stage_hold_last_camera_frame_id
                    != int(hold.camera_frame_id)
                ):
                    self._current_no_stage_hold_frame_changes += 1
                self._current_no_stage_hold_last_camera_frame_id = int(
                    hold.camera_frame_id
                )
            self._current_no_stage_hold_count += 1
            self._no_stage_policy_hold_count += 1
            self._no_stage_policy_hold_counts_by_reason[hold.reason] = (
                self._no_stage_policy_hold_counts_by_reason.get(hold.reason, 0)
                + 1
            )
            self._last_no_stage_policy_hold = {
                "sequence": int(hold.sequence),
                "reason": str(hold.reason),
                "camera_frame_id": int(hold.camera_frame_id),
                "observed_monotonic_s": observed,
                "retry_not_before_monotonic_s": float(
                    hold.retry_not_before_monotonic_s
                ),
                "scheduled_retry_delay_s": max(
                    0.0,
                    float(hold.retry_not_before_monotonic_s) - observed,
                ),
                "consecutive_index": self._current_no_stage_hold_count,
                "consecutive_duration_s": max(
                    0.0,
                    observed
                    - float(self._current_no_stage_hold_started_monotonic_s),
                ),
                "streak_first_camera_frame_id": (
                    self._current_no_stage_hold_first_camera_frame_id
                ),
                "streak_last_camera_frame_id": (
                    self._current_no_stage_hold_last_camera_frame_id
                ),
                "streak_frame_changes": self._current_no_stage_hold_frame_changes,
            }
            self._update_no_stage_hold_maxima_locked()

    def _require_no_stage_policy_hold_within_bound(
        self, now_monotonic_s: float
    ) -> None:
        """Fail with perception provenance before actuator watchdog expiry.

        The first hold observation is the immutable streak anchor.  Repeated
        holds, changing reasons/frames, and later retry suggestions cannot
        move the deadline.
        """

        now = _finite_float(
            now_monotonic_s,
            "no-stage hold liveness check",
        )
        with self._state_lock:
            started = self._current_no_stage_hold_started_monotonic_s
            if started is None:
                return
            duration_s = max(0.0, now - started)
            self._current_no_stage_hold_effective_end_monotonic_s = max(
                now,
                now
                if self._current_no_stage_hold_effective_end_monotonic_s is None
                else self._current_no_stage_hold_effective_end_monotonic_s,
            )
            self._update_no_stage_hold_maxima_locked()
            hold_deadline = (
                started + self._maximum_consecutive_no_stage_hold_s
            )
            if now < hold_deadline:
                return
            last = self._last_no_stage_policy_hold
            if last is None:
                raise BoundedC2HardwareRuntimeError(
                    "maximum consecutive no-stage observation hold exceeded "
                    f"after {duration_s:.6f}s "
                    f"(limit={self._maximum_consecutive_no_stage_hold_s:.6f}s)"
                )
            reason = str(last["reason"])
            sequence = int(last["sequence"])
            frame_id = int(last["camera_frame_id"])
            first_frame_id = self._current_no_stage_hold_first_camera_frame_id
            last_frame_id = self._current_no_stage_hold_last_camera_frame_id
            frame_changes = self._current_no_stage_hold_frame_changes
            source = self._policy_tick_source
        source_summary = ""
        if source is not None:
            try:
                summary = getattr(source, "diagnostics_summary", "")
                if summary:
                    source_summary = (
                        "; " + " ".join(str(summary).split())[:2000]
                    )
            except BaseException as exc:
                source_summary = (
                    "; perception diagnostics unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )
        raise BoundedC2HardwareRuntimeError(
            "maximum consecutive no-stage observation hold exceeded: "
            f"duration={duration_s:.6f}s "
            f"limit={self._maximum_consecutive_no_stage_hold_s:.6f}s "
            f"sequence={sequence} camera_frame_id={frame_id} "
            f"reason={reason} "
            f"streak_first_camera_frame_id={first_frame_id} "
            f"streak_last_camera_frame_id={last_frame_id} "
            f"streak_frame_changes={frame_changes}"
            f"{source_summary}"
        )

    def _no_stage_policy_hold_deadline_monotonic_s(self) -> Optional[float]:
        """Return the immutable deadline for the active hold streak."""

        with self._state_lock:
            started = self._current_no_stage_hold_started_monotonic_s
            if started is None:
                return None
            return started + self._maximum_consecutive_no_stage_hold_s

    def _close_no_stage_policy_hold_streak(
        self, ended_monotonic_s: float
    ) -> None:
        """Close the current hold streak when a command or terminal stop wins."""

        ended = _finite_float(
            ended_monotonic_s,
            "no-stage hold streak end",
        )
        with self._state_lock:
            if self._current_no_stage_hold_started_monotonic_s is None:
                return
            current_end = self._current_no_stage_hold_effective_end_monotonic_s
            self._current_no_stage_hold_effective_end_monotonic_s = max(
                ended,
                ended if current_end is None else current_end,
            )
            self._update_no_stage_hold_maxima_locked()
            self._current_no_stage_hold_started_monotonic_s = None
            self._current_no_stage_hold_effective_end_monotonic_s = None
            self._current_no_stage_hold_count = 0
            self._current_no_stage_hold_first_camera_frame_id = None
            self._current_no_stage_hold_last_camera_frame_id = None
            self._current_no_stage_hold_frame_changes = 0

    def _update_no_stage_hold_maxima_locked(self) -> None:
        started = self._current_no_stage_hold_started_monotonic_s
        ended = self._current_no_stage_hold_effective_end_monotonic_s
        if started is None or ended is None:
            return
        duration_s = max(0.0, ended - started)
        self._maximum_consecutive_no_stage_hold_count = max(
            self._maximum_consecutive_no_stage_hold_count,
            self._current_no_stage_hold_count,
        )
        self._maximum_consecutive_no_stage_hold_duration_s = max(
            self._maximum_consecutive_no_stage_hold_duration_s,
            duration_s,
        )

    def _rh56_feedback_audit_snapshot(self) -> Optional[Mapping[str, Any]]:
        """Copy the newest owner-cached RH56 feedback without performing IO."""

        owner = self._rh56_watchdog_owner
        if owner is None:
            return None
        try:
            try:
                snapshot = owner.feedback_history_snapshot()
            except TypeError:
                # Compatibility for original protocol fakes, whose keyword
                # was mandatory.  Production owners use their commissioned
                # default freshness bound on the first call above.
                snapshot = owner.feedback_history_snapshot(maximum_age_s=1.0)
            samples = tuple(getattr(snapshot, "samples", ()))
            if not samples:
                return None
            sample = samples[-1]
            feedback = getattr(sample, "feedback", None)
            if feedback is None:
                return None

            def values(name: str) -> Optional[list[int]]:
                raw = getattr(feedback, name, None)
                if raw is None:
                    return None
                return [int(item) for item in raw]

            captured_monotonic_s = getattr(
                feedback, "captured_monotonic_s", None
            )
            captured_realtime_s = getattr(sample, "captured_realtime_s", None)
            cached_monotonic_s = getattr(sample, "cached_monotonic_s", None)
            latest_age_s = getattr(snapshot, "latest_age_s", None)
            return {
                "angles": values("angles"),
                "positions": values("positions"),
                "statuses": values("statuses"),
                "currents_ma": values("currents_ma"),
                "forces_g": values("forces_g"),
                "errors": values("errors"),
                "temperatures_c": values("temperatures_c"),
                "captured_monotonic_s": (
                    None
                    if captured_monotonic_s is None
                    else float(captured_monotonic_s)
                ),
                "captured_realtime_s": (
                    None
                    if captured_realtime_s is None
                    else float(captured_realtime_s)
                ),
                "cached_monotonic_s": (
                    None
                    if cached_monotonic_s is None
                    else float(cached_monotonic_s)
                ),
                "latest_age_s": (
                    None if latest_age_s is None else float(latest_age_s)
                ),
                "fresh": bool(getattr(snapshot, "fresh", False)),
            }
        except BaseException:
            # This is supplemental evidence copied after the command's exact
            # write/readback ACK.  Owner health and freshness remain enforced
            # by the command path; a nonstandard legacy fake may lack fields.
            return None

    def _rh56_physical_tracking_audit_snapshot(
        self,
    ) -> Optional[Mapping[str, Any]]:
        owner = self._rh56_watchdog_owner
        method = (
            None
            if owner is None
            else getattr(owner, "physical_tracking_snapshot", None)
        )
        if not callable(method):
            # Original dependency-injected owner fakes predate physical
            # tracking.  Production owners expose the method and therefore
            # always emit the field below.
            return None
        return _plain_tracking_snapshot(method())

    def _wait_for_final_rh56_physical_tracking(
        self,
        *,
        hard_deadline_monotonic_s: float,
        external_stop_requested: threading.Event,
    ) -> Tuple[Optional[Mapping[str, Any]], float]:
        """Finish Franka cleanly, then hold RH56 until motion is evidenced.

        No policy source method, target publish, actuator execute, or ledger
        operation occurs here.  The production owner supplies a separately
        bounded sample-hold mode which keeps its short command watchdog alive
        while the original no-motion tracking deadline remains authoritative.
        Franka receives its existing one-shot clean STOP before the wait, so
        the native policy inter-target watchdog cannot race a hand-only settle.
        """

        snapshot = self._rh56_physical_tracking_audit_snapshot()
        if snapshot is None:
            return None, 0.0
        challenged = tuple(
            bool(item) for item in snapshot.get("challenged_axes", ())
        )
        verdict = str(snapshot.get("verdict", "not_exercised"))
        if not any(challenged) or verdict == "verified":
            return snapshot, 0.0
        if verdict == "failed":
            raise BoundedC2HardwareRuntimeError(
                "RH56 physical tracking failed before final settle"
            )
        if verdict != "challenged_unverified":
            raise BoundedC2HardwareRuntimeError(
                f"RH56 physical tracking returned invalid verdict={verdict}"
            )
        owner = self._rh56_watchdog_owner
        begin_settle = (
            None
            if owner is None
            else getattr(owner, "begin_final_tracking_settle", None)
        )
        if not callable(begin_settle):
            raise BoundedC2HardwareRuntimeError(
                "RH56 challenged tracking requires the bounded final-settle owner API"
            )
        settle_started = self._now()
        # There are no more policy targets after the admitted final sequence.
        # Stop the arm through the already-audited native STOP/STOP_PROOF path
        # before any hand-only setup can consume the native 500 ms
        # inter-target window.  This avoids both weakening that watchdog and
        # fabricating an extra action sequence while the slower hand settles.
        self.franka_session.request_clean_stop()
        owner_deadline = _finite_float(
            begin_settle(),
            "RH56 final tracking settle deadline",
        )
        while True:
            if (
                self._stop_requested.is_set()
                or external_stop_requested.is_set()
            ):
                raise BoundedC2HardwareRuntimeError(
                    "stop requested during final RH56 physical-tracking settle"
                )
            error = self._franka_failure()
            if error is not None:
                raise BoundedC2HardwareRuntimeError(
                    "Franka owner failed during final RH56 physical-tracking "
                    f"settle: {error}"
                ) from error
            if self._franka_thread is not None and self._franka_done.is_set():
                telemetry = self.franka_session.last_telemetry
                franka_stop_verified = bool(
                    telemetry is not None
                    and telemetry.stop_requested
                    and telemetry.stop_verified
                )
                if not franka_stop_verified:
                    raise BoundedC2HardwareRuntimeError(
                        "Franka did not return a verified clean stop during "
                        "final RH56 physical-tracking settle"
                    )
            self._require_rh56_healthy()
            snapshot = self._rh56_physical_tracking_audit_snapshot()
            if snapshot is None:
                raise BoundedC2HardwareRuntimeError(
                    "RH56 physical-tracking evidence disappeared during final settle"
                )
            verdict = str(snapshot.get("verdict", "not_exercised"))
            # STOP was already issued.  If STOP_PROOF has arrived, it was
            # validated above; otherwise the unconditional stop_and_verify()
            # cleanup owns its longer join/proof deadline.  Do not couple that
            # proof latency to RH56's shorter physical-progress deadline.
            if verdict == "verified":
                return snapshot, max(0.0, self._now() - settle_started)
            if verdict == "failed":
                raise BoundedC2HardwareRuntimeError(
                    "RH56 physical tracking failed during final settle"
                )
            if verdict != "challenged_unverified":
                raise BoundedC2HardwareRuntimeError(
                    "RH56 physical tracking lost its challenged state during "
                    f"final settle: verdict={verdict}"
                )
            now = self._now()
            if now >= hard_deadline_monotonic_s:
                raise BoundedC2HardwareRuntimeError(
                    "bounded run deadline expired during final RH56 "
                    "physical-tracking settle"
                )
            if now >= owner_deadline:
                # Never convert an elapsed, unverified challenge into PASS.
                # The owner has the same absolute bound and will take its
                # verified disable path; this runtime edge also latches both
                # stop paths if it observes the boundary first.
                raise BoundedC2HardwareRuntimeError(
                    "RH56 final physical-tracking settle expired without "
                    "verified motion"
                )
            self._require_live_boundary(now)
            self._sleep(
                min(
                    self._commit_poll_interval_s,
                    hard_deadline_monotonic_s - now,
                    owner_deadline - now,
                )
            )

    def _now(self) -> float:
        return _finite_float(self._monotonic(), "monotonic clock")

    def _require_live_boundary(self, now: float) -> None:
        self.admission.require_active(now_monotonic_s=now)
        self._live_safety_supervisor.require_motion(
            self.admission,
            now_monotonic_s=now,
        )
        # Do not trust a supervisor return as a substitute for the shared gate.
        self.admission.safety_gate.require_motion(
            run_id=self.admission.run_id,
            now_monotonic_s=now,
        )

    def _require_declared_safety_contract(self) -> None:
        names = (
            (
                "operator_supervision_confirmed",
                "independent_command_watchdogs_configured",
                "verified_dual_device_stop_supported",
            )
            if self._supervised_non_c2
            else (
                "physical_interlocks_configured",
                "independent_command_watchdogs_configured",
                "verified_dual_device_stop_supported",
            )
        )
        for name in names:
            if getattr(self, name) is not True:
                raise BoundedC2HardwareRuntimeError(
                    f"runtime safety contract is not active: {name}"
                )

    def _start_franka_owner(self) -> None:
        if self._franka_thread is not None:
            return

        def owner() -> None:
            try:
                telemetry = self.franka_session.run(
                    authorization=self.admission.authorization,
                    preflight_token=self.admission.franka_preflight_token,
                )
                with self._state_lock:
                    self._franka_telemetry = telemetry
            except BaseException as exc:
                with self._state_lock:
                    self._franka_error = exc
                    self._franka_telemetry = self.franka_session.last_telemetry
            finally:
                self._franka_done.set()

        thread = threading.Thread(
            target=owner,
            name=f"v94-franka-owner-{self.admission.run_id}",
            daemon=False,
        )
        self._franka_thread = thread
        thread.start()

    def _franka_failure(self) -> Optional[BaseException]:
        with self._state_lock:
            return self._franka_error

    def _rh56_fault_callback(self, error: object) -> None:
        """Immediate cross-owner stop edge invoked by the watchdog worker."""

        detail = str(error).strip() or "unspecified RH56 watchdog fault"
        with self._state_lock:
            # Retain the callback payload independently of the stop Event.
            # The Event is also used for clean external stop requests, so it
            # cannot by itself preserve whether this edge was a hardware
            # fault.  Production owners expose the same fault through
            # raise_if_faulted(); this retained copy closes the callback/check
            # race and keeps dependency-injected owners honest as well.
            if self._rh56_callback_fault is None:
                self._rh56_callback_fault = error
        self.admission.safety_gate.latch_fault(
            f"RH56 watchdog owner failed: {detail}"
        )
        self._stop_requested.set()
        self.franka_session.request_clean_stop()

    def _require_rh56_healthy(self) -> None:
        owner = self._rh56_watchdog_owner
        if owner is not None:
            owner.raise_if_faulted()
        with self._state_lock:
            callback_fault = self._rh56_callback_fault
        if callback_fault is not None:
            detail = (
                str(callback_fault).strip()
                or "unspecified RH56 watchdog fault"
            )
            raise BoundedC2HardwareRuntimeError(
                f"RH56 watchdog owner failed: {detail}"
            )

    def _start_rh56_watchdog_owner(self) -> None:
        factory = self._rh56_watchdog_owner_factory
        if factory is None:
            raise BoundedC2HardwareRuntimeError(
                "production RH56 watchdog owner factory is missing"
            )
        self._require_live_boundary(self._now())
        owner = factory(
            admission=self.admission,
            action_ledger=self.action_ledger,
            fault_callback=self._rh56_fault_callback,
        )
        if not isinstance(owner, ManagedRH56WatchdogOwner):
            raise BoundedC2HardwareRuntimeError(
                "RH56 owner factory returned an invalid watchdog owner"
            )
        self._rh56_watchdog_owner = owner
        self._require_live_boundary(self._now())
        owner.start()
        if owner.independent_watchdog_active is not True:
            raise BoundedC2HardwareRuntimeError(
                "RH56 owner did not activate its independent watchdog"
            )
        if owner.fault_callback_configured is not True:
            raise BoundedC2HardwareRuntimeError(
                "RH56 owner has no immediate Franka-stop fault callback"
            )
        if (
            self._supervised_non_c2
            and owner.first_command_readiness_verified is not True
        ):
            raise BoundedC2HardwareRuntimeError(
                "RH56 owner returned before complete-feedback readiness"
            )
        owner.raise_if_faulted()

    def _wait_for_franka_bootstrap(
        self,
        *,
        hard_deadline_monotonic_s: float,
        external_stop_requested: threading.Event,
    ) -> None:
        while True:
            self._require_rh56_healthy()
            error = self._franka_failure()
            if error is not None:
                raise BoundedC2HardwareRuntimeError(
                    f"Franka measured-hold bootstrap failed: {error}"
                ) from error
            if self._franka_done.is_set():
                raise BoundedC2HardwareRuntimeError(
                    "Franka owner stopped during measured-hold bootstrap"
                )
            if self._stop_requested.is_set() or external_stop_requested.is_set():
                raise BoundedC2HardwareRuntimeError(
                    "stop requested during measured-hold bootstrap"
                )
            samples = self.franka_session.pose_ring.snapshot()
            if self.franka_session.c2_bootstrap_ready and len(samples) >= 2:
                if not self._supervised_non_c2:
                    return
                # Idle readback may report success_rate=0 before the first
                # active write.  Keep measured-q hold active for 100 ms; the
                # session begins enforcing >=0.90 after that warm-up, so no
                # supervised policy target can be emitted on startup zeros.
                if samples[-1].monotonic_s - samples[0].monotonic_s >= 0.10:
                    return
            now = self._now()
            if now >= hard_deadline_monotonic_s:
                raise BoundedC2HardwareRuntimeError(
                    "bounded run deadline expired during Franka bootstrap"
                )
            self._require_live_boundary(now)
            self._sleep(self._commit_poll_interval_s)

    def _open_and_arm_rh56(self) -> None:
        if self._rh56_transport_factory is None or self._rh56_actuator_factory is None:
            raise BoundedC2HardwareRuntimeError("legacy fake RH56 path is incomplete")
        now = self._now()
        self._require_live_boundary(now)
        transport = self._rh56_transport_factory()
        if not isinstance(transport, ManagedRH56TransactionalTransport):
            raise BoundedC2HardwareRuntimeError(
                "RH56 factory did not return a managed transactional transport"
            )
        if transport.is_open:
            raise BoundedC2HardwareRuntimeError(
                "RH56 transport factory violated inert-construction boundary"
            )
        if transport.hardware_backed:
            raise BoundedC2HardwareRuntimeError(
                "hardware-backed RH56 requires the dedicated watchdog-owner path"
            )
        self._rh56_transport = transport

        # This is the final no-device boundary.  The call below is the first
        # permitted serial open/import/discovery point.
        self._require_live_boundary(self._now())
        opened = transport.open()
        if opened is not transport or not transport.is_open:
            raise BoundedC2HardwareRuntimeError(
                "RH56 transport open did not preserve exact ownership"
            )
        self._require_live_boundary(self._now())

        actuator = self._rh56_actuator_factory(transport, self.admission)
        if not isinstance(actuator, RH56TransactionalActuator):
            raise BoundedC2HardwareRuntimeError(
                "RH56 actuator factory returned an invalid actuator"
            )
        if actuator.state is not RH56ActuatorState.DISARMED:
            raise BoundedC2HardwareRuntimeError(
                "RH56 actuator did not start disarmed"
            )
        if actuator.hardware_backed is not transport.hardware_backed:
            raise BoundedC2HardwareRuntimeError(
                "RH56 actuator/transport hardware metadata differs"
            )
        if actuator.baud_rate != transport.baud_rate:
            raise BoundedC2HardwareRuntimeError(
                "RH56 actuator/transport baud metadata differs"
            )
        if not math.isclose(
            actuator.command_watchdog_timeout_s,
            self.admission.rh56_preflight.verified_command_watchdog_timeout_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise BoundedC2HardwareRuntimeError(
                "RH56 actuator watchdog is not bound to admission"
            )
        actuator.claim_for_current_thread()
        arm_method = (
            actuator.arm_supervised
            if self._supervised_non_c2
            else actuator.arm
        )
        arm_method(
            preflight=self.admission.rh56_preflight,
            authorization=self.admission.authorization,
            safety_gate=self.admission.safety_gate,
            run_id=self.admission.run_id,
            now_monotonic_s=self._now(),
        )
        self._rh56_actuator = actuator

    def _wait_for_dual_commit(
        self,
        command: ClosedLoopCommand,
        *,
        hard_deadline_monotonic_s: float,
        external_stop_requested: threading.Event,
    ) -> None:
        while self.action_ledger.last_committed_sequence < command.sequence:
            error = self._franka_failure()
            if error is not None:
                raise BoundedC2HardwareRuntimeError(
                    f"Franka owner failed before dual ACK: {error}"
                ) from error
            self._require_rh56_healthy()
            if self._franka_done.is_set():
                raise BoundedC2HardwareRuntimeError(
                    "Franka owner stopped before dual ACK"
                )
            if self._stop_requested.is_set() or external_stop_requested.is_set():
                raise BoundedC2HardwareRuntimeError(
                    "stop requested while a policy tick was partially acknowledged"
                )
            now = self._now()
            if now >= hard_deadline_monotonic_s:
                raise BoundedC2HardwareRuntimeError(
                    "bounded run deadline expired before dual ACK"
                )
            self._require_live_boundary(now)
            self.action_ledger.require_commit_deadline(now_monotonic_s=now)
            self._sleep(self._commit_poll_interval_s)
        if self.action_ledger.last_committed_sequence != command.sequence:
            raise BoundedC2HardwareRuntimeError(
                "dual-ACK ledger sequence skipped the pending policy tick"
            )
        if self.action_ledger.pending_sequence is not None:
            raise BoundedC2HardwareRuntimeError(
                "dual-ACK ledger retained a pending command after commit"
            )

    def _wait_for_franka_ack_before_rh56(
        self,
        command: ClosedLoopCommand,
        *,
        hard_deadline_monotonic_s: float,
        external_stop_requested: threading.Event,
    ) -> None:
        """Gate supervised RH56 IO on the exact Franka sequence ACK.

        The production supervised path uses this asymmetric barrier so a
        rejected or faulted Franka target cannot still move the hand.  Formal
        C2 retains its existing concurrent dual-ACK semantics and never calls
        this method.
        """

        if not self._supervised_non_c2:
            raise BoundedC2HardwareRuntimeError(
                "Franka-first ACK barrier is supervised-only"
            )
        while True:
            error = self._franka_failure()
            if error is not None:
                raise BoundedC2HardwareRuntimeError(
                    f"Franka owner failed before RH56 command: {error}"
                ) from error
            self._require_rh56_healthy()
            if self._franka_done.is_set():
                raise BoundedC2HardwareRuntimeError(
                    "Franka owner stopped before RH56 command"
                )
            if self._stop_requested.is_set() or external_stop_requested.is_set():
                raise BoundedC2HardwareRuntimeError(
                    "stop requested before Franka ACK; RH56 command withheld"
                )
            if self.action_ledger.consumer_acknowledged(
                "franka",
                sequence=command.sequence,
            ):
                # Recheck asynchronous terminal edges after observing the ACK
                # and before returning command authority to the RH56 caller.
                error = self._franka_failure()
                if error is not None:
                    raise BoundedC2HardwareRuntimeError(
                        f"Franka owner failed after ACK and before RH56 command: {error}"
                    ) from error
                if self._franka_done.is_set():
                    raise BoundedC2HardwareRuntimeError(
                        "Franka owner stopped after ACK and before RH56 command"
                    )
                if (
                    self._stop_requested.is_set()
                    or external_stop_requested.is_set()
                ):
                    raise BoundedC2HardwareRuntimeError(
                        "stop requested after Franka ACK; RH56 command withheld"
                    )
                self._require_rh56_healthy()
                self._require_live_boundary(self._now())
                return
            now = self._now()
            if now >= hard_deadline_monotonic_s:
                raise BoundedC2HardwareRuntimeError(
                    "bounded run deadline expired before Franka ACK; "
                    "RH56 command withheld"
                )
            self._require_live_boundary(now)
            self.action_ledger.require_commit_deadline(now_monotonic_s=now)
            self._sleep(self._commit_poll_interval_s)

    def _wait_for_policy_release(
        self,
        not_before_monotonic_s: float,
        *,
        hard_deadline_monotonic_s: float,
        external_stop_requested: threading.Event,
    ) -> bool:
        """Prevent catch-up bursts from running policy faster than training."""

        while True:
            error = self._franka_failure()
            if error is not None:
                raise BoundedC2HardwareRuntimeError(
                    f"Franka owner failed before next policy release: {error}"
                ) from error
            self._require_rh56_healthy()
            if self._stop_requested.is_set():
                # RH56's asynchronous callback sets this Event after
                # publishing its fault.  Recheck both owners after observing
                # the Event to close the race where the callback arrives
                # between the checks above and this branch.
                error = self._franka_failure()
                if error is not None:
                    raise BoundedC2HardwareRuntimeError(
                        "Franka owner failed before next policy release: "
                        f"{error}"
                    ) from error
                self._require_rh56_healthy()
                # Direct request_stop() without an owner fault is a clean
                # external stop API and therefore still returns normally.
                return False
            if external_stop_requested.is_set():
                # A genuine operator/SIGINT stop remains a normal bounded
                # early return.  The internal edge is checked first so a
                # simultaneous hardware fault cannot be rewritten as Ctrl+C.
                return False
            if self._franka_thread is not None and self._franka_done.is_set():
                raise BoundedC2HardwareRuntimeError(
                    "Franka owner stopped before next policy release"
                )
            now = self._now()
            self._require_no_stage_policy_hold_within_bound(now)
            if now >= hard_deadline_monotonic_s:
                raise BoundedC2HardwareRuntimeError(
                    "bounded run deadline expired before policy release"
                )
            if now >= not_before_monotonic_s:
                return True
            self._require_live_boundary(now)
            self._sleep(
                min(
                    self._commit_poll_interval_s,
                    not_before_monotonic_s - now,
                )
            )

    def _advance_phase_locked_policy_release(
        self,
        scheduled_release_monotonic_s: float,
        *,
        policy_period_s: float,
        completed_at_monotonic_s: float,
    ) -> float:
        """Advance an absolute policy grid without drift or catch-up bursts."""

        scheduled = _finite_float(
            scheduled_release_monotonic_s,
            "scheduled_release_monotonic_s",
        )
        period = _positive_float(policy_period_s, "policy_period_s")
        completed = _finite_float(
            completed_at_monotonic_s,
            "completed_at_monotonic_s",
        )
        lateness = max(0.0, completed - scheduled)
        next_release = scheduled + period
        skipped = 0
        if next_release <= completed:
            skipped = int(math.floor((completed - next_release) / period)) + 1
            next_release += skipped * period
        with self._state_lock:
            self._policy_schedule_skipped_slot_count += skipped
            self._policy_schedule_max_release_lateness_s = max(
                self._policy_schedule_max_release_lateness_s,
                lateness,
            )
        return next_release

    def _wait_for_final_franka_arrival(
        self,
        source: BoundedV94PolicyTickSource,
        *,
        hard_deadline_monotonic_s: float,
        external_stop_requested: threading.Event,
    ) -> Mapping[str, Any]:
        """Prove the last target arrived for diagnostic arrival-gated runs."""

        checker = getattr(source, "check_franka_arrival_gate", None)
        enabled = bool(getattr(source, "franka_arrival_gate_enabled", False))
        if not enabled or not callable(checker):
            return {"enabled": False, "arrived": None, "wait_s": 0.0}
        started = self._now()
        while True:
            error = self._franka_failure()
            if error is not None:
                raise BoundedC2HardwareRuntimeError(
                    f"Franka owner failed during final arrival gate: {error}"
                ) from error
            self._require_rh56_healthy()
            if self._stop_requested.is_set() or external_stop_requested.is_set():
                raise BoundedC2HardwareRuntimeError(
                    "stop requested during final Franka arrival gate"
                )
            if self._franka_thread is not None and self._franka_done.is_set():
                raise BoundedC2HardwareRuntimeError(
                    "Franka owner stopped during final arrival gate"
                )
            now = self._now()
            if now >= hard_deadline_monotonic_s:
                raise BoundedC2HardwareRuntimeError(
                    "bounded run deadline expired during final Franka arrival gate"
                )
            self._require_live_boundary(now)
            poses = self.franka_session.pose_ring.snapshot()
            if poses and checker(
                poses[-1].q_rad,
                now_monotonic_s=now,
            ):
                diagnostics = getattr(source, "diagnostics_snapshot", {})
                gate = (
                    diagnostics.get("franka_arrival_gate", {})
                    if isinstance(diagnostics, Mapping)
                    else {}
                )
                return {
                    "enabled": True,
                    "arrived": True,
                    "wait_s": max(0.0, self._now() - started),
                    "last_error_rad": (
                        gate.get("last_error_rad")
                        if isinstance(gate, Mapping)
                        else None
                    ),
                }
            self._sleep(self._commit_poll_interval_s)

    def _abort_pending(self, reason: str) -> None:
        command = self._pending_command
        if command is None:
            return
        try:
            source = self._policy_tick_source
            if source is not None:
                source.abort(command, reason=reason)
        except BaseException:
            pass
        if self.action_ledger.pending_sequence == command.sequence:
            failed_consumer = "rh56"
            if self._supervised_non_c2:
                try:
                    if not self.action_ledger.consumer_acknowledged(
                        "franka",
                        sequence=command.sequence,
                    ):
                        # The Franka-first supervised barrier withheld RH56;
                        # preserve that provenance instead of blaming a hand
                        # transaction that was never attempted.
                        failed_consumer = "franka"
                except ClosedLoopProtocolError:
                    # A device owner may already have latched the more exact
                    # first fault. ExecutedActionLedger.fail() preserves it.
                    pass
            try:
                self.action_ledger.fail(
                    failed_consumer,
                    sequence=command.sequence,
                    reason=reason,
                )
            except BaseException:
                pass

    def run(
        self,
        *,
        maximum_policy_steps: int,
        hard_deadline_monotonic_s: float,
        stop_requested: threading.Event,
    ) -> Mapping[str, Any]:
        with self._state_lock:
            if self._run_started:
                raise BoundedC2HardwareRuntimeError("bounded runtime is single-use")
            self._run_started = True
        maximum_steps = _positive_integer(
            maximum_policy_steps, "maximum_policy_steps"
        )
        hard_deadline = _finite_float(
            hard_deadline_monotonic_s, "hard_deadline_monotonic_s"
        )
        if maximum_steps != self.admission.requested_policy_steps:
            raise BoundedC2HardwareRuntimeError(
                "runtime policy-step bound differs from sealed admission"
            )
        if not math.isclose(
            hard_deadline,
            self.admission.hard_deadline_monotonic_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise BoundedC2HardwareRuntimeError(
                "runtime hard deadline differs from sealed admission"
            )

        try:
            # Repeat the orchestrator check so calling this reusable runtime
            # directly cannot bypass the final no-device safety contract.
            self._require_declared_safety_contract()
            self._require_live_boundary(self._now())
            source_factory = self._policy_tick_source_factory
            if source_factory is None:
                self._open_and_arm_rh56()
            else:
                # Camera initialization/warm-up can be expensive, but it has
                # no robot command authority.  Complete it before either
                # actuator owner enters its active session.
                source_factory.open_and_warm_camera(
                    self.admission,
                    hard_deadline_monotonic_s=hard_deadline,
                )
                self._require_live_boundary(self._now())
                prepare_trigger = getattr(
                    source_factory, "prepare_rollout_trigger", None
                )
                detect_trigger = getattr(
                    source_factory, "detect_rollout_trigger", None
                )
                split_trigger = bool(
                    callable(prepare_trigger) and callable(detect_trigger)
                    and getattr(
                        source_factory, "rollout_trigger_config", None
                    ) is not None
                )
                wait_for_trigger = getattr(
                    source_factory, "wait_for_rollout_trigger", None
                )
                if split_trigger:
                    prepare_trigger(
                        self.admission,
                        hard_deadline_monotonic_s=hard_deadline,
                        stop_requested=stop_requested,
                    )
                    self._require_live_boundary(self._now())
                elif callable(wait_for_trigger):
                    wait_for_trigger(
                        self.admission,
                        hard_deadline_monotonic_s=hard_deadline,
                        stop_requested=stop_requested,
                    )
                    self._require_live_boundary(self._now())
                owner_preparation_started_s = self._now()
                first_target_handoff_deadline = hard_deadline
                if self._supervised_non_c2 and split_trigger:
                    # Native ACTION_READY permits five seconds before the
                    # first numeric target.  Finish static owner preparation,
                    # policy warmup and release detection within four seconds
                    # measured from before owner start, retaining at least one
                    # second for the first inference/dual-ACK transaction.
                    first_target_handoff_deadline = min(
                        hard_deadline, owner_preparation_started_s + 4.0
                    )
                self._start_franka_owner()
                self._wait_for_franka_bootstrap(
                    hard_deadline_monotonic_s=first_target_handoff_deadline,
                    external_stop_requested=stop_requested,
                )
                # Do not spend any of RH56's strict 50 ms bootstrap heartbeat
                # on FCI startup or camera/pose alignment.  Franka is already
                # held at measured q while the next usable camera frame is
                # selected.
                alignment_deadline = first_target_handoff_deadline
                if self._supervised_non_c2:
                    # Native ACTION_READY starts a 5 s first-target watchdog.
                    # Reserve at most 1 s for a fresh aligned D435 frame, then
                    # leave the RH56 owner's bounded 3 s startup plus margin
                    # for source construction and the first policy inference.
                    alignment_deadline = min(
                        first_target_handoff_deadline, self._now() + 1.0
                    )
                source_factory.wait_for_camera_pose_alignment(
                    franka_session=self.franka_session,
                    hard_deadline_monotonic_s=alignment_deadline,
                )
                self._require_live_boundary(self._now())
                self._start_rh56_watchdog_owner()
                assert self._rh56_watchdog_owner is not None
                source = source_factory(
                    franka_session=self.franka_session,
                    rh56_feedback_source=self._rh56_watchdog_owner,
                    hard_deadline_monotonic_s=first_target_handoff_deadline,
                )
                if not isinstance(source, BoundedV94PolicyTickSource):
                    raise BoundedC2HardwareRuntimeError(
                        "late-bound policy factory returned an invalid source"
                    )
                self._policy_tick_source = source
                warmup = getattr(source, "warmup_before_first_action", None)
                if callable(warmup):
                    # Startup perception recovery and the first CUDA/model
                    # forward are completed before the formal no-stage hold
                    # clock begins.  The warmup source contract guarantees no
                    # policy history, mapper target, replay index, ledger, or
                    # actuator command is staged by this call.
                    warmup(
                        hard_deadline_monotonic_s=(
                            first_target_handoff_deadline
                        )
                    )
                    self._require_rh56_healthy()
                    self._require_live_boundary(self._now())
                if split_trigger:
                    detect_trigger(
                        self.admission,
                        hard_deadline_monotonic_s=(
                            first_target_handoff_deadline
                        ),
                        stop_requested=stop_requested,
                    )
                    self._require_rh56_healthy()
                    self._require_live_boundary(self._now())
            source = self._policy_tick_source
            if source is None:
                raise BoundedC2HardwareRuntimeError("policy tick source is missing")
            policy_period_s = 1.0 / self.admission.policy_rate_hz
            phase_locked_20hz = math.isclose(
                self.admission.policy_rate_hz,
                20.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            with self._state_lock:
                self._policy_schedule_mode = (
                    "phase_locked_no_catch_up"
                    if phase_locked_20hz
                    else "legacy_minimum_period"
                )
            next_policy_not_before = self._now()
            startup_steps = int(
                getattr(source, "startup_non_actuated_policy_steps", 0)
            )
            if startup_steps < 0:
                raise BoundedC2HardwareRuntimeError(
                    "startup non-actuated policy-step count cannot be negative"
                )
            prime_startup = getattr(
                source, "prime_non_actuated_startup_tick", None
            )
            if startup_steps > 0 and not callable(prime_startup):
                raise BoundedC2HardwareRuntimeError(
                    "policy requests non-actuated startup without a prime hook"
                )
            startup_complete = startup_steps == 0
            while self._startup_non_actuated_completed_steps < startup_steps:
                scheduled_release = next_policy_not_before
                if not self._wait_for_policy_release(
                    scheduled_release,
                    hard_deadline_monotonic_s=hard_deadline,
                    external_stop_requested=stop_requested,
                ):
                    break
                startup_now = self._now()
                self._require_no_stage_policy_hold_within_bound(startup_now)
                self._require_live_boundary(startup_now)
                assert callable(prime_startup)
                primed = prime_startup(
                    logical_index=(
                        self._startup_non_actuated_completed_steps
                    ),
                    now_monotonic_s=startup_now,
                    hard_deadline_monotonic_s=hard_deadline,
                )
                if isinstance(primed, BoundedPolicyTickHold):
                    if self.action_ledger.last_committed_sequence != 0:
                        raise BoundedC2HardwareRuntimeError(
                            "startup policy hold changed the hardware ledger"
                        )
                    self._record_no_stage_policy_hold(primed)
                    self._require_no_stage_policy_hold_within_bound(self._now())
                else:
                    if int(getattr(primed, "logical_index", -1)) != (
                        self._startup_non_actuated_completed_steps
                    ):
                        raise BoundedC2HardwareRuntimeError(
                            "startup policy tick returned the wrong logical index"
                        )
                    if (
                        self.action_ledger.last_committed_sequence != 0
                        or self.action_ledger.pending_sequence is not None
                    ):
                        raise BoundedC2HardwareRuntimeError(
                            "non-actuated startup mutated the hardware ledger"
                        )
                    self._close_no_stage_policy_hold_streak(
                        float(getattr(primed, "produced_monotonic_s"))
                    )
                    self._startup_non_actuated_completed_steps += 1
                next_policy_not_before = (
                    self._advance_phase_locked_policy_release(
                        scheduled_release,
                        policy_period_s=policy_period_s,
                        completed_at_monotonic_s=self._now(),
                    )
                )
            startup_complete = (
                self._startup_non_actuated_completed_steps == startup_steps
            )
            final_hold_not_before: Optional[float] = None
            replay_source_complete = False
            while (
                startup_complete
                and self.action_ledger.last_committed_sequence < maximum_steps
            ):
                scheduled_release = next_policy_not_before
                if not self._wait_for_policy_release(
                    scheduled_release,
                    hard_deadline_monotonic_s=hard_deadline,
                    external_stop_requested=stop_requested,
                ):
                    break
                now = self._now()
                self._require_no_stage_policy_hold_within_bound(now)
                self._require_live_boundary(now)
                committed_before = self.action_ledger.committed_snapshot()
                sequence = committed_before.sequence + 1
                previous = committed_before.executed_policy_action13
                prepared = source.prepare(
                    sequence=sequence,
                    previous_executed_action13=previous,
                    now_monotonic_s=now,
                    hard_deadline_monotonic_s=hard_deadline,
                )
                if isinstance(prepared, BoundedPolicyTickHold):
                    if prepared.sequence != sequence:
                        raise BoundedC2HardwareRuntimeError(
                            "policy hold sequence differs from runtime"
                        )
                    if self.action_ledger.pending_sequence is not None:
                        raise BoundedC2HardwareRuntimeError(
                            "policy hold occurred while an action was staged"
                        )
                    if (
                        self._supervised_non_c2
                        and self._rh56_watchdog_owner is not None
                    ):
                        self._require_rh56_healthy()
                    committed_after = self.action_ledger.committed_snapshot()
                    if (
                        committed_after.sequence != committed_before.sequence
                        or not np.array_equal(
                            committed_after.executed_policy_action13,
                            committed_before.executed_policy_action13,
                        )
                    ):
                        raise BoundedC2HardwareRuntimeError(
                            "no-stage policy hold changed the committed ledger"
                        )
                    self._record_no_stage_policy_hold(prepared)
                    hold_checked_at = self._now()
                    self._require_no_stage_policy_hold_within_bound(
                        hold_checked_at
                    )
                    if phase_locked_20hz:
                        # Missing perception does not move the 20 Hz clock.
                        # Keep the prior actuator target and retry on the next
                        # absolute slot rather than polling until a camera
                        # frame arrives and stretching this tick toward 100 ms.
                        next_policy_not_before = (
                            self._advance_phase_locked_policy_release(
                                scheduled_release,
                                policy_period_s=policy_period_s,
                                completed_at_monotonic_s=hold_checked_at,
                            )
                        )
                    else:
                        hold_deadline = (
                            self._no_stage_policy_hold_deadline_monotonic_s()
                        )
                        if hold_deadline is None:
                            raise BoundedC2HardwareRuntimeError(
                                "no-stage policy hold streak disappeared before retry"
                            )
                        next_policy_not_before = max(
                            hold_checked_at,
                            min(
                                prepared.retry_not_before_monotonic_s,
                                hold_deadline,
                            ),
                        )
                    continue
                command = prepared
                if not isinstance(command, ClosedLoopCommand):
                    raise BoundedC2HardwareRuntimeError(
                        "policy source returned no ClosedLoopCommand"
                    )
                if command.sequence != sequence:
                    raise BoundedC2HardwareRuntimeError(
                        "policy source command sequence differs from runtime"
                    )
                self._close_no_stage_policy_hold_streak(
                    command.produced_monotonic_s
                )
                if not np.array_equal(
                    command.previous_executed_action13_used, previous
                ):
                    raise BoundedC2HardwareRuntimeError(
                        "policy observation did not use exact dual-committed action"
                    )
                if (
                    self._supervised_non_c2
                    and self._rh56_watchdog_owner is not None
                ):
                    # Feedback and targets share one owner thread, so they can
                    # never overlap on the serial descriptor.  Recheck health
                    # before publishing the paired Franka target.
                    self._require_rh56_healthy()
                    self._require_live_boundary(self._now())
                if sequence == 1 and self._supervised_non_c2:
                    # This is the operator-visible motion boundary.  Camera,
                    # Franka/RH56 owners, bootstrap/alignment, first policy
                    # inference and all health checks are already complete.
                    # The very next operations stage and publish this first
                    # command, so the message is not emitted seconds early by
                    # the outer setup layer.
                    trigger_diagnostics = getattr(
                        source_factory, "rollout_trigger_diagnostics", {}
                    )
                    detected_at = None
                    if isinstance(trigger_diagnostics, Mapping):
                        detected_at = trigger_diagnostics.get(
                            "detected_monotonic_s"
                        )
                    trigger_suffix = ""
                    if detected_at is not None:
                        detected_value = _finite_float(
                            detected_at, "rollout trigger detection time"
                        )
                        self._rollout_trigger_to_first_action_s = max(
                            0.0, self._now() - detected_value
                        )
                        trigger_suffix = (
                            "; trigger_to_first_action="
                            f"{self._rollout_trigger_to_first_action_s:.3f}s"
                        )
                    print(
                        "[Rollout START] "
                        f"steps={maximum_steps} "
                        f"rate={self.admission.policy_rate_hz:g}Hz; "
                        "first action ready"
                        f"{trigger_suffix}; Ctrl+C requests verified stop",
                        flush=True,
                    )
                self._pending_command = command
                self.action_ledger.stage(command, now_monotonic_s=self._now())
                self.target_hold.publish(
                    FrankaJointTarget.from_closed_loop_command(
                        command,
                        source=(
                            FrankaTargetSource.SUPERVISED_V94
                            if self._supervised_non_c2
                            else FrankaTargetSource.C2_V94_POLICY
                        ),
                    )
                )
                if sequence == 1 and source_factory is None:
                    self._start_franka_owner()
                if self._supervised_non_c2:
                    self._wait_for_franka_ack_before_rh56(
                        command,
                        hard_deadline_monotonic_s=hard_deadline,
                        external_stop_requested=stop_requested,
                    )
                rh56_receipt = None
                if self._rh56_watchdog_owner is not None:
                    rh56_receipt = self._rh56_watchdog_owner.execute(command)
                    self._rh56_watchdog_owner.raise_if_faulted()
                else:
                    assert self._rh56_actuator is not None
                    rh56_receipt = self._rh56_actuator.execute(
                        command,
                        action_ledger=self.action_ledger,
                    )
                self._wait_for_dual_commit(
                    command,
                    hard_deadline_monotonic_s=hard_deadline,
                    external_stop_requested=stop_requested,
                )

                # The two hardware owners have now accepted this exact
                # command.  Persist its immutable execution evidence before
                # any policy bookkeeping, pose sampling or final-hold logic
                # can fail.  ``policy_source_commit_completed`` distinguishes
                # hardware execution from the subsequent model-state commit.
                committed_record = {
                    "sequence": sequence,
                    "produced_monotonic_s": command.produced_monotonic_s,
                    "observation_realtime_s": command.observation_realtime_s,
                    "previous_executed_action13_used": (
                        command.previous_executed_action13_used.tolist()
                    ),
                    "previous_policy_action13_used": (
                        command.previous_policy_action13_used.tolist()
                    ),
                    "raw_policy_action13": command.raw_policy_action13.tolist(),
                    "executed_policy_action13": (
                        command.executed_policy_action13.tolist()
                    ),
                    "franka_target_q_rad": command.franka_target_q_rad.tolist(),
                    "rh56_angle_set_register_order": (
                        command.rh56_angle_set_register_order.tolist()
                    ),
                    "rh56_write_response_received": getattr(
                        rh56_receipt,
                        "write_response_received",
                        None,
                    ),
                    "rh56_application_verified_by_readback": getattr(
                        rh56_receipt,
                        "application_verified_by_readback",
                        None,
                    ),
                    "rh56_numeric_write_performed": getattr(
                        rh56_receipt,
                        "numeric_write_performed",
                        None,
                    ),
                    "dual_ack_completed": True,
                    "policy_source_commit_completed": False,
                    "measured_franka_q_rad_at_commit": None,
                    "measured_rh56_angle_act_at_commit": None,
                    "measured_rh56_feedback_at_commit": None,
                }
                with self._state_lock:
                    self._committed_commands.append(committed_record)
                source.commit(
                    command,
                    action_ledger=self.action_ledger,
                )
                with self._state_lock:
                    if (
                        not self._committed_commands
                        or self._committed_commands[-1].get("sequence") != sequence
                    ):
                        raise BoundedC2HardwareRuntimeError(
                            "dual-ACK audit sequence was lost during source commit"
                        )
                    self._committed_commands[-1] = {
                        **self._committed_commands[-1],
                        "policy_source_commit_completed": True,
                    }
                if self.action_ledger.last_committed_sequence != sequence:
                    raise BoundedC2HardwareRuntimeError(
                        "policy source commit changed dual-ACK sequence"
                )
                poses = self.franka_session.pose_ring.snapshot()
                measured_q = None if not poses else poses[-1].q_rad.tolist()
                measured_hand_feedback = self._rh56_feedback_audit_snapshot()
                measured_hand = (
                    None
                    if measured_hand_feedback is None
                    else measured_hand_feedback.get("angles")
                )
                with self._state_lock:
                    if (
                        not self._committed_commands
                        or self._committed_commands[-1].get("sequence") != sequence
                    ):
                        raise BoundedC2HardwareRuntimeError(
                            "dual-ACK audit sequence was lost before measurement record"
                        )
                    self._committed_commands[-1] = {
                        **self._committed_commands[-1],
                        "measured_franka_q_rad_at_commit": measured_q,
                        "measured_rh56_angle_act_at_commit": measured_hand,
                        "measured_rh56_feedback_at_commit": measured_hand_feedback,
                    }
                if measured_hand_feedback is not None:
                    feedback_commit = getattr(
                        source,
                        "commit_rh56_feedback_after_dual_ack",
                        None,
                    )
                    if callable(feedback_commit):
                        feedback_commit(sequence, measured_hand_feedback)
                self._pending_command = None
                if phase_locked_20hz:
                    next_policy_not_before = (
                        self._advance_phase_locked_policy_release(
                            scheduled_release,
                            policy_period_s=policy_period_s,
                            completed_at_monotonic_s=self._now(),
                        )
                    )
                else:
                    next_policy_not_before = now + policy_period_s
                # A write ACK proves that the target entered the Franka
                # servo; it does not prove that the last policy target was
                # held for one training tick.  Remember a full post-ACK hold
                # boundary so a k=1 supervised run cannot stop immediately
                # after the shaper's first sub-millisecond write.
                final_hold_not_before = self._now() + policy_period_s
                replay_source_complete = bool(
                    getattr(source, "replay_complete", False)
                )
                if replay_source_complete:
                    break

            self._close_no_stage_policy_hold_streak(self._now())
            completed = self.action_ledger.last_committed_sequence
            if completed > maximum_steps:
                raise BoundedC2HardwareRuntimeError(
                    "dual-ACK ledger exceeded admitted k-step bound"
                )
            final_policy_hold_completed = False
            final_franka_arrival = {
                "enabled": False,
                "arrived": None,
                "wait_s": 0.0,
            }
            final_rh56_tracking_settle_s = 0.0
            rh56_physical_tracking: Optional[Mapping[str, Any]] = None
            if (
                self._supervised_non_c2
                and (completed == maximum_steps or replay_source_complete)
                and completed > 0
            ):
                assert final_hold_not_before is not None
                final_franka_arrival = self._wait_for_final_franka_arrival(
                    source,
                    hard_deadline_monotonic_s=hard_deadline,
                    external_stop_requested=stop_requested,
                )
                if not self._wait_for_policy_release(
                    final_hold_not_before,
                    hard_deadline_monotonic_s=hard_deadline,
                    external_stop_requested=stop_requested,
                ):
                    raise BoundedC2HardwareRuntimeError(
                        "stop requested during final "
                        f"{self.admission.policy_rate_hz:g} Hz policy-target hold"
                    )
                final_policy_hold_completed = True
                (
                    rh56_physical_tracking,
                    final_rh56_tracking_settle_s,
                ) = self._wait_for_final_rh56_physical_tracking(
                    hard_deadline_monotonic_s=hard_deadline,
                    external_stop_requested=stop_requested,
                )
                poses = self.franka_session.pose_ring.snapshot()
                final_measured_q = None if not poses else poses[-1].q_rad.tolist()
                final_hand_feedback = self._rh56_feedback_audit_snapshot()
                final_measured_hand = (
                    None
                    if final_hand_feedback is None
                    else final_hand_feedback.get("angles")
                )
                self._committed_commands[-1] = {
                    **self._committed_commands[-1],
                    "measured_franka_q_rad_after_final_hold": final_measured_q,
                    "measured_rh56_angle_act_after_final_hold": final_measured_hand,
                    "measured_rh56_feedback_after_final_hold": (
                        final_hand_feedback
                    ),
                }
            if rh56_physical_tracking is None:
                rh56_physical_tracking = (
                    self._rh56_physical_tracking_audit_snapshot()
                )
            challenged_axes = tuple(
                bool(item)
                for item in (
                    ()
                    if rh56_physical_tracking is None
                    else rh56_physical_tracking.get("challenged_axes", ())
                )
            )
            tracking_verdict = str(
                (
                    "not_exercised"
                    if rh56_physical_tracking is None
                    else rh56_physical_tracking.get(
                        "verdict", "not_exercised"
                    )
                )
            )
            if any(challenged_axes) and tracking_verdict != "verified":
                raise BoundedC2HardwareRuntimeError(
                    "RH56 physical tracking was challenged but not verified: "
                    f"verdict={tracking_verdict}"
                )
            diagnostics = self.runtime_diagnostics_snapshot
            result = {
                "completed_policy_steps": completed,
                "last_dual_ack_sequence": completed,
                "requested_policy_steps": maximum_steps,
                "stopped_early": completed < maximum_steps and not replay_source_complete,
                **diagnostics,
                "counting_basis": "executed_action_ledger_dual_ack_commit",
            }
            startup_records = getattr(
                source, "startup_non_actuated_records", ()
            )
            result["startup_non_actuated_policy_ticks"] = [
                {
                    "logical_index": int(record.logical_index),
                    "produced_monotonic_s": float(
                        record.produced_monotonic_s
                    ),
                    "camera_frame_id": int(record.camera_frame_id),
                    "previous_policy_action13_used": (
                        record.previous_policy_action13_used.tolist()
                    ),
                    "accepted_policy_action13": (
                        record.accepted_policy_action13.tolist()
                    ),
                    "hardware_command_staged": False,
                }
                for record in startup_records
            ]
            if final_franka_arrival.get("enabled") is True:
                result["final_franka_arrival_gate"] = final_franka_arrival
            if bool(getattr(source, "is_replay_source", False)):
                result.update(
                    {
                        "replay_source_complete": replay_source_complete,
                        "completed_replay_frames": int(
                            getattr(source, "completed_replay_frames", 0)
                        ),
                        "repeated_replay_target_commands": int(
                            getattr(source, "repeated_replay_target_commands", 0)
                        ),
                    }
                )
            if rh56_physical_tracking is not None:
                result["rh56_physical_tracking"] = rh56_physical_tracking
            if self._supervised_non_c2:
                result["final_policy_hold_s"] = policy_period_s
                result["final_policy_hold_completed"] = final_policy_hold_completed
                result["final_rh56_tracking_settle_s"] = (
                    final_rh56_tracking_settle_s
                )
                result["committed_commands"] = self.committed_commands_snapshot
            return result
        except BaseException as exc:
            # Preserve the elapsed streak through the first terminal fault.
            # Diagnostic clock failures must never replace the control fault.
            try:
                self._close_no_stage_policy_hold_streak(self._now())
            except BaseException:
                pass
            detail = f"{type(exc).__name__}: {exc}"
            pending_sequence = self.action_ledger.pending_sequence
            with self._state_lock:
                if (
                    self._failure_pending_command is None
                    and self._pending_command is not None
                    and self._pending_command.sequence == pending_sequence
                ):
                    self._failure_pending_command = self._pending_command
            self._abort_pending(detail)
            self.admission.safety_gate.latch_fault(
                f"bounded hardware runtime failed: {detail}"
            )
            self.request_stop("terminal bounded hardware runtime failure")
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, BoundedC2AdmissionError):
                raise
            raise BoundedC2HardwareRuntimeError(detail) from exc

    def request_stop(self, reason: str) -> None:
        del reason
        self._stop_requested.set()
        self.franka_session.request_clean_stop()
        owner = self._rh56_watchdog_owner
        if owner is not None:
            owner.request_stop()

    def stop_and_verify(self) -> BoundedC2StopProof:
        with self._state_lock:
            if self._stop_proof is not None:
                return self._stop_proof
        self.request_stop("verified dual-device stop")
        errors = []

        owner = self._rh56_watchdog_owner
        rh56_verified = self._rh56_actuator is None and owner is None
        if owner is not None:
            try:
                result = owner.stop_and_close(
                    timeout_s=self._franka_stop_join_timeout_s
                )
                rh56_verified = bool(
                    getattr(
                        result,
                        "rh56_disabled_verified",
                        False,
                    )
                )
                worker_stopped = bool(getattr(result, "worker_stopped", False))
                disable_attempted = bool(getattr(result, "disable_attempted", False))
                close_attempted = bool(getattr(result, "close_attempted", False))
                serial_closed = bool(getattr(result, "serial_closed", False))
                owner_fault = getattr(result, "fault", None)
                if self._supervised_non_c2:
                    if not worker_stopped:
                        errors.append("RH56 watchdog owner worker did not stop")
                    if not serial_closed:
                        errors.append("RH56 serial close or supervised setting restore failed")
                    if not disable_attempted:
                        errors.append("RH56 watchdog owner did not attempt disable")
                    if not close_attempted:
                        errors.append("RH56 watchdog owner did not attempt close")
                    if owner_fault is not None:
                        errors.append(f"RH56 watchdog owner fault: {owner_fault}")
                    rh56_verified = bool(
                        rh56_verified
                        and worker_stopped
                        and disable_attempted
                        and close_attempted
                        and serial_closed
                    )
                if not rh56_verified:
                    errors.append("RH56 watchdog owner did not verify disabled stop")
            except BaseException as exc:
                errors.append(f"RH56 watchdog-owner stop/close: {exc}")
                rh56_verified = False
        actuator = self._rh56_actuator
        if actuator is not None:
            try:
                if actuator.stop_confirmed:
                    rh56_verified = True
                else:
                    report = actuator.disable_and_verify()
                    rh56_verified = bool(report.verified)
            except BaseException as exc:
                errors.append(f"RH56 disable/stop verification: {exc}")
                rh56_verified = bool(actuator.stop_confirmed)

        thread = self._franka_thread
        franka_verified = thread is None
        if thread is not None:
            thread.join(timeout=self._franka_stop_join_timeout_s)
            if thread.is_alive():
                errors.append("Franka owner did not stop before join timeout")
                franka_verified = False
            else:
                telemetry = self.franka_session.last_telemetry
                with self._state_lock:
                    self._franka_telemetry = telemetry
                # Stop proof describes the physical stop transaction, not
                # whether the preceding run completed without a fault.  A
                # policy/timing failure still fails ``run()``, while a
                # successful unconditional stop remains truthfully proven.
                franka_verified = bool(
                    telemetry is not None
                    and telemetry.stop_requested
                    and telemetry.stop_verified
                )
                if not franka_verified:
                    error = self._franka_failure()
                    errors.append(
                        "Franka stop verification failed"
                        + ("" if error is None else f": {error}")
                    )

        transport = self._rh56_transport
        if transport is not None:
            try:
                transport.close()
            except BaseException as exc:
                errors.append(f"RH56 serial close: {exc}")
                rh56_verified = False

        source_factory = self._policy_tick_source_factory
        if source_factory is not None:
            try:
                source_factory.close()
            except BaseException as exc:
                errors.append(f"D435/policy source close: {exc}")

        proof = BoundedC2StopProof(
            franka_stop_verified=franka_verified,
            rh56_disabled_verified=rh56_verified,
        )
        with self._state_lock:
            self._stop_errors = tuple(errors)
            self._stop_proof = proof
        return proof


# Structural assertion for reviewers; it performs no runtime or device action.
_bounded_runtime_protocol_typecheck: BoundedC2Runtime


__all__ = [
    "BoundedC2HardwareRuntimeError",
    "BoundedC2LiveSafetySupervisor",
    "BoundedPolicyTickHold",
    "BoundedV94C2Runtime",
    "BoundedV94C2RuntimeFactory",
    "BoundedV94PolicyTickSource",
    "LinuxRH56TransportFactory",
    "ManagedRH56TransactionalTransport",
    "SupervisedV94Admission",
    "SupervisedV94RuntimeFactory",
]
