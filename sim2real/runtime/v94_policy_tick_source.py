"""Transactional V94 observation -> policy -> actuator-command wiring.

The source in this module has no hardware entry point and opens no device.  It
accepts an already-owned, side-effect-free policy observation snapshot,
rebuilds the checkpoint-owned four- or eight-frame history, runs the NumPy
rolling student, and
uses :class:`~sim2real.closed_loop_core.TransactionalV94ActionMapper` to form
one exact :class:`~sim2real.closed_loop_core.ClosedLoopCommand`.

Every mutable logical state is proposal/commit based:

* the newest proprio frame receives only the dual-committed action supplied by
  ``BoundedV94C2Runtime``;
* the four-frame history is a private candidate until both devices ACK;
* the action mapper already retains targets as a pending proposal; and
* normal hardware-tick cloud use is counted on dual commit, while the explicit
  non-actuated startup contract counts each accepted policy-only tick.

The packaged rolling student recreates its LSTM hidden/cell state from zero on
every call.  Its complete temporal state is therefore the external history
cloned here.  Arbitrary stateful policy wrappers and observation
providers that mutate logical continuity while taking a snapshot are rejected
at construction rather than treated as rollback-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Mapping, Optional, Protocol, Union, runtime_checkable

import numpy as np

from .bounded_c2_runtime import BoundedPolicyTickHold
from sim2real.closed_loop_core import (
    ClosedLoopCommand,
    ClosedLoopProtocolError,
    ExecutedActionLedger,
    TransactionalV94ActionMapper,
    TransactionalV94ActionProposal,
)
from sim2real.policy import (
    LEGACY_ACTION_CONTROLLER_CONTRACT_ID,
    QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
)
from sim2real.contracts.v94 import INITIAL_PREVIOUS_ACTION13
from sim2real.contracts.actions import (
    RH56HardwareCommandProposal,
    TransactionalRH56HardwareCommandShaper,
)

POLICY_TRANSACTIONAL_STATE_CONTRACT = (
    "external_four_frame_history_lstm_recomputed_from_zero_v1"
)
OBSERVATION_SNAPSHOT_CONTRACT = "side_effect_free_owned_policy_snapshot_v1"
CAMERA_REUSE_HOLD_REASON = "camera_frame_policy_reuse_limit"
OBJECT_POINTCLOUD_STALE_HOLD_REASON = "object_pointcloud_age_limit"
OBJECT_POINTCLOUD_INVALID_HOLD_REASON = "object_pointcloud_transient_invalid"
FRANKA_STATE_STALE_HOLD_REASON = "franka_state_action_age_limit"
FRANKA_ARRIVAL_GATE_HOLD_REASON = "franka_target_arrival_pending"
EXACT_REPLAY_ARM_HOLD_REASON = "exact_replay_arm_target_held"
QD_G015_STARTUP_NON_ACTUATED_POLICY_STEPS = 4


class V94PolicyTickSourceError(RuntimeError):
    """A terminal source, observation, inference, or transaction fault."""


class V94RecoverableObservationHold(RuntimeError):
    """A side-effect-free observation candidate that must stage no action."""

    def __init__(
        self,
        *,
        camera_frame_id: int,
        reason: str,
        fresh_actuator_snapshot: Optional["V94FreshActuatorSnapshot"] = None,
    ) -> None:
        self.camera_frame_id = _nonnegative_integer(camera_frame_id, "camera_frame_id")
        detail = str(reason).strip()
        if not detail:
            raise ValueError("recoverable observation hold reason is empty")
        if fresh_actuator_snapshot is not None and not isinstance(
            fresh_actuator_snapshot, V94FreshActuatorSnapshot
        ):
            raise TypeError(
                "fresh_actuator_snapshot must be V94FreshActuatorSnapshot"
            )
        self.reason = detail
        self.fresh_actuator_snapshot = fresh_actuator_snapshot
        super().__init__(detail)


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


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0.0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(numeric)


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _readonly_float_array(
    value: object, shape: tuple[int, ...], name: str
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have finite shape {shape}") from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have finite shape {shape}")
    result = result.copy()
    result.setflags(write=False)
    return result


def _readonly_policy_points(value: object, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (128,3) or (128,6)") from exc
    if (
        result.ndim != 2
        or result.shape[0] != 128
        or result.shape[1] not in (3, 6)
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(f"{name} must have finite shape (128,3) or (128,6)")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class V94FreshActuatorSnapshot:
    """Fresh robot state attached to an object-pointcloud no-stage hold."""

    measured_franka_q_rad: np.ndarray
    franka_state_captured_monotonic_s: float
    observation_realtime_s: float
    hold_arm_target: bool
    shaper_q_d_rad: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        measured = _readonly_float_array(
            self.measured_franka_q_rad,
            (7,),
            "fresh actuator measured_franka_q_rad",
        )
        captured = _finite_float(
            self.franka_state_captured_monotonic_s,
            "fresh actuator franka_state_captured_monotonic_s",
        )
        realtime = _finite_float(
            self.observation_realtime_s,
            "fresh actuator observation_realtime_s",
        )
        hold = _strict_bool(
            self.hold_arm_target, "fresh actuator hold_arm_target"
        )
        shaper = self.shaper_q_d_rad
        if shaper is not None:
            shaper = _readonly_float_array(
                shaper, (7,), "fresh actuator shaper_q_d_rad"
            )
        object.__setattr__(self, "measured_franka_q_rad", measured)
        object.__setattr__(
            self, "franka_state_captured_monotonic_s", captured
        )
        object.__setattr__(self, "observation_realtime_s", realtime)
        object.__setattr__(self, "hold_arm_target", hold)
        object.__setattr__(self, "shaper_q_d_rad", shaper)


@dataclass(frozen=True)
class V94PolicyObservation:
    """One immutable, policy-ready current observation without action feedback.

    ``proprio_prefix54`` is exactly indices ``0:54`` of the V94 proprio layout.
    The final 13 entries are intentionally absent so this source, rather than a
    camera/robot provider cache, must insert the ledger-committed action.

    Point-cloud/observation timestamps are host realtime/epoch values.
    ``franka_state_captured_monotonic_s`` uses the same host monotonic clock as
    command deadlines so freshness can be rechecked after policy inference.
    """

    pointcloud_xyzrgb_palm: np.ndarray
    pointcloud_valid: np.ndarray
    proprio_prefix54: np.ndarray
    measured_franka_q_rad: np.ndarray
    franka_state_captured_monotonic_s: float
    pointcloud_captured_realtime_s: float
    observation_realtime_s: float
    # ``camera_frame_id`` is the current formal D435 publication used for
    # transport liveness.  ``pointcloud_source_frame_id`` independently owns
    # content-reuse accounting.  During ``stale_palm`` the cloud deliberately
    # comes from an older native-palm capture, so an advancing transport frame
    # must not make unchanged policy content look new.
    camera_frame_id: int
    hold_arm_target: bool
    controller_state29: Optional[np.ndarray] = None
    # Current trajectory-generator desired position captured atomically with
    # measured q and controller_state29.  It is the arm-action reference for
    # the q_d-relative g015 contract, not a value reconstructed after policy
    # inference from a newer servo sample.
    shaper_q_d_rad: Optional[np.ndarray] = None
    pointcloud_status: str = "fresh"
    source_valid_points: int = 128
    pointcloud_source_frame_id: Optional[int] = None

    def __post_init__(self) -> None:
        points = _readonly_policy_points(
            self.pointcloud_xyzrgb_palm, "pointcloud_xyzrgb_palm"
        )
        valid = _readonly_float_array(self.pointcloud_valid, (128,), "pointcloud_valid")
        if np.any(valid < 0.0) or np.any(valid > 1.0):
            raise ValueError("pointcloud_valid must lie in [0,1]")
        if float(np.sum(valid)) < 1.0:
            raise ValueError("policy observation must contain a valid object point")
        prefix = _readonly_float_array(self.proprio_prefix54, (54,), "proprio_prefix54")
        measured_q = _readonly_float_array(
            self.measured_franka_q_rad, (7,), "measured_franka_q_rad"
        )
        franka_captured = _finite_float(
            self.franka_state_captured_monotonic_s,
            "franka_state_captured_monotonic_s",
        )
        captured = _finite_float(
            self.pointcloud_captured_realtime_s,
            "pointcloud_captured_realtime_s",
        )
        observed = _finite_float(self.observation_realtime_s, "observation_realtime_s")
        frame_id = _nonnegative_integer(self.camera_frame_id, "camera_frame_id")
        source_frame_id = (
            frame_id
            if self.pointcloud_source_frame_id is None
            else _nonnegative_integer(
                self.pointcloud_source_frame_id,
                "pointcloud_source_frame_id",
            )
        )
        hold_arm = _strict_bool(self.hold_arm_target, "hold_arm_target")
        controller_state = self.controller_state29
        if controller_state is not None:
            controller_state = _readonly_float_array(
                controller_state, (29,), "controller_state29"
            )
            if (
                np.any(controller_state[:14] < -4.0)
                or np.any(controller_state[:14] > 4.0)
                or np.any(controller_state[14:28] < -1.0)
                or np.any(controller_state[14:28] > 1.0)
                or controller_state[28] != np.float32(1.0)
            ):
                raise ValueError("controller_state29 violates the V258 contract")
        shaper_q_d = self.shaper_q_d_rad
        if shaper_q_d is not None:
            shaper_q_d = _readonly_float_array(
                shaper_q_d, (7,), "shaper_q_d_rad"
            )
        status = str(self.pointcloud_status).strip()
        if not status:
            raise ValueError("pointcloud_status must be non-empty")
        source_points = _nonnegative_integer(
            self.source_valid_points, "source_valid_points"
        )
        object.__setattr__(self, "pointcloud_xyzrgb_palm", points)
        object.__setattr__(self, "pointcloud_valid", valid)
        object.__setattr__(self, "proprio_prefix54", prefix)
        object.__setattr__(self, "measured_franka_q_rad", measured_q)
        object.__setattr__(self, "franka_state_captured_monotonic_s", franka_captured)
        object.__setattr__(self, "pointcloud_captured_realtime_s", captured)
        object.__setattr__(self, "observation_realtime_s", observed)
        object.__setattr__(self, "camera_frame_id", frame_id)
        object.__setattr__(self, "pointcloud_source_frame_id", source_frame_id)
        object.__setattr__(self, "hold_arm_target", hold_arm)
        object.__setattr__(self, "controller_state29", controller_state)
        object.__setattr__(self, "shaper_q_d_rad", shaper_q_d)
        object.__setattr__(self, "pointcloud_status", status)
        object.__setattr__(self, "source_valid_points", source_points)

    @property
    def point_features_palm(self) -> np.ndarray:
        """Neutral alias for XYZ and XYZRGB policy tensors."""

        return self.pointcloud_xyzrgb_palm

    @property
    def point_feature_dim(self) -> int:
        return int(self.pointcloud_xyzrgb_palm.shape[1])

    @property
    def point_feature_mode(self) -> str:
        return "xyz" if self.point_feature_dim == 3 else "xyzrgb"


@runtime_checkable
class SideEffectFreeV94ObservationProvider(Protocol):
    """Provider whose snapshot does not advance policy/velocity continuity."""

    transactional_snapshot_contract: str

    def snapshot(
        self,
        *,
        sequence: int,
        now_monotonic_s: float,
        hard_deadline_monotonic_s: float,
    ) -> V94PolicyObservation: ...


@runtime_checkable
class TransactionalStateV94Policy(Protocol):
    """Policy whose complete recurrent state is its supplied frame history."""

    transactional_state_contract: str

    def act(
        self,
        pointcloud: np.ndarray,
        valid: np.ndarray,
        proprio: np.ndarray,
    ) -> Any: ...


@dataclass(frozen=True)
class _HistoryCandidate:
    points: np.ndarray
    valid: np.ndarray
    proprio: np.ndarray


@dataclass(frozen=True)
class _PendingTick:
    command: ClosedLoopCommand
    mapper_proposal: TransactionalV94ActionProposal
    rh56_hardware_proposal: Optional[RH56HardwareCommandProposal]
    history: Optional[_HistoryCandidate]
    camera_frame_id: int
    pointcloud_source_frame_id: int
    measured_franka_q_rad: np.ndarray
    observation: Optional[V94PolicyObservation]
    ledger_previous_action13: np.ndarray
    occluded_closure_hand_only: bool = False
    occluded_closure_clearance_lift: bool = False
    frozen_post_closure_without_pointcloud: bool = False


@dataclass(frozen=True)
class NonActuatedPolicyTick:
    """One accepted policy tick that deliberately stages no hardware target."""

    logical_index: int
    produced_monotonic_s: float
    camera_frame_id: int
    previous_policy_action13_used: np.ndarray
    accepted_policy_action13: np.ndarray

    def __post_init__(self) -> None:
        index = _nonnegative_integer(self.logical_index, "logical_index")
        produced = _finite_float(
            self.produced_monotonic_s, "produced_monotonic_s"
        )
        frame_id = _nonnegative_integer(self.camera_frame_id, "camera_frame_id")
        previous = _readonly_float_array(
            self.previous_policy_action13_used,
            (13,),
            "previous_policy_action13_used",
        )
        accepted = _readonly_float_array(
            self.accepted_policy_action13,
            (13,),
            "accepted_policy_action13",
        )
        if np.any(accepted < -1.0) or np.any(accepted > 1.0):
            raise ValueError("accepted startup policy action must lie in [-1,1]")
        object.__setattr__(self, "logical_index", index)
        object.__setattr__(self, "produced_monotonic_s", produced)
        object.__setattr__(self, "camera_frame_id", frame_id)
        object.__setattr__(
            self, "previous_policy_action13_used", previous
        )
        object.__setattr__(self, "accepted_policy_action13", accepted)


class TransactionalBoundedV94PolicyTickSource:
    """Real-runtime tick source with dual-commit temporal/action semantics."""

    def __init__(
        self,
        *,
        observation_provider: SideEffectFreeV94ObservationProvider,
        policy: TransactionalStateV94Policy,
        action_mapper: TransactionalV94ActionMapper,
        rh56_hardware_command_shaper: Optional[
            TransactionalRH56HardwareCommandShaper
        ] = None,
        maximum_object_pointcloud_age_s: object = 0.20,
        maximum_pointcloud_future_skew_s: object = 0.010,
        maximum_franka_action_age_s: object = 0.025,
        maximum_franka_hard_age_s: object = 0.050,
        maximum_actions_per_camera_frame: int = 2,
        startup_non_actuated_policy_steps: object = 0,
        hold_retry_interval_s: object = 0.001,
        franka_arrival_gate_enabled: bool = False,
        franka_arrival_gate_tolerance_rad: object = 0.015,
        franka_arrival_gate_timeout_s: object = 0.350,
        policy_io_recorder: Optional[Any] = None,
        monotonic=time.monotonic,
        realtime=time.time,
    ) -> None:
        if not isinstance(observation_provider, SideEffectFreeV94ObservationProvider):
            raise TypeError(
                "observation_provider must implement SideEffectFreeV94ObservationProvider"
            )
        if (
            observation_provider.transactional_snapshot_contract
            != OBSERVATION_SNAPSHOT_CONTRACT
        ):
            raise ValueError(
                "observation provider is not certified side-effect-free for proposals"
            )
        if not isinstance(policy, TransactionalStateV94Policy):
            raise TypeError("policy must implement TransactionalStateV94Policy")
        if policy.transactional_state_contract != POLICY_TRANSACTIONAL_STATE_CONTRACT:
            raise ValueError("policy recurrent state cannot be rolled back safely")
        if not isinstance(action_mapper, TransactionalV94ActionMapper):
            raise TypeError("action_mapper must be TransactionalV94ActionMapper")
        if action_mapper.last_committed_sequence != 0 or action_mapper.pending_sequence:
            raise ValueError("action_mapper must be a fresh, uncommitted instance")
        if rh56_hardware_command_shaper is not None:
            if not isinstance(
                rh56_hardware_command_shaper,
                TransactionalRH56HardwareCommandShaper,
            ):
                raise TypeError("RH56 hardware command shaper has an invalid type")
            if (
                rh56_hardware_command_shaper.last_committed_sequence != 0
                or rh56_hardware_command_shaper.pending_sequence is not None
            ):
                raise ValueError(
                    "RH56 hardware command shaper must be fresh and uncommitted"
                )
        if not callable(monotonic) or not callable(realtime):
            raise TypeError("source clocks must be callable")
        if policy_io_recorder is not None and not callable(
            getattr(policy_io_recorder, "record_tick", None)
        ):
            raise TypeError("policy_io_recorder must expose record_tick()")
        maximum_actions = _positive_integer(
            maximum_actions_per_camera_frame,
            "maximum_actions_per_camera_frame",
        )
        # D435 is nominally 30 Hz.  The caller selects three uses at 20 Hz
        # (current plus the training-contract 0..2 delayed ticks) or six at
        # 60 Hz, bounding transport-frozen content while still matching the
        # low-rate checkpoint's point-cloud latency support.
        # A stale-palm cloud may have older
        # content by bundle contract, but its *current camera frame* still
        # advances and therefore does not evade this transport-liveness gate.
        if not 2 <= maximum_actions <= 6:
            raise ValueError(
                "V94 camera reuse contract requires 2..6 actions per frame"
            )

        self._provider = observation_provider
        self._policy = policy
        point_feature_dim = getattr(policy, "point_feature_dim", 6)
        if (
            isinstance(point_feature_dim, bool)
            or not isinstance(point_feature_dim, (int, np.integer))
            or int(point_feature_dim) not in (3, 6)
        ):
            raise ValueError("policy point_feature_dim must be 3 or 6")
        self.point_feature_dim = int(point_feature_dim)
        history_length = getattr(policy, "history_length", 4)
        proprio_dim = getattr(policy, "proprio_dim", 67)
        if (
            isinstance(history_length, bool)
            or not isinstance(history_length, (int, np.integer))
            or isinstance(proprio_dim, bool)
            or not isinstance(proprio_dim, (int, np.integer))
            or (int(history_length), int(proprio_dim))
            not in ((4, 67), (8, 67), (8, 96), (16, 96))
        ):
            raise ValueError(
                "policy history/proprio contract must be (4,67), (8,67), "
                "(8,96), or (16,96)"
            )
        self.history_length = int(history_length)
        self.proprio_dim = int(proprio_dim)
        action_controller = getattr(policy, "action_controller", None)
        contract_id = getattr(
            action_controller,
            "contract_id",
            LEGACY_ACTION_CONTROLLER_CONTRACT_ID,
        )
        if contract_id not in {
            LEGACY_ACTION_CONTROLLER_CONTRACT_ID,
            QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
        }:
            raise ValueError(
                f"unsupported policy action-controller contract: {contract_id!r}"
            )
        self.action_controller_contract_id = str(contract_id)
        if self.action_controller_contract_id == QD_G015_ACTION_CONTROLLER_CONTRACT_ID:
            if (
                self.history_length != 8
                or self.proprio_dim not in (67, 96)
                or self.point_feature_dim != 3
            ):
                raise ValueError(
                    "q_d g015 tick source requires XYZ points, history=8, "
                    "and proprio_dim=67 or 96"
                )
        elif (self.history_length, self.proprio_dim) == (8, 67):
            raise ValueError(
                "legacy policy does not define an (8,67) observation contract"
            )
        initial_previous = getattr(
            policy,
            "initial_previous_action13",
            INITIAL_PREVIOUS_ACTION13,
        )
        self.initial_previous_action13 = _readonly_float_array(
            initial_previous,
            (13,),
            "policy initial_previous_action13",
        )
        self._requires_shaper_q_d = (
            self.action_controller_contract_id
            == QD_G015_ACTION_CONTROLLER_CONTRACT_ID
        )
        self._mapper = action_mapper
        self._rh56_hardware_command_shaper = rh56_hardware_command_shaper
        self._policy_io_recorder = policy_io_recorder
        self.maximum_object_pointcloud_age_s = _positive_float(
            maximum_object_pointcloud_age_s,
            "maximum_object_pointcloud_age_s",
        )
        self.maximum_pointcloud_future_skew_s = _finite_float(
            maximum_pointcloud_future_skew_s,
            "maximum_pointcloud_future_skew_s",
        )
        if self.maximum_pointcloud_future_skew_s < 0.0:
            raise ValueError("maximum_pointcloud_future_skew_s cannot be negative")
        self.maximum_franka_action_age_s = _positive_float(
            maximum_franka_action_age_s,
            "maximum_franka_action_age_s",
        )
        self.maximum_franka_hard_age_s = _positive_float(
            maximum_franka_hard_age_s,
            "maximum_franka_hard_age_s",
        )
        if self.maximum_franka_action_age_s >= self.maximum_franka_hard_age_s:
            raise ValueError("Franka action age must be below the hard state-age limit")
        self.maximum_actions_per_camera_frame = maximum_actions
        configured_startup_steps = _nonnegative_integer(
            startup_non_actuated_policy_steps,
            "startup_non_actuated_policy_steps",
        )
        # Exact-action replay already carries its complete simulator target
        # sequence.  Prepending policy-only ticks would shift that trajectory.
        self.startup_non_actuated_policy_steps = (
            0 if self.is_replay_source else configured_startup_steps
        )
        self.hold_retry_interval_s = _positive_float(
            hold_retry_interval_s, "hold_retry_interval_s"
        )
        self.franka_arrival_gate_enabled = _strict_bool(
            franka_arrival_gate_enabled, "franka_arrival_gate_enabled"
        )
        self.franka_arrival_gate_tolerance_rad = _positive_float(
            franka_arrival_gate_tolerance_rad,
            "franka_arrival_gate_tolerance_rad",
        )
        if self.franka_arrival_gate_tolerance_rad > 0.030:
            raise ValueError(
                "franka_arrival_gate_tolerance_rad must not exceed 0.030"
            )
        self.franka_arrival_gate_timeout_s = _positive_float(
            franka_arrival_gate_timeout_s,
            "franka_arrival_gate_timeout_s",
        )
        if self.franka_arrival_gate_timeout_s > 0.400:
            raise ValueError(
                "franka_arrival_gate_timeout_s must not exceed 0.400"
            )
        # Arrival polling is diagnostic, not a new high-rate control loop.
        # Bound it to 50--200 Hz so waiting cannot hammer the full RGB-D/SAM2
        # observation assembler at the native servo's 1 kHz rate.
        self.franka_arrival_gate_retry_interval_s = min(
            0.020,
            max(0.005, float(self._mapper.control_dt_s) * 0.5),
        )
        self._monotonic = monotonic
        self._realtime = realtime

        self._lock = threading.Lock()
        self._history: Optional[_HistoryCandidate] = None
        self._policy_previous_action13 = self.initial_previous_action13
        self._startup_non_actuated_completed_steps = 0
        self._startup_non_actuated_records: list[NonActuatedPolicyTick] = []
        self._pending: Optional[_PendingTick] = None
        self._last_committed_sequence = 0
        self._last_committed_frame_id: Optional[int] = None
        self._last_accepted_pointcloud_source_frame_id: Optional[int] = None
        self._accepted_uses_for_last_frame = 0
        self._hold_count = 0
        self._fault_reason: Optional[str] = None
        self._startup_warmup_complete = False
        self._startup_warmup_inference_s: Optional[float] = None
        self._startup_warmup_frame_id: Optional[int] = None
        self._startup_warmup_retries = 0
        self._arrival_gate_target_q_rad: Optional[np.ndarray] = None
        self._arrival_gate_start_q_rad: Optional[np.ndarray] = None
        self._arrival_gate_reached_axes: Optional[np.ndarray] = None
        self._arrival_gate_started_monotonic_s: Optional[float] = None
        self._arrival_gate_hold_count = 0
        self._arrival_gate_arrived_target_count = 0
        self._arrival_gate_total_wait_s = 0.0
        self._arrival_gate_max_wait_s = 0.0
        self._arrival_gate_last_error_rad: Optional[float] = None
        self._arrival_gate_last_wait_s: Optional[float] = None

    @property
    def last_committed_sequence(self) -> int:
        with self._lock:
            return self._last_committed_sequence

    @property
    def pending_sequence(self) -> Optional[int]:
        with self._lock:
            return None if self._pending is None else self._pending.command.sequence

    @property
    def last_committed_frame_id(self) -> Optional[int]:
        with self._lock:
            return self._last_committed_frame_id

    @property
    def accepted_uses_for_last_frame(self) -> int:
        with self._lock:
            return self._accepted_uses_for_last_frame

    @property
    def replay_complete(self) -> bool:
        return bool(getattr(self._policy, "replay_complete", False))

    @property
    def is_replay_source(self) -> bool:
        return bool(getattr(self._policy, "is_replay_policy", False))

    @property
    def completed_replay_frames(self) -> int:
        return int(getattr(self._policy, "completed_replay_frames", 0))

    @property
    def repeated_replay_target_commands(self) -> int:
        return int(getattr(self._policy, "repeated_target_commands", 0))

    def _discard_policy_proposal(self, sequence: int) -> None:
        discard = getattr(self._policy, "discard_replay_proposal", None)
        if callable(discard):
            discard(sequence)

    @property
    def hold_count(self) -> int:
        with self._lock:
            return self._hold_count

    @property
    def fault_reason(self) -> Optional[str]:
        with self._lock:
            return self._fault_reason

    @property
    def diagnostics_snapshot(self) -> dict[str, object]:
        """Read-only observation provenance for terminal runtime audits."""

        provider_diagnostics = getattr(self._provider, "diagnostics_snapshot", None)
        startup = {
            "startup_policy_warmup_complete": bool(
                self._startup_warmup_complete
            ),
            "startup_policy_warmup_inference_s": (
                self._startup_warmup_inference_s
            ),
            "startup_policy_warmup_frame_id": self._startup_warmup_frame_id,
            "startup_policy_warmup_retries": int(self._startup_warmup_retries),
            "startup_non_actuated_policy_steps": int(
                self.startup_non_actuated_policy_steps
            ),
            "startup_non_actuated_completed_steps": int(
                self._startup_non_actuated_completed_steps
            ),
            "startup_non_actuated_contract": (
                "history_and_policy_previous_advance_without_mapper_ledger_or_"
                "actuator_target"
            ),
            "franka_arrival_gate": {
                "enabled": bool(self.franka_arrival_gate_enabled),
                "tolerance_rad": float(
                    self.franka_arrival_gate_tolerance_rad
                ),
                "per_target_timeout_s": float(
                    self.franka_arrival_gate_timeout_s
                ),
                "retry_interval_s": float(
                    self.franka_arrival_gate_retry_interval_s
                ),
                "hold_count": int(self._arrival_gate_hold_count),
                "arrived_target_count": int(
                    self._arrival_gate_arrived_target_count
                ),
                "total_wait_s": float(self._arrival_gate_total_wait_s),
                "max_wait_s": float(self._arrival_gate_max_wait_s),
                "last_wait_s": self._arrival_gate_last_wait_s,
                "last_error_rad": self._arrival_gate_last_error_rad,
                "criterion": "per_axis_tolerance_or_crossing_latched",
                "reached_axes": (
                    None
                    if self._arrival_gate_reached_axes is None
                    else self._arrival_gate_reached_axes.astype(bool).tolist()
                ),
                "target_pending": self._arrival_gate_target_q_rad is not None,
            },
        }
        policy_diagnostics = getattr(self._policy, "diagnostics_snapshot", None)
        if policy_diagnostics is not None:
            try:
                value = (
                    policy_diagnostics()
                    if callable(policy_diagnostics)
                    else policy_diagnostics
                )
                if isinstance(value, dict):
                    startup["replay_policy"] = dict(value)
            except Exception as exc:
                startup["replay_policy"] = {
                    "diagnostics_error": f"{type(exc).__name__}: {exc}"
                }
        if provider_diagnostics is None:
            return startup
        try:
            provider = (
                dict(provider_diagnostics)
                if not isinstance(provider_diagnostics, dict)
                else dict(provider_diagnostics)
            )
            provider.update(startup)
            return provider
        except BaseException as exc:
            startup["diagnostics_error"] = f"{type(exc).__name__}: {exc}"
            return startup

    @property
    def diagnostics_summary(self) -> str:
        summary = getattr(self._provider, "diagnostics_summary", None)
        if summary is None:
            return ""
        try:
            return " ".join(str(summary).split())[:2000]
        except BaseException as exc:
            return f"diagnostics unavailable: {type(exc).__name__}: {exc}"

    def committed_history(self) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        with self._lock:
            if self._history is None:
                return None
            return (
                self._history.points.copy(),
                self._history.valid.copy(),
                self._history.proprio.copy(),
            )

    @property
    def startup_non_actuated_records(self) -> tuple[NonActuatedPolicyTick, ...]:
        with self._lock:
            return tuple(self._startup_non_actuated_records)

    def _now_monotonic(self) -> float:
        return _finite_float(self._monotonic(), "source monotonic clock")

    def _now_realtime(self) -> float:
        return _finite_float(self._realtime(), "source realtime clock")

    def _require_healthy_locked(self) -> None:
        if self._fault_reason is not None:
            raise V94PolicyTickSourceError(
                f"policy tick source fault is latched: {self._fault_reason}"
            )

    def _latch_locked(self, reason: str) -> None:
        if self._fault_reason is None:
            self._fault_reason = str(reason).strip() or "unspecified policy tick fault"

    def _record_policy_io_locked(
        self,
        *,
        logical_policy_step: int,
        hardware_sequence: int,
        startup_non_actuated: bool,
        history: _HistoryCandidate,
        observation: V94PolicyObservation,
        policy_previous_action13: np.ndarray,
        ledger_previous_action13: np.ndarray,
        raw_model_action13: np.ndarray,
        sent_action13: np.ndarray,
        produced_monotonic_s: float,
        franka_target_q_rad: Optional[np.ndarray],
        rh56_target_register_order: Optional[np.ndarray],
        hardware_command_valid: bool,
    ) -> None:
        """Best-effort in-memory diagnostics after a logical tick is accepted.

        Recording is deliberately outside the policy/mapper transaction.  A
        diagnostic failure must never reject, abort, or otherwise alter a
        hardware command that has already reached the accepted/dual-ACK state.
        The recorder itself performs no disk I/O in this method.
        """

        recorder = self._policy_io_recorder
        if recorder is None:
            return
        try:
            recorder.record_tick(
                logical_policy_step=logical_policy_step,
                hardware_sequence=hardware_sequence,
                startup_non_actuated=startup_non_actuated,
                pointcloud_history_metric=history.points,
                pointcloud_valid_history=history.valid,
                proprio_history_raw=history.proprio,
                previous_policy_action13=policy_previous_action13,
                previous_ledger_action13=ledger_previous_action13,
                raw_model_action13=raw_model_action13,
                sent_action13=sent_action13,
                camera_frame_id=observation.camera_frame_id,
                pointcloud_source_frame_id=(
                    observation.pointcloud_source_frame_id
                ),
                pointcloud_status=observation.pointcloud_status,
                source_valid_points=observation.source_valid_points,
                observation_realtime_s=observation.observation_realtime_s,
                pointcloud_captured_realtime_s=(
                    observation.pointcloud_captured_realtime_s
                ),
                franka_state_captured_monotonic_s=(
                    observation.franka_state_captured_monotonic_s
                ),
                produced_monotonic_s=produced_monotonic_s,
                measured_franka_q_rad=observation.measured_franka_q_rad,
                shaper_q_d_rad=observation.shaper_q_d_rad,
                franka_target_q_rad=franka_target_q_rad,
                rh56_target_register_order=rh56_target_register_order,
                hardware_command_valid=hardware_command_valid,
                hold_arm_target=observation.hold_arm_target,
            )
        except Exception:
            # Third-party/fake recorder implementations are held to the same
            # fail-soft contract as the production recorder.  Do not print or
            # allocate an exception traceback on the real-time command path.
            return

    def _hold_locked(
        self,
        *,
        sequence: int,
        frame_id: int,
        reason: str,
        observed_monotonic_s: float,
        retry_interval_s: Optional[float] = None,
    ) -> BoundedPolicyTickHold:
        self._hold_count += 1
        retry_interval = (
            self.hold_retry_interval_s
            if retry_interval_s is None
            else _positive_float(retry_interval_s, "hold retry interval")
        )
        return BoundedPolicyTickHold(
            sequence=sequence,
            observed_monotonic_s=observed_monotonic_s,
            retry_not_before_monotonic_s=(
                observed_monotonic_s + retry_interval
            ),
            camera_frame_id=frame_id,
            reason=reason,
            policy_state_mutated=False,
            action_staged=False,
        )

    def _candidate_history_locked(
        self,
        observation: V94PolicyObservation,
        previous_executed_action13: np.ndarray,
    ) -> _HistoryCandidate:
        previous = _readonly_float_array(
            previous_executed_action13,
            (13,),
            "previous_executed_action13",
        )
        base67 = np.concatenate(
            [observation.proprio_prefix54, previous]
        ).astype(np.float32)
        if self.proprio_dim == 96:
            if observation.controller_state29 is None:
                raise V94PolicyTickSourceError(
                    "96D checkpoint requires same-sample Franka controller_state29"
                )
            current_proprio = np.concatenate(
                [base67, observation.controller_state29]
            ).astype(np.float32)
        else:
            current_proprio = base67
        if observation.pointcloud_xyzrgb_palm.shape[1] != self.point_feature_dim:
            raise V94PolicyTickSourceError(
                "observation/policy point feature mismatch: "
                f"observation={observation.pointcloud_xyzrgb_palm.shape[1]}D "
                f"policy={self.point_feature_dim}D"
            )
        if self._history is None:
            points = np.repeat(
                observation.pointcloud_xyzrgb_palm[None, :, :],
                self.history_length,
                axis=0,
            )
            valid = np.repeat(
                observation.pointcloud_valid[None, :], self.history_length, axis=0
            )
            proprio = np.repeat(
                current_proprio[None, :], self.history_length, axis=0
            )
        else:
            points = np.concatenate(
                [
                    self._history.points[1:],
                    observation.pointcloud_xyzrgb_palm[None, :, :],
                ],
                axis=0,
            )
            valid = np.concatenate(
                [
                    self._history.valid[1:],
                    observation.pointcloud_valid[None, :],
                ],
                axis=0,
            )
            proprio = np.concatenate(
                [self._history.proprio[1:], current_proprio[None, :]],
                axis=0,
            )
        candidate = _HistoryCandidate(
            points=_readonly_float_array(
                points,
                (self.history_length, 128, self.point_feature_dim),
                "point history",
            ),
            valid=_readonly_float_array(
                valid, (self.history_length, 128), "valid history"
            ),
            proprio=_readonly_float_array(
                proprio,
                (self.history_length, self.proprio_dim),
                "proprio history",
            ),
        )
        return candidate

    def _pointcloud_age(self, observation: V94PolicyObservation) -> float:
        now_realtime = self._now_realtime()
        observation_future = observation.observation_realtime_s - now_realtime
        if observation_future > self.maximum_pointcloud_future_skew_s:
            raise V94PolicyTickSourceError(
                "observation realtime timestamp exceeds the future-skew limit"
            )
        age = now_realtime - observation.pointcloud_captured_realtime_s
        if age < -self.maximum_pointcloud_future_skew_s:
            raise V94PolicyTickSourceError(
                "object point-cloud timestamp exceeds the future-skew limit"
            )
        if observation.pointcloud_status == "stale_palm":
            # DEPLOYMENT.md explicitly requires the previous masked cloud when
            # the current formal frame contains fewer than 16 valid object
            # pixels.  Its original capture timestamp/native-palm coordinates
            # are provenance, not a local freshness deadline.  Current-camera
            # publication liveness and reuse remain independently bounded by
            # D435ObjectCameraOwner and ``camera_frame_id`` above.
            return 0.0
        return age

    def _franka_state_age(
        self,
        observation: V94PolicyObservation,
        *,
        now_monotonic_s: float,
    ) -> float:
        now = _finite_float(now_monotonic_s, "Franka age observation time")
        age = now - observation.franka_state_captured_monotonic_s
        if age < -self.maximum_pointcloud_future_skew_s:
            raise V94PolicyTickSourceError(
                "Franka state timestamp exceeds the future-skew limit"
            )
        # A late parent-side STATE datagram must never authorize a new
        # command, but one isolated scheduling slip is not evidence that the
        # native 1 kHz Franka owner failed.  Every caller below already turns
        # any value above ``maximum_franka_action_age_s`` into a proposal-pure
        # no-stage hold.  Let the runtime's independent consecutive-hold
        # deadline decide whether freshness recovered instead of terminating
        # on the first packet that crosses the diagnostic hard-age boundary.
        # Future timestamps remain terminal because they violate clock
        # provenance rather than freshness.
        return age

    def _check_franka_arrival_gate_locked(
        self,
        measured_q_rad: object,
        *,
        now_monotonic_s: float,
    ) -> bool:
        """Return whether the last committed Franka target has arrived.

        This check never stages an action or advances policy/history state.
        The target remains owned by the native 1 kHz sample-hold while the
        low-rate policy is paused.
        """

        if not self.franka_arrival_gate_enabled:
            return True
        target = self._arrival_gate_target_q_rad
        started = self._arrival_gate_started_monotonic_s
        if target is None:
            return True
        if started is None:
            raise V94PolicyTickSourceError(
                "Franka arrival gate target has no commit timestamp"
            )
        measured = _readonly_float_array(
            measured_q_rad, (7,), "arrival-gate measured_franka_q_rad"
        )
        now = _finite_float(now_monotonic_s, "arrival-gate monotonic time")
        if now < started:
            raise V94PolicyTickSourceError("arrival-gate clock regressed")
        start = self._arrival_gate_start_q_rad
        reached_axes = self._arrival_gate_reached_axes
        if start is None or reached_axes is None:
            raise V94PolicyTickSourceError(
                "Franka arrival gate target has no per-axis start state"
            )
        signed_start_error = target.astype(np.float64) - start.astype(np.float64)
        signed_error = target.astype(np.float64) - measured.astype(np.float64)
        error_by_axis = np.abs(signed_error)
        # A jerk-limited multi-axis step does not necessarily put every joint
        # inside a tight position band at the same instant.  Remember each
        # joint once it either enters that band or crosses its target.  This
        # makes the gate mean "every requested joint has physically reached
        # its target at least once" instead of waiting for an accidental
        # seven-axis simultaneous convergence window.
        reached_axes |= (
            (error_by_axis <= self.franka_arrival_gate_tolerance_rad)
            | (signed_start_error * signed_error <= 0.0)
        )
        error = float(np.max(error_by_axis))
        elapsed = float(now - started)
        self._arrival_gate_last_error_rad = error
        self._arrival_gate_last_wait_s = elapsed
        if bool(np.all(reached_axes)):
            self._arrival_gate_arrived_target_count += 1
            self._arrival_gate_total_wait_s += elapsed
            self._arrival_gate_max_wait_s = max(
                self._arrival_gate_max_wait_s, elapsed
            )
            self._arrival_gate_target_q_rad = None
            self._arrival_gate_start_q_rad = None
            self._arrival_gate_reached_axes = None
            self._arrival_gate_started_monotonic_s = None
            return True
        if elapsed >= self.franka_arrival_gate_timeout_s:
            unreached_error = np.where(reached_axes, -1.0, error_by_axis)
            axis = int(np.argmax(unreached_error))
            reason = (
                "Franka arrival gate timed out: "
                f"wait={elapsed:.6f}s "
                f"limit={self.franka_arrival_gate_timeout_s:.6f}s "
                f"Linf={error:.9f}rad axis={axis + 1} "
                f"tolerance={self.franka_arrival_gate_tolerance_rad:.9f}rad "
                f"reached_axes={int(np.count_nonzero(reached_axes))}/7"
            )
            self._latch_locked(reason)
            raise V94PolicyTickSourceError(reason)
        return False

    def check_franka_arrival_gate(
        self,
        measured_q_rad: object,
        *,
        now_monotonic_s: float,
    ) -> bool:
        """Thread-safe final-target arrival check used by the runtime."""

        with self._lock:
            self._require_healthy_locked()
            return self._check_franka_arrival_gate_locked(
                measured_q_rad,
                now_monotonic_s=now_monotonic_s,
            )

    def warmup_before_first_action(
        self,
        *,
        hard_deadline_monotonic_s: float,
        maximum_warmup_s: float = 2.0,
    ) -> None:
        """Warm the policy on valid live input without staging an action.

        D435/SAM2 recovery and the first CUDA/model forward are startup work,
        not a closed-loop observation outage.  This method waits for a valid
        source frame, performs one deliberately discarded forward, then waits
        for a newer frame with a fresh Franka sample.  No mapper, history,
        replay index, camera-use counter, or actuator target is mutated.
        """

        hard_deadline = _finite_float(
            hard_deadline_monotonic_s, "hard_deadline_monotonic_s"
        )
        warmup_limit = _positive_float(maximum_warmup_s, "maximum_warmup_s")
        with self._lock:
            self._require_healthy_locked()
            if self._startup_warmup_complete:
                raise V94PolicyTickSourceError("policy startup warmup already ran")
            if (
                self._last_committed_sequence != 0
                or self._pending is not None
                or self._history is not None
            ):
                raise V94PolicyTickSourceError(
                    "policy startup warmup requires a pristine tick source"
                )
            # Exact action replay has no CUDA/model cold path.  Calling its
            # sequence API here would create a transactional replay proposal,
            # so leave it untouched.
            if self.is_replay_source:
                self._startup_warmup_complete = True
                return

            deadline = min(hard_deadline, self._now_monotonic() + warmup_limit)
            previous = self.initial_previous_action13.copy()
            warmed_frame_id: Optional[int] = None

            while self._now_monotonic() < deadline:
                now = self._now_monotonic()
                try:
                    observation = self._provider.snapshot(
                        sequence=1,
                        now_monotonic_s=now,
                        hard_deadline_monotonic_s=hard_deadline,
                    )
                    after_snapshot = self._now_monotonic()
                    if (
                        observation.pointcloud_status != "fresh"
                        or self._pointcloud_age(observation)
                        > self.maximum_object_pointcloud_age_s
                        or self._franka_state_age(
                            observation, now_monotonic_s=after_snapshot
                        )
                        > self.maximum_franka_action_age_s
                    ):
                        self._startup_warmup_retries += 1
                        time.sleep(self.hold_retry_interval_s)
                        continue
                except V94RecoverableObservationHold:
                    self._startup_warmup_retries += 1
                    time.sleep(self.hold_retry_interval_s)
                    continue

                candidate = self._candidate_history_locked(observation, previous)
                inference_started = self._now_monotonic()
                output = self._policy.act(
                    candidate.points,
                    candidate.valid,
                    candidate.proprio,
                )
                raw_action = _readonly_float_array(
                    getattr(output, "action13", None),
                    (13,),
                    "discarded warmup policy output.action13",
                )
                if np.any(raw_action < -1.0) or np.any(raw_action > 1.0):
                    raise V94PolicyTickSourceError(
                        "discarded warmup policy action must lie in [-1,1]"
                    )
                inference_finished = self._now_monotonic()
                self._startup_warmup_inference_s = (
                    inference_finished - inference_started
                )
                warmed_frame_id = int(observation.camera_frame_id)
                self._startup_warmup_frame_id = warmed_frame_id
                break

            if warmed_frame_id is None:
                raise V94PolicyTickSourceError(
                    "timed out waiting for a valid observation for discarded "
                    "policy startup warmup"
                )

            # The warmup frame and its robot sample may have expired during
            # the cold forward.  Do not return to the action loop until a
            # genuinely newer camera frame is paired with a fresh arm state.
            while self._now_monotonic() < deadline:
                now = self._now_monotonic()
                try:
                    observation = self._provider.snapshot(
                        sequence=1,
                        now_monotonic_s=now,
                        hard_deadline_monotonic_s=hard_deadline,
                    )
                    checked = self._now_monotonic()
                    ready = (
                        int(observation.camera_frame_id) > warmed_frame_id
                        and observation.pointcloud_status == "fresh"
                        and self._pointcloud_age(observation)
                        <= self.maximum_object_pointcloud_age_s
                        and self._franka_state_age(
                            observation, now_monotonic_s=checked
                        )
                        <= self.maximum_franka_action_age_s
                    )
                    if ready:
                        self._startup_warmup_complete = True
                        return
                except V94RecoverableObservationHold:
                    pass
                self._startup_warmup_retries += 1
                time.sleep(self.hold_retry_interval_s)

            raise V94PolicyTickSourceError(
                "timed out waiting for a fresh post-warmup camera/Franka sample"
            )

    def prime_non_actuated_startup_tick(
        self,
        *,
        logical_index: int,
        now_monotonic_s: float,
        hard_deadline_monotonic_s: float,
    ) -> Union[NonActuatedPolicyTick, BoundedPolicyTickHold]:
        """Advance one training-aligned startup tick without mapping an action.

        The simulator evaluates the policy for its first four 20 Hz ticks but
        keeps Franka at reset and RH56 open.  This method reproduces the policy
        side of that contract while leaving the action mapper, RH56 shaper,
        dual-ACK ledger and both actuator targets untouched.
        """

        index = _nonnegative_integer(logical_index, "logical_index")
        caller_now = _finite_float(now_monotonic_s, "now_monotonic_s")
        hard_deadline = _finite_float(
            hard_deadline_monotonic_s, "hard_deadline_monotonic_s"
        )
        with self._lock:
            self._require_healthy_locked()
            if self.startup_non_actuated_policy_steps <= 0:
                raise V94PolicyTickSourceError(
                    "non-actuated startup is not enabled for this policy"
                )
            if index != self._startup_non_actuated_completed_steps:
                raise V94PolicyTickSourceError(
                    "non-actuated startup index mismatch: "
                    f"expected={self._startup_non_actuated_completed_steps}, "
                    f"actual={index}"
                )
            if index >= self.startup_non_actuated_policy_steps:
                raise V94PolicyTickSourceError(
                    "non-actuated startup already completed"
                )
            if self._last_committed_sequence != 0 or self._pending is not None:
                raise V94PolicyTickSourceError(
                    "non-actuated startup must precede every hardware command"
                )
            if caller_now >= hard_deadline:
                raise V94PolicyTickSourceError(
                    "hard deadline expired before non-actuated startup tick"
                )

            try:
                observation = self._provider.snapshot(
                    sequence=1,
                    now_monotonic_s=caller_now,
                    hard_deadline_monotonic_s=hard_deadline,
                )
                if not isinstance(observation, V94PolicyObservation):
                    raise TypeError(
                        "observation provider returned no V94PolicyObservation"
                    )
                after_snapshot = self._now_monotonic()
                frame_id = int(observation.camera_frame_id)
                pointcloud_source_frame_id = int(
                    observation.pointcloud_source_frame_id
                )
                if after_snapshot < caller_now:
                    raise V94PolicyTickSourceError("source monotonic clock regressed")
                if after_snapshot >= hard_deadline:
                    raise V94PolicyTickSourceError(
                        "hard deadline expired while acquiring startup observation"
                    )
                if (
                    self._franka_state_age(
                        observation, now_monotonic_s=after_snapshot
                    )
                    > self.maximum_franka_action_age_s
                ):
                    return self._hold_locked(
                        sequence=1,
                        frame_id=frame_id,
                        reason=FRANKA_STATE_STALE_HOLD_REASON,
                        observed_monotonic_s=after_snapshot,
                    )
                if (
                    self._last_committed_frame_id is not None
                    and frame_id < self._last_committed_frame_id
                ):
                    raise V94PolicyTickSourceError(
                        "camera frame regressed during non-actuated startup"
                    )
                uses = (
                    self._accepted_uses_for_last_frame
                    if pointcloud_source_frame_id
                    == self._last_accepted_pointcloud_source_frame_id
                    else 0
                )
                if uses >= self.maximum_actions_per_camera_frame:
                    return self._hold_locked(
                        sequence=1,
                        frame_id=frame_id,
                        reason=CAMERA_REUSE_HOLD_REASON,
                        observed_monotonic_s=after_snapshot,
                    )
                if (
                    self._pointcloud_age(observation)
                    > self.maximum_object_pointcloud_age_s
                ):
                    return self._hold_locked(
                        sequence=1,
                        frame_id=frame_id,
                        reason=OBJECT_POINTCLOUD_STALE_HOLD_REASON,
                        observed_monotonic_s=after_snapshot,
                    )

                previous = self._policy_previous_action13.copy()
                candidate = self._candidate_history_locked(
                    observation, previous
                )
                output = self._policy.act(
                    candidate.points,
                    candidate.valid,
                    candidate.proprio,
                )
                raw_action = _readonly_float_array(
                    getattr(output, "action13", None),
                    (13,),
                    "startup policy output.action13",
                )
                if np.any(raw_action < -1.0) or np.any(raw_action > 1.0):
                    raise V94PolicyTickSourceError(
                        "startup policy action must lie in [-1,1]"
                    )
                after_policy = self._now_monotonic()
                if after_policy >= hard_deadline:
                    raise V94PolicyTickSourceError(
                        "hard deadline expired during startup policy inference"
                    )
                if (
                    self._pointcloud_age(observation)
                    > self.maximum_object_pointcloud_age_s
                ):
                    return self._hold_locked(
                        sequence=1,
                        frame_id=frame_id,
                        reason=OBJECT_POINTCLOUD_STALE_HOLD_REASON,
                        observed_monotonic_s=after_policy,
                    )
                if (
                    self._franka_state_age(
                        observation, now_monotonic_s=after_policy
                    )
                    > self.maximum_franka_action_age_s
                ):
                    return self._hold_locked(
                        sequence=1,
                        frame_id=frame_id,
                        reason=FRANKA_STATE_STALE_HOLD_REASON,
                        observed_monotonic_s=after_policy,
                    )

                accepted = raw_action.copy()
                if observation.hold_arm_target:
                    accepted[:7] = np.float32(0.0)
                accepted.setflags(write=False)
                record = NonActuatedPolicyTick(
                    logical_index=index,
                    produced_monotonic_s=after_policy,
                    camera_frame_id=frame_id,
                    previous_policy_action13_used=previous,
                    accepted_policy_action13=accepted,
                )
                self._history = candidate
                self._policy_previous_action13 = accepted
                if (
                    pointcloud_source_frame_id
                    == self._last_accepted_pointcloud_source_frame_id
                ):
                    self._accepted_uses_for_last_frame += 1
                else:
                    self._last_accepted_pointcloud_source_frame_id = (
                        pointcloud_source_frame_id
                    )
                    self._accepted_uses_for_last_frame = 1
                self._last_committed_frame_id = frame_id
                self._startup_non_actuated_completed_steps += 1
                self._startup_non_actuated_records.append(record)
                self._record_policy_io_locked(
                    logical_policy_step=index,
                    hardware_sequence=-1,
                    startup_non_actuated=True,
                    history=candidate,
                    observation=observation,
                    policy_previous_action13=previous,
                    ledger_previous_action13=self.initial_previous_action13,
                    raw_model_action13=raw_action,
                    sent_action13=accepted,
                    produced_monotonic_s=after_policy,
                    franka_target_q_rad=None,
                    rh56_target_register_order=None,
                    hardware_command_valid=False,
                )
                return record
            except V94RecoverableObservationHold as exc:
                return self._hold_locked(
                    sequence=1,
                    frame_id=exc.camera_frame_id,
                    reason=exc.reason,
                    observed_monotonic_s=self._now_monotonic(),
                )
            except BaseException as exc:
                self._latch_locked(f"{type(exc).__name__}: {exc}")
                raise

    def _prepare_occluded_closure_hand_only_locked(
        self,
        *,
        sequence: int,
        previous_executed_action13: np.ndarray,
        camera_frame_id: int,
        hold_reason: str,
        actuator_snapshot: Optional[V94FreshActuatorSnapshot],
        caller_now_monotonic_s: float,
        hard_deadline_monotonic_s: float,
    ) -> Optional[ClosedLoopCommand]:
        """Stage a bounded RH56-only completion after visual closure commit."""

        if not str(hold_reason).startswith(
            (
                "guarded_stale_palm_fail_closed:",
                "object_pointcloud_transient_invalid:",
            )
        ):
            return None
        action_for_occluded_closure = getattr(
            self._policy, "action_for_occluded_closure", None
        )
        if not callable(action_for_occluded_closure):
            return None
        # A recorder entry without a fresh observation would falsely claim a
        # point-cloud/model inference tick.  Keep that audit mode fail-closed.
        if self._policy_io_recorder is not None:
            return None
        if self._last_committed_sequence <= 0 or self._history is None:
            return None
        if self._now_monotonic() >= hard_deadline_monotonic_s:
            raise V94PolicyTickSourceError(
                "hard deadline expired before occluded closure completion"
            )

        output = action_for_occluded_closure(
            sequence,
            observation_hold_reason=hold_reason,
        )
        if output is None:
            return None
        raw_action = _readonly_float_array(
            getattr(output, "action13", None),
            (13,),
            "occluded closure output.action13",
        )
        if np.any(raw_action < -1.0) or np.any(raw_action > 1.0):
            raise V94PolicyTickSourceError(
                "occluded closure policy action must lie in [-1,1]"
            )
        exact_arm = getattr(output, "exact_franka_target_q_rad", None)
        exact_hand = getattr(
            output, "exact_rh56_angle_set_register_order", None
        )
        if exact_arm is None or exact_hand is None:
            raise V94PolicyTickSourceError(
                "occluded closure must provide exact Franka/RH56 targets"
            )
        if not bool(getattr(output, "bypass_rh56_host_slew", False)):
            raise V94PolicyTickSourceError(
                "occluded closure must use the reviewed exact RH56 template"
            )
        clearance_lift = bool(
            getattr(output, "occluded_clearance_lift", False)
        )
        committed_arm, _committed_hand = self._mapper.committed_targets()
        requested_arm = _readonly_float_array(
            exact_arm, (7,), "occluded closure exact Franka target"
        )
        if not clearance_lift and not np.array_equal(
            requested_arm, committed_arm
        ):
            raise V94PolicyTickSourceError(
                "occluded closure attempted to move the Franka target"
            )
        measured_q = committed_arm
        observation_realtime_s = self._now_realtime()
        if clearance_lift:
            if actuator_snapshot is None:
                self._discard_policy_proposal(sequence)
                return None
            observed_now = self._now_monotonic()
            franka_age = (
                observed_now
                - actuator_snapshot.franka_state_captured_monotonic_s
            )
            if franka_age < -self.maximum_pointcloud_future_skew_s:
                raise V94PolicyTickSourceError(
                    "occluded clearance-lift Franka state exceeds future skew"
                )
            if franka_age > self.maximum_franka_hard_age_s:
                raise V94PolicyTickSourceError(
                    "occluded clearance-lift Franka state exceeded hard age"
                )
            if (
                franka_age > self.maximum_franka_action_age_s
                or actuator_snapshot.hold_arm_target
            ):
                self._discard_policy_proposal(sequence)
                return None
            if not self._check_franka_arrival_gate_locked(
                actuator_snapshot.measured_franka_q_rad,
                now_monotonic_s=observed_now,
            ):
                self._arrival_gate_hold_count += 1
                self._discard_policy_proposal(sequence)
                return None
            measured_q = actuator_snapshot.measured_franka_q_rad
            observation_realtime_s = (
                actuator_snapshot.observation_realtime_s
            )
        mapper_proposal = self._mapper.propose_exact_targets(
            sequence,
            raw_action,
            franka_target_q_rad=requested_arm,
            rh56_angle_set_register_order=exact_hand,
            measured_q_rad=measured_q,
            hold_arm_target=not clearance_lift,
        )
        produced = self._now_monotonic()
        if produced < caller_now_monotonic_s:
            raise V94PolicyTickSourceError("source monotonic clock regressed")
        if produced >= hard_deadline_monotonic_s:
            self._mapper.discard_unstaged(mapper_proposal)
            self._discard_policy_proposal(sequence)
            raise V94PolicyTickSourceError(
                "hard deadline expired during occluded closure mapping"
            )
        mapped = mapper_proposal.mapped
        if not np.array_equal(mapped.franka_target_q_rad, requested_arm):
            raise V94PolicyTickSourceError(
                "occluded closure mapper changed the exact Franka target"
            )
        if not clearance_lift and not np.array_equal(
            mapped.franka_target_q_rad, committed_arm
        ):
            raise V94PolicyTickSourceError(
                "occluded closure mapper failed to hold the Franka target"
            )
        command = ClosedLoopCommand(
            sequence=sequence,
            produced_monotonic_s=produced,
            observation_realtime_s=observation_realtime_s,
            previous_executed_action13_used=previous_executed_action13,
            raw_policy_action13=raw_action,
            executed_policy_action13=mapped.executed_policy_action13,
            franka_target_q_rad=mapped.franka_target_q_rad,
            rh56_angle_set_register_order=(
                mapped.rh56_angle_set_register_order
            ),
            hold_arm_target=not clearance_lift,
            previous_policy_action13_used=self._policy_previous_action13,
        )
        self._pending = _PendingTick(
            command=command,
            mapper_proposal=mapper_proposal,
            rh56_hardware_proposal=None,
            history=self._history,
            camera_frame_id=int(camera_frame_id),
            pointcloud_source_frame_id=int(
                self._last_accepted_pointcloud_source_frame_id or 0
            ),
            measured_franka_q_rad=measured_q,
            observation=None,
            ledger_previous_action13=previous_executed_action13,
            occluded_closure_hand_only=True,
            occluded_closure_clearance_lift=clearance_lift,
        )
        return command

    def _prepare_frozen_post_closure_locked(
        self,
        *,
        sequence: int,
        previous_executed_action13: np.ndarray,
        camera_frame_id: int,
        hold_reason: str,
        actuator_snapshot: Optional[V94FreshActuatorSnapshot],
        caller_now_monotonic_s: float,
        hard_deadline_monotonic_s: float,
    ) -> Optional[ClosedLoopCommand]:
        """Stage a frozen lift tick using fresh robot state, never stale PCD."""

        if actuator_snapshot is None or self._policy_io_recorder is not None:
            return None
        action_for_frozen_post_closure = getattr(
            self._policy, "action_for_frozen_post_closure", None
        )
        if not callable(action_for_frozen_post_closure):
            return None
        if self._last_committed_sequence <= 0 or self._history is None:
            return None
        observed_now = self._now_monotonic()
        franka_age = (
            observed_now
            - actuator_snapshot.franka_state_captured_monotonic_s
        )
        if franka_age < -self.maximum_pointcloud_future_skew_s:
            raise V94PolicyTickSourceError(
                "frozen lift Franka state exceeds the future-skew limit"
            )
        if franka_age > self.maximum_franka_hard_age_s:
            raise V94PolicyTickSourceError(
                "frozen lift Franka state exceeded the hard age limit"
            )
        if (
            franka_age > self.maximum_franka_action_age_s
            or actuator_snapshot.hold_arm_target
        ):
            return None
        if not self._check_franka_arrival_gate_locked(
            actuator_snapshot.measured_franka_q_rad,
            now_monotonic_s=observed_now,
        ):
            self._arrival_gate_hold_count += 1
            return None
        if observed_now >= hard_deadline_monotonic_s:
            raise V94PolicyTickSourceError(
                "hard deadline expired before frozen post-closure motion"
            )

        output = action_for_frozen_post_closure(
            sequence,
            observation_hold_reason=hold_reason,
        )
        if output is None:
            return None
        raw_action = _readonly_float_array(
            getattr(output, "action13", None),
            (13,),
            "frozen post-closure output.action13",
        )
        if np.any(raw_action < -1.0) or np.any(raw_action > 1.0):
            raise V94PolicyTickSourceError(
                "frozen post-closure action must lie in [-1,1]"
            )
        exact_arm = getattr(output, "exact_franka_target_q_rad", None)
        exact_hand = getattr(
            output, "exact_rh56_angle_set_register_order", None
        )
        if exact_arm is None or exact_hand is None:
            raise V94PolicyTickSourceError(
                "frozen post-closure motion must provide exact targets"
            )
        if not bool(getattr(output, "bypass_rh56_host_slew", False)):
            raise V94PolicyTickSourceError(
                "frozen post-closure motion lost its exact RH56 template"
            )
        mapper_proposal = self._mapper.propose_exact_targets(
            sequence,
            raw_action,
            franka_target_q_rad=exact_arm,
            rh56_angle_set_register_order=exact_hand,
            measured_q_rad=actuator_snapshot.measured_franka_q_rad,
            hold_arm_target=False,
        )
        produced = self._now_monotonic()
        if produced < caller_now_monotonic_s:
            raise V94PolicyTickSourceError("source monotonic clock regressed")
        if produced >= hard_deadline_monotonic_s:
            self._mapper.discard_unstaged(mapper_proposal)
            self._discard_policy_proposal(sequence)
            raise V94PolicyTickSourceError(
                "hard deadline expired during frozen post-closure mapping"
            )
        mapped = mapper_proposal.mapped
        command = ClosedLoopCommand(
            sequence=sequence,
            produced_monotonic_s=produced,
            observation_realtime_s=(
                actuator_snapshot.observation_realtime_s
            ),
            previous_executed_action13_used=previous_executed_action13,
            raw_policy_action13=raw_action,
            executed_policy_action13=mapped.executed_policy_action13,
            franka_target_q_rad=mapped.franka_target_q_rad,
            rh56_angle_set_register_order=(
                mapped.rh56_angle_set_register_order
            ),
            hold_arm_target=False,
            previous_policy_action13_used=self._policy_previous_action13,
        )
        self._pending = _PendingTick(
            command=command,
            mapper_proposal=mapper_proposal,
            rh56_hardware_proposal=None,
            history=self._history,
            camera_frame_id=int(camera_frame_id),
            pointcloud_source_frame_id=int(
                self._last_accepted_pointcloud_source_frame_id or 0
            ),
            measured_franka_q_rad=(
                actuator_snapshot.measured_franka_q_rad
            ),
            observation=None,
            ledger_previous_action13=previous_executed_action13,
            frozen_post_closure_without_pointcloud=True,
        )
        return command

    def prepare(
        self,
        *,
        sequence: int,
        previous_executed_action13: np.ndarray,
        now_monotonic_s: float,
        hard_deadline_monotonic_s: float,
    ) -> Union[ClosedLoopCommand, BoundedPolicyTickHold]:
        requested_sequence = _positive_integer(sequence, "sequence")
        caller_now = _finite_float(now_monotonic_s, "now_monotonic_s")
        hard_deadline = _finite_float(
            hard_deadline_monotonic_s, "hard_deadline_monotonic_s"
        )
        previous = _readonly_float_array(
            previous_executed_action13,
            (13,),
            "previous_executed_action13",
        )
        with self._lock:
            self._require_healthy_locked()
            if self._pending is not None:
                self._latch_locked("prepare called while a policy proposal is pending")
                raise V94PolicyTickSourceError(self._fault_reason)
            expected = self._last_committed_sequence + 1
            if requested_sequence != expected:
                self._latch_locked(
                    f"policy sequence mismatch: expected={expected}, "
                    f"actual={requested_sequence}"
                )
                raise V94PolicyTickSourceError(self._fault_reason)
            expected_ledger_previous = (
                self.initial_previous_action13
                if self._last_committed_sequence == 0
                else self._policy_previous_action13
            )
            if not np.array_equal(previous, expected_ledger_previous):
                if self._last_committed_sequence == 0:
                    self._latch_locked(
                        "initial previous action differs from checkpoint controller "
                        f"contract {self.action_controller_contract_id}"
                    )
                else:
                    self._latch_locked(
                        "hardware-ledger previous action differs from the expected "
                        f"transactional value for "
                        f"{self.action_controller_contract_id}"
                    )
                raise V94PolicyTickSourceError(self._fault_reason)
            policy_previous = self._policy_previous_action13.copy()
            if caller_now >= hard_deadline:
                raise V94PolicyTickSourceError(
                    "hard deadline expired before observation proposal"
                )

            try:
                observation = self._provider.snapshot(
                    sequence=requested_sequence,
                    now_monotonic_s=caller_now,
                    hard_deadline_monotonic_s=hard_deadline,
                )
                if not isinstance(observation, V94PolicyObservation):
                    raise TypeError(
                        "observation provider returned no V94PolicyObservation"
                    )
                after_snapshot = self._now_monotonic()
                if after_snapshot < caller_now:
                    raise V94PolicyTickSourceError("source monotonic clock regressed")
                if after_snapshot >= hard_deadline:
                    raise V94PolicyTickSourceError(
                        "hard deadline expired while acquiring observation"
                    )

                frame_id = observation.camera_frame_id
                pointcloud_source_frame_id = int(
                    observation.pointcloud_source_frame_id
                )
                if (
                    self._franka_state_age(
                        observation,
                        now_monotonic_s=after_snapshot,
                    )
                    > self.maximum_franka_action_age_s
                ):
                    return self._hold_locked(
                        sequence=requested_sequence,
                        frame_id=frame_id,
                        reason=FRANKA_STATE_STALE_HOLD_REASON,
                        observed_monotonic_s=after_snapshot,
                    )
                if not self._check_franka_arrival_gate_locked(
                    observation.measured_franka_q_rad,
                    now_monotonic_s=after_snapshot,
                ):
                    self._arrival_gate_hold_count += 1
                    return self._hold_locked(
                        sequence=requested_sequence,
                        frame_id=frame_id,
                        reason=FRANKA_ARRIVAL_GATE_HOLD_REASON,
                        observed_monotonic_s=after_snapshot,
                        retry_interval_s=(
                            self.franka_arrival_gate_retry_interval_s
                        ),
                    )
                if (
                    self._last_committed_frame_id is not None
                    and frame_id < self._last_committed_frame_id
                ):
                    raise V94PolicyTickSourceError(
                        "camera frame regressed behind the last committed action"
                    )
                uses = (
                    self._accepted_uses_for_last_frame
                    if pointcloud_source_frame_id
                    == self._last_accepted_pointcloud_source_frame_id
                    else 0
                )
                if uses >= self.maximum_actions_per_camera_frame:
                    return self._hold_locked(
                        sequence=requested_sequence,
                        frame_id=frame_id,
                        reason=CAMERA_REUSE_HOLD_REASON,
                        observed_monotonic_s=after_snapshot,
                    )

                if (
                    self._pointcloud_age(observation)
                    > self.maximum_object_pointcloud_age_s
                ):
                    return self._hold_locked(
                        sequence=requested_sequence,
                        frame_id=frame_id,
                        reason=OBJECT_POINTCLOUD_STALE_HOLD_REASON,
                        observed_monotonic_s=self._now_monotonic(),
                    )

                candidate = self._candidate_history_locked(
                    observation, policy_previous
                )
                action_for_sequence = getattr(self._policy, "action_for_sequence", None)
                if callable(action_for_sequence):
                    output = action_for_sequence(
                        requested_sequence,
                        candidate.points,
                        candidate.valid,
                        candidate.proprio,
                    )
                else:
                    output = self._policy.act(
                        candidate.points,
                        candidate.valid,
                        candidate.proprio,
                    )
                raw_action = _readonly_float_array(
                    getattr(output, "action13", None),
                    (13,),
                    "policy output.action13",
                )
                if np.any(raw_action < -1.0) or np.any(raw_action > 1.0):
                    raise V94PolicyTickSourceError(
                        "V94 raw policy action must lie in [-1,1]"
                    )
                after_policy = self._now_monotonic()
                if after_policy >= hard_deadline:
                    raise V94PolicyTickSourceError(
                        "hard deadline expired during policy inference"
                    )
                # A frame may be eligible before inference and expire during
                # it.  Since both policy and history are proposal-pure, this is
                # still a recoverable no-stage hold.
                if (
                    self._pointcloud_age(observation)
                    > self.maximum_object_pointcloud_age_s
                ):
                    self._discard_policy_proposal(requested_sequence)
                    return self._hold_locked(
                        sequence=requested_sequence,
                        frame_id=frame_id,
                        reason=OBJECT_POINTCLOUD_STALE_HOLD_REASON,
                        observed_monotonic_s=after_policy,
                    )
                if (
                    self._franka_state_age(
                        observation,
                        now_monotonic_s=after_policy,
                    )
                    > self.maximum_franka_action_age_s
                ):
                    self._discard_policy_proposal(requested_sequence)
                    return self._hold_locked(
                        sequence=requested_sequence,
                        frame_id=frame_id,
                        reason=FRANKA_STATE_STALE_HOLD_REASON,
                        observed_monotonic_s=after_policy,
                    )

                exact_franka_target = getattr(
                    output, "exact_franka_target_q_rad", None
                )
                exact_rh56_target = getattr(
                    output, "exact_rh56_angle_set_register_order", None
                )
                if (exact_franka_target is None) != (exact_rh56_target is None):
                    raise V94PolicyTickSourceError(
                        "exact replay must provide both Franka and RH56 targets"
                    )
                # An exact replay frame is one atomic Franka/RH56 waypoint.
                # If the live observation requires the Franka target to be
                # held, mapping the frame would retain the old arm target but
                # still commit the replay policy's new frame index.  A later
                # healthy observation could then skip an unsent waypoint and
                # exceed the native per-target delta envelope.  Discard the
                # proposal and retry the same ledger sequence instead; no
                # history, replay index, arm target, or hand target advances.
                if exact_franka_target is not None and observation.hold_arm_target:
                    self._discard_policy_proposal(requested_sequence)
                    return self._hold_locked(
                        sequence=requested_sequence,
                        frame_id=frame_id,
                        reason=EXACT_REPLAY_ARM_HOLD_REASON,
                        observed_monotonic_s=after_policy,
                    )
                if exact_franka_target is None:
                    mapper_kwargs: dict[str, object] = {}
                    if self._requires_shaper_q_d:
                        if observation.shaper_q_d_rad is None:
                            raise V94PolicyTickSourceError(
                                "q_d-relative controller contract requires "
                                "same-snapshot shaper_q_d_rad"
                            )
                        mapper_kwargs["shaper_q_d_rad"] = (
                            observation.shaper_q_d_rad
                        )
                    mapper_proposal = self._mapper.propose(
                        requested_sequence,
                        raw_action,
                        measured_q_rad=observation.measured_franka_q_rad,
                        hold_arm_target=observation.hold_arm_target,
                        **mapper_kwargs,
                    )
                else:
                    mapper_proposal = self._mapper.propose_exact_targets(
                        requested_sequence,
                        raw_action,
                        franka_target_q_rad=exact_franka_target,
                        rh56_angle_set_register_order=exact_rh56_target,
                        measured_q_rad=observation.measured_franka_q_rad,
                        hold_arm_target=observation.hold_arm_target,
                    )
                produced = self._now_monotonic()
                if produced < caller_now:
                    raise V94PolicyTickSourceError("source monotonic clock regressed")
                if produced >= hard_deadline:
                    self._mapper.discard_unstaged(mapper_proposal)
                    raise V94PolicyTickSourceError(
                        "hard deadline expired while mapping policy action"
                    )
                mapped = mapper_proposal.mapped
                rh56_hardware_proposal = None
                rh56_hardware_target = mapped.rh56_angle_set_register_order
                bypass_rh56_host_slew = bool(
                    getattr(output, "bypass_rh56_host_slew", False)
                )
                if (
                    self._rh56_hardware_command_shaper is not None
                    and not bypass_rh56_host_slew
                ):
                    rh56_hardware_proposal = self._rh56_hardware_command_shaper.propose(
                        requested_sequence,
                        mapped.rh56_angle_set_register_order,
                    )
                    rh56_hardware_target = (
                        rh56_hardware_proposal.angle_set_register_order
                    )
                command = ClosedLoopCommand(
                    sequence=requested_sequence,
                    produced_monotonic_s=produced,
                    observation_realtime_s=observation.observation_realtime_s,
                    previous_executed_action13_used=previous,
                    raw_policy_action13=raw_action,
                    executed_policy_action13=mapped.executed_policy_action13,
                    franka_target_q_rad=mapped.franka_target_q_rad,
                    rh56_angle_set_register_order=rh56_hardware_target,
                    hold_arm_target=observation.hold_arm_target,
                    previous_policy_action13_used=policy_previous,
                )
                self._pending = _PendingTick(
                    command=command,
                    mapper_proposal=mapper_proposal,
                    rh56_hardware_proposal=rh56_hardware_proposal,
                    history=candidate,
                    camera_frame_id=frame_id,
                    pointcloud_source_frame_id=pointcloud_source_frame_id,
                    measured_franka_q_rad=observation.measured_franka_q_rad,
                    observation=observation,
                    ledger_previous_action13=previous,
                )
                return command
            except V94RecoverableObservationHold as exc:
                # The provider explicitly guarantees that this rejection
                # occurred before policy/mapper mutation.  Retry the same
                # ledger sequence on a newer camera frame; no stale cloud or
                # no-op actuator command is substituted.  The sole exception
                # is a reviewed, bounded late RH56 closure completion after
                # the visual spatial/TTC gate already committed grasp:
                # it holds the exact last Franka target and consumes no stale
                # point cloud.
                try:
                    completion = (
                        self._prepare_occluded_closure_hand_only_locked(
                            sequence=requested_sequence,
                            previous_executed_action13=previous,
                            camera_frame_id=exc.camera_frame_id,
                            hold_reason=exc.reason,
                            actuator_snapshot=exc.fresh_actuator_snapshot,
                            caller_now_monotonic_s=caller_now,
                            hard_deadline_monotonic_s=hard_deadline,
                        )
                    )
                except BaseException as completion_error:
                    self._latch_locked(
                        f"{type(completion_error).__name__}: "
                        f"{completion_error}"
                    )
                    raise
                if completion is not None:
                    return completion
                try:
                    frozen_post_closure = (
                        self._prepare_frozen_post_closure_locked(
                            sequence=requested_sequence,
                            previous_executed_action13=previous,
                            camera_frame_id=exc.camera_frame_id,
                            hold_reason=exc.reason,
                            actuator_snapshot=(
                                exc.fresh_actuator_snapshot
                            ),
                            caller_now_monotonic_s=caller_now,
                            hard_deadline_monotonic_s=hard_deadline,
                        )
                    )
                except BaseException as frozen_error:
                    self._latch_locked(
                        f"{type(frozen_error).__name__}: {frozen_error}"
                    )
                    raise
                if frozen_post_closure is not None:
                    return frozen_post_closure
                return self._hold_locked(
                    sequence=requested_sequence,
                    frame_id=exc.camera_frame_id,
                    reason=exc.reason,
                    observed_monotonic_s=self._now_monotonic(),
                )
            except BaseException as exc:
                # Explicit observation holds returned above are not faults.
                # Every exception, including an action-mapper rejection, makes
                # this source terminal so a caller cannot reuse uncertain
                # proposal state.
                self._latch_locked(f"{type(exc).__name__}: {exc}")
                raise

    def commit(
        self,
        command: ClosedLoopCommand,
        *,
        action_ledger: ExecutedActionLedger,
    ) -> None:
        if not isinstance(action_ledger, ExecutedActionLedger):
            raise TypeError("action_ledger must be ExecutedActionLedger")
        committed = action_ledger.committed_snapshot()
        with self._lock:
            self._require_healthy_locked()
            pending = self._pending
            if pending is None or pending.command is not command:
                self._latch_locked("commit did not receive the exact pending command")
                raise V94PolicyTickSourceError(self._fault_reason)
            if committed.sequence != command.sequence:
                self._latch_locked(
                    "policy state cannot commit before exact ledger dual ACK"
                )
                raise V94PolicyTickSourceError(self._fault_reason)
            if not np.array_equal(
                committed.executed_policy_action13,
                command.executed_policy_action13,
            ):
                self._latch_locked(
                    "ledger committed action differs from policy command"
                )
                raise V94PolicyTickSourceError(self._fault_reason)
            try:
                self._mapper.commit(
                    pending.mapper_proposal,
                    action_ledger=action_ledger,
                )
                if (
                    self._rh56_hardware_command_shaper is not None
                    and pending.rh56_hardware_proposal is not None
                ):
                    self._rh56_hardware_command_shaper.commit(
                        pending.rh56_hardware_proposal
                    )
                commit_policy = getattr(
                    self._policy, "commit_replay_proposal", None
                )
                if callable(commit_policy):
                    commit_policy(command.sequence)
                if pending.occluded_closure_hand_only:
                    if pending.observation is not None:
                        raise AssertionError(
                            "occluded closure must not carry a stale observation"
                        )
                    clearance_lift = bool(
                        pending.occluded_closure_clearance_lift
                    )
                    if command.hold_arm_target == clearance_lift:
                        raise AssertionError(
                            "occluded closure lost its Franka hold/lift contract"
                        )
                    committed_arm, _committed_hand = (
                        self._mapper.committed_targets()
                    )
                    if not np.array_equal(
                        committed_arm, command.franka_target_q_rad
                    ):
                        raise AssertionError(
                            "occluded closure changed the committed Franka target"
                        )
                    # Deliberately do not advance point-cloud history, camera
                    # provenance, or frame-reuse counts: no observation was
                    # consumed.  Ordinarily only the RH56 target advances; a
                    # reviewed clearance-lift tick may also raise the Franka
                    # target using a current actuator snapshot.
                    self._policy_previous_action13 = (
                        command.executed_policy_action13
                    )
                    self._last_committed_sequence = command.sequence
                    if self.franka_arrival_gate_enabled:
                        self._arrival_gate_target_q_rad = _readonly_float_array(
                            command.franka_target_q_rad,
                            (7,),
                            "occluded closure arrival target",
                        )
                        self._arrival_gate_start_q_rad = _readonly_float_array(
                            (
                                pending.measured_franka_q_rad
                                if clearance_lift
                                else command.franka_target_q_rad
                            ),
                            (7,),
                            "occluded closure arrival start",
                        )
                        self._arrival_gate_reached_axes = np.ones(
                            7, dtype=bool
                        )
                        self._arrival_gate_started_monotonic_s = (
                            self._now_monotonic()
                        )
                    self._pending = None
                    return
                if pending.frozen_post_closure_without_pointcloud:
                    if pending.observation is not None:
                        raise AssertionError(
                            "frozen post-closure tick carried a stale observation"
                        )
                    if command.hold_arm_target:
                        raise AssertionError(
                            "frozen lift unexpectedly held the Franka target"
                        )
                    self._policy_previous_action13 = (
                        command.executed_policy_action13
                    )
                    self._last_committed_sequence = command.sequence
                    # The camera and robot state are current, but no object
                    # cloud was consumed.  Preserve history/source reuse while
                    # advancing transport liveness and the exact frozen lift.
                    self._last_committed_frame_id = pending.camera_frame_id
                    if self.franka_arrival_gate_enabled:
                        self._arrival_gate_target_q_rad = _readonly_float_array(
                            command.franka_target_q_rad,
                            (7,),
                            "frozen lift arrival target",
                        )
                        self._arrival_gate_start_q_rad = _readonly_float_array(
                            pending.measured_franka_q_rad,
                            (7,),
                            "frozen lift arrival start",
                        )
                        self._arrival_gate_reached_axes = np.less_equal(
                            np.abs(
                                self._arrival_gate_target_q_rad.astype(
                                    np.float64
                                )
                                - self._arrival_gate_start_q_rad.astype(
                                    np.float64
                                )
                            ),
                            self.franka_arrival_gate_tolerance_rad,
                        )
                        self._arrival_gate_started_monotonic_s = (
                            self._now_monotonic()
                        )
                    self._pending = None
                    return
                # Candidate arrays were already shape/finite validated.  These
                # assignments cannot fail after mapper commit and complete the
                # one logical transaction without running user/provider code.
                if pending.history is None or pending.observation is None:
                    raise AssertionError(
                        "ordinary policy commit lost its observation transaction"
                    )
                self._history = pending.history
                self._policy_previous_action13 = (
                    command.executed_policy_action13
                )
                if (
                    pending.pointcloud_source_frame_id
                    == self._last_accepted_pointcloud_source_frame_id
                ):
                    self._accepted_uses_for_last_frame += 1
                else:
                    self._last_accepted_pointcloud_source_frame_id = (
                        pending.pointcloud_source_frame_id
                    )
                    self._accepted_uses_for_last_frame = 1
                self._last_committed_frame_id = pending.camera_frame_id
                if (
                    self._accepted_uses_for_last_frame
                    > self.maximum_actions_per_camera_frame
                ):
                    raise AssertionError("camera reuse commit guard failed")
                self._last_committed_sequence = command.sequence
                if self.franka_arrival_gate_enabled:
                    self._arrival_gate_target_q_rad = _readonly_float_array(
                        command.franka_target_q_rad,
                        (7,),
                        "committed arrival-gate Franka target",
                    )
                    self._arrival_gate_start_q_rad = _readonly_float_array(
                        pending.measured_franka_q_rad,
                        (7,),
                        "committed arrival-gate Franka start",
                    )
                    self._arrival_gate_reached_axes = np.less_equal(
                        np.abs(
                            self._arrival_gate_target_q_rad.astype(np.float64)
                            - self._arrival_gate_start_q_rad.astype(np.float64)
                        ),
                        self.franka_arrival_gate_tolerance_rad,
                    )
                    self._arrival_gate_started_monotonic_s = (
                        self._now_monotonic()
                    )
                self._pending = None
                self._record_policy_io_locked(
                    logical_policy_step=(
                        self._startup_non_actuated_completed_steps
                        + command.sequence
                        - 1
                    ),
                    hardware_sequence=command.sequence,
                    startup_non_actuated=False,
                    history=pending.history,
                    observation=pending.observation,
                    policy_previous_action13=(
                        command.previous_policy_action13_used
                    ),
                    ledger_previous_action13=(
                        pending.ledger_previous_action13
                    ),
                    raw_model_action13=command.raw_policy_action13,
                    sent_action13=command.executed_policy_action13,
                    produced_monotonic_s=command.produced_monotonic_s,
                    franka_target_q_rad=command.franka_target_q_rad,
                    rh56_target_register_order=(
                        command.rh56_angle_set_register_order
                    ),
                    hardware_command_valid=True,
                )
            except BaseException as exc:
                self._latch_locked(f"{type(exc).__name__}: {exc}")
                raise

    def commit_rh56_feedback_after_dual_ack(
        self,
        sequence: int,
        feedback: Mapping[str, object],
    ) -> None:
        """Bind owner-cached physical hand feedback to one committed tick."""

        if isinstance(sequence, bool) or not isinstance(
            sequence, (int, np.integer)
        ):
            raise ValueError("RH56 feedback sequence must be an integer")
        if not isinstance(feedback, Mapping):
            raise TypeError("RH56 feedback must be a mapping")
        with self._lock:
            self._require_healthy_locked()
            if self._pending is not None:
                self._latch_locked(
                    "RH56 feedback callback overlapped a pending command"
                )
                raise V94PolicyTickSourceError(self._fault_reason)
            if int(sequence) != self._last_committed_sequence:
                self._latch_locked(
                    "RH56 feedback callback differs from the committed sequence"
                )
                raise V94PolicyTickSourceError(self._fault_reason)
            observer = getattr(
                self._policy, "commit_rh56_feedback_after_dual_ack", None
            )
            if not callable(observer):
                return
            try:
                observer(int(sequence), dict(feedback))
            except BaseException as exc:
                self._latch_locked(f"{type(exc).__name__}: {exc}")
                raise

    def abort(self, command: ClosedLoopCommand, *, reason: str) -> None:
        detail = str(reason).strip() or "unspecified staged policy abort"
        with self._lock:
            pending = self._pending
            if pending is None or pending.command is not command:
                self._latch_locked(
                    "abort did not receive the exact pending command: " + detail
                )
                return
            # Never turn a partial device failure into permission to reuse the
            # candidate.  The committed history and mapper targets remain at
            # the prior sequence; the pending proposal is intentionally left
            # unreachable behind this terminal source fault.
            self._latch_locked("staged policy command aborted: " + detail)


__all__ = [
    "CAMERA_REUSE_HOLD_REASON",
    "FRANKA_ARRIVAL_GATE_HOLD_REASON",
    "FRANKA_STATE_STALE_HOLD_REASON",
    "OBJECT_POINTCLOUD_INVALID_HOLD_REASON",
    "OBJECT_POINTCLOUD_STALE_HOLD_REASON",
    "OBSERVATION_SNAPSHOT_CONTRACT",
    "POLICY_TRANSACTIONAL_STATE_CONTRACT",
    "QD_G015_STARTUP_NON_ACTUATED_POLICY_STEPS",
    "NonActuatedPolicyTick",
    "SideEffectFreeV94ObservationProvider",
    "TransactionalBoundedV94PolicyTickSource",
    "TransactionalStateV94Policy",
    "V94FreshActuatorSnapshot",
    "V94PolicyObservation",
    "V94PolicyTickSourceError",
    "V94RecoverableObservationHold",
]
