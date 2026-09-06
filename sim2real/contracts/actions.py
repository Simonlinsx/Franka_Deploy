"""Exact V94 13-D policy-action to FR3/RH56 target mapping."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Optional, Tuple

import numpy as np

from .v94 import (
    POLICY_HAND_ORDER,
    Q_HAND_SEMANTIC_CLOSE_RAD,
    REGISTER_HAND_ORDER,
)


LEGACY_FRANKA_ACTION_CONTRACT_ID = "v94_inspire_semantic_13d"
QD_G015_FRANKA_ACTION_CONTRACT_ID = (
    "franka_inspire_qd_g015_student_controller_v1"
)
QD_G015_EFFECTIVE_ARM_GAIN_RAD = np.float32(0.06)
SUPPORTED_FRANKA_ACTION_CONTRACT_IDS = frozenset(
    {
        LEGACY_FRANKA_ACTION_CONTRACT_ID,
        QD_G015_FRANKA_ACTION_CONTRACT_ID,
    }
)


def contracted_joint_limits_float64(
    joint_limits_rad: object,
    *,
    margin_rad: object = 0.0,
) -> np.ndarray:
    """Return the authoritative margin-contracted Franka interval.

    The commissioning profile and native safety contract are binary64.  The
    contraction must therefore happen before any conversion to the policy's
    binary32 target representation; doing the addition after a float32 cast
    can move a decimal endpoint a few ULPs outside the native interval.
    """

    source = np.asarray(joint_limits_rad, dtype=np.float64)
    if source.shape != (7, 2) or not np.all(np.isfinite(source)):
        raise ValueError("joint_limits_rad must have finite shape (7,2)")
    if np.any(source[:, 0] >= source[:, 1]):
        raise ValueError("joint_limits_rad lower bounds must be below upper bounds")
    try:
        margin = float(margin_rad)
    except (TypeError, ValueError) as exc:
        raise ValueError("joint limit margin must be finite and non-negative") from exc
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("joint limit margin must be finite and non-negative")

    contracted = source.copy()
    contracted[:, 0] += margin
    contracted[:, 1] -= margin
    if np.any(contracted[:, 0] >= contracted[:, 1]):
        raise ValueError("joint limit margin consumes a joint interval")
    return contracted


def inward_float32_joint_limits(
    joint_limits_rad: object,
    *,
    margin_rad: object = 0.0,
) -> np.ndarray:
    """Return float32 endpoints that remain inside the binary64 contract."""

    contracted = contracted_joint_limits_float64(
        joint_limits_rad,
        margin_rad=margin_rad,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        limits = contracted.astype(np.float32)
    if not np.all(np.isfinite(limits)):
        raise ValueError("joint_limits_rad cannot be represented as finite float32")

    lower_outward = limits[:, 0].astype(np.float64) < contracted[:, 0]
    upper_outward = limits[:, 1].astype(np.float64) > contracted[:, 1]
    limits[lower_outward, 0] = np.nextafter(
        limits[lower_outward, 0], np.float32(np.inf)
    )
    limits[upper_outward, 1] = np.nextafter(
        limits[upper_outward, 1], np.float32(-np.inf)
    )
    if (
        not np.all(np.isfinite(limits))
        or np.any(limits[:, 0] >= limits[:, 1])
        or np.any(limits[:, 0].astype(np.float64) < contracted[:, 0])
        or np.any(limits[:, 1].astype(np.float64) > contracted[:, 1])
    ):
        raise ValueError("joint limits have no safe float32 representation")
    return limits


def normalize_franka_action_contract_id(value: object) -> str:
    contract_id = str(value).strip()
    if contract_id not in SUPPORTED_FRANKA_ACTION_CONTRACT_IDS:
        raise ValueError(f"unsupported Franka action contract: {contract_id!r}")
    return contract_id


@dataclass(frozen=True)
class V94MappedAction:
    raw_policy_action13: np.ndarray
    executed_policy_action13: np.ndarray
    franka_target_q_rad: np.ndarray
    rh56_target_q_policy_order_rad: np.ndarray
    rh56_angle_set_register_order: np.ndarray
    clipped: bool
    reasons: Tuple[str, ...]
    hold_arm_target: bool


def _readonly_register_vector(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (6,) or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain six finite integer values")
    numeric = raw.astype(np.float64)
    if not np.all(numeric == np.rint(numeric)):
        raise ValueError(f"{name} must contain integer values")
    result = numeric.astype(np.int32)
    if np.any(result < 0) or np.any(result > 1000):
        raise ValueError(f"{name} must lie in [0,1000]")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RH56HardwareCommandProposal:
    """One rollback-safe 20 Hz device-target proposal.

    The policy and its previous-action chain remain at the selected 20/60 Hz
    contract.  The physical RH56 setpoint is always released at its proven
    20 Hz streaming rate.
    """

    sequence: int
    desired_angle_set_register_order: np.ndarray
    bounded_desired_angle_set_register_order: np.ndarray
    angle_set_register_order: np.ndarray
    prior_angle_set_register_order: np.ndarray
    hardware_update_due: bool

    def __post_init__(self) -> None:
        sequence = int(self.sequence)
        if sequence <= 0:
            raise ValueError("RH56 hardware proposal sequence must be positive")
        object.__setattr__(self, "sequence", sequence)
        for field_name in (
            "desired_angle_set_register_order",
            "bounded_desired_angle_set_register_order",
            "angle_set_register_order",
            "prior_angle_set_register_order",
        ):
            object.__setattr__(
                self,
                field_name,
                _readonly_register_vector(getattr(self, field_name), field_name),
            )
        if not isinstance(self.hardware_update_due, (bool, np.bool_)):
            raise ValueError("hardware_update_due must be boolean")
        object.__setattr__(
            self, "hardware_update_due", bool(self.hardware_update_due)
        )


class TransactionalRH56HardwareCommandShaper:
    """Convert supported policy-rate targets into 20 Hz RH56 setpoints.

    The installed RH56 streaming path has physical evidence for 20 Hz target
    updates with bounded register-space slew.  Sending every intermediate
    policy target can continually restart the hand firmware's motion planner.
    This shaper therefore publishes the newest desired target only on an
    integer policy/update boundary and holds the last committed device target
    on intervening policy ticks.

    Proposals are committed only after the existing dual-device action ledger
    commits the corresponding command.  A failed or unstaged command cannot
    advance the physical target schedule.
    """

    def __init__(
        self,
        *,
        initial_angle_set_register_order: object,
        policy_rate_hz: object,
        hardware_rate_hz: object,
        maximum_register_delta_per_update: object,
        minimum_angle_set_register_order: object = (0,) * 6,
        maximum_angle_set_register_order: object = (1000,) * 6,
    ) -> None:
        try:
            policy_rate = float(policy_rate_hz)
            hardware_rate = float(hardware_rate_hz)
        except (TypeError, ValueError) as exc:
            raise ValueError("RH56 policy/hardware rates must be finite") from exc
        if (
            not math.isfinite(policy_rate)
            or not math.isfinite(hardware_rate)
            or policy_rate <= 0.0
            or hardware_rate <= 0.0
            or hardware_rate > policy_rate
        ):
            raise ValueError("RH56 policy/hardware rates must be positive and ordered")
        ratio = policy_rate / hardware_rate
        divider = int(round(ratio))
        if divider < 1 or not math.isclose(
            ratio, float(divider), rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "RH56 hardware rate must divide the policy rate exactly"
            )
        maximum_delta = _readonly_register_vector(
            maximum_register_delta_per_update,
            "maximum_register_delta_per_update",
        )
        if np.any(maximum_delta <= 0):
            raise ValueError("maximum_register_delta_per_update must be positive")
        lower = _readonly_register_vector(
            minimum_angle_set_register_order,
            "minimum_angle_set_register_order",
        )
        upper = _readonly_register_vector(
            maximum_angle_set_register_order,
            "maximum_angle_set_register_order",
        )
        if np.any(lower >= upper):
            raise ValueError("RH56 hardware command intervals are invalid")
        initial = _readonly_register_vector(
            initial_angle_set_register_order,
            "initial_angle_set_register_order",
        )
        if np.any(initial < lower) or np.any(initial > upper):
            raise ValueError(
                "initial RH56 hardware target is outside the calibrated interval"
            )

        self.policy_rate_hz = policy_rate
        self.hardware_rate_hz = hardware_rate
        self.policy_ticks_per_hardware_update = divider
        self.maximum_register_delta_per_update = maximum_delta
        self.minimum_angle_set_register_order = lower
        self.maximum_angle_set_register_order = upper
        self._lock = threading.Lock()
        self._target = initial
        self._last_committed_sequence = 0
        self._pending: Optional[RH56HardwareCommandProposal] = None
        self._fault_reason: Optional[str] = None

    @property
    def last_committed_sequence(self) -> int:
        with self._lock:
            return self._last_committed_sequence

    @property
    def pending_sequence(self) -> Optional[int]:
        with self._lock:
            return None if self._pending is None else self._pending.sequence

    @property
    def fault_reason(self) -> Optional[str]:
        with self._lock:
            return self._fault_reason

    def committed_target(self) -> np.ndarray:
        with self._lock:
            return self._target.copy()

    def _fault_locked(self, reason: str) -> None:
        if self._fault_reason is None:
            self._fault_reason = str(reason).strip() or "unspecified RH56 shaper fault"

    def _require_healthy_locked(self) -> None:
        if self._fault_reason is not None:
            raise RuntimeError(
                f"RH56 hardware command shaper fault is latched: "
                f"{self._fault_reason}"
            )

    def propose(
        self,
        sequence: int,
        desired_angle_set_register_order: object,
    ) -> RH56HardwareCommandProposal:
        proposal_sequence = int(sequence)
        with self._lock:
            self._require_healthy_locked()
            if self._pending is not None:
                self._fault_locked(
                    "cannot propose while another RH56 hardware target is pending"
                )
                raise RuntimeError(self._fault_reason)
            expected = self._last_committed_sequence + 1
            if proposal_sequence != expected:
                self._fault_locked(
                    f"RH56 hardware proposal sequence mismatch: expected={expected}, "
                    f"actual={proposal_sequence}"
                )
                raise RuntimeError(self._fault_reason)
            desired = _readonly_register_vector(
                desired_angle_set_register_order,
                "desired_angle_set_register_order",
            )
            bounded = np.clip(
                desired,
                self.minimum_angle_set_register_order,
                self.maximum_angle_set_register_order,
            ).astype(np.int32)
            due = (
                (proposal_sequence - 1)
                % self.policy_ticks_per_hardware_update
                == 0
            )
            prior = self._target.copy()
            if due:
                delta = np.clip(
                    bounded.astype(np.int64) - prior.astype(np.int64),
                    -self.maximum_register_delta_per_update.astype(np.int64),
                    self.maximum_register_delta_per_update.astype(np.int64),
                )
                target = (prior.astype(np.int64) + delta).astype(np.int32)
            else:
                target = prior
            proposal = RH56HardwareCommandProposal(
                sequence=proposal_sequence,
                desired_angle_set_register_order=desired,
                bounded_desired_angle_set_register_order=bounded,
                angle_set_register_order=target,
                prior_angle_set_register_order=prior,
                hardware_update_due=due,
            )
            self._pending = proposal
            return proposal

    def discard_unstaged(self, proposal: RH56HardwareCommandProposal) -> None:
        with self._lock:
            self._require_healthy_locked()
            if self._pending is not proposal:
                self._fault_locked(
                    "only the exact pending RH56 hardware proposal can be discarded"
                )
                raise RuntimeError(self._fault_reason)
            self._pending = None

    def commit(self, proposal: RH56HardwareCommandProposal) -> None:
        with self._lock:
            self._require_healthy_locked()
            if self._pending is not proposal:
                self._fault_locked(
                    "RH56 hardware commit did not receive the pending proposal"
                )
                raise RuntimeError(self._fault_reason)
            if proposal.sequence != self._last_committed_sequence + 1:
                self._fault_locked(
                    "RH56 hardware proposal sequence changed before commit"
                )
                raise RuntimeError(self._fault_reason)
            if not np.array_equal(
                self._target, proposal.prior_angle_set_register_order
            ):
                self._fault_locked(
                    "RH56 committed hardware target changed after proposal"
                )
                raise RuntimeError(self._fault_reason)
            self._target = _readonly_register_vector(
                proposal.angle_set_register_order,
                "committed RH56 hardware target",
            )
            self._last_committed_sequence = proposal.sequence
            self._pending = None

    def abort(self, proposal: RH56HardwareCommandProposal, *, reason: str) -> None:
        detail = str(reason).strip() or "unspecified RH56 hardware proposal abort"
        with self._lock:
            if self._pending is not proposal:
                self._fault_locked(
                    "RH56 hardware abort did not receive the pending proposal: "
                    + detail
                )
                raise RuntimeError(self._fault_reason)
            self._fault_locked(detail)
            self._pending = None


class V94ActionMapper:
    """Stateful pre-action mapper with versioned simulator target semantics.

    The legacy contract accumulates the arm target from the previously
    committed target.  The q_d g015 contract deliberately does not: every arm
    proposal is based on the explicitly supplied, same-snapshot software
    shaper ``q_d``.  Hand target state remains persistent in both modes.
    """

    def __init__(
        self,
        *,
        initial_arm_target_q_rad: np.ndarray,
        initial_hand_target_q_policy_order_rad: np.ndarray,
        joint_limits_rad: np.ndarray,
        q_hand_close_rad: np.ndarray = Q_HAND_SEMANTIC_CLOSE_RAD,
        measured_arm_envelope_rad: float = 0.05,
        arm_raw_gain_rad: float = 0.015,
        target_filter_alpha: float = 0.20,
        hand_target_filter_alpha: Optional[float] = None,
        maximum_arm_target_step_rad: float = 0.015,
        maximum_hand_target_step_rad: float = 0.05,
        franka_action_contract_id: str = LEGACY_FRANKA_ACTION_CONTRACT_ID,
    ) -> None:
        self.franka_action_contract_id = normalize_franka_action_contract_id(
            franka_action_contract_id
        )
        self.arm_target = self._vector(
            initial_arm_target_q_rad, 7, "initial_arm_target_q_rad"
        )
        self.hand_target = self._vector(
            initial_hand_target_q_policy_order_rad,
            6,
            "initial_hand_target_q_policy_order_rad",
        )
        # Policy targets are intentionally computed and stored as float32,
        # while the commissioned Franka safety interval is a float64
        # contract.  A decimal bound such as J4=-2.9921 is not exactly
        # representable in binary32: the nearest float32 is
        # -2.9921000003814697, which lies just *outside* the real lower bound.
        # Preserve the original double interval and move only an outward-
        # rounded endpoint one binary32 step inward.  This neither widens the
        # commissioned interval nor changes endpoints that already rounded
        # inward.
        limits = inward_float32_joint_limits(joint_limits_rad)
        self.joint_limits = limits.copy()
        self.q_hand_close = self._vector(q_hand_close_rad, 6, "q_hand_close_rad")
        if np.any(self.q_hand_close <= 0.0):
            raise ValueError("q_hand_close_rad must be positive")
        hand_alpha = (
            target_filter_alpha
            if hand_target_filter_alpha is None
            else hand_target_filter_alpha
        )
        arm_scalars = (
            measured_arm_envelope_rad,
            arm_raw_gain_rad,
            target_filter_alpha,
            maximum_arm_target_step_rad,
        )
        hand_scalars = (hand_alpha, maximum_hand_target_step_rad)
        if not all(
            np.isfinite(value) and float(value) > 0.0 for value in hand_scalars
        ):
            raise ValueError("action mapper limits must be finite and positive")
        if self.franka_action_contract_id == LEGACY_FRANKA_ACTION_CONTRACT_ID:
            if not all(
                np.isfinite(value) and float(value) > 0.0 for value in arm_scalars
            ):
                raise ValueError("action mapper limits must be finite and positive")
        elif not all(np.isfinite(value) for value in arm_scalars):
            raise ValueError("q_d-relative arm mapper parameters must be finite")
        if float(target_filter_alpha) > 1.0 or float(hand_alpha) > 1.0:
            raise ValueError("target filter alpha must not exceed one")
        self.measured_arm_envelope = np.float32(measured_arm_envelope_rad)
        self.arm_raw_gain = np.float32(arm_raw_gain_rad)
        self.alpha = np.float32(target_filter_alpha)
        self.hand_alpha = np.float32(hand_alpha)
        self.maximum_arm_step = np.float32(maximum_arm_target_step_rad)
        self.maximum_hand_step = np.float32(maximum_hand_target_step_rad)

    @staticmethod
    def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain {size} finite values")
        return result.copy()

    @staticmethod
    def policy_to_register_order(value: np.ndarray) -> np.ndarray:
        policy = np.asarray(value)
        if policy.shape != (6,):
            raise ValueError("policy-order hand vector must have shape (6,)")
        by_name = dict(zip(POLICY_HAND_ORDER, policy.tolist()))
        return np.asarray([by_name[name] for name in REGISTER_HAND_ORDER])

    def map(
        self,
        action13: np.ndarray,
        *,
        measured_q_rad: np.ndarray,
        shaper_q_d_rad: Optional[np.ndarray] = None,
        hold_arm_target: bool = False,
    ) -> V94MappedAction:
        raw = self._vector(action13, 13, "action13")
        measured = self._vector(measured_q_rad, 7, "measured_q_rad")
        reasons = []
        executed = np.clip(raw, np.float32(-1.0), np.float32(1.0))
        if not np.array_equal(executed, raw):
            reasons.append("policy action clipped to [-1,1]")
        executed = executed.astype(np.float32, copy=True)
        if hold_arm_target:
            if np.any(executed[:7] != 0.0):
                reasons.append("external hold gate zeroed Franka action")
            executed[:7] = 0.0

        previous_arm = self.arm_target.copy()
        if self.franka_action_contract_id == QD_G015_FRANKA_ACTION_CONTRACT_ID:
            if shaper_q_d_rad is None:
                raise ValueError(
                    "q_d g015 Franka action mapping requires shaper_q_d_rad"
                )
            shaper_q_d = self._vector(shaper_q_d_rad, 7, "shaper_q_d_rad")
            next_arm = (
                shaper_q_d + QD_G015_EFFECTIVE_ARM_GAIN_RAD * executed[:7]
            )
            bounded_arm = np.clip(
                next_arm, self.joint_limits[:, 0], self.joint_limits[:, 1]
            )
            if not np.array_equal(bounded_arm, next_arm):
                reasons.append("Franka joint-limit interval clipped")
        else:
            raw_arm_target = previous_arm + self.arm_raw_gain * executed[:7]
            next_arm = (
                self.alpha * raw_arm_target
                + (np.float32(1.0) - self.alpha) * previous_arm
            )
            arm_delta = np.clip(
                next_arm - previous_arm,
                -self.maximum_arm_step,
                self.maximum_arm_step,
            )
            if not np.array_equal(arm_delta, next_arm - previous_arm):
                reasons.append("Franka per-tick target step clipped")
            next_arm = previous_arm + arm_delta
            measured_lower = measured - self.measured_arm_envelope
            measured_upper = measured + self.measured_arm_envelope
            safe_lower = np.maximum(self.joint_limits[:, 0], measured_lower)
            safe_upper = np.minimum(self.joint_limits[:, 1], measured_upper)
            bounded_arm = np.clip(next_arm, safe_lower, safe_upper)
            if not np.array_equal(bounded_arm, next_arm):
                reasons.append("Franka measured-q/joint-limit envelope clipped")
        self.arm_target = bounded_arm.astype(np.float32, copy=True)

        fraction = np.clip(
            (executed[7:] + np.float32(1.0)) * np.float32(0.5),
            np.float32(0.0),
            np.float32(1.0),
        )
        raw_hand = fraction * self.q_hand_close
        previous_hand = self.hand_target.copy()
        filtered_hand = (
            self.hand_alpha * raw_hand
            + (np.float32(1.0) - self.hand_alpha) * previous_hand
        )
        hand_delta = np.clip(
            filtered_hand - previous_hand,
            -self.maximum_hand_step,
            self.maximum_hand_step,
        )
        if not np.array_equal(hand_delta, filtered_hand - previous_hand):
            reasons.append("RH56 virtual target step clipped")
        next_hand = np.clip(
            previous_hand + hand_delta, np.float32(0.0), self.q_hand_close
        )
        if not np.array_equal(next_hand, previous_hand + hand_delta):
            reasons.append("RH56 semantic joint limits clipped")
        self.hand_target = next_hand.astype(np.float32, copy=True)

        register_policy_order = np.rint(
            np.float32(1000.0)
            * (np.float32(1.0) - self.hand_target / self.q_hand_close)
        ).astype(np.int32)
        registers = self.policy_to_register_order(register_policy_order).astype(
            np.int32
        )
        if np.any(registers < 0) or np.any(registers > 1000):
            raise RuntimeError("mapped RH56 register target is outside [0,1000]")
        return V94MappedAction(
            raw_policy_action13=raw,
            executed_policy_action13=executed,
            franka_target_q_rad=self.arm_target.copy(),
            rh56_target_q_policy_order_rad=self.hand_target.copy(),
            rh56_angle_set_register_order=registers,
            clipped=bool(reasons),
            reasons=tuple(reasons),
            hold_arm_target=bool(hold_arm_target),
        )


__all__ = [
    "LEGACY_FRANKA_ACTION_CONTRACT_ID",
    "QD_G015_EFFECTIVE_ARM_GAIN_RAD",
    "QD_G015_FRANKA_ACTION_CONTRACT_ID",
    "RH56HardwareCommandProposal",
    "SUPPORTED_FRANKA_ACTION_CONTRACT_IDS",
    "TransactionalRH56HardwareCommandShaper",
    "V94ActionMapper",
    "V94MappedAction",
    "normalize_franka_action_contract_id",
]
