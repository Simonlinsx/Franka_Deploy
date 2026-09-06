"""Rollback-safe V94 action-to-target mapping transaction."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from sim2real.contracts.actions import (
    LEGACY_FRANKA_ACTION_CONTRACT_ID,
    QD_G015_FRANKA_ACTION_CONTRACT_ID,
    V94ActionMapper,
    V94MappedAction,
    contracted_joint_limits_float64,
    inward_float32_joint_limits,
    normalize_franka_action_contract_id,
)

from .authorization import ClosedLoopProtocolError
from .commands import TransactionalV94ActionProposal
from .ledger import ExecutedActionLedger
from .validation import (
    _finite_scalar,
    _optional_positive_rate_vector,
    _positive_integer,
    _readonly_float_vector,
    _readonly_register_vector,
)


class TransactionalV94ActionMapper:
    """Rollback-safe wrapper around the exact V94 target mapping.

    A proposal runs the existing mapper in a temporary state.  The persistent
    arm/hand targets are updated only when :meth:`commit` observes an atomic
    :class:`ExecutedActionCommit` for the same proposal.  That commit can only
    be produced by :class:`ExecutedActionLedger` after both actuator owners
    acknowledge the command.

    ``joint_limit_margin_rad`` contracts the supplied limits before they reach
    :class:`V94ActionMapper`, matching the safe interval enforced by the FCI
    loop.  For the legacy accumulated-target contract, optional commissioned
    rate vectors reject, rather than silently reshape, a target transition
    that exceeds the verified policy-rate envelope.  The q_d g015 contract
    intentionally has no high-level target-rate guard; its separately bounded
    1 kHz generator owns velocity/acceleration/jerk shaping.
    """

    def __init__(
        self,
        *,
        initial_arm_target_q_rad: np.ndarray,
        initial_hand_target_q_policy_order_rad: np.ndarray,
        joint_limits_rad: np.ndarray,
        control_dt_s: float,
        q_hand_close_rad: np.ndarray,
        joint_limit_margin_rad: float = 0.0,
        commissioned_max_arm_target_rate_rad_s: object = None,
        commissioned_max_hand_target_rate_rad_s: object = None,
        measured_arm_envelope_rad: float = 0.05,
        arm_raw_gain_rad: float = 0.015,
        target_filter_alpha: float = 0.20,
        hand_target_filter_alpha: object = None,
        maximum_arm_target_step_rad: float = 0.015,
        maximum_hand_target_step_rad: float = 0.05,
        franka_action_contract_id: str = LEGACY_FRANKA_ACTION_CONTRACT_ID,
    ) -> None:
        control_dt = _finite_scalar(control_dt_s, "control_dt_s")
        margin = _finite_scalar(joint_limit_margin_rad, "joint_limit_margin_rad")
        if control_dt <= 0.0:
            raise ValueError("control_dt_s must be positive")
        if margin < 0.0:
            raise ValueError("joint_limit_margin_rad cannot be negative")
        safe_limits_f64 = contracted_joint_limits_float64(
            joint_limits_rad,
            margin_rad=margin,
        )
        safe_limits = inward_float32_joint_limits(
            joint_limits_rad,
            margin_rad=margin,
        )

        self.control_dt_s = control_dt
        self.joint_limit_margin_rad = margin
        self.franka_action_contract_id = normalize_franka_action_contract_id(
            franka_action_contract_id
        )
        self._joint_limits_f64 = safe_limits_f64
        self._joint_limits = safe_limits
        # Exact replay payloads are stored as float32.  Retain the legacy
        # nearest-float representation solely to recognize an endpoint whose
        # quantization landed outside the authoritative binary64 interval.
        # The command itself is always canonicalized to ``_joint_limits``.
        self._rounded_joint_limits = safe_limits_f64.astype(np.float32)
        self._q_hand_close = _readonly_float_vector(
            q_hand_close_rad, 6, "q_hand_close_rad"
        )
        if np.any(self._q_hand_close <= 0.0):
            raise ValueError("q_hand_close_rad must be positive")
        self._maximum_arm_rate = _optional_positive_rate_vector(
            commissioned_max_arm_target_rate_rad_s,
            7,
            "commissioned_max_arm_target_rate_rad_s",
        )
        self._maximum_hand_rate = _optional_positive_rate_vector(
            commissioned_max_hand_target_rate_rad_s,
            6,
            "commissioned_max_hand_target_rate_rad_s",
        )
        scalar_parameters = {
            "measured_arm_envelope_rad": measured_arm_envelope_rad,
            "arm_raw_gain_rad": arm_raw_gain_rad,
            "target_filter_alpha": target_filter_alpha,
            "hand_target_filter_alpha": (
                target_filter_alpha
                if hand_target_filter_alpha is None
                else hand_target_filter_alpha
            ),
            "maximum_arm_target_step_rad": maximum_arm_target_step_rad,
            "maximum_hand_target_step_rad": maximum_hand_target_step_rad,
        }
        self._mapper_parameters = {
            name: _finite_scalar(value, name)
            for name, value in scalar_parameters.items()
        }
        if self.franka_action_contract_id == LEGACY_FRANKA_ACTION_CONTRACT_ID:
            if any(value <= 0.0 for value in self._mapper_parameters.values()):
                raise ValueError("action mapper parameters must be positive")
        else:
            if (
                self._mapper_parameters["arm_raw_gain_rad"] < 0.0
                or self._mapper_parameters["target_filter_alpha"] < 0.0
                or self._mapper_parameters["maximum_arm_target_step_rad"] < 0.0
                or self._mapper_parameters["measured_arm_envelope_rad"] < 0.0
                or self._mapper_parameters["hand_target_filter_alpha"] <= 0.0
                or self._mapper_parameters["maximum_hand_target_step_rad"] <= 0.0
            ):
                raise ValueError("q_d-relative mapper parameters are invalid")
        if (
            self._mapper_parameters["target_filter_alpha"] > 1.0
            or self._mapper_parameters["hand_target_filter_alpha"] > 1.0
        ):
            raise ValueError("target filter alpha must not exceed one")

        self._arm_target = _readonly_float_vector(
            initial_arm_target_q_rad, 7, "initial_arm_target_q_rad"
        )
        self._hand_target = _readonly_float_vector(
            initial_hand_target_q_policy_order_rad,
            6,
            "initial_hand_target_q_policy_order_rad",
        )
        if np.any(self._arm_target < self._joint_limits[:, 0]) or np.any(
            self._arm_target > self._joint_limits[:, 1]
        ):
            raise ValueError(
                "initial_arm_target_q_rad is outside the margin-contracted limits"
            )
        if np.any(self._hand_target < 0.0) or np.any(
            self._hand_target > self._q_hand_close
        ):
            raise ValueError(
                "initial_hand_target_q_policy_order_rad is outside semantic limits"
            )
        self._lock = threading.Lock()
        self._last_committed_sequence = 0
        self._pending: Optional[TransactionalV94ActionProposal] = None
        self._fault_reason: Optional[str] = None

    @property
    def fault_reason(self) -> Optional[str]:
        with self._lock:
            return self._fault_reason

    @property
    def last_committed_sequence(self) -> int:
        with self._lock:
            return self._last_committed_sequence

    @property
    def pending_sequence(self) -> Optional[int]:
        with self._lock:
            return None if self._pending is None else self._pending.sequence

    def committed_targets(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._arm_target.copy(), self._hand_target.copy()

    def _fault_locked(self, reason: str) -> None:
        if self._fault_reason is None:
            self._fault_reason = str(reason).strip() or "unspecified mapper fault"

    def _require_healthy_locked(self) -> None:
        if self._fault_reason is not None:
            raise ClosedLoopProtocolError(
                f"transactional action mapper fault is latched: {self._fault_reason}"
            )

    @staticmethod
    def _rate_violation(
        actual: np.ndarray,
        maximum: Optional[np.ndarray],
        *,
        label: str,
    ) -> Optional[str]:
        if maximum is None:
            return None
        excess = np.flatnonzero(
            np.abs(np.asarray(actual, dtype=np.float64))
            > maximum + np.float64(1.0e-7)
        )
        if len(excess) == 0:
            return None
        index = int(excess[0])
        return (
            f"{label} target rate exceeds commissioned limit at axis {index + 1}: "
            f"actual={abs(float(actual[index])):.9f}rad/s, "
            f"limit={float(maximum[index]):.9f}rad/s"
        )

    def propose(
        self,
        sequence: int,
        action13: np.ndarray,
        *,
        measured_q_rad: np.ndarray,
        shaper_q_d_rad: object = None,
        hold_arm_target: bool = False,
    ) -> TransactionalV94ActionProposal:
        proposal_sequence = _positive_integer(sequence, "proposal sequence")
        with self._lock:
            self._require_healthy_locked()
            if self._pending is not None:
                reason = "cannot propose while another mapper proposal is pending"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            expected = self._last_committed_sequence + 1
            if proposal_sequence != expected:
                reason = (
                    f"mapper proposal sequence mismatch: expected={expected}, "
                    f"actual={proposal_sequence}"
                )
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            prior_arm = self._arm_target.copy()
            prior_hand = self._hand_target.copy()
            q_d = None
            if self.franka_action_contract_id == QD_G015_FRANKA_ACTION_CONTRACT_ID:
                if shaper_q_d_rad is None:
                    reason = (
                        "q_d g015 Franka action proposal requires shaper_q_d_rad"
                    )
                    self._fault_locked(reason)
                    raise ClosedLoopProtocolError(reason)
                try:
                    q_d = _readonly_float_vector(
                        shaper_q_d_rad, 7, "shaper_q_d_rad"
                    )
                except ValueError as exc:
                    reason = f"invalid q_d g015 shaper state: {exc}"
                    self._fault_locked(reason)
                    raise ClosedLoopProtocolError(reason) from exc
            temporary = V94ActionMapper(
                initial_arm_target_q_rad=prior_arm,
                initial_hand_target_q_policy_order_rad=prior_hand,
                joint_limits_rad=self._joint_limits,
                q_hand_close_rad=self._q_hand_close,
                franka_action_contract_id=self.franka_action_contract_id,
                **self._mapper_parameters,
            )
            mapped = temporary.map(
                action13,
                measured_q_rad=measured_q_rad,
                shaper_q_d_rad=q_d,
                hold_arm_target=bool(hold_arm_target),
            )
            arm_rate = (
                np.asarray(mapped.franka_target_q_rad, dtype=np.float64)
                - prior_arm.astype(np.float64)
            ) / self.control_dt_s
            hand_rate = (
                np.asarray(
                    mapped.rh56_target_q_policy_order_rad, dtype=np.float64
                )
                - prior_hand.astype(np.float64)
            ) / self.control_dt_s
            arm_violation = None
            if self.franka_action_contract_id == LEGACY_FRANKA_ACTION_CONTRACT_ID:
                arm_violation = self._rate_violation(
                    arm_rate, self._maximum_arm_rate, label="Franka"
                )
            violation = arm_violation or self._rate_violation(
                hand_rate, self._maximum_hand_rate, label="RH56"
            )
            if violation is not None:
                self._fault_locked(violation)
                raise ClosedLoopProtocolError(violation)
            proposal = TransactionalV94ActionProposal(
                sequence=proposal_sequence,
                mapped=mapped,
                prior_arm_target_q_rad=prior_arm,
                prior_hand_target_q_policy_order_rad=prior_hand,
                arm_target_rate_rad_s=arm_rate,
                hand_target_rate_rad_s=hand_rate,
            )
            self._pending = proposal
            return proposal

    @staticmethod
    def _register_order_to_policy_order(value: np.ndarray) -> np.ndarray:
        """Return RH56 register values in V94 semantic policy order."""

        registers = _readonly_register_vector(value, "exact RH56 target")
        # register: little, ring, middle, index, thumb_bending, thumb_rotation
        # policy:   thumb_rotation, thumb_bending, index, middle, ring, little
        return registers[np.asarray([5, 4, 3, 2, 1, 0], dtype=np.int64)]

    def propose_exact_targets(
        self,
        sequence: int,
        action13: np.ndarray,
        *,
        franka_target_q_rad: np.ndarray,
        rh56_angle_set_register_order: np.ndarray,
        measured_q_rad: np.ndarray,
        hold_arm_target: bool = False,
    ) -> TransactionalV94ActionProposal:
        """Propose simulator-recorded actuator targets without remapping them.

        This path is intentionally limited to replay payloads that carry both
        exact target arrays.  ``action13`` remains the dual-committed previous
        action used by the observation contract, but it is not mapped a second
        time. Continuous Franka motion is generated by the native V225 1 kHz
        interpolator followed by libfranka's official rate limiter.
        """

        proposal_sequence = _positive_integer(sequence, "proposal sequence")
        raw_action = _readonly_float_vector(action13, 13, "action13")
        if np.any(raw_action < -1.0) or np.any(raw_action > 1.0):
            raise ValueError("exact replay action13 must lie in [-1,1]")
        requested_arm = _readonly_float_vector(
            franka_target_q_rad, 7, "exact franka_target_q_rad"
        )
        measured = _readonly_float_vector(measured_q_rad, 7, "measured_q_rad")
        del measured  # freshness/tracking is enforced by the observation/native owners
        registers = _readonly_register_vector(
            rh56_angle_set_register_order,
            "exact rh56_angle_set_register_order",
        )
        with self._lock:
            self._require_healthy_locked()
            if self._pending is not None:
                reason = "cannot propose while another mapper proposal is pending"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            expected = self._last_committed_sequence + 1
            if proposal_sequence != expected:
                reason = (
                    f"mapper proposal sequence mismatch: expected={expected}, "
                    f"actual={proposal_sequence}"
                )
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            prior_arm = self._arm_target.copy()
            prior_hand = self._hand_target.copy()
            next_arm = prior_arm if bool(hold_arm_target) else requested_arm.copy()
            boundary_canonicalized = False
            if not bool(hold_arm_target):
                lower_quantized_outward = (
                    self._rounded_joint_limits[:, 0].astype(np.float64)
                    < self._joint_limits_f64[:, 0]
                )
                upper_quantized_outward = (
                    self._rounded_joint_limits[:, 1].astype(np.float64)
                    > self._joint_limits_f64[:, 1]
                )
                use_lower = lower_quantized_outward & (
                    next_arm == self._rounded_joint_limits[:, 0]
                )
                use_upper = upper_quantized_outward & (
                    next_arm == self._rounded_joint_limits[:, 1]
                )
                next_arm[use_lower] = self._joint_limits[use_lower, 0]
                next_arm[use_upper] = self._joint_limits[use_upper, 1]
                boundary_canonicalized = bool(np.any(use_lower | use_upper))
            if np.any(next_arm < self._joint_limits[:, 0]) or np.any(
                next_arm > self._joint_limits[:, 1]
            ):
                reason = "exact Franka replay target is outside joint limits"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)

            register_policy = self._register_order_to_policy_order(registers)
            next_hand = (
                self._q_hand_close
                * (
                    np.float32(1.0)
                    - register_policy.astype(np.float32) / np.float32(1000.0)
                )
            ).astype(np.float32)
            arm_rate = (
                next_arm.astype(np.float64) - prior_arm.astype(np.float64)
            ) / self.control_dt_s
            hand_rate = (
                next_hand.astype(np.float64) - prior_hand.astype(np.float64)
            ) / self.control_dt_s
            reasons = []
            if hold_arm_target:
                reasons.append("external hold gate held exact Franka replay target")
            if boundary_canonicalized:
                reasons.append("Franka float32 boundary canonicalized inward")
            mapped = V94MappedAction(
                raw_policy_action13=raw_action,
                executed_policy_action13=raw_action,
                franka_target_q_rad=next_arm,
                rh56_target_q_policy_order_rad=next_hand,
                rh56_angle_set_register_order=registers,
                clipped=bool(reasons),
                reasons=tuple(reasons),
                hold_arm_target=bool(hold_arm_target),
            )
            proposal = TransactionalV94ActionProposal(
                sequence=proposal_sequence,
                mapped=mapped,
                prior_arm_target_q_rad=prior_arm,
                prior_hand_target_q_policy_order_rad=prior_hand,
                arm_target_rate_rad_s=arm_rate,
                hand_target_rate_rad_s=hand_rate,
            )
            self._pending = proposal
            return proposal

    def discard_unstaged(self, proposal: TransactionalV94ActionProposal) -> None:
        """Discard an exact proposal before it is published to either actuator."""

        with self._lock:
            self._require_healthy_locked()
            if self._pending is not proposal:
                reason = "only the exact pending mapper proposal can be discarded"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            self._pending = None

    def commit(
        self,
        proposal: TransactionalV94ActionProposal,
        *,
        action_ledger: ExecutedActionLedger,
    ) -> None:
        """Commit only after the ledger proves this action was dual-acknowledged."""

        if not isinstance(action_ledger, ExecutedActionLedger):
            raise TypeError("action_ledger must be ExecutedActionLedger")
        committed = action_ledger.committed_snapshot()
        with self._lock:
            self._require_healthy_locked()
            if self._pending is not proposal:
                reason = "mapper commit did not receive the exact pending proposal"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            expected = self._last_committed_sequence + 1
            if proposal.sequence != expected:
                reason = "pending mapper proposal sequence changed before commit"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            if committed.sequence != proposal.sequence:
                raise ClosedLoopProtocolError(
                    "mapper proposal has not been dual-acknowledged by the ledger"
                )
            if not np.array_equal(
                committed.executed_policy_action13,
                proposal.mapped.executed_policy_action13,
            ):
                reason = "ledger committed a different action for this mapper sequence"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            if not np.array_equal(
                self._arm_target, proposal.prior_arm_target_q_rad
            ) or not np.array_equal(
                self._hand_target, proposal.prior_hand_target_q_policy_order_rad
            ):
                reason = "mapper committed targets changed after proposal"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            self._arm_target = _readonly_float_vector(
                proposal.mapped.franka_target_q_rad,
                7,
                "committed franka_target_q_rad",
            )
            self._hand_target = _readonly_float_vector(
                proposal.mapped.rh56_target_q_policy_order_rad,
                6,
                "committed rh56_target_q_policy_order_rad",
            )
            self._last_committed_sequence = proposal.sequence
            self._pending = None

__all__ = ["TransactionalV94ActionMapper"]
