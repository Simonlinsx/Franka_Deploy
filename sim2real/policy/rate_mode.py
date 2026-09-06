"""Single source of truth for the supported real-policy rate modes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from sim2real.contracts.v94 import (
    POLICY_HAND_ORDER,
    Q_HAND_SEMANTIC_CLOSE_RAD,
    REGISTER_HAND_ORDER,
)


SUPPORTED_POLICY_RATES_HZ = (20.0, 60.0)
RH56_TARGET_RATE_HZ = 20.0
MAXIMUM_SUPERVISED_NOMINAL_DURATION_S = 12.0
ABSOLUTE_MAXIMUM_SUPERVISED_STEPS = 720


def rh56_register_delta_envelope(
    *,
    policy_rate_hz: float,
    hardware_rate_hz: float,
    maximum_semantic_target_step_rad: float,
) -> tuple[int, ...]:
    """Map a checkpoint's semantic hand-step bound into register order."""

    ratio = float(policy_rate_hz) / float(hardware_rate_hz)
    ticks_per_update = int(round(ratio))
    if ticks_per_update < 1 or not math.isclose(
        ratio, float(ticks_per_update), rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("RH56 target rate must divide the policy rate exactly")
    step = float(maximum_semantic_target_step_rad)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("maximum semantic RH56 target step must be positive")
    by_name = dict(zip(POLICY_HAND_ORDER, Q_HAND_SEMANTIC_CLOSE_RAD.tolist()))
    close_register_order = np.asarray(
        [by_name[name] for name in REGISTER_HAND_ORDER], dtype=np.float64
    )
    return tuple(
        int(item)
        for item in np.ceil(
            ticks_per_update * 1000.0 * step / close_register_order
        ).astype(np.int32)
    )


@dataclass(frozen=True)
class PolicyRateMode:
    name: str
    policy_rate_hz: float
    control_dt_s: float
    rh56_target_rate_hz: float
    policy_ticks_per_rh56_update: int
    rh56_max_register_delta_per_update: tuple[int, ...]
    camera_max_policy_actions_per_frame: int
    maximum_supervised_steps: int

    @property
    def rh56_target_schedule_semantics(self) -> str:
        tick_name = {1: "one", 3: "three"}.get(
            self.policy_ticks_per_rh56_update,
            str(self.policy_ticks_per_rh56_update),
        )
        return (
            "20hz_latest_legal_absolute_target;"
            f"v94_{tick_name}_policy_tick_"
            "transparent_envelope"
        )


def resolve_policy_rate_mode(value: object) -> PolicyRateMode:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("policy rate must be exactly 20 or 60 Hz")
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("policy rate must be exactly 20 or 60 Hz") from exc
    if not math.isfinite(rate) or not any(
        math.isclose(rate, allowed, rel_tol=0.0, abs_tol=1.0e-12)
        for allowed in SUPPORTED_POLICY_RATES_HZ
    ):
        raise ValueError("policy rate must be exactly 20 or 60 Hz")
    rate = next(
        allowed
        for allowed in SUPPORTED_POLICY_RATES_HZ
        if math.isclose(rate, allowed, rel_tol=0.0, abs_tol=1.0e-12)
    )
    ratio = rate / RH56_TARGET_RATE_HZ
    ticks_per_update = int(round(ratio))
    if ticks_per_update < 1 or not math.isclose(
        ratio, float(ticks_per_update), rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("RH56 target rate must divide the policy rate exactly")

    register_envelope = rh56_register_delta_envelope(
        policy_rate_hz=rate,
        hardware_rate_hz=RH56_TARGET_RATE_HZ,
        maximum_semantic_target_step_rad=0.05,
    )
    # Training admits point-cloud latency 0..2 policy steps.  At 20 Hz that
    # means the original sample plus two further 50 ms ticks (three policy
    # uses total).  Keep the reviewed six-use 60 Hz transport bound unchanged.
    camera_reuse = (
        3
        if math.isclose(rate, 20.0, rel_tol=0.0, abs_tol=1.0e-12)
        else max(2, min(6, int(math.ceil(rate * 0.10))))
    )
    maximum_steps = min(
        ABSOLUTE_MAXIMUM_SUPERVISED_STEPS,
        int(math.floor(rate * MAXIMUM_SUPERVISED_NOMINAL_DURATION_S)),
    )
    return PolicyRateMode(
        name=f"{int(rate)}hz",
        policy_rate_hz=rate,
        control_dt_s=1.0 / rate,
        rh56_target_rate_hz=RH56_TARGET_RATE_HZ,
        policy_ticks_per_rh56_update=ticks_per_update,
        rh56_max_register_delta_per_update=register_envelope,
        camera_max_policy_actions_per_frame=camera_reuse,
        maximum_supervised_steps=maximum_steps,
    )


DEFAULT_POLICY_RATE_MODE = resolve_policy_rate_mode(60.0)


__all__ = [
    "ABSOLUTE_MAXIMUM_SUPERVISED_STEPS",
    "DEFAULT_POLICY_RATE_MODE",
    "MAXIMUM_SUPERVISED_NOMINAL_DURATION_S",
    "PolicyRateMode",
    "RH56_TARGET_RATE_HZ",
    "SUPPORTED_POLICY_RATES_HZ",
    "resolve_policy_rate_mode",
    "rh56_register_delta_envelope",
]
