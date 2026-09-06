"""Automatic reset phase for the supervised V94 real-robot entry point.

Importing this module is hardware inert.  Device-facing reset functions are
called only by :func:`run_v94_execution_reset`, after the caller has completed
the deployment bundle/profile/native-binary validation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dexgrasp.apps import reset_franka_v94_training as franka_training_reset
from dexgrasp.apps.reset_installed_rh56_open import (
    RH56ResetOpenProof,
    run_installed_rh56_reset_open,
)
from anydex_pipeline.control_config import verify_adapter_assets

from sim2real.rh56_profile_contract import load_commissioned_rh56_force_set_g


RESET_DWELL_S = 0.5
# Match the supervised RH56 running envelope.  The installed device's own
# CURRENT_LIMIT register (normally 1400 mA) remains the authoritative hard
# protection and is intersected with this host limit by the reset driver.
# The earlier 400 mA commissioning threshold rejected normal direct q6 reset
# motion (526 mA observed) even though STATUS/ERROR remained healthy.
RESET_MAX_AXIS_CURRENT_MA = 1000
RESET_MAX_FRANKA_START_DELTA_RAD = 1.21
# Automatic execution reset is an unloaded, operator-supervised return to the
# same hash-bound q_home.  The generic commissioning reset remains at its
# profile's 0.05 rad/s and 6 s minimum; only this execution-local copy is made
# faster.  The cosine time law, joint bounds, arrival proof and verified stop
# are unchanged.
RESET_FRANKA_MAX_JOINT_SPEED_RAD_S = 0.20
RESET_FRANKA_MIN_SEGMENT_DURATION_S = 2.0
# The supervised runtime independently requires the first fresh Franka sample
# to be within 0.01 rad of q_home.  Use half that envelope for the automatic
# reset so its stop sample cannot legitimately pass at the runtime boundary.
# This is deliberately local to the V94 automatic reset; the generic installed
# hand reset retains its commissioned profile tolerance.
RESET_FRANKA_ARRIVAL_TOLERANCE_RAD = 0.005
RESET_RH56_OPEN_FORCE_G = 80


class V94ExecutionResetError(RuntimeError):
    """The automatic reset could not establish the policy start state."""


def _rh56_reset_motion_profile(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the reset-only 40/80 profile without mutating runtime settings.

    A thrown-task profile may commission 500 g for policy execution, but the
    reusable installed-hand reset has a separate fixed no-contact 40/80 g
    motion contract.  Validate the task-scoped force evidence before making
    the reset-local downgrade; an unlisted profile cannot acquire this path.
    """

    inspire = profile.get("inspire")
    if not isinstance(inspire, Mapping):
        # Preserve lightweight injected test profiles; the real reset runner
        # independently requires the complete inspire mapping.
        return profile
    try:
        configured_force = int(inspire["force_limit_g"])
    except (KeyError, TypeError, ValueError) as exc:
        raise V94ExecutionResetError(
            "RH56 reset profile has no valid runtime force limit"
        ) from exc
    if configured_force > RESET_RH56_OPEN_FORCE_G:
        try:
            commissioned_force = load_commissioned_rh56_force_set_g(profile)
        except ValueError as exc:
            raise V94ExecutionResetError(
                "RH56 runtime force is not commissioned for reset-local downgrade"
            ) from exc
        if commissioned_force != configured_force:
            raise V94ExecutionResetError(
                "RH56 runtime force differs from its commissioned task contract"
            )
    reset_profile = copy.deepcopy(dict(profile))
    reset_profile["inspire"]["force_limit_g"] = min(
        configured_force, RESET_RH56_OPEN_FORCE_G
    )
    return reset_profile


@dataclass(frozen=True)
class FrankaV94HomeResetProof:
    target_q_rad: tuple[float, ...]
    target_sha256_f64_le: str
    initial_linf_delta_rad: float
    maximum_start_delta_rad: float
    final_linf_error_rad: float
    franka_stop_verified: bool = True
    rh56_open_disabled_verified: bool = True


@dataclass(frozen=True)
class V94ExecutionResetProof:
    rh56: RH56ResetOpenProof
    franka: FrankaV94HomeResetProof
    dwell_requested_s: float
    dwell_elapsed_s: float


def _same_serialized_q_home(first: np.ndarray, second: np.ndarray) -> bool:
    """Compare q_home in the bundle's canonical float32 storage domain.

    The V94 handoff stores the simulation initial joint positions as float32,
    while the commissioning JSON spells the same values as short decimal
    float64 literals.  Comparing the decoded arrays as float64 therefore
    rejects only their harmless decimal round-trip tails.  Exact float32 byte
    equality retains a strict contract and still rejects any representable
    change to the policy/reset reference.
    """

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    return bool(
        left.shape == (7,)
        and right.shape == (7,)
        and np.all(np.isfinite(left))
        and np.all(np.isfinite(right))
        and np.array_equal(left.astype("<f4"), right.astype("<f4"))
    )


def _verify_v94_reset_profile_and_assets(
    profile: Mapping[str, Any],
    profile_source: Path,
    contract_q_home_rad: np.ndarray,
) -> None:
    """Reuse the adapter checker without making historical evidence a gate.

    The legacy generic validator only accepts a null evidence path.  V94's
    non-null path is documentation; live reset correctness is instead proven
    from the verified bundle target and fresh post-reset hardware state.
    """

    expected = np.asarray(contract_q_home_rad, dtype=np.float64)
    try:
        profile_target = np.asarray(
            profile["franka"]["default_q_rad"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise V94ExecutionResetError(
            "reset profile has no finite seven-axis default_q_rad"
        ) from exc
    if not _same_serialized_q_home(profile_target, expected):
        raise V94ExecutionResetError(
            "reset profile default_q_rad differs from the prepared task q_home"
        )
    adapter_profile = copy.deepcopy(dict(profile))
    try:
        adapter_profile["franka"]["default_q_provenance"][
            "evidence_artifact"
        ] = None
    except (KeyError, TypeError) as exc:
        raise V94ExecutionResetError(
            "V94 reset profile default-q provenance is malformed"
        ) from exc
    verify_adapter_assets(adapter_profile, profile_source)


def _default_franka_reset(
    config: Mapping[str, Any],
    *,
    bundle_path: Path,
    contract_q_home_rad: np.ndarray,
    maximum_start_delta_rad: float,
) -> FrankaV94HomeResetProof:
    # Keep verifying the bundle because it still owns joint limits, observation
    # layout and action mapping.  A resolved task profile may, however, replace
    # only q_home with its separately SHA-bound reset contract.
    _bundle_target, _metadata = franka_training_reset._training_target(bundle_path)
    expected = np.asarray(contract_q_home_rad, dtype=np.float64)
    try:
        profile_target = np.asarray(
            config["franka"]["default_q_rad"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise V94ExecutionResetError("reset profile q_home is malformed") from exc
    if expected.shape != (7,) or not np.all(np.isfinite(expected)):
        raise V94ExecutionResetError(
            "prepared task contract q_home is malformed"
        )
    if not _same_serialized_q_home(profile_target, expected):
        raise V94ExecutionResetError(
            "reset profile q_home differs from the prepared task contract"
        )
    runtime_config = franka_training_reset._reset_config(config, expected)
    runtime_config["franka"]["default_max_joint_velocity_rad_s"] = (
        RESET_FRANKA_MAX_JOINT_SPEED_RAD_S
    )
    runtime_config["franka"]["default_min_duration_s"] = (
        RESET_FRANKA_MIN_SEGMENT_DURATION_S
    )
    runtime_config["franka"]["default_arrival_tolerance_rad"] = min(
        float(runtime_config["franka"]["default_arrival_tolerance_rad"]),
        RESET_FRANKA_ARRIVAL_TOLERANCE_RAD,
    )
    reset_proof = franka_training_reset.DEFAULT_RESET.run_reset(
        runtime_config,
        maximum_start_delta_rad=maximum_start_delta_rad,
    )
    return FrankaV94HomeResetProof(
        target_q_rad=tuple(float(value) for value in expected),
        target_sha256_f64_le=hashlib.sha256(
            np.ascontiguousarray(expected.astype("<f8")).tobytes()
        ).hexdigest(),
        initial_linf_delta_rad=float(reset_proof.initial_linf_delta_rad),
        maximum_start_delta_rad=float(maximum_start_delta_rad),
        final_linf_error_rad=float(reset_proof.final_linf_error_rad),
    )


def validate_v94_execution_reset_proof(
    proof: V94ExecutionResetProof,
    *,
    contract_q_home_rad: Sequence[float],
    maximum_start_delta_rad: float,
) -> V94ExecutionResetProof:
    """Require an internally consistent, finite reset proof."""

    expected_target = np.asarray(contract_q_home_rad, dtype=np.float64)
    try:
        actual_target = np.asarray(proof.franka.target_q_rad, dtype=np.float64)
        maximum = float(maximum_start_delta_rad)
        proved_maximum = float(proof.franka.maximum_start_delta_rad)
        initial_delta = float(proof.franka.initial_linf_delta_rad)
        final_error = float(proof.franka.final_linf_error_rad)
        dwell_requested = float(proof.dwell_requested_s)
        dwell_elapsed = float(proof.dwell_elapsed_s)
    except (AttributeError, TypeError, ValueError) as exc:
        raise V94ExecutionResetError("automatic reset proof is malformed") from exc
    if (
        not isinstance(proof, V94ExecutionResetProof)
        or expected_target.shape != (7,)
        or actual_target.shape != (7,)
        or not np.all(np.isfinite(expected_target))
        or not np.all(np.isfinite(actual_target))
        or not np.array_equal(actual_target, expected_target)
    ):
        raise V94ExecutionResetError(
            "automatic reset proof target differs from the V94 contract"
        )
    expected_sha256 = hashlib.sha256(
        np.ascontiguousarray(expected_target.astype("<f8")).tobytes()
    ).hexdigest()
    actual_sha256 = str(proof.franka.target_sha256_f64_le)
    if (
        re.fullmatch(r"[0-9a-f]{64}", actual_sha256) is None
        or actual_sha256 != expected_sha256
    ):
        raise V94ExecutionResetError(
            "automatic reset proof q_home SHA-256 is invalid"
        )
    if (
        not np.isfinite(maximum)
        or maximum <= 0.0
        or not np.isfinite(proved_maximum)
        or proved_maximum != maximum
        or not np.isfinite(initial_delta)
        or not 0.0 <= initial_delta <= maximum
        or not np.isfinite(final_error)
        or final_error < 0.0
    ):
        raise V94ExecutionResetError(
            "automatic Franka reset proof is outside the formal start envelope"
        )
    if final_error > RESET_FRANKA_ARRIVAL_TOLERANCE_RAD:
        raise V94ExecutionResetError(
            "automatic Franka reset proof final error "
            f"{final_error:.9f}rad exceeds the V94 reset arrival envelope "
            f"{RESET_FRANKA_ARRIVAL_TOLERANCE_RAD:.9f}rad"
        )
    if (
        not np.isfinite(dwell_requested)
        or not np.isfinite(dwell_elapsed)
        or dwell_requested < RESET_DWELL_S
        or dwell_elapsed + 1.0e-9 < dwell_requested
    ):
        raise V94ExecutionResetError(
            "automatic reset proof has an invalid stationary dwell"
        )
    if (
        proof.rh56.rh56_disabled_verified is not True
        or proof.rh56.franka_read_only_verified is not True
        or proof.franka.franka_stop_verified is not True
        or proof.franka.rh56_open_disabled_verified is not True
    ):
        raise V94ExecutionResetError(
            "automatic reset proof lacks verified dual-device stop"
        )
    return proof


def run_v94_execution_reset(
    *,
    profile: Mapping[str, Any],
    profile_path: Path,
    bundle_path: Path,
    contract_q_home_rad: Sequence[float],
    rh56_reset_runner: Callable[..., RH56ResetOpenProof] = (
        run_installed_rh56_reset_open
    ),
    franka_reset_runner: Callable[..., FrankaV94HomeResetProof] = (
        _default_franka_reset
    ),
    maximum_franka_start_delta_rad: float = (
        RESET_MAX_FRANKA_START_DELTA_RAD
    ),
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> V94ExecutionResetProof:
    """Reset RH56, reset Franka, then prove a minimum command-free dwell.

    Both reset runners are synchronous: a successful return means their device
    handles have completed verified-stop cleanup and have been released.  No
    policy object, camera owner, runtime actuator owner, or motion
    authorization is constructed in this phase.
    """

    profile_source = Path(profile_path).expanduser().resolve()
    bundle_source = Path(bundle_path).expanduser().resolve()
    target = np.asarray(contract_q_home_rad, dtype=np.float64)
    if target.shape != (7,) or not np.all(np.isfinite(target)):
        raise V94ExecutionResetError("prepared V94 q_home is malformed")
    try:
        maximum_numeric = float(maximum_franka_start_delta_rad)
    except (TypeError, ValueError) as exc:
        raise V94ExecutionResetError(
            "maximum Franka reset start delta must be finite and positive"
        ) from exc
    if (
        isinstance(maximum_franka_start_delta_rad, (bool, np.bool_))
        or not np.isfinite(maximum_numeric)
        or maximum_numeric <= 0.0
    ):
        raise V94ExecutionResetError(
            "maximum Franka reset start delta must be finite and positive"
        )
    maximum_franka_start_delta_rad = maximum_numeric

    # These remain offline checks and precede the first device factory.
    _verify_v94_reset_profile_and_assets(profile, profile_source, target)

    print(
        "[Automatic reset] RH56 canonical open/disabled "
        f"(strict current cap {RESET_MAX_AXIS_CURRENT_MA}mA)",
        flush=True,
    )
    rh56_proof = rh56_reset_runner(
        _rh56_reset_motion_profile(profile),
        max_axis_current_ma=RESET_MAX_AXIS_CURRENT_MA,
    )
    if (
        not isinstance(rh56_proof, RH56ResetOpenProof)
        or rh56_proof.rh56_disabled_verified is not True
        or rh56_proof.franka_read_only_verified is not True
    ):
        raise V94ExecutionResetError("RH56 reset returned no verified-stop proof")

    print(
        "[Automatic reset] Franka -> prepared task contract q_home "
        f"(arrival<={RESET_FRANKA_ARRIVAL_TOLERANCE_RAD:.3f}rad)",
        flush=True,
    )
    franka_proof = franka_reset_runner(
        profile,
        bundle_path=bundle_source,
        contract_q_home_rad=target,
        maximum_start_delta_rad=maximum_franka_start_delta_rad,
    )
    if (
        not isinstance(franka_proof, FrankaV94HomeResetProof)
        or franka_proof.franka_stop_verified is not True
        or franka_proof.rh56_open_disabled_verified is not True
    ):
        raise V94ExecutionResetError("Franka reset returned no verified-stop proof")

    dwell_started = float(monotonic())
    sleep(RESET_DWELL_S)
    dwell_elapsed = float(monotonic()) - dwell_started
    if not np.isfinite(dwell_elapsed) or dwell_elapsed + 1.0e-9 < RESET_DWELL_S:
        raise V94ExecutionResetError(
            "post-reset stationary dwell was shorter than 0.5s"
        )
    proof = V94ExecutionResetProof(
        rh56=rh56_proof,
        franka=franka_proof,
        dwell_requested_s=RESET_DWELL_S,
        dwell_elapsed_s=dwell_elapsed,
    )
    validated = validate_v94_execution_reset_proof(
        proof,
        contract_q_home_rad=target,
        maximum_start_delta_rad=maximum_franka_start_delta_rad,
    )
    print(
        f"[Automatic reset PASS] both devices stopped; dwell={dwell_elapsed:.3f}s",
        flush=True,
    )
    return validated


__all__ = [
    "FrankaV94HomeResetProof",
    "RESET_DWELL_S",
    "RESET_FRANKA_ARRIVAL_TOLERANCE_RAD",
    "RESET_FRANKA_MAX_JOINT_SPEED_RAD_S",
    "RESET_FRANKA_MIN_SEGMENT_DURATION_S",
    "RESET_RH56_OPEN_FORCE_G",
    "RESET_MAX_FRANKA_START_DELTA_RAD",
    "RESET_MAX_AXIS_CURRENT_MA",
    "V94ExecutionResetError",
    "V94ExecutionResetProof",
    "run_v94_execution_reset",
    "validate_v94_execution_reset_proof",
]
