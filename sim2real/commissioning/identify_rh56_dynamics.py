"""Identify an effective free-space RH56 response at a fixed SPEED_SET.

This tool never opens a Franka interface.  It moves one selected RH56 axis by
a small ANGLE_SET step, records ANGLE_ACT at the fastest rate supported by the
serial link, returns to the measured start angle, and disables all six targets.

The fitted model is a velocity-limited second-order response::

    q_ddot = omega_n**2 * (q_target - q) - 2*zeta*omega_n*q_dot

It reports ``K/I_eff`` and ``D/I_eff``.  Absolute stiffness and damping are
only candidates after choosing an effective simulated armature/inertia.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.optimize import least_squares

from examples.inspire_rh56_test import (
    JOINTS,
    JOINT_LABELS,
    REG_ANGLE_ACT,
    REG_ANGLE_SET,
    REG_CURRENT,
    REG_FORCE_SET,
    REG_SPEED_SET,
    RH56Error,
    RH56Hand,
    LinuxSerial,
    _disable_all_targets_verified,
    find_serial_port,
)


CONFIRMATION = "RH56_FREE_SPACE_STEP_IDENTIFICATION"
DEFAULT_SPEED_SET = 600
DEFAULT_STEP_UNITS = 150
DEFAULT_DURATION_S = 1.25
DEFAULT_SETTLE_S = 0.35
DEFAULT_MAX_CURRENT_MA = 400
SIM_EFFECTIVE_INERTIA_KG_M2 = 0.01


class RH56DynamicsIdentificationError(RuntimeError):
    """The acquisition or fit could not be completed safely."""


def _simulate_normalized_step(
    times_s: np.ndarray,
    *,
    omega_n_rad_s: float,
    damping_ratio: float,
    velocity_limit_steps_s: float,
    delay_s: float,
) -> np.ndarray:
    """Simulate a unit step using a small fixed integration step."""

    if times_s.size == 0:
        return np.empty(0, dtype=np.float64)
    output = np.empty(times_s.size, dtype=np.float64)
    q = 0.0
    dq = 0.0
    previous_t = 0.0
    for sample_index, sample_t in enumerate(times_s):
        remaining = max(0.0, float(sample_t) - previous_t)
        while remaining > 0.0:
            dt = min(0.001, remaining)
            target = 1.0 if previous_t + dt >= delay_s else 0.0
            ddq = (
                omega_n_rad_s * omega_n_rad_s * (target - q)
                - 2.0 * damping_ratio * omega_n_rad_s * dq
            )
            dq = float(
                np.clip(
                    dq + ddq * dt,
                    -velocity_limit_steps_s,
                    velocity_limit_steps_s,
                )
            )
            q += dq * dt
            previous_t += dt
            remaining -= dt
        output[sample_index] = q
    return output


def fit_step_response(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Fit one captured leg and return inertia-normalized dynamics."""

    if len(samples) < 8:
        raise RH56DynamicsIdentificationError(
            f"need at least 8 samples for a fit, got {len(samples)}"
        )
    times = np.asarray([row["t_s"] for row in samples], dtype=np.float64)
    actual = np.asarray([row["angle_actual"] for row in samples], dtype=np.float64)
    start = float(samples[0]["start_angle"])
    command_target = float(samples[0]["target_angle"])
    command_amplitude = command_target - start
    steady_count = max(8, len(samples) // 5)
    steady_actual = float(np.median(actual[-steady_count:]))
    observed_amplitude = steady_actual - start
    if abs(command_amplitude) < 20.0:
        raise RH56DynamicsIdentificationError(
            "commanded step amplitude is too small for identification: "
            f"{command_amplitude:.1f} units"
        )
    if abs(observed_amplitude) < 20.0:
        raise RH56DynamicsIdentificationError(
            "observed steady-state step is too small for identification: "
            f"{observed_amplitude:.1f} units"
        )
    times = times - times[0]
    # Fit the transient shape against the measured steady endpoint.  RH56's
    # ANGLE_SET->ANGLE_ACT map has direction/position-dependent static bias;
    # forcing unity DC gain corrupts omega/zeta.  Report that gain separately.
    measured = (actual - start) / observed_amplitude

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        omega_n, zeta, velocity_limit, delay = np.exp(log_parameters)
        predicted = _simulate_normalized_step(
            times,
            omega_n_rad_s=float(omega_n),
            damping_ratio=float(zeta),
            velocity_limit_steps_s=float(velocity_limit),
            delay_s=float(delay),
        )
        return predicted - measured

    lower = np.log([0.5, 0.05, 0.1, 0.001])
    upper = np.log([150.0, 8.0, 50.0, 0.25])
    # SPEED_SET can make omega/zeta/vmax partly correlated.  Use several
    # physically distinct starting points and retain the best response fit.
    starts = (
        (8.0, 0.7, 2.0, 0.010),
        (18.0, 1.0, 5.0, 0.025),
        (35.0, 1.5, 12.0, 0.050),
    )
    solutions = [
        least_squares(
            residual,
            np.log(start),
            bounds=(lower, upper),
            diff_step=0.03,
            max_nfev=300,
        )
        for start in starts
    ]
    solution = min(solutions, key=lambda candidate: float(np.sum(candidate.fun**2)))
    omega_n, zeta, velocity_limit, delay = np.exp(solution.x)
    predicted = measured + solution.fun
    residual_sum = float(np.sum((predicted - measured) ** 2))
    centered_sum = float(np.sum((measured - float(np.mean(measured))) ** 2))
    r_squared = 1.0 - residual_sum / centered_sum if centered_sum > 0.0 else 0.0
    k_over_i = float(omega_n * omega_n)
    d_over_i = float(2.0 * zeta * omega_n)
    return {
        "sample_count": len(samples),
        "sample_rate_hz_median": float(1.0 / np.median(np.diff(times))),
        "commanded_step_units": command_amplitude,
        "observed_steady_step_units": observed_amplitude,
        "steady_angle_actual": steady_actual,
        "steady_error_to_command_units": steady_actual - command_target,
        "command_to_actual_dc_gain": observed_amplitude / command_amplitude,
        "omega_n_rad_s": float(omega_n),
        "natural_frequency_hz": float(omega_n / (2.0 * math.pi)),
        "damping_ratio": float(zeta),
        "normalized_velocity_limit_steps_s": float(velocity_limit),
        "angle_velocity_limit_units_s": float(
            velocity_limit * abs(observed_amplitude)
        ),
        "delay_s": float(delay),
        "K_over_I_s-2": k_over_i,
        "D_over_I_s-1": d_over_i,
        "candidate_sim_armature_kg_m2": SIM_EFFECTIVE_INERTIA_KG_M2,
        "candidate_stiffness_nm_per_rad": (
            k_over_i * SIM_EFFECTIVE_INERTIA_KG_M2
        ),
        "candidate_damping_nms_per_rad": (
            d_over_i * SIM_EFFECTIVE_INERTIA_KG_M2
        ),
        "r_squared": r_squared,
        "fit_accepted": bool(r_squared >= 0.90),
    }


def _write_disabled(hand: RH56Hand) -> None:
    _disable_all_targets_verified(hand)


def _capture_leg(
    hand: RH56Hand,
    *,
    joint_index: int,
    target_angle: int,
    duration_s: float,
    max_current_ma: int,
) -> List[Dict[str, Any]]:
    start_angle = int(hand.read_short(REG_ANGLE_ACT + 2 * joint_index))
    targets = [-1] * 6
    targets[joint_index] = int(target_angle)
    hand.write_six_shorts(REG_ANGLE_SET, targets, retries=0)
    if tuple(hand.read_six_shorts(REG_ANGLE_SET, retries=0)) != tuple(targets):
        raise RH56DynamicsIdentificationError("ANGLE_SET exact readback mismatch")

    samples: List[Dict[str, Any]] = []
    started = time.monotonic()
    next_current_poll = started
    currents = (0,) * 6
    while True:
        now = time.monotonic()
        if now - started >= duration_s:
            break
        angle = int(hand.read_short(REG_ANGLE_ACT + 2 * joint_index))
        captured = time.monotonic()
        if captured >= next_current_poll:
            currents = tuple(hand.read_six_shorts(REG_CURRENT, retries=0))
            next_current_poll = captured + 0.05
            if max(abs(value) for value in currents) > max_current_ma:
                raise RH56DynamicsIdentificationError(
                    "running current exceeded limit: "
                    f"currents={currents}, limit={max_current_ma}mA"
                )
        samples.append(
            {
                "t_s": captured - started,
                "start_angle": int(start_angle),
                "target_angle": int(target_angle),
                "angle_actual": angle,
                "selected_current_ma": int(currents[joint_index]),
            }
        )
    return samples


def acquire(args: argparse.Namespace) -> Dict[str, Any]:
    joint_index = JOINTS.index(args.joint)
    port = args.port or find_serial_port()
    with LinuxSerial(port, args.baud, args.timeout, args.debug) as serial_port:
        hand = RH56Hand(serial_port, args.id)
        snapshot = hand.snapshot()
        if any(snapshot["errors"]):
            raise RH56DynamicsIdentificationError(
                f"RH56 reports errors: {snapshot['errors']}"
            )
        if any(abs(value) > 100 for value in snapshot["currents"]):
            raise RH56DynamicsIdentificationError(
                f"RH56 is not unloaded/idle: currents={snapshot['currents']}"
            )
        if max(snapshot["temperatures"]) >= 50:
            raise RH56DynamicsIdentificationError(
                f"RH56 is too warm: temperatures={snapshot['temperatures']}"
            )
        if tuple(snapshot["angle_targets"]) != (-1,) * 6:
            raise RH56DynamicsIdentificationError(
                "all six ANGLE_SET values must be -1 before identification"
            )

        original_speeds = tuple(snapshot["speeds"])
        original_forces = tuple(snapshot["force_limits"])
        samples: List[Dict[str, Any]] = []
        initial_angle = int(snapshot["angles"][joint_index])
        baseline_target = (
            int(args.baseline_target)
            if args.baseline_target is not None
            else initial_angle
        )
        signed_step = (
            -abs(args.step_units)
            if baseline_target >= 500
            else abs(args.step_units)
        )
        target_angle = int(baseline_target + signed_step)
        if not 0 <= target_angle <= 1000:
            raise RH56DynamicsIdentificationError(
                f"step target {target_angle} is outside ANGLE_SET 0..1000"
            )
        try:
            hand.write_six_shorts(REG_FORCE_SET, (args.force_set,) * 6, retries=0)
            if tuple(hand.read_six_shorts(REG_FORCE_SET, retries=0)) != (
                args.force_set,
            ) * 6:
                raise RH56DynamicsIdentificationError("FORCE_SET readback mismatch")
            # q6 and other nonlinear axes can be conditioned away from a hard
            # endpoint before the measured step.  Conditioning is deliberately
            # slower and is excluded from the fitted samples.
            if args.baseline_target is not None:
                hand.write_six_shorts(
                    REG_SPEED_SET, (args.conditioning_speed_set,) * 6, retries=0
                )
                if tuple(hand.read_six_shorts(REG_SPEED_SET, retries=0)) != (
                    args.conditioning_speed_set,
                ) * 6:
                    raise RH56DynamicsIdentificationError(
                        "conditioning SPEED_SET readback mismatch"
                    )
                _capture_leg(
                    hand,
                    joint_index=joint_index,
                    target_angle=baseline_target,
                    duration_s=args.conditioning_duration,
                    max_current_ma=args.max_current_ma,
                )
                _write_disabled(hand)
                time.sleep(args.settle)

            hand.write_six_shorts(REG_SPEED_SET, (args.speed_set,) * 6, retries=0)
            if tuple(hand.read_six_shorts(REG_SPEED_SET, retries=0)) != (
                args.speed_set,
            ) * 6:
                raise RH56DynamicsIdentificationError("SPEED_SET readback mismatch")
            for trial in range(1, args.trials + 1):
                outbound = _capture_leg(
                    hand,
                    joint_index=joint_index,
                    target_angle=target_angle,
                    duration_s=args.duration,
                    max_current_ma=args.max_current_ma,
                )
                for row in outbound:
                    row.update(trial=trial, leg="outbound")
                samples.extend(outbound)
                time.sleep(args.settle)
                returned = _capture_leg(
                    hand,
                    joint_index=joint_index,
                    target_angle=baseline_target,
                    duration_s=args.duration,
                    max_current_ma=args.max_current_ma,
                )
                for row in returned:
                    row.update(trial=trial, leg="return")
                samples.extend(returned)
                _write_disabled(hand)
                time.sleep(args.settle)
        finally:
            # Attempt every cleanup independently: a transient stop/readback
            # error must not prevent restoration of the device settings.
            cleanup_errors = []
            try:
                _write_disabled(hand)
            except BaseException as exc:
                cleanup_errors.append(f"disable failed: {exc}")
            try:
                hand.write_six_shorts(REG_SPEED_SET, original_speeds, retries=1)
            except BaseException as exc:
                cleanup_errors.append(f"SPEED_SET restore failed: {exc}")
            try:
                hand.write_six_shorts(REG_FORCE_SET, original_forces, retries=1)
            except BaseException as exc:
                cleanup_errors.append(f"FORCE_SET restore failed: {exc}")
            if cleanup_errors:
                raise RH56DynamicsIdentificationError(
                    "cleanup incomplete; cut RH56 power if it is moving: "
                    + "; ".join(cleanup_errors)
                )

    fits = []
    for trial in range(1, args.trials + 1):
        for leg in ("outbound", "return"):
            leg_samples = [
                row for row in samples if row["trial"] == trial and row["leg"] == leg
            ]
            fit = fit_step_response(leg_samples)
            fit.update(trial=trial, leg=leg)
            fits.append(fit)
    accepted = [fit for fit in fits if fit["fit_accepted"]]
    aggregate: Dict[str, Any] = {
        "accepted_leg_count": len(accepted),
        "total_leg_count": len(fits),
    }
    for key in (
        "omega_n_rad_s",
        "natural_frequency_hz",
        "damping_ratio",
        "K_over_I_s-2",
        "D_over_I_s-1",
        "candidate_stiffness_nm_per_rad",
        "candidate_damping_nms_per_rad",
        "angle_velocity_limit_units_s",
        "delay_s",
    ):
        values = [float(fit[key]) for fit in accepted]
        aggregate[key + "_median"] = float(np.median(values)) if values else None
    return {
        "contract": "rh56_free_space_velocity_limited_second_order_v1",
        "joint": args.joint,
        "joint_label": JOINT_LABELS[joint_index],
        "port": port,
        "speed_set": args.speed_set,
        "force_set_g": args.force_set,
        "step_units": abs(target_angle - baseline_target),
        "initial_angle": initial_angle,
        "baseline_target": baseline_target,
        "target_angle": target_angle,
        "fits": fits,
        "aggregate": aggregate,
        "samples": samples,
    }


def _write_artifacts(result: Dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    csv_path = output.with_suffix(".csv")
    fields = (
        "trial",
        "leg",
        "t_s",
        "start_angle",
        "target_angle",
        "angle_actual",
        "selected_current_ma",
    )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in result["samples"])
    print(f"[SAVED] {output}")
    print(f"[SAVED] {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", choices=JOINTS, required=True)
    parser.add_argument("--speed-set", type=int, default=DEFAULT_SPEED_SET)
    parser.add_argument("--step-units", type=int, default=DEFAULT_STEP_UNITS)
    parser.add_argument(
        "--baseline-target",
        type=int,
        help=(
            "optional interior ANGLE_SET used as the return target; the axis is "
            "conditioned there before measured trials"
        ),
    )
    parser.add_argument("--conditioning-speed-set", type=int, default=200)
    parser.add_argument("--conditioning-duration", type=float, default=3.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S)
    parser.add_argument("--force-set", type=int, default=500)
    parser.add_argument("--max-current-ma", type=int, default=DEFAULT_MAX_CURRENT_MA)
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-free-space", metavar="TOKEN", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.confirm_free_space != CONFIRMATION:
        raise SystemExit(f"--confirm-free-space must equal {CONFIRMATION}")
    if not 1 <= args.speed_set <= 1000:
        raise SystemExit("--speed-set must be in 1..1000")
    if not 100 <= args.step_units <= 250:
        raise SystemExit("--step-units must be in 100..250")
    if args.baseline_target is not None and not 100 <= args.baseline_target <= 900:
        raise SystemExit("--baseline-target must be in 100..900")
    if not 1 <= args.conditioning_speed_set <= 600:
        raise SystemExit("--conditioning-speed-set must be in 1..600")
    if not 1.0 <= args.conditioning_duration <= 8.0:
        raise SystemExit("--conditioning-duration must be in 1..8 seconds")
    if not 1 <= args.trials <= 10:
        raise SystemExit("--trials must be in 1..10")
    if not 0.75 <= args.duration <= 3.0:
        raise SystemExit("--duration must be in 0.75..3.0 seconds")
    if not 100 <= args.max_current_ma <= 1400:
        raise SystemExit("--max-current-ma must be in 100..1400")
    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    output = args.output or Path(
        f"dexgrasp/runs/rh56_dynamics_speed{args.speed_set}_{args.joint}_{run_stamp}.json"
    )
    try:
        result = acquire(args)
        _write_artifacts(result, output)
        print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))
        if result["aggregate"]["accepted_leg_count"] == 0:
            print(
                "[WARN] no leg reached R^2>=0.90; keep the raw CSV and do not "
                "copy candidate K/D into simulation"
            )
        return 0
    except (RH56Error, RH56DynamicsIdentificationError) as exc:
        print(f"[FAILED] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
