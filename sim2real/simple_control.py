#!/usr/bin/env python3
"""Run a deterministic sim2real policy preview with all hardware writes blocked."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    workspace = Path(__file__).resolve().parents[1]
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from sim2real.baseline import ActionLimiter, BaselinePolicy  # type: ignore
    from sim2real.config import (  # type: ignore
        DEFAULT_CONFIG_PATH,
        load_runtime_config,
        observation_spec_from_config,
    )
    from sim2real.contracts import assemble_observation  # type: ignore
    from sim2real.io import ObservationReader  # type: ignore
else:
    from .baseline import ActionLimiter, BaselinePolicy
    from .config import (
        DEFAULT_CONFIG_PATH,
        load_runtime_config,
        observation_spec_from_config,
    )
    from .contracts import assemble_observation
    from .io import ObservationReader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview a simple bounded policy. This stage never writes Franka or RH56 commands."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=BaselinePolicy.MODES, default="hold")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--policy-hz", type=float)
    parser.add_argument("--arm-amplitude-rad", type=float, default=0.01)
    parser.add_argument("--arm-period-s", type=float, default=4.0)
    parser.add_argument(
        "--live-readonly",
        action="store_true",
        help="read real Franka/RH56/object-PCD observations, still without writes",
    )
    parser.add_argument("--without-inspire", action="store_true")
    parser.add_argument("--without-object-pcd", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="reserved for the commissioned checkpoint executor; currently refused",
    )
    return parser


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _initial_hand(profile) -> np.ndarray:
    return np.asarray(profile["inspire"]["open_targets"], dtype=np.int32)


def _make_limiter(runtime, profile, initial_q, initial_hand) -> ActionLimiter:
    baseline = runtime["baseline"]
    franka = profile["franka"]
    return ActionLimiter(
        initial_q_rad=initial_q,
        joint_limits_rad=franka["joint_limits_rad"],
        joint_limit_margin_rad=float(franka["joint_limit_margin_rad"]),
        max_arm_step_rad=float(baseline["max_arm_step_rad"]),
        max_arm_episode_delta_rad=float(baseline["max_arm_episode_delta_rad"]),
        initial_hand_angles=initial_hand,
        max_hand_step_units=int(baseline["max_hand_step_units"]),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        runtime, profile, _ = load_runtime_config(args.config)
        baseline_cfg = runtime["baseline"]
        if args.execute:
            blocker = baseline_cfg.get(
                "hardware_write_blocker", "hardware executor is not commissioned"
            )
            raise RuntimeError(
                "hardware execution is intentionally blocked in stage one: "
                + str(blocker)
            )
        if baseline_cfg.get("hardware_writes_enabled") is not False:
            raise ValueError(
                "stage-one config must keep baseline.hardware_writes_enabled=false"
            )
        hz = _positive(
            baseline_cfg["policy_hz"] if args.policy_hz is None else args.policy_hz,
            "policy_hz",
        )
        duration = _positive(
            baseline_cfg["duration_s"] if args.duration is None else args.duration,
            "duration",
        )
        policy = BaselinePolicy(
            args.mode,
            policy_hz=hz,
            arm_amplitude_rad=args.arm_amplitude_rad,
            arm_period_s=args.arm_period_s,
        )
        use_inspire = args.live_readonly and not args.without_inspire
        use_object = args.live_readonly and not args.without_object_pcd
        spec = observation_spec_from_config(
            runtime,
            profile,
            require_franka=args.live_readonly,
            require_inspire=use_inspire,
            require_object_pcd=use_object,
        )
        reader = None
        try:
            if args.live_readonly:
                reader = ObservationReader(
                    runtime,
                    profile,
                    use_franka=True,
                    use_inspire=use_inspire,
                    use_object_pcd=use_object,
                )
                reader.start()
                first_franka, first_hand, first_object = reader.read_raw()
                first_observation = assemble_observation(
                    captured_at_s=time.time(),
                    spec=spec,
                    franka=first_franka,
                    inspire=first_hand,
                    object_pcd=first_object,
                )
                if first_franka is None:
                    raise RuntimeError("live preview did not return Franka state")
                initial_q = first_franka.q
                initial_hand = (
                    first_hand.angles
                    if first_hand is not None
                    else _initial_hand(profile)
                )
            else:
                first_observation = None
                initial_q = np.asarray(
                    profile["franka"]["default_q_rad"], dtype=np.float64
                )
                initial_hand = _initial_hand(profile)

            limiter = _make_limiter(runtime, profile, initial_q, initial_hand)
            started = time.monotonic()
            next_tick = started
            step = 0
            observation = first_observation
            while True:
                now = time.monotonic()
                elapsed = now - started
                if elapsed >= duration:
                    break
                if reader is not None and step > 0:
                    franka, hand, obj = reader.read_raw()
                    observation = assemble_observation(
                        captured_at_s=time.time(),
                        spec=spec,
                        franka=franka,
                        inspire=hand,
                        object_pcd=obj,
                    )
                raw = policy.act(observation, elapsed_s=elapsed)
                limited = limiter.apply(raw)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "t": round(elapsed, 6),
                            "observation_valid": (
                                True if observation is None else observation.valid
                            ),
                            "source": raw.source,
                            "arm_delta_rad": limited.arm_joint_delta_rad.tolist(),
                            "arm_target_q_rad": limited.arm_target_q_rad.tolist(),
                            "hand_targets": limited.hand_targets.tolist(),
                            "clipped": limited.clipped,
                            "clip_reasons": list(limited.reasons),
                            "write": False,
                        },
                        sort_keys=True,
                    )
                )
                step += 1
                next_tick += 1.0 / hz
                wait_s = next_tick - time.monotonic()
                if wait_s > 0.0:
                    time.sleep(wait_s)
        finally:
            if reader is not None:
                reader.close()
        print(
            "[dry-run complete] no Franka control handle was created and no RH56 "
            "register was written",
            file=sys.stderr,
        )
        return 0
    except KeyboardInterrupt:
        return 130
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
