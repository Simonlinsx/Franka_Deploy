#!/usr/bin/env python3
"""Replay a validated simulation action sequence through the real runtime."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

from .action_replay import load_replay_actions
from sim2real.policy.rate_mode import resolve_policy_rate_mode
from .deployment.runner import (
    DeploymentRequest,
    SupervisedV94RunError,
    build_deployment_request,
    build_deployment_summary,
    build_parser as build_policy_parser,
    execute_deployment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_policy_parser()
    parser.description = (
        "Replay normalized simulation actions through the same observation, "
        "reset, Franka/RH56 mapping, shaping, ACK, stop, and audit runtime as "
        "normal deployment. Only checkpoint inference is replaced."
    )
    parser.add_argument(
        "--actions",
        type=Path,
        required=True,
        help=(
            "validated [K,13] .json/.npy/.npz/.csv/.txt sequence, or the "
            "simulation-export .zip replay bundle"
        ),
    )
    # ``build_policy_parser`` owns the shared --arrival-gated and tolerance
    # spellings.  Replay keeps its existing exact-target repeat semantics;
    # normal checkpoint deployment uses the same flags as a no-inference hold.
    parser.set_defaults(steps=None, arrival_tolerance_rad="0.015")
    return parser


def build_replay_request(args: argparse.Namespace) -> DeploymentRequest:
    mode = resolve_policy_rate_mode(args.policy_rate_hz)
    action_path = args.actions.expanduser().resolve()
    sequence = load_replay_actions(
        action_path,
        expected_policy_rate_hz=mode.policy_rate_hz,
    )
    try:
        arrival_tolerance = float(str(args.arrival_tolerance_rad).strip())
    except (TypeError, ValueError) as exc:
        raise SupervisedV94RunError(
            "--arrival-tolerance-rad must be in 0.005..0.030"
        ) from exc
    if not 0.005 <= arrival_tolerance <= 0.030:
        raise SupervisedV94RunError(
            "--arrival-tolerance-rad must be in 0.005..0.030"
        )
    if bool(args.arrival_gated) and (
        sequence.recorded_franka_target_q_rad is None
        or sequence.recorded_rh56_angle_set_register_order is None
    ):
        raise SupervisedV94RunError(
            "--arrival-gated requires replay data with exact Franka and RH56 targets"
        )
    if sequence.tabletop_intercept is not None and not bool(args.arrival_gated):
        raise SupervisedV94RunError(
            "tabletop intercept replay requires --arrival-gated"
        )
    if sequence.tabletop_online_planner is not None and bool(args.arrival_gated):
        raise SupervisedV94RunError(
            "online tabletop planner runs at policy-rate 20 Hz and must not "
            "use --arrival-gated"
        )
    if args.steps is None:
        selected_steps = sequence.action_count
    else:
        try:
            selected_steps = int(str(args.steps).strip())
        except (TypeError, ValueError) as exc:
            raise SupervisedV94RunError(
                "--steps must be an integer no larger than the action count"
            ) from exc
    if not 1 <= selected_steps <= sequence.action_count:
        raise SupervisedV94RunError(
            f"--steps must remain in 1..{sequence.action_count} for this action file"
        )
    if (
        sequence.tabletop_intercept is not None
        or sequence.tabletop_online_planner is not None
    ) and selected_steps != sequence.action_count:
        raise SupervisedV94RunError(
            "tabletop planner replay requires the complete action plan"
        )
    if selected_steps > mode.maximum_supervised_steps:
        raise SupervisedV94RunError(
            f"{mode.name} replay permits at most "
            f"{mode.maximum_supervised_steps} steps per supervised run"
        )
    args.steps = str(selected_steps)
    base = build_deployment_request(args)
    return replace(
        base,
        replay_actions=action_path,
        replay_actions_sha256=sequence.sha256,
        replay_action_count=sequence.action_count,
        replay_arrival_gated=bool(args.arrival_gated),
        replay_arrival_tolerance_rad=arrival_tolerance,
    )


def build_replay_summary(request: DeploymentRequest) -> Mapping[str, object]:
    return build_deployment_summary(request)


def execute_replay(request: DeploymentRequest) -> Mapping[str, object]:
    if request.replay_actions is None:
        raise SupervisedV94RunError("replay request has no --actions file")
    return execute_deployment(request)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        request = build_replay_request(build_parser().parse_args(argv))
        summary = dict(build_replay_summary(request))
        if not request.execute:
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        result = dict(execute_replay(request))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, SupervisedV94RunError) as exc:
        print(f"V94 action replay: REFUSED: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "V94 action replay: interrupted; verified stop was requested",
            file=sys.stderr,
        )
        return 130
    except SystemExit:
        raise
    except BaseException as exc:
        print(
            f"V94 action replay: FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


__all__ = [
    "build_parser",
    "build_replay_request",
    "build_replay_summary",
    "execute_replay",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
