#!/usr/bin/env python3
"""Hardware-free materializer/verifier for the commissioned V94 RH56 profile."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT / "src", WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anydex_pipeline.rh56_commissioning import atomic_write_evidence
from anydex_pipeline.v94_rh56_profile_bridge import (
    verify_materialized_v94_rh56_profile,
    verify_v94_rh56_profile_bridge,
)


DEFAULT_EVIDENCE = (
    ROOT / "runs/rh56_policy_seq286_20hz_coupled_20260724_codex01.json"
)
DEFAULT_SOURCE = ROOT / "configs/fr3_rh56_v7_commissioning.json"
DEFAULT_V94_BASE = ROOT / "configs/fr3_rh56_v94_commissioning.json"
DEFAULT_OUTPUT = (
    ROOT / "configs/fr3_rh56_v94_seq286_20hz_commissioned.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="materialize or verify the evidence-derived V94 RH56 profile"
    )
    parser.add_argument("command", choices=("materialize", "verify"))
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--v94-base-profile", type=Path, default=DEFAULT_V94_BASE)
    parser.add_argument("--output-profile", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_profile.expanduser().resolve()
    if args.command == "materialize":
        if output.parent != args.v94_base_profile.expanduser().resolve().parent:
            raise ValueError("output profile must remain beside the V94 base profile")
        bridge = verify_v94_rh56_profile_bridge(
            args.evidence,
            source_config_path=args.source_config,
            v94_base_profile_path=args.v94_base_profile,
        )
        if not bridge.passed or bridge.derived_profile is None:
            for blocker in bridge.blockers:
                print(f"[bridge blocker] {blocker}", file=sys.stderr)
            return 2
        digest = atomic_write_evidence(output, bridge.derived_profile)
        print(f"[materialize] profile={output}")
        print(f"[materialize] sha256={digest}")

    verified = verify_materialized_v94_rh56_profile(
        args.evidence,
        output,
        source_config_path=args.source_config,
        v94_base_profile_path=args.v94_base_profile,
    )
    if not verified.passed:
        for blocker in verified.blockers:
            print(f"[verify blocker] {blocker}", file=sys.stderr)
        return 2
    print("[verify] PASS")
    print(f"[verify] profile={output}")
    print(f"[verify] proposal={verified.proposal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
