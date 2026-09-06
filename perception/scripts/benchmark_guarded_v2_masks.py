#!/usr/bin/env python3
"""Run the hardware-free guarded_v2 recorded-video regression benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_pcd.evaluation.guarded_v2_offline import (  # noqa: E402
    OfflineBenchmarkError,
    evaluate_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved binary masks on recorded videos; never opens camera, "
            "Franka or RH56 interfaces."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=None,
        help=(
            "override every historical candidate with "
            "ROOT/<case>/masks and optional ROOT/<case>/states.jsonl"
        ),
    )
    parser.add_argument("--candidate-name", default="guarded_v2")
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help=(
            "select a case only with --diagnostic-partial; a partial run can "
            "never receive production PASS"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--diagnostic-partial",
        action="store_true",
        help=(
            "verify replay summaries/digests for selected cases but mark the "
            "whole result diagnostic and production-ineligible"
        ),
    )
    mode.add_argument(
        "--diagnostic-legacy-candidate",
        action="store_true",
        help=(
            "legacy mask-directory compatibility without replay provenance; "
            "always diagnostic and never a production PASS"
        ),
    )
    args = parser.parse_args()
    candidate_root_mode = (
        "diagnostic_partial"
        if args.diagnostic_partial
        else (
            "diagnostic_legacy"
            if args.diagnostic_legacy_candidate
            else "production"
        )
    )
    try:
        result = evaluate_manifest(
            args.manifest,
            args.output,
            candidate_root=args.candidate_root,
            candidate_name=str(args.candidate_name),
            candidate_root_mode=candidate_root_mode,
            selected_cases=args.case,
        )
    except (OfflineBenchmarkError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"guarded_v2 offline benchmark: FAILED: {type(exc).__name__}: {exc}")
        return 2
    print(
        json.dumps(
            {
                "result": "PASS" if result["passed"] else "FAILED_CHECKS",
                "case_count": result["case_count"],
                "summary": str(args.output.expanduser().resolve() / "summary.json"),
                "report": str(args.output.expanduser().resolve() / "REPORT.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
