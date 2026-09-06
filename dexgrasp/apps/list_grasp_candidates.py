#!/usr/bin/env python3
"""List every saved candidate without opening a camera, GUI, or robot driver."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anydex_pipeline.candidate_summary import candidate_summary_lines  # noqa: E402
from anydex_pipeline.snapshot import load_snapshot_npz  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print fixed candidate indices, poses, scores, types, and RH56 targets"
    )
    parser.add_argument("snapshot", type=Path)
    parser.add_argument(
        "--selected-index",
        type=int,
        default=None,
        help="mark this candidate with '*'; default uses NPZ selection or index 0",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        snapshot = load_snapshot_npz(args.snapshot.expanduser().resolve())
        selected = (
            int(args.selected_index)
            if args.selected_index is not None
            else max(0, int(snapshot.grasps.selected_index))
        )
        if not 0 <= selected < snapshot.grasps.count:
            raise ValueError(
                "selected index {} is outside candidates 0..{}".format(
                    selected, snapshot.grasps.count - 1
                )
            )
        print(
            "[safety] OFFLINE LIST ONLY: no camera, GUI, Franka, or Inspire transport"
        )
        for line in candidate_summary_lines(
            snapshot,
            selected_index=selected,
            requested_top_k=snapshot.grasps.count,
        ):
            print(line)
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print("[list-candidates] failed: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
