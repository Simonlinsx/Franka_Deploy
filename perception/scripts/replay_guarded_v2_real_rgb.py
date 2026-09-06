#!/usr/bin/env python3
"""Export production guarded_v2 masks from historical colour-only videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_pcd.evaluation.guarded_v2_real_rgb_replay import (  # noqa: E402
    RealRGBReplayError,
    replay_manifest,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT / "configs" / "guarded_v2_real_rgb_replay_cases.json"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "d435_default.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the production guarded_v2 2-D state machine on saved RGB MP4s. "
            "Uses neutral synthetic depth; opens no hardware interface."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="case name; repeat to select several (default: all)",
    )
    parser.add_argument("--neutral-depth-m", type=float, default=1.0)
    parser.add_argument(
        "--sam2-image-size",
        type=int,
        default=None,
        help="offline A/B override; e.g. 448 versus production default 512",
    )
    parser.add_argument(
        "--as-fast-as-possible",
        action="store_true",
        help="diagnostic only; default preserves the exact 20 Hz wall schedule",
    )
    parser.add_argument(
        "--diagnostic-semantic-sam2-direct",
        action="store_true",
        help=(
            "A/B diagnosis only: let temporal SAM2 publish directly. This "
            "never counts as guarded_v2 production acceptance."
        ),
    )
    args = parser.parse_args()
    try:
        result = replay_manifest(
            args.manifest,
            args.candidate_root,
            config_path=args.config,
            selected_cases=args.case,
            neutral_depth_m=float(args.neutral_depth_m),
            realtime=not bool(args.as_fast_as_possible),
            diagnostic_semantic_sam2_direct=bool(
                args.diagnostic_semantic_sam2_direct
            ),
            sam2_image_size_override=args.sam2_image_size,
        )
    except (
        FileExistsError,
        OSError,
        RealRGBReplayError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"guarded_v2 real-RGB replay: FAILED: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
