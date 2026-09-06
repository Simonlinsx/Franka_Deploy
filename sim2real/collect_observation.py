#!/usr/bin/env python3
"""Collect synchronized-enough diagnostic observations from the real setup.

This command is strictly read-only.  It never creates a Franka control handle
and never writes an RH56 register.  The object-PCD publisher must already be
running when that source is enabled.
"""

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
    from sim2real.config import (  # type: ignore
        DEFAULT_CONFIG_PATH,
        load_runtime_config,
        observation_spec_from_config,
    )
    from sim2real.contracts import assemble_observation  # type: ignore
    from sim2real.io import ObservationReader  # type: ignore
else:
    from .config import (
        DEFAULT_CONFIG_PATH,
        load_runtime_config,
        observation_spec_from_config,
    )
    from .contracts import assemble_observation
    from .io import ObservationReader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read Franka + Inspire + object-PCD observations without motion"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--hz", type=float, default=2.0)
    parser.add_argument(
        "--without-franka",
        action="store_true",
        help="do not connect to the Franka; useful for camera/hand bench checks",
    )
    parser.add_argument(
        "--without-inspire",
        action="store_true",
        help="do not open the RH56 serial port",
    )
    parser.add_argument(
        "--without-object-pcd",
        action="store_true",
        help="do not subscribe to the object-PCD publisher",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "save one sample as .npz; with --count > 1 this must be an output "
            "directory"
        ),
    )
    parser.add_argument(
        "--require-valid",
        action="store_true",
        help="return a non-zero exit code if any requested sample is invalid",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if not np.isfinite(args.hz) or args.hz <= 0.0:
        raise ValueError("--hz must be finite and positive")
    if args.output is not None and args.count == 1 and args.output.suffix != ".npz":
        raise ValueError("single-sample --output must end in .npz")
    if args.output is not None and args.count > 1 and args.output.suffix:
        raise ValueError("multi-sample --output must be a directory")


def _save_sample(sample, output: Path, index: int, count: int) -> Path:
    if count == 1:
        destination = output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
    else:
        directory = output.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"observation_{index:06d}.npz"
    np.savez_compressed(destination, **sample.to_npz_payload())
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
        runtime, profile, _ = load_runtime_config(args.config)
        use_franka = (
            bool(runtime["franka"].get("enabled", True)) and not args.without_franka
        )
        use_inspire = (
            bool(runtime["inspire"].get("enabled", True)) and not args.without_inspire
        )
        use_object = (
            bool(runtime["object_pcd"].get("enabled", True))
            and not args.without_object_pcd
        )
        spec = observation_spec_from_config(
            runtime,
            profile,
            require_franka=use_franka,
            require_inspire=use_inspire,
            require_object_pcd=use_object,
        )
        reader = ObservationReader(
            runtime,
            profile,
            use_franka=use_franka,
            use_inspire=use_inspire,
            use_object_pcd=use_object,
        )
        any_invalid = False
        period_s = 1.0 / args.hz
        with reader:
            for index in range(args.count):
                loop_started = time.monotonic()
                franka, inspire, object_pcd = reader.read_raw()
                sample = assemble_observation(
                    captured_at_s=time.time(),
                    spec=spec,
                    franka=franka,
                    inspire=inspire,
                    object_pcd=object_pcd,
                )
                any_invalid = any_invalid or not sample.valid
                print(json.dumps(sample.summary(), ensure_ascii=False, sort_keys=True))
                if args.output is not None:
                    saved = _save_sample(sample, args.output, index, args.count)
                    print(f"[saved] {saved}", file=sys.stderr)
                remaining = period_s - (time.monotonic() - loop_started)
                if index + 1 < args.count and remaining > 0.0:
                    time.sleep(remaining)
        return 1 if args.require_valid and any_invalid else 0
    except KeyboardInterrupt:
        return 130
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
