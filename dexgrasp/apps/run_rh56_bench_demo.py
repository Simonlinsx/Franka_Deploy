#!/usr/bin/env python3
"""Supervised, unmounted RH56 bench demonstration.

The hand must be mechanically clamped.  This program never connects Franka.
It performs a conservative partial close, disables all six outputs, then opens
the hand again in a fresh connection and leaves every ANGLE_SET at -1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT / "src", WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


CLAMP_TOKEN = "RH56_MECHANICALLY_CLAMPED"
POWER_TOKEN = "RH56_24V_CUTOFF_READY"
CLEAR_TOKEN = "RH56_WORKSPACE_CLEAR"
DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
CLOSE_TARGET = (700, 700, 700, 700, 800, 900)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an unmounted RH56 partial-close demonstration and finish fully "
            "open with all six ANGLE_SET outputs disabled. Franka is never opened."
        )
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument(
        "--close-targets",
        type=int,
        nargs=6,
        default=list(CLOSE_TARGET),
        metavar=("PINKY", "RING", "MIDDLE", "INDEX", "THUMB_BEND", "THUMB_ROTATE"),
    )
    parser.add_argument("--confirm-clamped", metavar=CLAMP_TOKEN)
    parser.add_argument("--confirm-24v-cutoff", metavar=POWER_TOKEN)
    parser.add_argument("--confirm-workspace-clear", metavar=CLEAR_TOKEN)
    return parser


def _require_confirmations(args: argparse.Namespace) -> None:
    expected = (
        ("--confirm-clamped", args.confirm_clamped, CLAMP_TOKEN),
        ("--confirm-24v-cutoff", args.confirm_24v_cutoff, POWER_TOKEN),
        ("--confirm-workspace-clear", args.confirm_workspace_clear, CLEAR_TOKEN),
    )
    missing = [
        "{} {}".format(flag, token)
        for flag, actual, token in expected
        if actual != token
    ]
    if missing:
        raise ValueError("exact confirmations required: " + "; ".join(missing))
    if not 0.0 <= float(args.hold_seconds) <= 5.0:
        raise ValueError("--hold-seconds must be in 0..5")
    targets = tuple(int(value) for value in args.close_targets)
    if len(targets) != 6 or any(value < 0 or value > 1000 for value in targets):
        raise ValueError("--close-targets must contain six values in 0..1000")
    if not 900 <= targets[5] <= 1000:
        raise ValueError("thumb-rotate close target must stay in validated 900..1000")


def _connect(port: str):
    # Lazy import: missing confirmations cannot import or connect the driver.
    from anydex_pipeline.inspire_sequence_driver import RH56SequenceDriver

    return RH56SequenceDriver.connect(
        port=port,
        baud=115200,
        hand_id=1,
        thumb_rotate_range=(900, 1000),
        motion_timeout_s=20.0,
        angle_tolerance=25,
    )


def run_demo(port: str, hold_seconds: float, close_target=CLOSE_TARGET) -> None:
    close_target = tuple(int(value) for value in close_target)
    print("[RH56 bench] Franka is not connected")
    print("[RH56 bench] stage 1/5: verify/open all six axes")
    first = _connect(port)
    try:
        first.open_and_verify((1000,) * 6)
        print("[RH56 bench] stage 2/5: preshape thumb rotation to 900")
        first.preshape_thumb(900)
        print(
            "[RH56 bench] stage 3/5: partial close to {} at speed=40 force=80g".format(
                list(close_target)
            )
        )
        first.close_bends_and_hold(close_target)
        time.sleep(float(hold_seconds))
    finally:
        first.close()

    print("[RH56 bench] stage 4/5: numeric output disabled; reopen in fresh session")
    second = _connect(port)
    try:
        second.open_and_verify((1000,) * 6)
    finally:
        second.close()
    print("[RH56 bench] stage 5/5: fully open, ANGLE_SET=[-1]*6, settings restored")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _require_confirmations(args)
        run_demo(
            str(args.port), float(args.hold_seconds), tuple(args.close_targets)
        )
    except KeyboardInterrupt:
        print("[RH56 bench] interrupted; cut 24 V if any motion remains", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print("[RH56 bench] failed: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
