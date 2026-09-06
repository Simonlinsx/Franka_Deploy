#!/usr/bin/env python3
"""Create or verify a hardware-free continuous telemetry session manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anydex_pipeline.telemetry_session_manifest import (  # noqa: E402
    EXECUTION_COMMANDS,
    build_telemetry_session_manifest,
    load_telemetry_session_manifest,
    write_telemetry_session_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create/verify the immutable identity manifest shared by the native "
            "continuous telemetry producer and its viewer. This command is "
            "offline and never opens hardware."
        )
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    create = subparsers.add_parser("create", help="derive and atomically write a manifest")
    create.add_argument("--snapshot", type=Path, required=True)
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--audit-artifact", type=Path, required=True)
    create.add_argument(
        "--producer-build",
        type=Path,
        required=True,
        help="exact native telemetry producer binary/build artifact to bind",
    )
    create.add_argument("--command", choices=EXECUTION_COMMANDS, required=True)
    create.add_argument("--selected-index", type=int, required=True)
    create.add_argument(
        "--run-uuid",
        default=None,
        help="canonical UUID; omitted generates a new UUID4",
    )
    create.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify", help="strictly replay manifest content and every bound file"
    )
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def _print_identity(loaded) -> None:
    identity = loaded.identity
    print("[telemetry-session] run_uuid={}".format(identity.run_uuid))
    print(
        "[telemetry-session] execution_contract_sha256={}".format(
            identity.execution_contract_sha256
        )
    )
    print(
        "[telemetry-session] source_snapshot_sha256={}".format(
            identity.source_snapshot_sha256
        )
    )
    print(
        "[telemetry-session] control_config_sha256={}".format(
            identity.control_config_sha256
        )
    )
    print(
        "[telemetry-session] calibration_sha256={}".format(
            identity.calibration_sha256
        )
    )
    print(
        "[telemetry-session] producer_build_sha256={}".format(
            identity.producer_build_sha256
        )
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.operation == "create":
            payload = build_telemetry_session_manifest(
                snapshot_path=args.snapshot,
                control_config_path=args.config,
                audit_artifact_path=args.audit_artifact,
                producer_build_path=args.producer_build,
                command=args.command,
                selected_index=args.selected_index,
                run_uuid=args.run_uuid,
            )
            output = write_telemetry_session_manifest(args.output, payload)
            loaded = load_telemetry_session_manifest(output, verify_files=True)
            print("[telemetry-session] created={}".format(output))
        else:
            loaded = load_telemetry_session_manifest(
                args.manifest, verify_files=True
            )
            print("[telemetry-session] verified={}".format(loaded.path))
        print(
            "[telemetry-session] manifest_file_sha256={}".format(
                loaded.manifest_file_sha256
            )
        )
        _print_identity(loaded)
        print("[telemetry-session] motion_authorized=false")
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print("[telemetry-session] rejected: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
