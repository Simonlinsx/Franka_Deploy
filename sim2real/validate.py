#!/usr/bin/env python3
"""Stable offline validation entry point for a deployment bundle/checkpoint."""

from __future__ import annotations

from .deployment.verify import (
    V94CheckpointCompatibilityReport,
    V94VerificationReport,
    build_parser,
    main,
    verify_v94_bundle,
    verify_v94_checkpoint_override,
    verify_v94_checkpoint_payload,
)

__all__ = [
    "V94CheckpointCompatibilityReport",
    "V94VerificationReport",
    "build_parser",
    "main",
    "verify_v94_bundle",
    "verify_v94_checkpoint_override",
    "verify_v94_checkpoint_payload",
]


if __name__ == "__main__":
    raise SystemExit(main())
