#!/usr/bin/env python3
"""Stable command-line entry point for V94 real-robot deployment.

Use ``python -m sim2real.deployment`` for new integrations. This compatibility
module preserves ``python -m sim2real.deploy`` and the old import path.
"""

from __future__ import annotations

from .deployment.cli import (
    DeploymentAdmissionError,
    DeploymentExecutionError,
    DeploymentRequest,
    build_deployment_request,
    build_deployment_summary,
    build_parser,
    execute_deployment,
    main,
)

__all__ = [
    "DeploymentAdmissionError",
    "DeploymentExecutionError",
    "DeploymentRequest",
    "build_deployment_request",
    "build_deployment_summary",
    "build_parser",
    "execute_deployment",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
