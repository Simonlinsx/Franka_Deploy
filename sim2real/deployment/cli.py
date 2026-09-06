#!/usr/bin/env python3
"""Command-line entry point for supervised model deployment."""

from sim2real.deployment.runner import (
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
