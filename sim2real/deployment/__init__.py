"""Stable public API for model admission and supervised deployment.

The command facade is loaded lazily so importing a bundle or safety primitive
does not eagerly import the complete runtime and its device adapters.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = (
    "DeploymentAdmissionError",
    "DeploymentExecutionError",
    "DeploymentRequest",
    "build_deployment_request",
    "execute_deployment",
)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    implementation = import_module(".cli", __name__)
    return getattr(implementation, name)
