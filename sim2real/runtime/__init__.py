"""Supervised observation/action scheduling and hardware-owner runtime.

The public composition API is loaded lazily so importing a small runtime
primitive does not eagerly import policy, camera, or device adapters.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = (
    "FreshHardwarePreflight",
    "PreparedSupervisedV94Artifacts",
    "SupervisedV94RuntimeError",
    "VerifiedObjectROI",
    "prepare_supervised_v94",
    "prepare_supervised_v94_artifacts",
    "run_supervised_v94",
)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    implementation = import_module(".supervised_v94_runtime", __name__)
    return getattr(implementation, name)
