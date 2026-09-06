"""Task-level contracts and launch composition.

The launcher facade is lazy so ``python -m sim2real.tasks.launcher`` does not
pre-import the same module while Python is preparing to execute it.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = (
    "TaskLauncherError",
    "build_task_command",
    "materialize_task_config",
)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    implementation = import_module(".launcher", __name__)
    return getattr(implementation, name)
