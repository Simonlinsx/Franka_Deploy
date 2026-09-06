"""Tabletop task composition helpers."""

from pathlib import Path
from typing import Mapping, Sequence

from .launcher import build_task_command, materialize_task_config


def materialize_tabletop_config() -> tuple[Path, Mapping[str, object]]:
    """Materialize the maintained tabletop task configuration."""

    return materialize_task_config("tabletop")


def build_tabletop_command(
    mode: str,
    arguments: Sequence[str] = (),
) -> tuple[list[str], Path, Mapping[str, object]]:
    """Build a validated tabletop task command without executing it."""

    return build_task_command("tabletop", mode, arguments)


__all__ = ["build_tabletop_command", "materialize_tabletop_config"]
