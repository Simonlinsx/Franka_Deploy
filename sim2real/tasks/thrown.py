"""Thrown-object task contracts."""

from pathlib import Path
from typing import Mapping, Sequence

from sim2real.tasks.thrown_contract import (
    ThrownTaskContractError,
    V57CameraVisibility,
    V57Curriculum,
    V57ThrownTaskContract,
    apply_v57_task_contract,
    assess_v57_camera_visibility,
    load_v57_thrown_task_contract,
    resolve_v57_thrown_task_contract,
    v57_task_summary,
)

from .launcher import build_task_command, materialize_task_config


def materialize_thrown_config(
    task_name: str = "thrown_object",
) -> tuple[Path, Mapping[str, object]]:
    """Materialize one maintained thrown-object task configuration."""

    return materialize_task_config(task_name)


def build_thrown_command(
    mode: str,
    arguments: Sequence[str] = (),
    *,
    task_name: str = "thrown_object",
) -> tuple[list[str], Path, Mapping[str, object]]:
    """Build a validated thrown-object command without executing it."""

    return build_task_command(task_name, mode, arguments)

__all__ = [
    "ThrownTaskContractError",
    "V57CameraVisibility",
    "V57Curriculum",
    "V57ThrownTaskContract",
    "apply_v57_task_contract",
    "assess_v57_camera_visibility",
    "build_thrown_command",
    "load_v57_thrown_task_contract",
    "materialize_thrown_config",
    "resolve_v57_thrown_task_contract",
    "v57_task_summary",
]
