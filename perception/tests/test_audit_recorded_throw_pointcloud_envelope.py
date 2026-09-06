from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT = (
    WORKSPACE
    / "perception"
    / "scripts"
    / "audit_recorded_throw_pointcloud_envelope.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "audit_recorded_throw_pointcloud_envelope", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_percentile_is_explicit_for_empty_and_finite_values() -> None:
    module = _module()
    assert module._percentile([], 50) is None
    assert module._percentile([1.0, 2.0, 3.0], 50) == pytest.approx(2.0)
