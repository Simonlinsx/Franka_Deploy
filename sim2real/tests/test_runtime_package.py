"""Import-boundary checks for the runtime package."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "public_symbol"),
    [
        ("bounded_c2_orchestrator", "BoundedC2Orchestrator"),
        ("bounded_c2_runtime", "BoundedV94C2Runtime"),
        ("v94_policy_tick_source", "TransactionalBoundedV94PolicyTickSource"),
        ("v94_live_observation_owner", "D435ObjectCameraOwner"),
        ("supervised_v94_runtime", "run_supervised_v94"),
    ],
)
def test_canonical_runtime_module_exports_public_symbol(
    module_name: str,
    public_symbol: str,
) -> None:
    canonical = importlib.import_module(f"sim2real.runtime.{module_name}")

    assert getattr(canonical, public_symbol) is not None
