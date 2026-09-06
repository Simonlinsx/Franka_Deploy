"""Checks for policy and contract package organization."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("canonical_name", "public_symbol"),
    [
        (
            "sim2real.policy.io_recorder",
            "PolicyIORecorder",
        ),
        ("sim2real.policy.rate_mode", "PolicyRateMode"),
        ("sim2real.contracts.v94", "V94Contract"),
        ("sim2real.contracts.actions", "V94ActionMapper"),
    ],
)
def test_canonical_support_module_exports_public_symbol(
    canonical_name: str,
    public_symbol: str,
) -> None:
    canonical = importlib.import_module(canonical_name)

    assert getattr(canonical, public_symbol) is not None


def test_existing_policy_and_contract_import_paths_remain_public_packages() -> None:
    policy = importlib.import_module("sim2real.policy")
    contracts = importlib.import_module("sim2real.contracts")

    assert policy.RollingStudentPolicy.__module__ == "sim2real.policy"
    assert contracts.ObservationSpec.__module__ == "sim2real.contracts"
