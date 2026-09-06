"""Checks for commissioning module organization."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "public_symbol"),
    [
        ("calibrate_rh56_force", "main"),
        ("calibrate_rh56_force_baseline", "analyze_poses"),
        ("d435_timing_probe", "main"),
        ("derive_fr3_rh56_end_effector_dynamics", "main"),
        ("identify_fr3_rh56_static_load", "identify"),
        ("identify_rh56_dynamics", "fit_step_response"),
        ("rh56_v94_microprobe", "derive_v94_mapper_probe_endpoint"),
        ("rh56_v94_all_axis_microprobe", "derive_all_axis_mapper_contract"),
        ("rh56_v94_first_tick_vector_probe", "derive_first_tick_contract"),
    ],
)
def test_canonical_commissioning_module_exports_public_symbol(
    module_name: str,
    public_symbol: str,
) -> None:
    canonical = importlib.import_module(f"sim2real.commissioning.{module_name}")

    assert getattr(canonical, public_symbol) is not None


def test_importing_commissioning_package_has_no_eager_device_modules() -> None:
    package = importlib.import_module("sim2real.commissioning")

    assert package.__all__ == ()
