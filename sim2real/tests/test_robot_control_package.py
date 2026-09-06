"""Checks for the canonical FR3/RH56 device boundary."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("canonical_name", "public_symbol"),
    [
        (
            "robot_control.franka.session",
            "FrankaPersistentSession",
        ),
        (
            "robot_control.franka.backend",
            "PylibfrankaBackendFactory",
        ),
        (
            "robot_control.franka.native_session",
            "FrankaNativeSupervisedSessionProxy",
        ),
        (
            "robot_control.rh56.linux_transport",
            "LinuxRH56TransactionalTransport",
        ),
        (
            "robot_control.rh56.actuator",
            "RH56TransactionalActuator",
        ),
        (
            "robot_control.rh56.watchdog",
            "RH56WatchdogOwner",
        ),
    ],
)
def test_canonical_hardware_module_exports_public_symbol(
    canonical_name: str,
    public_symbol: str,
) -> None:
    canonical = importlib.import_module(canonical_name)

    assert getattr(canonical, public_symbol) is not None


def test_public_device_packages_export_canonical_types() -> None:
    from robot_control.franka import FrankaPersistentSession
    from robot_control.franka.session import (
        FrankaPersistentSession as CanonicalFrankaPersistentSession,
    )
    from robot_control.rh56 import RH56WatchdogOwner
    from robot_control.rh56.watchdog import (
        RH56WatchdogOwner as CanonicalRH56WatchdogOwner,
    )

    assert FrankaPersistentSession is CanonicalFrankaPersistentSession
    assert RH56WatchdogOwner is CanonicalRH56WatchdogOwner
