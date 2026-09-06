"""Checks for the domain-organized diagnostic implementations."""

from __future__ import annotations

import importlib

import pytest

from sim2real.workspace_paths import DEFAULT_V94_DEPLOY_BUNDLE, WORKSPACE_ROOT


@pytest.mark.parametrize(
    ("module_name", "public_symbol"),
    (
        ("analyze_v94_pointcloud_ab", "analyze_pointcloud_ab"),
        ("audit_thrown_v60_candidate", "run"),
        ("audit_thrown_v61_bundle", "audit_bundle"),
        ("audit_v57_thrown_alignment", "build_alignment_audit"),
        ("compare_policy_io", "compare_policy_io"),
        ("compare_v94_action_trends", "compare_action_trends"),
        ("export_policy_io_action_replay", "export_policy_io_action_replay"),
        ("replay_v94", "replay_recording"),
    ),
)
def test_canonical_diagnostic_module_exports_public_entry_point(
    module_name: str,
    public_symbol: str,
) -> None:
    canonical = importlib.import_module(f"sim2real.diagnostics.{module_name}")

    assert callable(canonical.main)
    assert getattr(canonical, public_symbol) is not None


def test_default_bundle_uses_canonical_data_fixture_root() -> None:
    expected = WORKSPACE_ROOT / "data/test_fixtures/sim2real/deploy.zip"

    assert DEFAULT_V94_DEPLOY_BUNDLE == expected
    assert DEFAULT_V94_DEPLOY_BUNDLE.is_file()
