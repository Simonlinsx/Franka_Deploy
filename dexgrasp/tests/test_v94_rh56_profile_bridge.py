from __future__ import annotations

import json
from pathlib import Path

from anydex_pipeline.v94_rh56_profile_bridge import (
    verify_materialized_v94_rh56_profile,
    verify_v94_rh56_profile_bridge,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "runs/rh56_policy_seq286_20hz_coupled_20260724_codex01.json"
SOURCE = ROOT / "configs/fr3_rh56_v7_commissioning.json"
V94_BASE = ROOT / "configs/fr3_rh56_v94_commissioning.json"
MATERIALIZED = (
    ROOT / "configs/fr3_rh56_v94_seq286_20hz_commissioned.json"
)


def test_checked_in_v94_profile_is_exact_evidence_bridge() -> None:
    result = verify_materialized_v94_rh56_profile(
        EVIDENCE,
        MATERIALIZED,
        source_config_path=SOURCE,
        v94_base_profile_path=V94_BASE,
    )
    assert result.passed, result.blockers
    assert result.proposal is not None
    assert result.proposal["inspire.thumb_rotate_validated_realtime_range"] == [
        416,
        1000,
    ]
    assert result.proposal["inspire.six_axis_coupled_closure_commissioned"] is True
    assert result.proposal["inspire.commissioned_air_closure_targets"] == [
        [61, 17, 524, 758, 422, 416]
    ]

    base = json.loads(V94_BASE.read_text(encoding="utf-8"))
    derived = json.loads(MATERIALIZED.read_text(encoding="utf-8"))
    assert derived["franka"] == base["franka"]
    assert derived["profile_id"] == base["profile_id"]
    assert derived["policy_contract"] == base["policy_contract"]


def test_bridge_rejects_unrelated_v94_hardware_change(tmp_path: Path) -> None:
    value = json.loads(V94_BASE.read_text(encoding="utf-8"))
    value["calibration"]["camera_serial"] = "different-camera"
    changed = tmp_path / "changed_v94.json"
    changed.write_text(json.dumps(value), encoding="utf-8")

    result = verify_v94_rh56_profile_bridge(
        EVIDENCE,
        source_config_path=SOURCE,
        v94_base_profile_path=changed,
    )
    assert not result.passed
    assert any("outside allowed reset-only fields" in item for item in result.blockers)


def test_bridge_rejects_materialized_profile_tamper(tmp_path: Path) -> None:
    value = json.loads(MATERIALIZED.read_text(encoding="utf-8"))
    value["inspire"]["thumb_rotate_validated_realtime_range"] = [400, 1000]
    changed = tmp_path / "changed_materialized.json"
    changed.write_text(json.dumps(value), encoding="utf-8")

    result = verify_materialized_v94_rh56_profile(
        EVIDENCE,
        changed,
        source_config_path=SOURCE,
        v94_base_profile_path=V94_BASE,
    )
    assert not result.passed
    assert result.blockers == (
        "materialized V94 profile is not the exact bridge-derived update",
    )
