from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from anydex_pipeline.hppfcl_installed_tool_backend import InstalledReturnNeighborhood
from anydex_pipeline.installed_scene_filter import bounded_residual_self_returns


ROOT = Path(__file__).resolve().parents[1]
FRESH_FILTER = (
    ROOT
    / "runs/live_scene_installed_filtered_candidate51_selfguard_v6_20260721.npz"
)
FRESH_DIAGNOSTIC = (
    ROOT
    / "runs/candidate51_installed_air_collision_diagnostic_selfguard_v6_20260721.json"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _neighborhood(count: int) -> InstalledReturnNeighborhood:
    return InstalledReturnNeighborhood(
        candidate_indices=np.arange(count, dtype=np.int64),
        reason_labels=tuple("FR3_visual_link2" for _ in range(count)),
        surface_distances_m=np.linspace(0.0, 0.014, count, dtype=np.float64),
        candidate_test_count=count,
        maximum_surface_distance_m=0.015,
        distance_method="HPP-FCL triangle-mesh to 1nm sphere, radius corrected",
    )


def test_residual_growth_requires_model_appearance_and_connectivity():
    # Index 0 is an exact conservative point-cube/mesh intersection seed.
    # 1/2 form a white connected self-return continuation.  Index 3 is a
    # nearby dark obstacle, and 4 is white/model-near but disconnected from
    # the seed component.  Both adversarial obstacles must remain.
    points = np.asarray(
        [
            [0.000, 0.000, 0.000],
            [0.007, 0.000, 0.000],
            [0.014, 0.000, 0.000],
            [0.006, 0.001, 0.000],
            [0.000, 0.015, 0.000],
        ],
        dtype=np.float64,
    )
    white = [0.98, 1.00, 0.99]
    colors = np.asarray([white, white, white, [0.05, 0.05, 0.05], white])
    removed, evidence = bounded_residual_self_returns(
        points,
        colors,
        np.asarray([0], dtype=np.int64),
        ("FR3_visual_link2",),
        _neighborhood(len(points)),
    )
    assert removed.tolist() == [1, 2]
    assert 3 not in removed  # close, but appearance-incompatible obstacle
    assert 4 not in removed  # same appearance/model-near, but disconnected
    assert evidence["residual_removed_count"] == 2
    assert evidence["residual_removed_surface_distance_max_m"] <= 0.015


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("connectivity_radius_m", 0.0101),
        ("palette_color_max_l2", 0.181),
        ("maximum_geodesic_m", 0.0401),
    ],
)
def test_residual_growth_policy_upper_bounds_are_locked(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        bounded_residual_self_returns(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[1.0, 1.0, 1.0]]),
            np.asarray([0], dtype=np.int64),
            ("FR3_visual_link2",),
            _neighborhood(1),
            **{keyword: value},
        )


@pytest.mark.skipif(
    not FRESH_FILTER.is_file() or not FRESH_DIAGNOSTIC.is_file(),
    reason="bound 2026-07-21 installed fresh-scene regression is unavailable",
)
def test_real_fresh_capture_replay_remains_locked_on_unresolved_rh56_return():
    """The bounded filter may improve evidence but must not forge a pass."""

    with np.load(FRESH_FILTER, allow_pickle=False) as archive:
        evidence_path = Path(str(archive["filter_evidence_path"].item()))
        assert _file_sha256(evidence_path) == str(
            archive["filter_evidence_sha256"].item()
        )
        assert bool(archive["scene_excludes_object"].item()) is True
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for path_key, hash_key in (
        ("live_scene_path", "live_scene_sha256"),
        ("snapshot_path", "snapshot_sha256"),
        ("adapter_path", "adapter_sha256"),
    ):
        assert _file_sha256(Path(evidence[path_key])) == evidence[hash_key]
    residual = evidence["installed_return_filter"]["residual_self_return_filter"]
    assert residual["enabled"] is True
    # Historical v6 evidence remains immutable; current evidence is validated
    # separately by the strict installed-tool audit schema.
    assert residual["hard_maximum_model_surface_distance_m"] == 0.020
    assert residual["maximum_model_surface_distance_m"] <= 0.020
    assert residual["residual_removed_surface_distance_max_m"] <= residual[
        "maximum_model_surface_distance_m"
    ]
    assert evidence["motion_authorized"] is False
    assert evidence["authoritative_for_unseen_camera_space"] is False

    diagnostic = json.loads(FRESH_DIAGNOSTIC.read_text(encoding="utf-8"))
    checks = {item["check_id"]: item for item in diagnostic["checks"]}
    assert checks["fr3_scene_path"]["minimum_signed_distance_m"] > 0.002
    unresolved = checks["rh56_open_scene_path"]
    assert unresolved["minimum_signed_distance_m"] < 0.0
    assert unresolved["diagnostic_policy_passed"] is False
    assert unresolved["observed_pairs"] == [
        "Link111 / scene",
        "Link44 / scene",
    ]
    assert diagnostic["motion_authorized"] is False
    assert diagnostic["summary"]["all_geometric_and_authority_policies_passed"] is False
