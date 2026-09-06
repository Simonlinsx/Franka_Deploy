from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anydex_pipeline.air_target_poses import derive_air_target_poses
from anydex_pipeline.snapshot import load_snapshot_npz


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
SNAPSHOT = (
    ROOT / "runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz"
)


def _pose(xyz=(0.0, 0.0, 0.0)) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return result


def test_air_target_contract_retreats_final_then_pregrasp_along_minus_approach():
    canonical = _pose([0.5, 0.1, 0.2])
    hand = _pose([0.6, 0.2, 0.3])
    mount = _pose([0.0, 0.0, 0.05])

    targets = derive_air_target_poses(
        canonical,
        hand,
        [2.0, 0.0, 0.0],
        mount,
        retreat_distance_m=0.08,
        pregrasp_extra_distance_m=0.01,
    )

    np.testing.assert_allclose(targets.approach_reference, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        targets.T_reference_EE_nominal[:3, 3], [0.6, 0.2, 0.25]
    )
    np.testing.assert_allclose(
        targets.T_reference_EE_final_air[:3, 3], [0.52, 0.2, 0.25]
    )
    np.testing.assert_allclose(
        targets.T_reference_EE_pregrasp[:3, 3], [0.51, 0.2, 0.25]
    )
    np.testing.assert_allclose(
        targets.T_reference_hand_final_air,
        targets.T_reference_EE_final_air @ mount,
    )


@pytest.mark.parametrize(
    "retreat,extra,message",
    [
        (0.0, 0.01, "retreat"),
        (0.08, 0.0, "pregrasp"),
        (float("nan"), 0.01, "retreat"),
    ],
)
def test_air_target_contract_rejects_nonpositive_or_nonfinite_distances(
    retreat, extra, message
):
    with pytest.raises(ValueError, match=message):
        derive_air_target_poses(
            _pose(),
            _pose(),
            [1.0, 0.0, 0.0],
            _pose(),
            retreat_distance_m=retreat,
            pregrasp_extra_distance_m=extra,
        )


@pytest.mark.skipif(
    not SNAPSHOT.is_file(), reason="candidate-51 workstation snapshot is unavailable"
)
def test_candidate51_air_targets_match_the_audit_and_executor_contract():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    snapshot = load_snapshot_npz(SNAPSHOT)
    index = 51

    targets = derive_air_target_poses(
        snapshot.grasps.canonical_poses[index],
        snapshot.grasps.hand_poses[index],
        snapshot.grasps.approach_axis_local,
        np.asarray(config["tool"]["T_EE_hand"], dtype=np.float64),
        retreat_distance_m=float(config["grasp"]["air_retreat_distance_m"]),
        pregrasp_extra_distance_m=float(
            config["grasp"]["air_pregrasp_distance_m"]
        ),
    )

    np.testing.assert_allclose(
        targets.T_reference_EE_nominal[:3, 3],
        [0.59703523, 0.11803622, 0.31821600],
        atol=5.0e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        targets.T_reference_EE_final_air[:3, 3],
        [0.66101691, 0.15337982, 0.35073023],
        atol=5.0e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        targets.T_reference_EE_pregrasp[:3, 3],
        [0.66901461, 0.15779777, 0.35479451],
        atol=5.0e-8,
        rtol=0.0,
    )
