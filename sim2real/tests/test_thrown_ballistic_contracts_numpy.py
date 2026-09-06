from pathlib import Path

import numpy as np

from sim2real.tasks.ballistics import (
    PACKAGED_SOURCE_SHA256,
    deployable_thrown_v35_future_contract,
    deployable_visual_ballistic_future_contract,
)


ALIGNMENT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data/checkpoints/thrown"
    / "thrown_v61_sixexpert_visualflight_perceptiondr025_cmp31p9_demo_candidate_20260815"
    / "runtime_alignment/runtime_alignment"
)


def test_v61_numpy_contracts_match_frozen_golden_vectors():
    payload = np.load(ALIGNMENT_ROOT / "golden_vectors.npz", allow_pickle=False)
    actual_v1 = deployable_thrown_v35_future_contract(
        payload["predicted_compact_privileged"], payload["proprio_seq"][:, -1]
    )
    actual_v2 = deployable_visual_ballistic_future_contract(
        payload["pointcloud_seq"], payload["valid_seq"], payload["proprio_seq"]
    )

    # v1 is byte-identical.  NumPy and PyTorch use different SIMD reduction
    # orders for the v2 128-point centroid, giving at most ~2e-6 rounding in
    # the 17-D observation while preserving every discrete gate/selection.
    np.testing.assert_array_equal(actual_v1, payload["expected_v1"])
    np.testing.assert_allclose(
        actual_v2, payload["expected_v2"], rtol=0.0, atol=2.1e-6
    )


def test_v61_numpy_contract_is_bound_to_packaged_reference_source():
    assert PACKAGED_SOURCE_SHA256 == (
        "547c570ba93593541e6e1db466463e759b155c683211a007c2de69bc7ed9b956"
    )
