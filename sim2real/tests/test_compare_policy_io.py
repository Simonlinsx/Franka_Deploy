import json
from pathlib import Path

import numpy as np
import pytest

from sim2real.diagnostics.compare_policy_io import compare_policy_io, main


SHA_A = "a" * 64
SHA_B = "b" * 64


def _policy_io(
    path: Path,
    *,
    steps: int = 3,
    checkpoint_sha256: str = SHA_A,
    point_permutation: bool = False,
    point_bias: float = 0.0,
) -> Path:
    history, points, point_dim, proprio_dim = 2, 5, 3, 6
    metric = (
        np.arange(steps * history * points * point_dim, dtype=np.float32).reshape(
            steps, history, points, point_dim
        )
        / 100.0
        + np.float32(point_bias)
    )
    if point_permutation:
        metric = metric[:, :, ::-1, :].copy()
    valid = np.ones((steps, history, points), dtype=np.float32)
    valid[:, :, -1] = 0.0
    point_mean = np.asarray([[[[0.1, -0.2, 0.3]]]], dtype=np.float32)
    point_std = np.asarray([[[[0.5, 1.0, 2.0]]]], dtype=np.float32)
    point_normalized = ((metric - point_mean) / point_std).astype(np.float32)

    proprio = (
        np.arange(steps * history * proprio_dim, dtype=np.float32).reshape(
            steps, history, proprio_dim
        )
        / 20.0
    )
    proprio_mean = np.linspace(-0.2, 0.2, proprio_dim, dtype=np.float32).reshape(
        1, 1, proprio_dim
    )
    proprio_std = np.linspace(0.5, 1.0, proprio_dim, dtype=np.float32).reshape(
        1, 1, proprio_dim
    )
    proprio_normalized = ((proprio - proprio_mean) / proprio_std).astype(np.float32)
    previous = np.linspace(-0.5, 0.5, steps * 13, dtype=np.float32).reshape(steps, 13)
    action = np.linspace(-1.0, 1.0, steps * 13, dtype=np.float32).reshape(steps, 13)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        input_pointcloud_history_metric=metric,
        input_pointcloud_history_normalized=point_normalized,
        input_pointcloud_valid_history=valid,
        input_proprio_history_raw=proprio,
        input_proprio_history_normalized=proprio_normalized,
        input_previous_executed_action=previous,
        output_model_action=action,
        output_action_sent_to_env=action * np.float32(0.9),
        constant__normalization_pointcloud_mean=point_mean,
        constant__normalization_pointcloud_std=point_std,
        constant__normalization_proprio_mean=proprio_mean,
        constant__normalization_proprio_std=proprio_std,
        constant__checkpoint_sha256=np.asarray(checkpoint_sha256),
    )
    return path


def test_identical_policy_io_reports_normalization_and_offline_semantics(tmp_path):
    sim = _policy_io(tmp_path / "sim" / "policy_io.npz")
    real = _policy_io(
        tmp_path / "real.npz",
        point_permutation=True,
    )

    report = compare_policy_io(real, sim)

    assert report["checkpoint"] == {
        "real_sha256": SHA_A,
        "simulation_sha256_values": [SHA_A],
        "match": True,
        "status": "match",
    }
    assert report["schema"]["all_core_shapes_compatible_ignoring_step_count"]
    assert report["normalization"]["real_reconstruction"]["pointcloud"][
        "within_float32_tolerance"
    ]
    assert report["normalization"]["real_reconstruction"]["proprio"][
        "within_float32_tolerance"
    ]
    constant = report["normalization"]["constants_real_vs_sim"][
        "constant__normalization_pointcloud_mean"
    ]
    assert constant["shape_compatible"] is True
    assert constant["match"] is True
    assert "status" not in constant
    semantics = report["comparison_semantics"]
    assert semantics["distributional_only"] is True
    assert semantics["exact_tick_alignment"] is False
    assert semantics["physical_control_performed"] is False
    assert semantics["physical_motion_authorized"] is False
    descriptor = report["pointcloud_permutation_invariant"]
    assert descriptor["point_order_assumed_aligned"] is False
    assert descriptor["pointwise_rmse_computed"] is False
    assert descriptor["descriptors"]["centroid_xyz_m"]["real"]["count"] > 0
    assert descriptor["descriptors"]["centroid_xyz_m"][
        "last_axis_distribution"
    ]["last_axis_dimension"] == 3
    for scalar_name in (
        "valid_point_count",
        "valid_fraction",
        "radius_median_m",
        "radius_p95_m",
    ):
        assert (
            "last_axis_distribution"
            not in descriptor["descriptors"][scalar_name]
        )


def test_sim_directory_is_recursive_and_aggregates_traces(tmp_path):
    real = _policy_io(tmp_path / "real.npz", steps=2, checkpoint_sha256=SHA_A)
    root = tmp_path / "successful_cases"
    _policy_io(root / "case_0" / "policy_io.npz", steps=3, checkpoint_sha256=SHA_B)
    _policy_io(root / "nested" / "case_1" / "policy_io.npz", steps=4, checkpoint_sha256=SHA_B)

    report = compare_policy_io(real, root)

    assert report["sources"]["simulation_archive_count"] == 2
    assert report["sources"]["simulation_total_steps"] == 7
    assert report["checkpoint"]["status"] == "mismatch"
    assert report["checkpoint"]["match"] is False
    action = report["distributional_fields"]["model_action13"]
    assert action["available"] is True
    assert action["sim"]["count"] == 7 * 13
    assert action["last_axis_distribution"]["last_axis_dimension"] == 13


def test_core_schema_and_positive_normalizer_are_enforced(tmp_path):
    real = _policy_io(tmp_path / "real.npz")
    sim = _policy_io(tmp_path / "sim.npz")
    with np.load(real, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    values.pop("input_proprio_history_normalized")
    np.savez_compressed(real, **values)
    with pytest.raises(ValueError, match="missing core policy-I/O fields"):
        compare_policy_io(real, sim)

    real = _policy_io(tmp_path / "real_bad_std.npz")
    with np.load(real, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    values["constant__normalization_pointcloud_std"][..., 0] = 0.0
    np.savez_compressed(real, **values)
    with pytest.raises(ValueError, match="strictly positive"):
        compare_policy_io(real, sim)


def test_cli_atomically_writes_json_and_rejects_pickle_archive(tmp_path, capsys):
    real = _policy_io(tmp_path / "real.npz")
    sim = _policy_io(tmp_path / "sim.npz")
    output = tmp_path / "reports" / "comparison.json"
    assert main(["--real", str(real), "--sim", str(sim), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"] == "sim2real_policy_io_distributional_comparison_v1"
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))
    assert "distributional_only" in capsys.readouterr().out

    bad = tmp_path / "bad.npz"
    np.savez_compressed(
        bad,
        input_pointcloud_history_metric=np.asarray([{"unsafe": True}], dtype=object),
    )
    assert main(["--real", str(bad), "--sim", str(sim)]) == 2
    assert "invalid no-pickle" in capsys.readouterr().err
