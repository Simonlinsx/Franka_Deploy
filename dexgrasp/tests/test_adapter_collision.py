from __future__ import annotations

import numpy as np
import pytest

from anydex_pipeline.adapter_collision import audit_adapter_sweep_against_points


def _pose(x=0.0, yaw=0.0):
    transform = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    transform[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    transform[0, 3] = x
    return transform


def test_adapter_envelope_detects_point_inside_disk_or_spigot_height():
    points = np.asarray([[0.0, 0.0, 0.005], [0.0, 0.0, 0.017], [0.1, 0.0, 0.0]])
    report = audit_adapter_sweep_against_points(
        points,
        T_reference_EE_start=np.eye(4),
        T_reference_EE_end=np.eye(4),
        T_EE_adapter=np.eye(4),
        margin_m=0.0,
        trajectory_samples=2,
    )
    assert not report.collision_free
    assert report.colliding_point_indices == (0, 1)
    assert not report.authoritative


def test_adapter_sweep_checks_middle_not_only_endpoints():
    # Adapter moves from x=-0.10 to +0.10; this point is only inside near alpha=.5.
    points = np.asarray([[0.0, 0.0, 0.005]])
    report = audit_adapter_sweep_against_points(
        points,
        T_reference_EE_start=_pose(-0.10),
        T_reference_EE_end=_pose(0.10, yaw=np.pi / 2),
        T_EE_adapter=np.eye(4),
        margin_m=0.0,
        trajectory_samples=21,
    )
    assert report.colliding_point_indices == (0,)
    assert 10 in report.colliding_sample_indices


def test_clear_sparse_points_do_not_become_authoritative_collision_proof():
    report = audit_adapter_sweep_against_points(
        [[0.2, 0.2, 0.2]],
        T_reference_EE_start=np.eye(4),
        T_reference_EE_end=_pose(0.01),
        T_EE_adapter=np.eye(4),
    )
    assert report.collision_free
    assert report.authoritative is False


@pytest.mark.parametrize("samples", [0, 1, True])
def test_invalid_sample_counts_are_rejected(samples):
    with pytest.raises(ValueError, match="integer >= 2"):
        audit_adapter_sweep_against_points(
            np.empty((0, 3)),
            T_reference_EE_start=np.eye(4),
            T_reference_EE_end=np.eye(4),
            T_EE_adapter=np.eye(4),
            trajectory_samples=samples,
        )

