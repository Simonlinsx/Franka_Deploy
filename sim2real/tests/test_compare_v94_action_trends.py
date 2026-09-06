from pathlib import Path

import numpy as np
import pytest

from sim2real.diagnostics.compare_v94_action_trends import compare_action_trends
from sim2real.deployment.bundle import DeployBundle


BUNDLE = Path(__file__).resolve().parents[2] / "data/test_fixtures/sim2real/deploy.zip"
INITIAL_NPZ = "alignment/reset_idle_open/initial_observations_and_student_response.npz"


def _shadow(path: Path, *, hardware_writes: bool = False) -> Path:
    initial = DeployBundle(BUNDLE).load_npz(INITIAL_NPZ)
    action = initial["policy_action13"][:4].copy()
    np.savez_compressed(
        path,
        raw_policy_action13=action,
        host_action_monotonic_s=100.0 + np.arange(4) / 60.0,
        pointcloud_frame_id=np.arange(4, dtype=np.int64),
        proposal_mode=np.asarray(["one_step_from_measured_idle"] * 4),
        hardware_writes=np.asarray(hardware_writes),
        robot_command_writes=np.asarray(False),
        write=np.zeros(4, dtype=bool),
    )
    return path


def test_identical_reset_rows_report_direction_without_authorizing_motion(tmp_path):
    report, arrays = compare_action_trends(_shadow(tmp_path / "shadow.npz"))
    direction = report["reset_local_direction"]
    assert direction["all_13_axes_match_sim_reset_dominant_sign"]
    assert direction["nearest_sim_reset_action13_linf"]["max"] == 0.0
    assert direction["arm7_cosine_to_sim_reset_median"]["median"] > 0.999
    assert report["comparison_semantics"]["time_aligned_to_closed_loop"] is False
    assert report["interpretation"]["closed_loop_action_trend_validated"] is False
    assert report["interpretation"]["semantic_action_correctness_claimed"] is False
    assert report["interpretation"]["physical_motion_authorized"] is False
    assert arrays["real_action13"].shape == (4, 13)


def test_comparison_refuses_capture_reporting_hardware_writes(tmp_path):
    with pytest.raises(ValueError, match="hardware_writes=true"):
        compare_action_trends(
            _shadow(tmp_path / "unsafe_shadow.npz", hardware_writes=True)
        )
