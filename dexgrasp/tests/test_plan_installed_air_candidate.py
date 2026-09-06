from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    ROOT
    / "runs"
    / "d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz"
)
CONFIG = ROOT / "configs" / "fr3_rh56_v7_commissioning.json"
URDF = Path("/home/qiaoguanren/code/libfranka/test/fr3.urdf")


def _load_app():
    name = "test_plan_installed_air_candidate_app"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "apps" / "plan_installed_air_candidate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


APP = _load_app()


def test_profile_default_is_the_provenance_bound_cable_friendly_pose():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    np.testing.assert_allclose(
        config["franka"]["default_q_rad"],
        [-0.1118436, -0.1207545, 0.0739457, -1.7431009, 0.046354, 1.6809169, 0.8117281],
        atol=0.0,
        rtol=0.0,
    )
    assert config["franka"]["default_q_provenance"]["motion_authorized"] is False


def test_joint_path_obeys_linf_step_and_preserves_waypoints():
    start = np.zeros(7)
    middle = np.asarray([0.0, 0.011, 0.0, 0.0, 0.0, 0.0, 0.0])
    end = np.asarray([0.0, 0.011, -0.006, 0.0, 0.0, 0.0, 0.0])

    path, intervals = APP.build_joint_path((start, middle, end), 0.005)

    assert intervals == (3, 2)
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[3], middle)
    np.testing.assert_allclose(path[-1], end)
    assert float(np.max(np.abs(np.diff(path, axis=0)))) <= 0.005


@pytest.mark.skipif(
    not SNAPSHOT.is_file() or not CONFIG.is_file() or not URDF.is_file(),
    reason="candidate-51 integration inputs are not present on this workstation",
)
def test_candidate51_cli_is_reproducible_and_fail_closed(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    command = [
        str(ROOT / "scripts" / "plan_installed_air_candidate.sh"),
        "--snapshot",
        str(SNAPSHOT),
        "--config",
        str(CONFIG),
        "--candidate-index",
        "51",
        "--q7-rad",
        "1.3525",
        "--start-q-rad",
        "-0.1118436",
        "-0.1207545",
        "0.0739457",
        "-1.7431009",
        "0.0463540",
        "1.6809169",
        "0.8117281",
    ]
    subprocess.run(command + ["--output", str(first)], check=True)
    subprocess.run(command + ["--output", str(second)], check=True)

    assert first.read_bytes() == second.read_bytes()
    artifact = json.loads(first.read_text(encoding="utf-8"))
    assert artifact["motion_authorized"] is False
    assert artifact["candidate"]["index"] == 51
    assert artifact["candidate"]["source_index"] == 54
    assert artifact["candidate"]["hand_targets"] == [0, 358, 799, 911, 922, 646]
    np.testing.assert_allclose(
        artifact["joint_plan"]["q_pregrasp_rad"],
        [
            -0.006605840521754,
            0.568400418236178,
            0.409275843740853,
            -1.504338651687827,
            -0.925398889002316,
            1.762354174376624,
            1.3525,
        ],
        atol=5e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        artifact["joint_plan"]["q_final_air_rad"],
        [
            0.033813125565760,
            0.531050228219339,
            0.340096683888818,
            -1.557399996752551,
            -0.890714434115401,
            1.804651188696697,
            1.3525,
        ],
        atol=5e-8,
        rtol=0.0,
    )
    assert artifact["fk_residual"]["passed"] is True
    assert artifact["joint_plan"]["waypoint_order"] == [
        "start",
        "default",
        "pregrasp",
        "final_air",
    ]
    assert (
        artifact["joint_plan"]["sampling_algorithm"]
        == "numpy_linspace_float64_v1"
    )
    np.testing.assert_allclose(
        artifact["joint_plan"]["q_start_rad"],
        artifact["joint_plan"]["q_default_rad"],
        atol=0.0,
        rtol=0.0,
    )
    assert artifact["joint_plan"]["segment_interval_counts"] == [1, 195, 14]
    assert artifact["joint_plan"]["joint_limits_with_margin_passed"] is True
    assert artifact["joint_plan"]["actual_max_joint_step_rad"] <= 0.005
    assert artifact["joint_plan"]["q7_start_rad"] == pytest.approx(0.8117281)
    assert artifact["joint_plan"]["q7_default_rad"] == pytest.approx(0.8117281)
    assert artifact["joint_plan"]["q7_start_to_default_rotation_rad"] == 0.0
    assert artifact["joint_plan"]["q7_default_to_target_rotation_rad"] == pytest.approx(
        0.5407719
    )
    assert artifact["joint_plan"]["q7_absolute_rotation_rad"] == pytest.approx(
        0.5407719
    )
    assert artifact["cable_constraint"]["q7_zero_assumed"] is False
    assert artifact["cable_constraint"]["cable_geometry_or_slack_verified"] is False
    assert artifact["collision_check"]["bare_fr3_self_collision_free"] is True
    assert artifact["collision_check"]["scene_collision_checked"] is False
    assert artifact["collision_check"]["adapter_collision_checked"] is False
    assert artifact["collision_check"]["rh56_collision_checked"] is False

    original = first.read_bytes()
    duplicate = subprocess.run(
        command + ["--output", str(first)], capture_output=True, text=True
    )
    assert duplicate.returncode != 0
    assert "refusing to overwrite" in duplicate.stderr
    assert first.read_bytes() == original
