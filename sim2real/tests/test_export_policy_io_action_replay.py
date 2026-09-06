from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from sim2real.action_replay import load_replay_actions
from sim2real.diagnostics.export_policy_io_action_replay import (
    export_policy_io_action_replay,
)


def _write_source(path: Path, *, success: bool = True, tick_delta: float = 0.01):
    count = 3
    actions = np.zeros((count, 13), dtype=np.float32)
    actions[:, 0] = [0.0, 0.5, 1.0]
    arm_position = np.zeros((count, 7), dtype=np.float32)
    arm_target = np.zeros((count, 7), dtype=np.float32)
    arm_target[:, 0] = np.arange(count, dtype=np.float32) * np.float32(tick_delta)
    rh56 = np.full((count, 6), 1000, dtype=np.int32)
    np.savez(
        path,
        output_action_sent_to_env=actions,
        sim_arm_joint_target_rad=arm_target,
        sim_arm_joint_position_rad=arm_position,
        rh56_angle_set_register_order=rh56,
        time_s=np.arange(count, dtype=np.float64) * 0.05,
        sim_success=np.asarray([False, False, success]),
        sim_stable_hold=np.asarray([False, True, True]),
    )
    return actions, arm_target, rh56


def test_successful_sim_policy_io_exports_exact_target_replay(tmp_path: Path):
    source = tmp_path / "policy_io.npz"
    output = tmp_path / "replay.zip"
    actions, arm_target, rh56 = _write_source(source)

    result = export_policy_io_action_replay(source, output)
    replay = load_replay_actions(output, expected_policy_rate_hz=20)

    assert result["frames"] == 3
    assert result["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    np.testing.assert_array_equal(replay.actions13, actions)
    np.testing.assert_array_equal(replay.recorded_franka_target_q_rad, arm_target)
    np.testing.assert_array_equal(
        replay.recorded_rh56_angle_set_register_order, rh56
    )
    with zipfile.ZipFile(output) as bundle:
        metadata = json.loads(bundle.read("replay/metadata.json"))
    assert metadata["source_final_success"] is True
    assert metadata["source_policy_io_sha256"] == result["source_sha256"]


def test_export_rejects_trace_without_final_success(tmp_path: Path):
    source = tmp_path / "policy_io.npz"
    _write_source(source, success=False)
    with pytest.raises(ValueError, match="successful stable-hold"):
        export_policy_io_action_replay(source, tmp_path / "replay.zip")


def test_export_rejects_target_above_supervised_tick_bound(tmp_path: Path):
    source = tmp_path / "policy_io.npz"
    _write_source(source, tick_delta=0.021)
    with pytest.raises(ValueError, match="0.020 rad/tick"):
        export_policy_io_action_replay(source, tmp_path / "replay.zip")
