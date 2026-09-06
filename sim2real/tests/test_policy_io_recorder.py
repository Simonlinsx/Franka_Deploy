from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim2real.policy.io_recorder import (
    POLICY_IO_SCHEMA_VERSION,
    PolicyIORecorder,
    default_policy_io_path,
)


HISTORY = 4
POINTS = 128
POINT_DIM = 3
PROPRIO_DIM = 67


def _recorder(path: Path, *, maximum_records: int = 3) -> PolicyIORecorder:
    return PolicyIORecorder(
        path,
        maximum_records=maximum_records,
        pointcloud_mean=np.asarray([[[[1.0, 2.0, 3.0]]]], dtype=np.float32),
        pointcloud_std=np.asarray([[[[2.0, 4.0, 5.0]]]], dtype=np.float32),
        proprio_mean=np.arange(PROPRIO_DIM, dtype=np.float32).reshape(1, 1, -1),
        proprio_std=np.full((1, 1, PROPRIO_DIM), 2.0, dtype=np.float32),
        q_hand_close_rad=np.asarray(
            [1.25, 0.599, 0.95, 0.95, 1.05, 1.10], dtype=np.float32
        ),
        metadata={
            "run_id": "unit-test",
            "checkpoint_sha256": "a" * 64,
            "history_length": HISTORY,
            "num_object_points": POINTS,
            "path_value": Path("checkpoint.pt"),
        },
    )


def _tick_kwargs(index: int = 0) -> dict[str, object]:
    points = (
        np.arange(HISTORY * POINTS * POINT_DIM, dtype=np.float32)
        .reshape(HISTORY, POINTS, POINT_DIM)
        / np.float32(100.0)
        + np.float32(index)
    )
    valid = np.ones((HISTORY, POINTS), dtype=np.float32)
    proprio = (
        np.arange(HISTORY * PROPRIO_DIM, dtype=np.float32)
        .reshape(HISTORY, PROPRIO_DIM)
        / np.float32(10.0)
    )
    previous_from_input = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
    proprio[-1, 54:67] = previous_from_input
    return {
        "logical_policy_step": index,
        "hardware_sequence": -1 if index == 0 else index,
        "startup_non_actuated": index == 0,
        "pointcloud_history_metric": points,
        "pointcloud_valid_history": valid,
        "proprio_history_raw": proprio,
        "previous_policy_action13": np.full(13, index / 10.0, dtype=np.float32),
        "previous_ledger_action13": np.full(13, -0.25, dtype=np.float32),
        "raw_model_action13": np.linspace(-0.9, 0.9, 13, dtype=np.float32),
        "sent_action13": np.linspace(-0.8, 0.8, 13, dtype=np.float32),
        "camera_frame_id": 20 + index,
        "pointcloud_source_frame_id": 10 + index,
        "pointcloud_status": "fresh",
        "source_valid_points": 961,
        "observation_realtime_s": 1000.0 + index * 0.05,
        "pointcloud_captured_realtime_s": 999.98 + index * 0.05,
        "franka_state_captured_monotonic_s": 50.0 + index * 0.05,
        "produced_monotonic_s": 50.02 + index * 0.05,
        "measured_franka_q_rad": np.arange(7, dtype=np.float32) / 10.0,
        "shaper_q_d_rad": None if index == 0 else np.full(7, 0.2, dtype=np.float32),
        "franka_target_q_rad": None
        if index == 0
        else np.full(7, 0.3, dtype=np.float32),
        "rh56_target_register_order": None
        if index == 0
        else np.asarray([900, 800, 700, 600, 500, 400], dtype=np.int64),
        "hardware_command_valid": index != 0,
        "hold_arm_target": False,
    }


def test_records_exact_copies_and_defers_all_io_until_save(tmp_path: Path) -> None:
    output = tmp_path / "trace.npz"
    recorder = _recorder(output)
    first = _tick_kwargs(0)
    second = _tick_kwargs(1)
    expected_points = np.asarray(first["pointcloud_history_metric"]).copy()
    expected_proprio = np.asarray(first["proprio_history_raw"]).copy()

    assert recorder.record_tick(**first)
    assert recorder.record_tick(**second)
    assert not output.exists()

    # The caller may immediately recycle its candidate buffers.  Recording
    # must retain the exact accepted values, not references to those buffers.
    np.asarray(first["pointcloud_history_metric"])[:] = -999.0
    np.asarray(first["proprio_history_raw"])[:] = -999.0

    stats = recorder.save()
    assert stats["saved"] is True
    assert stats["recorded_ticks"] == 2
    assert stats["startup_non_actuated_ticks"] == 1
    assert stats["hardware_command_ticks"] == 1
    assert output.is_file()

    with np.load(output, allow_pickle=False) as archive:
        assert np.array_equal(
            archive["input_pointcloud_history_metric"][0], expected_points
        )
        assert np.array_equal(
            archive["input_pointcloud_history_pre_temporal_alignment"],
            archive["input_pointcloud_history_metric"],
        )
        assert np.array_equal(
            archive["input_proprio_history_raw"][0], expected_proprio
        )
        expected_normalized_points = (
            expected_points - np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        ) / np.asarray([2.0, 4.0, 5.0], dtype=np.float32)
        assert np.array_equal(
            archive["input_pointcloud_history_normalized"][0],
            expected_normalized_points,
        )
        expected_normalized_proprio = (
            expected_proprio - np.arange(PROPRIO_DIM, dtype=np.float32)
        ) / np.float32(2.0)
        assert np.array_equal(
            archive["input_proprio_history_normalized"][0],
            expected_normalized_proprio,
        )
        assert np.array_equal(
            archive["input_previous_executed_action"][0],
            expected_proprio[-1, 54:67],
        )
        assert np.array_equal(
            archive["output_policy_action_after_sample_bias_clamp"],
            archive["output_model_action"],
        )
        assert np.array_equal(
            archive["rh56_angle_set_register_order"][0], np.full(6, -1)
        )
        assert np.array_equal(
            archive["rh56_angle_set_register_order"][1],
            np.asarray([900, 800, 700, 600, 500, 400]),
        )
        assert np.allclose(
            archive["rh56_close_fraction_policy_order"][1],
            np.asarray([0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32),
        )
        assert archive["real_pointcloud_status"].dtype.kind == "U"
        assert archive["constant__metadata_json"].dtype.kind == "U"
        assert archive["constant__checkpoint_sha256"].item() == "a" * 64
        assert archive["constant__policy_io_schema_version"].item() == (
            POLICY_IO_SCHEMA_VERSION
        )
        for key in archive.files:
            assert archive[key].dtype != object, key


def test_record_tick_is_fail_soft_and_latches_first_error(tmp_path: Path) -> None:
    output = tmp_path / "failed.npz"
    recorder = _recorder(output)
    bad = _tick_kwargs()
    bad["pointcloud_history_metric"] = np.zeros((3, POINTS, 3), dtype=np.float32)

    assert recorder.record_tick(**bad) is False
    assert "history length" in (recorder.record_error or "")
    assert recorder.record_tick(**_tick_kwargs(1)) is False
    assert not output.exists()

    stats = recorder.save()
    assert stats["recorded_ticks"] == 0
    assert stats["dropped_records"] == 2
    with np.load(output, allow_pickle=False) as archive:
        assert archive["input_pointcloud_history_metric"].shape == (
            0,
            HISTORY,
            POINTS,
            POINT_DIM,
        )
        assert archive["input_proprio_history_raw"].shape == (
            0,
            HISTORY,
            PROPRIO_DIM,
        )
        assert archive["output_action_sent_to_env"].shape == (0, 13)


def test_capacity_is_bounded_and_save_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "bounded.npz"
    recorder = _recorder(output, maximum_records=1)
    assert recorder.record_tick(**_tick_kwargs(0))
    assert recorder.record_tick(**_tick_kwargs(1)) is False
    assert "capacity" in (recorder.record_error or "")
    first_stats = recorder.save()
    assert first_stats["recorded_ticks"] == 1
    assert recorder.save() == first_stats  # an already completed save is idempotent

    with pytest.raises(FileExistsError, match="already exists"):
        _recorder(output)


def test_zero_tick_archive_and_default_path_contract(tmp_path: Path) -> None:
    output = tmp_path / "empty.npz"
    recorder = _recorder(output)
    recorder.save()
    with np.load(output, allow_pickle=False) as archive:
        assert archive["policy_step"].shape == (0,)
        assert archive["real_pointcloud_status"].shape == (0,)
        assert archive["real_pointcloud_status"].dtype.kind == "U"

    expected = (
        Path(__file__).resolve().parents[2]
        / "dexgrasp"
        / "runs"
        / "run-123_policy_io.npz"
    )
    assert default_policy_io_path("run-123") == expected
    with pytest.raises(ValueError):
        default_policy_io_path("../escape")
