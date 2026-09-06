from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from sim2real.action_replay import (
    CANONICAL_ACTION_ORDER,
    ReplayActionSequence,
    TransactionalReplayActionPolicy,
    load_replay_actions_payload,
    summarize_replay_actions,
)
from sim2real.replay_actions import (
    build_parser,
    build_replay_request,
    build_replay_summary,
)
from sim2real.deployment.runner import SupervisedV94RunError


def _json_payload(actions, **metadata) -> bytes:
    return json.dumps({"actions": actions, **metadata}).encode("utf-8")


def test_json_replay_contract_and_stateless_sequence_lookup():
    actions = np.linspace(-1.0, 1.0, 26, dtype=np.float32).reshape(2, 13)
    sequence = load_replay_actions_payload(
        _json_payload(
            actions.tolist(),
            policy_rate_hz=60,
            action_order=list(CANONICAL_ACTION_ORDER),
        ),
        suffix=".json",
        expected_policy_rate_hz=60,
        selected_steps=2,
    )
    policy = TransactionalReplayActionPolicy(
        sequence, point_feature_dim=3, selected_steps=2
    )
    inputs = (
        np.zeros((4, 128, 3), dtype=np.float32),
        np.ones((4, 128), dtype=np.float32),
        np.zeros((4, 67), dtype=np.float32),
    )
    # Proposal retries for the same ledger sequence are pure and exact.
    np.testing.assert_array_equal(
        policy.action_for_sequence(1, *inputs).action13,
        policy.action_for_sequence(1, *inputs).action13,
    )
    np.testing.assert_array_equal(
        policy.action_for_sequence(2, *inputs).action13, actions[1]
    )
    with pytest.raises(ValueError, match="outside"):
        policy.action_for_sequence(3, *inputs)


def test_npy_and_npz_replay_are_pickle_free_and_equivalent():
    actions = np.zeros((3, 13), dtype=np.float32)
    npy = io.BytesIO()
    np.save(npy, actions, allow_pickle=False)
    loaded_npy = load_replay_actions_payload(
        npy.getvalue(), suffix=".npy", expected_policy_rate_hz=20
    )
    npz = io.BytesIO()
    np.savez(
        npz,
        actions=actions,
        policy_rate_hz=np.asarray(20.0),
        action_order=np.asarray(CANONICAL_ACTION_ORDER),
    )
    loaded_npz = load_replay_actions_payload(
        npz.getvalue(), suffix=".npz", expected_policy_rate_hz=20
    )
    np.testing.assert_array_equal(loaded_npy.actions13, loaded_npz.actions13)


@pytest.mark.parametrize(
    ("actions", "match"),
    [
        ([[0.0] * 12], "shape"),
        ([[0.0] * 12 + [1.01]], "without clipping"),
        ([[0.0] * 12 + [float("nan")]], "NaN"),
    ],
)
def test_malformed_or_out_of_range_actions_are_rejected(actions, match):
    with pytest.raises(ValueError, match=match):
        load_replay_actions_payload(
            _json_payload(actions),
            suffix=".json",
            expected_policy_rate_hz=60,
        )


def test_declared_replay_rate_must_match_cli_rate():
    with pytest.raises(ValueError, match="disagrees"):
        load_replay_actions_payload(
            _json_payload([[0.0] * 13], policy_rate_hz=20),
            suffix=".json",
            expected_policy_rate_hz=60,
        )


def test_replay_cli_defaults_to_all_actions_and_reports_runtime_parity(tmp_path: Path):
    action_path = tmp_path / "success_actions.json"
    action_path.write_text(
        json.dumps({"policy_rate_hz": 60, "actions": [[0.0] * 13] * 3})
    )
    args = build_parser().parse_args(["--actions", str(action_path)])
    request = build_replay_request(args)
    assert request.steps == 3
    assert request.replay_action_count == 3
    summary = build_replay_summary(request)
    assert summary["action_source"]["kind"] == "validated_simulation_action_replay"
    assert summary["action_source"]["selected_actions"] == 3
    assert summary["action_source"]["advance_semantics"] == (
        "advance_only_after_exact_dual_device_ack"
    )
    assert summary["action_source"]["execution_contract"] == (
        "exact_recorded_actuator_targets_when_present_otherwise_"
        "normalized_action13_through_current_deployment_mapper"
    )


def test_replay_cli_allows_prefix_but_not_more_rows(tmp_path: Path):
    action_path = tmp_path / "actions.json"
    action_path.write_text(json.dumps([[0.0] * 13] * 4))
    request = build_replay_request(
        build_parser().parse_args(["--actions", str(action_path), "--steps", "2"])
    )
    assert request.steps == 2
    with pytest.raises(SupervisedV94RunError, match="1..4"):
        build_replay_request(
            build_parser().parse_args(["--actions", str(action_path), "--steps", "5"])
        )


def test_20hz_replay_does_not_require_an_unrelated_policy_checkpoint(tmp_path: Path):
    from sim2real.runtime.supervised_v94_runtime import prepare_supervised_v94_artifacts

    action_path = tmp_path / "actions_20hz.json"
    action_path.write_text(
        json.dumps({"policy_rate_hz": 20, "actions": [[0.0] * 13] * 2})
    )
    request = build_replay_request(
        build_parser().parse_args(
            [
                "--actions",
                str(action_path),
                "--policy-rate-hz",
                "20",
                "--run-id",
                "replay-20hz-offline-artifact-test",
            ]
        )
    )
    prepared = prepare_supervised_v94_artifacts(request)
    assert prepared.checkpoint_source == "bundle_primary"
    assert prepared.replay_action_sha256 == request.replay_actions_sha256
    assert prepared.replay_action_count == 2


def _v205_zip_payload(actions: np.ndarray) -> bytes:
    count = int(actions.shape[0])
    npz = io.BytesIO()
    np.savez(
        npz,
        time_s=np.arange(count, dtype=np.float32) * np.float32(0.05),
        policy_action=actions,
        franka_joint_target_rad=np.zeros((count, 7), dtype=np.float32),
        inspire_angle_set_register_order=np.full(
            (count, 6), 1000, dtype=np.int64
        ),
    )
    metadata = {
        "control_hz": 20.0,
        "control_dt_s": 0.05,
        "frames": count,
        "action_contract": (
            "7D Franka incremental joint target + "
            "6D Inspire absolute motor target"
        ),
        "inspire_policy_order": list(CANONICAL_ACTION_ORDER[7:]),
        "inspire_register_order": [
            "little",
            "ring",
            "middle",
            "index",
            "thumb_bending",
            "thumb_rotation",
        ],
        "recommended_replay_fields": {
            "franka": "franka_joint_target_rad",
            "inspire": "inspire_angle_set_register_order",
        },
    }
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("replay/metadata.json", json.dumps(metadata))
        bundle.writestr("replay/data.npz", npz.getvalue())
    return payload.getvalue()


def test_v205_zip_is_loaded_without_extraction_and_targets_are_audit_only():
    actions = np.zeros((3, 13), dtype=np.float32)
    actions[1, 0] = 1.0
    payload = _v205_zip_payload(actions)
    sequence = load_replay_actions_payload(
        payload,
        suffix=".zip",
        expected_policy_rate_hz=20,
        selected_steps=3,
    )
    assert sequence.source_format == "v205_replay_zip"
    assert sequence.declared_policy_rate_hz == pytest.approx(20.0)
    np.testing.assert_array_equal(sequence.actions13, actions)
    assert sequence.recorded_franka_target_q_rad.shape == (3, 7)
    assert sequence.recorded_rh56_angle_set_register_order.shape == (3, 6)
    policy = TransactionalReplayActionPolicy(
        sequence, point_feature_dim=3, selected_steps=3
    )
    output = policy.action_for_sequence(
        2,
        np.zeros((4, 128, 3), dtype=np.float32),
        np.ones((4, 128), dtype=np.float32),
        np.zeros((4, 67), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        output.exact_franka_target_q_rad,
        sequence.recorded_franka_target_q_rad[1],
    )
    np.testing.assert_array_equal(
        output.exact_rh56_angle_set_register_order,
        sequence.recorded_rh56_angle_set_register_order[1],
    )
    assert output.bypass_rh56_host_slew is True
    preview = summarize_replay_actions(sequence, selected_steps=3)
    assert preview["recorded_target_audit"]["franka_current_mapper_equivalent"] is (
        False
    )


def test_v205_zip_rate_must_match_cli():
    with pytest.raises(ValueError, match="disagrees"):
        load_replay_actions_payload(
            _v205_zip_payload(np.zeros((2, 13), dtype=np.float32)),
            suffix=".zip",
            expected_policy_rate_hz=60,
        )


def test_arrival_gated_exact_replay_repeats_until_measured_franka_arrives():
    actions = np.zeros((2, 13), dtype=np.float32)
    targets = np.zeros((2, 7), dtype=np.float32)
    targets[0, 0] = np.float32(0.10)
    targets[1, 0] = np.float32(0.20)
    sequence = ReplayActionSequence(
        actions13=actions,
        sha256="0" * 64,
        source_format="test",
        declared_policy_rate_hz=20.0,
        recorded_franka_target_q_rad=targets,
        recorded_rh56_angle_set_register_order=np.full(
            (2, 6), 1000, dtype=np.int32
        ),
    )
    policy = TransactionalReplayActionPolicy(
        sequence,
        point_feature_dim=3,
        selected_steps=2,
        arrival_gated=True,
        arrival_tolerance_rad=0.015,
        q_home_rad=np.zeros(7, dtype=np.float32),
    )
    points = np.zeros((4, 128, 3), dtype=np.float32)
    valid = np.ones((4, 128), dtype=np.float32)
    proprio = np.zeros((4, 67), dtype=np.float32)

    first = policy.action_for_sequence(1, points, valid, proprio)
    assert first.replay_frame_index == 0
    policy.commit_replay_proposal(1)

    repeated = policy.action_for_sequence(2, points, valid, proprio)
    assert repeated.replay_frame_index == 0
    assert repeated.replay_arrival_error_rad == pytest.approx(0.10)
    policy.commit_replay_proposal(2)

    proprio[-1, 0] = np.float32(0.10)
    advanced = policy.action_for_sequence(3, points, valid, proprio)
    assert advanced.replay_frame_index == 1
    policy.commit_replay_proposal(3)

    proprio[-1, 0] = np.float32(0.20)
    finished = policy.action_for_sequence(4, points, valid, proprio)
    assert finished.replay_frame_index == 1
    assert finished.replay_final_target_arrived is True
    policy.commit_replay_proposal(4)

    assert policy.replay_complete is True
    assert policy.completed_replay_frames == 2
    assert policy.repeated_target_commands == 2
