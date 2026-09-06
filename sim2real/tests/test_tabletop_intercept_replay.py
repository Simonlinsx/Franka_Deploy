from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
import zipfile

import numpy as np
import pytest

from sim2real.action_replay import (
    CANONICAL_ACTION_ORDER,
    ReplayActionSequence,
    TabletopInterceptReplayConfig,
    TransactionalReplayActionPolicy,
    load_replay_actions_payload,
)
from sim2real.diagnostics.build_tabletop_intercept_replays import (
    DEFAULT_BUNDLE,
    DEFAULT_SPHERE_SOURCE,
    DEFAULT_SOURCE,
    build_bundles,
)
from sim2real.deployment.bundle import DeployBundle, load_checkpoint_safely
from sim2real.policy import ActionControllerParameters
from sim2real.replay_actions import build_parser, build_replay_request
from sim2real.deployment.runner import SupervisedV94RunError
from motion_planning.online_tabletop import (
    _FingertipTableClearanceConflict,
    _OrientationCorrectionConflict,
    TransactionalTabletopOnlinePlannerPolicy,
    cartesian_joint_correction,
    panda_T_base_policy_palm,
    pose_preserving_joint_target,
)
from sim2real.contracts.v94 import V94Contract


TABLETOP_CHECKPOINT_ROOT = Path(
    "data/checkpoints/table/inspire_directional_demo_v364_v371_speed600_20hz_noflow_20260816/"
    "inspire_directional_demo_v364_v371_speed600_20hz_noflow_20260816/"
    "runtime_checkpoints"
)
TABLETOP_CHECKPOINT = (
    TABLETOP_CHECKPOINT_ROOT / "v364_cylinder_+Y_0.02-0.20mps_inference.pt"
)


def _config(*, collision: bool = False, max_wait_ticks: int = 20):
    return TabletopInterceptReplayConfig(
        preposition_end_index=1,
        control_dt_s=0.05,
        fit_sample_count=4,
        motion_axis_palm=(0.0, 1.0, 0.0),
        intercept_center_palm_m=(0.0, 0.0, 0.0),
        speed_min_m_s=0.05,
        speed_max_m_s=0.20,
        heading_cos_min=0.90,
        max_fit_residual_m=0.002,
        max_lateral_miss_m=0.01,
        trigger_ttc_max_s=0.25,
        max_wait_ticks=max_wait_ticks,
        collision_replan=collision,
        collision_min_velocity_change_m_s=0.05,
    )


def _policy(*, config=None):
    count = 5
    sequence = ReplayActionSequence(
        actions13=np.zeros((count, 13), dtype=np.float32),
        sha256="0" * 64,
        source_format="test",
        declared_policy_rate_hz=20.0,
        recorded_franka_target_q_rad=np.zeros((count, 7), dtype=np.float32),
        recorded_rh56_angle_set_register_order=np.full(
            (count, 6), 1000, dtype=np.int32
        ),
        tabletop_intercept=_config() if config is None else config,
    )
    return TransactionalReplayActionPolicy(
        sequence,
        point_feature_dim=3,
        selected_steps=count,
        arrival_gated=True,
        q_home_rad=np.zeros(7, dtype=np.float32),
    )


def _observation(center):
    points = np.zeros((4, 128, 3), dtype=np.float32)
    points[-1] = np.asarray(center, dtype=np.float32)
    valid = np.ones((4, 128), dtype=np.float32)
    proprio = np.zeros((4, 67), dtype=np.float32)
    return points, valid, proprio


def _advance_to_preposition(policy):
    observation = _observation((0.0, -0.1, 0.0))
    assert policy.action_for_sequence(1, *observation).replay_frame_index == 0
    policy.commit_replay_proposal(1)
    assert policy.action_for_sequence(2, *observation).replay_frame_index == 1
    policy.commit_replay_proposal(2)


def test_intercept_wait_is_transactional_and_triggers_only_after_stable_fit():
    policy = _policy()
    _advance_to_preposition(policy)

    centers = (-0.035, -0.030, -0.025, -0.020)
    for sequence, y in enumerate(centers, start=3):
        output = policy.action_for_sequence(
            sequence, *_observation((0.0, y, 0.0))
        )
        expected = 2 if sequence == 6 else 1
        assert output.replay_frame_index == expected
        if sequence == 4:
            policy.discard_replay_proposal(sequence)
            assert policy.diagnostics_snapshot["samples"] == 1
            output = policy.action_for_sequence(
                sequence, *_observation((0.0, y, 0.0))
            )
            assert output.replay_frame_index == 1
        policy.commit_replay_proposal(sequence)

    diagnostics = policy.diagnostics_snapshot
    assert diagnostics["armed"] is True
    assert diagnostics["triggered"] is True
    assert diagnostics["speed_m_s"] == pytest.approx(0.1, abs=1.0e-5)
    assert diagnostics["heading_cos"] == pytest.approx(1.0)
    assert diagnostics["state"] == "trigger_ready"

    # Closure/lift resumes at the demonstrated 20 Hz cadence; it does not
    # insert an arrival stop between every post-trigger waypoint.
    points, valid, proprio = _observation((0.0, 0.0, 0.0))
    proprio[-1, :7] = np.float32(0.5)
    streamed = policy.action_for_sequence(7, points, valid, proprio)
    assert streamed.replay_frame_index == 3


def test_intercept_preposition_streams_but_waits_for_endpoint_arrival():
    policy = _policy()
    points, valid, proprio = _observation((0.0, -0.1, 0.0))
    proprio[-1, :7] = np.float32(0.5)

    first = policy.action_for_sequence(1, points, valid, proprio)
    assert first.replay_frame_index == 0
    policy.commit_replay_proposal(1)
    streamed = policy.action_for_sequence(2, points, valid, proprio)
    assert streamed.replay_frame_index == 1
    policy.commit_replay_proposal(2)

    endpoint_wait = policy.action_for_sequence(3, points, valid, proprio)
    assert endpoint_wait.replay_frame_index == 1
    assert endpoint_wait.replay_arrival_error_rad == pytest.approx(0.5)


def test_reverse_motion_never_advances_and_wait_expires_fail_closed():
    policy = _policy(config=_config(max_wait_ticks=4))
    _advance_to_preposition(policy)
    for sequence, y in enumerate((0.02, 0.015, 0.010, 0.005), start=3):
        output = policy.action_for_sequence(
            sequence, *_observation((0.0, y, 0.0))
        )
        assert output.replay_frame_index == 1
        policy.commit_replay_proposal(sequence)
    assert policy.diagnostics_snapshot["triggered"] is False
    assert policy.diagnostics_snapshot["heading_cos"] == pytest.approx(-1.0)
    with pytest.raises(RuntimeError, match="wait expired"):
        policy.action_for_sequence(7, *_observation((0.0, 0.0, 0.0)))


def test_collision_plan_requires_velocity_change_and_fresh_post_impact_fit():
    config = TabletopInterceptReplayConfig(
        **{
            **_config(collision=True, max_wait_ticks=60).__dict__,
            "speed_max_m_s": 0.25,
        }
    )
    policy = _policy(config=config)
    _advance_to_preposition(policy)
    sequence = 3
    # A stable 0.15 m/s approach establishes the pre-impact velocity but may
    # never directly trigger a collision-mode grasp.
    for y in (-0.1500, -0.1425, -0.1350, -0.1275, -0.1200):
        output = policy.action_for_sequence(
            sequence, *_observation((0.0, y, 0.0))
        )
        assert output.replay_frame_index == 1
        policy.commit_replay_proposal(sequence)
        sequence += 1
    assert policy.diagnostics_snapshot["collision_seen"] is False

    # The board slows the object to 0.07 m/s.  The first mixed window detects
    # the velocity change; only a subsequent all-post-impact fit may trigger.
    triggered = False
    y = -0.1165
    for _ in range(35):
        output = policy.action_for_sequence(
            sequence, *_observation((0.0, y, 0.0))
        )
        policy.commit_replay_proposal(sequence)
        sequence += 1
        y += 0.0035
        if output.replay_frame_index == 2:
            triggered = True
            break
    assert triggered is True
    assert policy.diagnostics_snapshot["collision_seen"] is True
    assert policy.diagnostics_snapshot["triggered"] is True
    assert policy.diagnostics_snapshot["post_collision_samples"] >= 4


def _zip_with_intercept_metadata() -> bytes:
    count = 5
    npz = io.BytesIO()
    np.savez(
        npz,
        time_s=np.arange(count, dtype=np.float64) * 0.05,
        policy_action=np.zeros((count, 13), dtype=np.float32),
        franka_joint_target_rad=np.zeros((count, 7), dtype=np.float32),
        inspire_angle_set_register_order=np.full(
            (count, 6), 1000, dtype=np.int32
        ),
    )
    config = _config()
    metadata = {
        "control_hz": 20.0,
        "control_dt_s": 0.05,
        "frames": count,
        "action_contract": "exact target tabletop intercept",
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
        "tabletop_intercept": {
            "version": "tabletop_intercept_replay_v1",
            **config.__dict__,
        },
    }
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("replay/metadata.json", json.dumps(metadata))
        bundle.writestr("replay/data.npz", npz.getvalue())
    return payload.getvalue()


def test_zip_intercept_metadata_is_strictly_decoded():
    sequence = load_replay_actions_payload(
        _zip_with_intercept_metadata(),
        suffix=".zip",
        expected_policy_rate_hz=20,
    )
    assert sequence.tabletop_intercept is not None
    assert sequence.tabletop_intercept.preposition_end_index == 1
    assert sequence.tabletop_intercept.motion_axis_palm == (0.0, 1.0, 0.0)


def test_intercept_bundle_refuses_fixed_rate_or_partial_execution():
    sequence = load_replay_actions_payload(
        _zip_with_intercept_metadata(),
        suffix=".zip",
        expected_policy_rate_hz=20,
    )
    with pytest.raises(ValueError, match="arrival-gated"):
        TransactionalReplayActionPolicy(
            sequence, point_feature_dim=3, selected_steps=5
        )
    with pytest.raises(ValueError, match="complete action plan"):
        TransactionalReplayActionPolicy(
            sequence,
            point_feature_dim=3,
            selected_steps=4,
            arrival_gated=True,
        )


def test_cli_rejects_missing_arrival_gate_before_hardware_preparation(tmp_path):
    path = tmp_path / "intercept.zip"
    path.write_bytes(_zip_with_intercept_metadata())
    with pytest.raises(SupervisedV94RunError, match="requires --arrival-gated"):
        build_replay_request(
            build_parser().parse_args(
                ["--actions", str(path), "--policy-rate-hz", "20"]
            )
        )


def test_real_success_template_builds_nine_table_clear_plans(tmp_path):
    manifest = build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    assert manifest["kind"] == "tabletop_online_motion_planner_manifest_v5"
    assert manifest["hardware_writes"] is False
    assert len(manifest["bundles"]) == 9
    assert manifest["preposition_check"]["finger_closure"] is False
    assert manifest["preposition_check"]["lift"] is False
    assert manifest["preposition_end_index"] == 93
    safety = manifest["safety_audit"]
    assert safety["maximum_franka_tick_delta_rad"] <= 0.015
    assert safety["minimum_fingertip_table_clearance_m"] >= 0.020
    assert safety["minimum_palm_table_clearance_m"] >= 0.12
    online_safety = manifest["online_safety_audit"]
    assert online_safety["maximum_franka_tick_delta_rad"] <= 0.10
    assert online_safety["minimum_fingertip_table_clearance_m"] >= 0.020
    sphere_safety = manifest["sphere_online_safety_audit"]
    assert sphere_safety["maximum_franka_tick_delta_rad"] <= 0.10
    assert sphere_safety["minimum_fingertip_table_clearance_m"] >= 0.025
    assert manifest["sphere_reference_max_q_home_delta_rad"] < 1.10
    assert manifest["sphere_open_approach_lift_m"] == 0.030
    assert len(manifest["sphere_capture_object_offset_palm_m"]) == 3
    assert manifest["sphere_grasp_source"][
        "first_strict_pre_lift_grasp_index"
    ] == 64
    assert manifest["sphere_grasp_source"][
        "open_preshape_register_order"
    ] == [1000, 1000, 1000, 1000, 1000, 0]
    assert manifest["sphere_grasp_source"]["closure_duration_s"] == pytest.approx(
        0.70
    )
    assert max(
        manifest["sphere_grasp_source"][
            "maximum_closure_target_delta_per_tick_register_order"
        ]
    ) <= 132
    assert manifest["online_template_start_index"] == 4
    assert manifest["online_template_catch_index"] == 54
    assert manifest["online_closure_end_index"] == 67
    assert manifest["online_lift_end_index"] == 79
    assert manifest["dynamic_reference_max_q_home_delta_rad"] < 0.92
    assert sum(bool(item["collision_replan"]) for item in manifest["bundles"]) == 1
    for item in manifest["bundles"]:
        loaded = load_replay_actions_payload(
            (tmp_path / f"{item['name']}.zip").read_bytes(),
            suffix=".zip",
            expected_policy_rate_hz=20,
        )
        assert loaded.action_count == 98
        assert loaded.tabletop_intercept is None
        assert loaded.tabletop_online_planner is not None
        sphere = item["name"].startswith("sphere_")
        assert loaded.tabletop_online_planner.version == (
            "tabletop_online_intercept_planner_v5"
            if sphere
            else "tabletop_online_intercept_planner_v4"
        )
        assert loaded.tabletop_online_planner.minimum_fingertip_clearance_m == 0.015
        assert loaded.tabletop_online_planner.contact_prediction_latency_s == (
            pytest.approx(0.120 if sphere else 0.050)
        )
        assert loaded.tabletop_online_planner.max_linear_prediction_s == (
            1.00
            if item["name"] == "sphere_posy_board_collision"
            else 0.40
        )
        assert loaded.tabletop_online_planner.closure_trigger_ttc_s == (
            0.75 if sphere else 0.55
        )
        assert loaded.tabletop_online_planner.grasp_vertical_offset_m == (
            -0.015 if sphere else -0.030
        )
        if sphere:
            assert (
                loaded.tabletop_online_planner.capture_object_offset_palm_m
                is not None
            )
            assert loaded.tabletop_online_planner.arm_tracking_speed_rad_s == 0.50
            assert (
                loaded.tabletop_online_planner.minimum_intercept_horizon_s
                == 0.15
            )
            assert loaded.tabletop_online_planner.open_approach_lift_m == 0.030
        else:
            assert (
                loaded.tabletop_online_planner.capture_object_offset_palm_m
                is None
            )
        closure_end_index = loaded.tabletop_online_planner.closure_end_index
        target = loaded.recorded_franka_target_q_rad
        assert target is not None
        # A 0.5 rad/s joint reaches this nominal approach before the first
        # closure target.  The old schedule started and finished closure while
        # the arm still had roughly 0.44/0.27 rad of unavoidable tracking lag.
        simulated_measured = target[0].astype(np.float64).copy()
        error_at_closure_start = None
        for index, tick_target in enumerate(target[: closure_end_index + 1]):
            simulated_measured += np.clip(
                tick_target - simulated_measured,
                -0.025,
                0.025,
            )
            if index == loaded.tabletop_online_planner.template_catch_index:
                error_at_closure_start = float(
                    np.max(np.abs(tick_target - simulated_measured))
                )
        assert error_at_closure_start is not None
        assert error_at_closure_start <= (0.050 if sphere else 0.020)
        assert np.max(
            np.abs(target[closure_end_index] - simulated_measured)
        ) <= 0.020


def test_online_pose_solver_refines_a_safe_local_numerical_plateau():
    contract = V94Contract.from_bundle(
        DeployBundle(DEFAULT_BUNDLE)
    ).with_runtime_policy_rate_hz(20)
    seed = contract.q_home_rad.astype(np.float64)
    seed_palm = panda_T_base_policy_palm(
        seed, contract.T_flange_policy_palm
    )
    translation = np.asarray([0.15, -0.20, -0.03], dtype=np.float64)
    kwargs = {
        "T_flange_policy_palm": contract.T_flange_policy_palm,
        "joint_limits_rad": contract.joint_limits_rad,
        "joint_limit_margin_rad": 0.050,
        "damping": 0.04,
        "target_rotation_base": seed_palm[:3, :3],
    }

    with pytest.raises(RuntimeError, match="did not converge"):
        pose_preserving_joint_target(seed, translation, **kwargs)

    refined = pose_preserving_joint_target(
        seed,
        translation,
        local_precision_refinement=True,
        **kwargs,
    )
    refined_palm = panda_T_base_policy_palm(
        refined, contract.T_flange_policy_palm
    )
    assert np.linalg.norm(
        refined_palm[:3, 3] - (seed_palm[:3, 3] + translation)
    ) <= 5.0e-4
    assert np.linalg.norm(
        refined_palm[:3, :3] - seed_palm[:3, :3]
    ) <= 3.0e-3


def _online_policy(path: Path, *, legacy: bool = False):
    sequence = load_replay_actions_payload(
        path.read_bytes(), suffix=".zip", expected_policy_rate_hz=20
    )
    if legacy:
        assert sequence.tabletop_online_planner is not None
        with np.load(DEFAULT_SOURCE, allow_pickle=False) as archive:
            source_target = archive["real_franka_target_q_rad"].copy()
        contract_for_template = V94Contract.from_bundle(
            DeployBundle(DEFAULT_BUNDLE)
        ).with_runtime_policy_rate_hz(20)
        reference_q = source_target[47].astype(np.float64)
        reference_q += cartesian_joint_correction(
            reference_q,
            np.asarray([0.0, 0.0, 0.012]),
            T_flange_policy_palm=contract_for_template.T_flange_policy_palm,
            damping=0.04,
        )
        lift_q = pose_preserving_joint_target(
            reference_q,
            np.asarray([0.0, 0.0, 0.120]),
            T_flange_policy_palm=contract_for_template.T_flange_policy_palm,
            joint_limits_rad=contract_for_template.joint_limits_rad,
            joint_limit_margin_rad=0.050,
            damping=0.04,
        )
        target = sequence.recorded_franka_target_q_rad.copy()
        target[:4] = contract_for_template.q_home_rad
        for index in range(4, 55):
            fraction = (index - 4) / 50.0
            blend = fraction**3 * (
                10.0 - 15.0 * fraction + 6.0 * fraction**2
            )
            target[index] = contract_for_template.q_home_rad + blend * (
                reference_q - contract_for_template.q_home_rad
            )
        target[55:68] = reference_q
        for index in range(68, 80):
            fraction = (index - 67) / 12.0
            blend = fraction**3 * (
                10.0 - 15.0 * fraction + 6.0 * fraction**2
            )
            target[index] = reference_q + blend * (lift_q - reference_q)
        target[80:] = lift_q
        sequence = replace(
            sequence,
            recorded_franka_target_q_rad=target,
            tabletop_online_planner=replace(
                sequence.tabletop_online_planner,
                version="tabletop_online_intercept_planner_v2",
            ),
        )
    checkpoint = load_checkpoint_safely(TABLETOP_CHECKPOINT.read_bytes())
    contract = V94Contract.from_bundle(DeployBundle(DEFAULT_BUNDLE)).with_runtime_policy_rate_hz(20)
    policy = TransactionalTabletopOnlinePlannerPolicy(
        sequence,
        contract=contract,
        action_controller=ActionControllerParameters.from_metadata(
            checkpoint.metadata
        ),
        point_feature_dim=3,
        history_length=8,
        proprio_dim=96,
        selected_steps=sequence.action_count,
    )
    return policy, sequence, contract


def test_v5_sphere_uses_successful_grasp_geometry_not_cylinder_shape(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    sphere, sphere_sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    _cylinder, cylinder_sequence, _contract = _online_policy(
        tmp_path / "cylinder_posy_low.zip"
    )
    config = sphere_sequence.tabletop_online_planner
    assert config is not None
    assert config.version == "tabletop_online_intercept_planner_v5"
    reference_q = sphere_sequence.recorded_franka_target_q_rad[
        config.closure_end_index
    ]
    reference_palm = panda_T_base_policy_palm(
        reference_q, contract.T_flange_policy_palm
    )
    reconstructed = (
        reference_palm[:3, 3]
        + reference_palm[:3, :3]
        @ np.asarray(config.capture_object_offset_palm_m, dtype=np.float64)
    )
    assert np.allclose(
        reconstructed,
        config.reference_intercept_center_base_m,
        atol=5.0e-4,
        rtol=0.0,
    )
    approach_q = sphere_sequence.recorded_franka_target_q_rad[
        config.template_catch_index - 1
    ]
    approach_palm = panda_T_base_policy_palm(
        approach_q, contract.T_flange_policy_palm
    )
    assert approach_palm[2, 3] - reference_palm[2, 3] == pytest.approx(
        config.open_approach_lift_m, abs=1.0e-3
    )
    with np.load(DEFAULT_SPHERE_SOURCE, allow_pickle=False) as source:
        successful_hand = source[
            "inspire_angle_set_register_order"
        ][64]
    sphere_hand = sphere_sequence.recorded_rh56_angle_set_register_order
    cylinder_hand = cylinder_sequence.recorded_rh56_angle_set_register_order
    assert np.array_equal(
        sphere_hand[config.template_catch_index - 1],
        np.asarray([1000, 1000, 1000, 1000, 1000, 0]),
    )
    assert np.all(
        sphere_hand[: config.template_catch_index, :5] == 1000
    )
    assert np.array_equal(
        sphere_hand[config.closure_end_index], successful_hand
    )
    assert np.all(
        sphere_hand[config.closure_end_index :] == successful_hand
    )
    assert not np.array_equal(
        sphere_hand[config.closure_end_index],
        cylinder_hand[config.closure_end_index],
    )


def test_v5_prediction_horizon_recedes_with_measured_arm_time_to_go(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    far_horizon, far_error = policy._mpc_prediction_horizon(
        contract.q_home_rad
    )
    arrived_horizon, arrived_error = policy._mpc_prediction_horizon(
        policy._committed_goal_q
    )
    assert far_error > arrived_error
    assert far_horizon == config.max_linear_prediction_s
    assert arrived_error == pytest.approx(0.0)
    assert arrived_horizon == config.minimum_intercept_horizon_s
    closure_intervals = (
        config.closure_end_index - config.template_catch_index + 1
    )
    assert policy._mpc_grasp_completion_horizon(
        phase="approach", closure_step=-1
    ) == pytest.approx(
        closure_intervals * config.control_dt_s
        + config.contact_prediction_latency_s
    )
    assert policy._mpc_grasp_completion_horizon(
        phase="closure", closure_step=0
    ) == pytest.approx(
        (closure_intervals - 1) * config.control_dt_s
        + config.contact_prediction_latency_s
    )
    assert policy._mpc_grasp_completion_horizon(
        phase="closure", closure_step=closure_intervals - 2
    ) == pytest.approx(
        config.control_dt_s + config.contact_prediction_latency_s
    )
    with pytest.raises(
        RuntimeError, match="grasp-completion closure step is invalid"
    ):
        policy._mpc_grasp_completion_horizon(
            phase="closure", closure_step=-1
        )


def _synthetic_history_for_base_trajectory(
    *,
    index,
    start_center_base,
    velocity_base,
    proprio_history,
    contract,
):
    points = np.empty((8, 128, 3), dtype=np.float32)
    for history_index, trajectory_index in enumerate(range(index - 7, index + 1)):
        measured_q = (
            np.asarray(proprio_history[history_index, :7], dtype=np.float64)
            + contract.q_home_rad
        )
        inverse_palm = np.linalg.inv(
            panda_T_base_policy_palm(
                measured_q, contract.T_flange_policy_palm
            )
        )
        center = np.asarray(start_center_base, dtype=np.float64) + (
            np.asarray(velocity_base, dtype=np.float64)
            * trajectory_index
            * 0.05
        )
        points[history_index] = (
            inverse_palm @ np.concatenate([center, [1.0]])
        )[:3]
    return points


def _synthetic_history_for_base_center_function(
    *,
    index,
    center_at,
    proprio_history,
    contract,
):
    points = np.empty((8, 128, 3), dtype=np.float32)
    for history_index, trajectory_index in enumerate(range(index - 7, index + 1)):
        measured_q = (
            np.asarray(proprio_history[history_index, :7], dtype=np.float64)
            + contract.q_home_rad
        )
        inverse_palm = np.linalg.inv(
            panda_T_base_policy_palm(
                measured_q, contract.T_flange_policy_palm
            )
        )
        center = np.asarray(center_at(trajectory_index), dtype=np.float64)
        points[history_index] = (
            inverse_palm @ np.concatenate([center, [1.0]])
        )[:3]
    return points


def test_online_intercept_uses_each_history_samples_matching_palm_pose(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, _sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_low.zip", legacy=True
    )
    with np.load(DEFAULT_SOURCE, allow_pickle=False) as archive:
        proprio = archive["input_proprio_history_raw"][34].copy()

    center_base = np.asarray([0.590, -0.135, 0.052], dtype=np.float64)
    points = np.empty((8, 128, 3), dtype=np.float32)
    for history_index in range(8):
        measured_q = proprio[history_index, :7] + contract.q_home_rad
        inverse_palm = np.linalg.inv(
            panda_T_base_policy_palm(
                measured_q, contract.T_flange_policy_palm
            )
        )
        points[history_index] = (
            inverse_palm @ np.concatenate([center_base, [1.0]])
        )[:3]

    centers = policy._centers_base(
        points,
        np.ones((8, 128), dtype=np.float32),
        proprio,
    )
    np.testing.assert_allclose(
        centers,
        np.repeat(center_base[None, :], 8, axis=0),
        atol=2.0e-6,
    )


def test_online_intercept_locks_valid_initial_tabletop_height_against_depth_jitter(
    tmp_path,
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_low.zip", legacy=True
    )
    with np.load(DEFAULT_SOURCE, allow_pickle=False) as archive:
        proprio = archive["input_proprio_history_raw"].copy()
    config = sequence.tabletop_online_planner
    assert config is not None
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    valid = np.ones((8, 128), dtype=np.float32)

    first_proprio = proprio[0]
    first_points = _synthetic_history_for_base_trajectory(
        index=0,
        start_center_base=reference + np.asarray([0.0, 0.0, -0.014]),
        velocity_base=np.zeros(3),
        proprio_history=first_proprio,
        contract=contract,
    )
    policy.action_for_sequence(1, first_points, valid, first_proprio)
    policy.commit_replay_proposal(1)
    assert policy.diagnostics_snapshot["tabletop_height_source"] == (
        "first_committed_planar_observation"
    )

    jittered_proprio = proprio[1]
    jittered_points = _synthetic_history_for_base_trajectory(
        index=1,
        start_center_base=reference + np.asarray([0.0, 0.0, -0.025]),
        velocity_base=np.zeros(3),
        proprio_history=jittered_proprio,
        contract=contract,
    )
    policy.action_for_sequence(2, jittered_points, valid, jittered_proprio)
    policy.commit_replay_proposal(2)
    diagnostics = policy.diagnostics_snapshot
    assert diagnostics["tabletop_height_source"] == (
        "locked_tabletop_episode_height"
    )
    assert diagnostics["observed_height_correction_m"] == pytest.approx(-0.025)
    assert diagnostics["tabletop_height_correction_m"] == pytest.approx(-0.014)
    assert diagnostics["desired_position_correction_base_m"][2] == pytest.approx(
        -0.014
    )


def test_online_intercept_still_rejects_invalid_initial_tabletop_height(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_low.zip", legacy=True
    )
    with np.load(DEFAULT_SOURCE, allow_pickle=False) as archive:
        proprio = archive["input_proprio_history_raw"][0].copy()
    config = sequence.tabletop_online_planner
    assert config is not None
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    points = _synthetic_history_for_base_trajectory(
        index=0,
        start_center_base=reference + np.asarray([0.0, 0.0, -0.016]),
        velocity_base=np.zeros(3),
        proprio_history=proprio,
        contract=contract,
    )
    with pytest.raises(RuntimeError, match="outside the commissioned Z workspace"):
        policy.action_for_sequence(
            1,
            points,
            np.ones((8, 128), dtype=np.float32),
            proprio,
        )


def test_v4_starts_table_clear_approach_on_ramp_motion_without_closing(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_low.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    proprio = np.zeros((8, 96), dtype=np.float32)
    points = _synthetic_history_for_base_trajectory(
        index=0,
        start_center_base=reference + np.asarray([0.0, -0.20, 0.050]),
        velocity_base=np.asarray([0.0, 0.10, 0.0]),
        proprio_history=proprio,
        contract=contract,
    )
    output = policy.action_for_sequence(
        1,
        points,
        np.ones((8, 128), dtype=np.float32),
        proprio,
    )
    assert output.replay_frame_index == 0
    policy.commit_replay_proposal(1)
    diagnostics = policy.diagnostics_snapshot
    assert diagnostics["entry_height_outside_capture_corridor"] is True
    assert diagnostics["tabletop_height_source"] == (
        "calibrated_table_height_during_entry"
    )
    assert diagnostics["tabletop_height_correction_m"] == 0.0
    assert diagnostics["capture_gate_accepted"] is False
    assert diagnostics["adaptive_phase"] == "approach"


def test_online_intercept_tracks_low_speed_then_closes_and_lifts_transactionally(
    tmp_path,
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_low.zip", legacy=True
    )
    with np.load(DEFAULT_SOURCE, allow_pickle=False) as archive:
        proprio = archive["input_proprio_history_raw"].copy()

    config = sequence.tabletop_online_planner
    assert config is not None
    valid = np.ones((8, 128), dtype=np.float32)
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    velocity = np.asarray([0.0, 0.14, 0.0], dtype=np.float64)
    motion_ticks = 30
    start_center = reference - velocity * motion_ticks * config.control_dt_s

    def center_at(trajectory_index):
        bounded = int(np.clip(trajectory_index, 0, motion_ticks))
        return start_center + velocity * bounded * config.control_dt_s
    minimum_clearance = float("inf")
    corrections = []
    maximum_target_step = 0.0
    previous_target = None
    relative_at_contact = None
    palm_height_at_contact = None
    palm_height_after_lift = None
    for index in range(sequence.action_count):
        source_index = min(index, len(proprio) - 1)
        source_proprio = proprio[source_index]
        points = _synthetic_history_for_base_center_function(
            index=index,
            center_at=center_at,
            proprio_history=source_proprio,
            contract=contract,
        )
        output = policy.action_for_sequence(
            index + 1,
            points,
            valid,
            source_proprio,
        )
        target = output.exact_franka_target_q_rad.astype(np.float64)
        if previous_target is not None:
            maximum_target_step = max(
                maximum_target_step,
                float(np.max(np.abs(target - previous_target))),
            )
        previous_target = target
        corrections.append(
            float(
                np.max(
                    np.abs(
                        output.exact_franka_target_q_rad
                        - sequence.recorded_franka_target_q_rad[index]
                    )
                )
            )
        )
        policy.commit_replay_proposal(index + 1)
        target_palm = panda_T_base_policy_palm(
            target, contract.T_flange_policy_palm
        )
        if index == config.template_catch_index:
            object_center = center_at(index)
            relative_at_contact = object_center - target_palm[:3, 3]
            palm_height_at_contact = float(target_palm[2, 3])
        if index == config.lift_end_index:
            palm_height_after_lift = float(target_palm[2, 3])
        minimum_clearance = min(
            minimum_clearance,
            float(policy.diagnostics_snapshot["minimum_fingertip_clearance_m"]),
        )
    assert policy.replay_complete is True
    assert policy.completed_replay_frames == sequence.action_count
    assert max(corrections) <= 0.651
    assert max(corrections) >= 0.05
    assert maximum_target_step <= 0.100001
    assert minimum_clearance >= 0.015
    assert relative_at_contact is not None
    np.testing.assert_allclose(
        relative_at_contact,
        np.asarray([0.09, -0.01, -0.18]),
        atol=0.025,
    )
    assert palm_height_at_contact is not None
    assert palm_height_after_lift is not None
    assert palm_height_after_lift - palm_height_at_contact >= 0.10


def test_online_intercept_tracks_inbound_object_from_outside_reachable_radius(
    tmp_path,
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_low.zip", legacy=True
    )
    with np.load(DEFAULT_SOURCE, allow_pickle=False) as archive:
        proprio = archive["input_proprio_history_raw"].copy()
    config = sequence.tabletop_online_planner
    assert config is not None
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    velocity = np.asarray([0.0, 0.18, 0.0], dtype=np.float64)
    motion_ticks = 35
    start = reference - velocity * motion_ticks * config.control_dt_s

    def center_at(trajectory_index):
        bounded = int(np.clip(trajectory_index, 0, motion_ticks))
        return start + velocity * bounded * config.control_dt_s

    valid = np.ones((8, 128), dtype=np.float32)
    saw_boundary_approach = False
    entered_before_closure = False
    for index in range(sequence.action_count):
        source_proprio = proprio[min(index, len(proprio) - 1)]
        points = _synthetic_history_for_base_center_function(
            index=index,
            center_at=center_at,
            proprio_history=source_proprio,
            contract=contract,
        )
        policy.action_for_sequence(index + 1, points, valid, source_proprio)
        policy.commit_replay_proposal(index + 1)
        projected = bool(
            policy.diagnostics_snapshot.get(
                "workspace_boundary_projection", False
            )
        )
        saw_boundary_approach |= projected
        if index == config.template_catch_index:
            entered_before_closure = not projected

    assert policy.replay_complete is True
    assert saw_boundary_approach is True
    assert entered_before_closure is True


def test_online_intercept_matches_recorded_decelerating_cylinder_trace(tmp_path):
    """Replay the base-plane trace from the 20260819-015658 mask video."""

    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_low.zip", legacy=True
    )
    with np.load(DEFAULT_SOURCE, allow_pickle=False) as archive:
        proprio = archive["input_proprio_history_raw"].copy()
    config = sequence.tabletop_online_planner
    assert config is not None
    valid = np.ones((8, 128), dtype=np.float32)
    key_index = np.asarray([0, 3, 10, 20, 30, 34, 43, 58], dtype=np.float64)
    key_centers = np.asarray(
        [
            [0.586, -0.351, 0.051],
            [0.588, -0.294, 0.051],
            [0.593, -0.217, 0.051],
            [0.599, -0.110, 0.052],
            [0.605, -0.035, 0.052],
            [0.606, -0.014, 0.052],
            [0.608, 0.027, 0.052],
            [0.610, 0.057, 0.052],
        ],
        dtype=np.float64,
    )

    def center_at(index):
        clipped = float(np.clip(index, key_index[0], key_index[-1]))
        return np.asarray(
            [
                np.interp(clipped, key_index, key_centers[:, axis])
                for axis in range(3)
            ]
        )

    previous_target = None
    maximum_target_step = 0.0
    minimum_clearance = float("inf")
    relative_at_contact = None
    for index in range(sequence.action_count):
        source_proprio = proprio[min(index, len(proprio) - 1)]
        points = np.empty((8, 128, 3), dtype=np.float32)
        for history_index, trajectory_index in enumerate(
            range(index - 7, index + 1)
        ):
            measured_q = (
                source_proprio[history_index, :7] + contract.q_home_rad
            )
            inverse_palm = np.linalg.inv(
                panda_T_base_policy_palm(
                    measured_q, contract.T_flange_policy_palm
                )
            )
            points[history_index] = (
                inverse_palm
                @ np.concatenate([center_at(trajectory_index), [1.0]])
            )[:3]
        output = policy.action_for_sequence(
            index + 1, points, valid, source_proprio
        )
        target = output.exact_franka_target_q_rad.astype(np.float64)
        if previous_target is not None:
            maximum_target_step = max(
                maximum_target_step,
                float(np.max(np.abs(target - previous_target))),
            )
        previous_target = target
        policy.commit_replay_proposal(index + 1)
        diagnostics = policy.diagnostics_snapshot
        minimum_clearance = min(
            minimum_clearance,
            float(diagnostics["minimum_fingertip_clearance_m"]),
        )
        if index == config.template_catch_index:
            palm = panda_T_base_policy_palm(
                target, contract.T_flange_policy_palm
            )
            relative_at_contact = center_at(index) - palm[:3, 3]
    assert policy.replay_complete is True
    assert maximum_target_step <= 0.100001
    assert minimum_clearance >= 0.015
    np.testing.assert_allclose(
        relative_at_contact,
        np.asarray([0.093, -0.006, -0.180]),
        atol=0.015,
    )


def test_online_template_responds_to_live_base_frame_object_shift(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    original, sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_low.zip", legacy=True
    )
    shifted, _, _ = _online_policy(
        tmp_path / "cylinder_posy_low.zip"
    )
    with np.load(DEFAULT_SOURCE, allow_pickle=False) as archive:
        points = archive["input_pointcloud_history_metric"][4].copy()
        valid = archive["input_pointcloud_valid_history"][4].copy()
        proprio = archive["input_proprio_history_raw"][4].copy()

    # Advance into the smooth approach so the same live object shift must
    # alter an actually commanded target, not merely the staged catch goal.
    measured_q = proprio[-1, :7] + contract.q_home_rad
    palm = panda_T_base_policy_palm(
        measured_q, contract.T_flange_policy_palm
    )
    shift_palm = palm[:3, :3].T @ np.asarray([0.0, 0.020, 0.0])
    shifted_points = points.copy()
    shifted_points[:, :, :3] += shift_palm
    for sequence_id in range(1, 14):
        baseline_points = points
        moved_points = shifted_points
        original.action_for_sequence(sequence_id, baseline_points, valid, proprio)
        original.commit_replay_proposal(sequence_id)
        shifted.action_for_sequence(sequence_id, moved_points, valid, proprio)
        shifted.commit_replay_proposal(sequence_id)
    baseline = original.action_for_sequence(14, points, valid, proprio)
    moved = shifted.action_for_sequence(14, shifted_points, valid, proprio)
    assert not np.array_equal(
        baseline.exact_franka_target_q_rad,
        moved.exact_franka_target_q_rad,
    )
    assert np.max(
        np.abs(
            baseline.exact_franka_target_q_rad
            - moved.exact_franka_target_q_rad
        )
    ) >= 1.0e-3


def test_online_intercept_discard_does_not_advance_goal_or_capture(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    staged, sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_low.zip", legacy=True
    )
    control, _, _ = _online_policy(
        tmp_path / "cylinder_posy_low.zip", legacy=True
    )
    with np.load(DEFAULT_SOURCE, allow_pickle=False) as archive:
        proprio = archive["input_proprio_history_raw"].copy()
    config = sequence.tabletop_online_planner
    assert config is not None
    valid = np.ones((8, 128), dtype=np.float32)
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    velocity = np.asarray([0.0, 0.14, 0.0], dtype=np.float64)
    motion_ticks = 24
    start = reference - velocity * motion_ticks * config.control_dt_s

    def center_at(trajectory_index, *, shift_y=0.0):
        bounded = int(np.clip(trajectory_index, 0, motion_ticks))
        return (
            start
            + velocity * bounded * config.control_dt_s
            + np.asarray([0.0, shift_y, 0.0])
        )

    def observation(index, *, shift_y=0.0):
        source_proprio = proprio[min(index, len(proprio) - 1)]
        points = _synthetic_history_for_base_center_function(
            index=index,
            center_at=lambda trajectory_index: center_at(
                trajectory_index, shift_y=shift_y
            ),
            proprio_history=source_proprio,
            contract=contract,
        )
        return points, source_proprio

    for index in range(10):
        points, source_proprio = observation(index)
        for policy in (staged, control):
            policy.action_for_sequence(
                index + 1, points, valid, source_proprio
            )
            policy.commit_replay_proposal(index + 1)
    shifted_points, source_proprio = observation(10, shift_y=0.020)
    staged.action_for_sequence(11, shifted_points, valid, source_proprio)
    staged.discard_replay_proposal(11)
    baseline_points, source_proprio = observation(10)
    retried = staged.action_for_sequence(
        11, baseline_points, valid, source_proprio
    )
    expected = control.action_for_sequence(
        11, baseline_points, valid, source_proprio
    )
    np.testing.assert_array_equal(
        retried.exact_franka_target_q_rad,
        expected.exact_franka_target_q_rad,
    )
    staged.commit_replay_proposal(11)
    control.commit_replay_proposal(11)

    for index in range(11, config.closure_end_index):
        points, source_proprio = observation(index)
        staged.action_for_sequence(
            index + 1, points, valid, source_proprio
        )
        staged.commit_replay_proposal(index + 1)
    points, source_proprio = observation(config.closure_end_index)
    staged.action_for_sequence(
        config.closure_end_index + 1,
        points,
        valid,
        source_proprio,
    )
    assert staged._capture_q is None
    assert staged._lift_q is None
    staged.discard_replay_proposal(config.closure_end_index + 1)
    assert staged._capture_q is None
    assert staged._lift_q is None


def test_online_intercept_refits_after_board_collision_velocity_change(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip", legacy=True
    )
    with np.load(DEFAULT_SOURCE, allow_pickle=False) as archive:
        proprio = archive["input_proprio_history_raw"].copy()
    config = sequence.tabletop_online_planner
    assert config is not None and config.collision_replan is True
    valid = np.ones((8, 128), dtype=np.float32)
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    collision_index = 12
    before_velocity = np.asarray([0.0, 0.10, 0.0])
    after_velocity = np.asarray([0.0, -0.04, 0.0])
    pre_displacement = before_velocity * collision_index * 0.05
    post_displacement = (
        after_velocity
        * (config.template_catch_index - collision_index)
        * 0.05
    )
    start = reference - pre_displacement - post_displacement

    def center_at(index):
        bounded_index = min(index, config.template_catch_index)
        if bounded_index <= collision_index:
            return start + before_velocity * bounded_index * 0.05
        return (
            start
            + pre_displacement
            + after_velocity * (bounded_index - collision_index) * 0.05
        )

    saw_post_collision_fit = False
    relative_at_contact = None
    for index in range(sequence.action_count):
        source_proprio = proprio[min(index, len(proprio) - 1)]
        points = np.empty((8, 128, 3), dtype=np.float32)
        for history_index, trajectory_index in enumerate(
            range(index - 7, index + 1)
        ):
            measured_q = (
                source_proprio[history_index, :7] + contract.q_home_rad
            )
            inverse_palm = np.linalg.inv(
                panda_T_base_policy_palm(
                    measured_q, contract.T_flange_policy_palm
                )
            )
            points[history_index] = (
                inverse_palm
                @ np.concatenate([center_at(trajectory_index), [1.0]])
            )[:3]
        output = policy.action_for_sequence(
            index + 1, points, valid, source_proprio
        )
        target = output.exact_franka_target_q_rad.astype(np.float64)
        policy.commit_replay_proposal(index + 1)
        diagnostics = policy.diagnostics_snapshot
        if index >= collision_index + config.fit_sample_count:
            velocity = np.asarray(diagnostics.get("velocity_base_m_s", [0, 0, 0]))
            saw_post_collision_fit |= bool(
                diagnostics.get("velocity_source") == "live_fit"
                and velocity[1] <= -0.03
            )
        if index == config.template_catch_index:
            palm = panda_T_base_policy_palm(
                target, contract.T_flange_policy_palm
            )
            relative_at_contact = center_at(index) - palm[:3, 3]
    assert saw_post_collision_fit is True
    np.testing.assert_allclose(
        relative_at_contact,
        np.asarray([0.09, 0.0, -0.18]),
        atol=0.030,
    )
def test_online_template_cli_rejects_arrival_gate(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    path = tmp_path / "cylinder_posy_low.zip"
    with pytest.raises(SupervisedV94RunError, match="must not use"):
        build_replay_request(
            build_parser().parse_args(
                [
                    "--actions",
                    str(path),
                    "--checkpoint",
                    str(TABLETOP_CHECKPOINT),
                    "--policy-rate-hz",
                    "20",
                    "--arrival-gated",
                ]
            )
        )


@pytest.mark.parametrize(
    ("bundle_name", "checkpoint_name", "direction"),
    (
        (
            "cylinder_posy_high.zip",
            "v366_cylinder_+Y_0.20-0.40mps_inference.pt",
            1.0,
        ),
        (
            "cylinder_negy_high.zip",
            "v367_cylinder_-Y_0.20-0.40mps_inference.pt",
            -1.0,
        ),
    ),
)
def test_online_template_tracks_synthetic_high_speed_in_both_directions(
    tmp_path,
    bundle_name,
    checkpoint_name,
    direction,
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    sequence = load_replay_actions_payload(
        (tmp_path / bundle_name).read_bytes(),
        suffix=".zip",
        expected_policy_rate_hz=20,
    )
    checkpoint = load_checkpoint_safely(
        (TABLETOP_CHECKPOINT_ROOT / checkpoint_name).read_bytes()
    )
    contract = V94Contract.from_bundle(
        DeployBundle(DEFAULT_BUNDLE)
    ).with_runtime_policy_rate_hz(20)
    policy = TransactionalTabletopOnlinePlannerPolicy(
        sequence,
        contract=contract,
        action_controller=ActionControllerParameters.from_metadata(
            checkpoint.metadata
        ),
        point_feature_dim=3,
        history_length=8,
        proprio_dim=96,
        selected_steps=sequence.action_count,
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    def center_at(index):
        # A physically visible high-speed pass: 160 mm at 0.40 m/s, followed
        # by frictional stop at the capture corridor while the arm catches up.
        bounded = int(np.clip(index, 0, 8))
        center = reference.copy()
        center[1] += direction * (-0.16 + 0.40 * bounded * 0.05)
        return center

    records = _run_adaptive_physical_rollout(
        policy=policy,
        sequence=sequence,
        contract=contract,
        center_at=center_at,
    )
    assert policy.replay_complete is True
    assert min(
        item["diagnostics"]["minimum_fingertip_clearance_m"]
        for item in records
    ) >= 0.015
    assert sum(
        item["diagnostics"]["velocity_source"] == "live_fit"
        for item in records
    ) >= 4
    closure = next(
        item for item in records
        if item["diagnostics"]["adaptive_phase"] == "closure"
    )
    assert closure["diagnostics"]["capture_gate_accepted"] is True
    assert closure["diagnostics"]["capture_arm_ready"] is True


def _run_adaptive_physical_rollout(
    *,
    policy,
    sequence,
    contract,
    center_at,
    measured_joint_step_rad=0.025,
):
    """Run the v3 planner against a bounded 0.5 rad/s measured-arm model."""

    q_history = [
        contract.q_home_rad.astype(np.float64).copy() for _ in range(8)
    ]
    valid = np.ones((8, 128), dtype=np.float32)
    records = []
    for index in range(sequence.action_count):
        proprio = np.zeros((8, 96), dtype=np.float32)
        points = np.empty((8, 128, 3), dtype=np.float32)
        for history_index, (trajectory_index, measured_q) in enumerate(
            zip(range(index - 7, index + 1), q_history)
        ):
            proprio[history_index, :7] = (
                measured_q - contract.q_home_rad
            )
            inverse_palm = np.linalg.inv(
                panda_T_base_policy_palm(
                    measured_q, contract.T_flange_policy_palm
                )
            )
            center = np.asarray(center_at(trajectory_index), dtype=np.float64)
            points[history_index] = (
                inverse_palm @ np.concatenate([center, [1.0]])
            )[:3]
        output = policy.action_for_sequence(
            index + 1, points, valid, proprio
        )
        target = output.exact_franka_target_q_rad.astype(np.float64)
        policy.commit_replay_proposal(index + 1)
        records.append(
            {
                "runtime_index": index,
                "template_index": int(output.replay_frame_index),
                "target_q_rad": target.copy(),
                "measured_q_rad": q_history[-1].copy(),
                "diagnostics": dict(policy.diagnostics_snapshot),
            }
        )
        measured_next = q_history[-1] + np.clip(
            target - q_history[-1],
            -float(measured_joint_step_rad),
            float(measured_joint_step_rad),
        )
        q_history = q_history[1:] + [measured_next]
    return records


def test_v5_receding_horizon_tracks_a_release_and_post_collision_change(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )

    def center_at(index):
        # The release and post-board segment deliberately use unrelated
        # velocity vectors.  No replay tick or fixed arrival pose is supplied
        # to the controller.
        time_s = max(0, int(index)) * config.control_dt_s
        if time_s < 0.55:
            return reference + np.asarray(
                [-0.050, -0.200 + 0.320 * time_s, 0.0]
            )
        after_collision_s = min(time_s - 0.55, 0.30)
        return reference + np.asarray(
            [
                -0.050 + 0.110 * after_collision_s,
                -0.024 + 0.200 * after_collision_s,
                0.0,
            ]
        )

    records = _run_adaptive_physical_rollout(
        policy=policy,
        sequence=sequence,
        contract=contract,
        center_at=center_at,
    )
    horizons = [
        float(item["diagnostics"]["prediction_horizon_s"])
        for item in records
        if "prediction_horizon_s" in item["diagnostics"]
    ]
    assert policy.replay_complete is True
    assert horizons[0] == config.max_linear_prediction_s
    # The physical-arm surrogate retains a small tracking error, so the
    # measured-arm horizon may settle just above the configured floor after
    # adding the commissioned observation latency.  It must still converge
    # to that floor within one 20 Hz control interval.
    assert min(horizons) >= config.minimum_intercept_horizon_s
    assert min(horizons) <= (
        config.minimum_intercept_horizon_s + config.control_dt_s
    )
    assert any(
        item["diagnostics"].get("velocity_source") == "live_fit"
        for item in records
    )
    assert any(
        item["diagnostics"].get("adaptive_phase") == "closure"
        for item in records
    )
    closure = [
        item
        for item in records
        if item["diagnostics"].get("adaptive_phase") == "closure"
    ]
    closure_target_limit = (
        config.arm_tracking_speed_rad_s * config.control_dt_s
    )
    assert all(
        item["diagnostics"]["target_step_rad"]
        <= closure_target_limit + 2.0e-7
        for item in closure
    )
    assert all(
        isinstance(
            item["diagnostics"]["closure_target_slew_applied"], bool
        )
        for item in closure
    )
    closure_intervals = (
        config.closure_end_index - config.template_catch_index + 1
    )
    first_completion_horizon = (
        closure_intervals * config.control_dt_s
        + config.contact_prediction_latency_s
    )
    second_completion_horizon = (
        (closure_intervals - 1) * config.control_dt_s
        + config.contact_prediction_latency_s
    )
    assert closure[0]["diagnostics"]["grasp_completion_horizon_s"] == (
        pytest.approx(first_completion_horizon)
    )
    assert closure[0]["diagnostics"][
        "grasp_completion_horizon_applied_s"
    ] == pytest.approx(0.0)
    assert closure[1]["diagnostics"][
        "grasp_completion_horizon_applied_s"
    ] == pytest.approx(second_completion_horizon)
    assert closure[1]["diagnostics"]["prediction_horizon_s"] >= (
        second_completion_horizon
    )
    second_closure_diagnostics = closure[1]["diagnostics"]
    predicted_delta = np.asarray(
        second_closure_diagnostics["predicted_intercept_center_base_m"],
        dtype=np.float64,
    ) - np.asarray(
        second_closure_diagnostics["object_center_base_m"],
        dtype=np.float64,
    )
    assert predicted_delta == pytest.approx(
        np.asarray(
            second_closure_diagnostics["planning_velocity_base_m_s"],
            dtype=np.float64,
        )
        * second_closure_diagnostics["prediction_horizon_s"],
        abs=1.0e-8,
    )
    completion_horizons = [
        item["diagnostics"]["grasp_completion_horizon_s"]
        for item in closure
    ]
    assert completion_horizons == sorted(
        completion_horizons, reverse=True
    )
    trace = policy.diagnostics_snapshot["committed_trace"]
    assert len(trace) == len(records)
    assert any(
        item.get("grasp_completion_horizon_applied_s")
        == pytest.approx(second_completion_horizon)
        for item in trace
    )


def test_v5_approach_orientation_conflict_holds_last_safe_target_and_replans(
    tmp_path, monkeypatch
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    valid = np.ones((8, 128), dtype=np.float32)
    proprio = np.zeros((8, 96), dtype=np.float32)
    points = _synthetic_history_for_base_trajectory(
        index=0,
        start_center_base=reference,
        velocity_base=np.zeros(3),
        proprio_history=proprio,
        contract=contract,
    )
    policy.action_for_sequence(1, points, valid, proprio)
    policy.commit_replay_proposal(1)

    committed_target = policy._committed_target_q.copy()
    committed_nominal = policy._committed_nominal_target_q.copy()
    committed_goal = policy._committed_goal_q.copy()
    committed_template = int(policy._committed_template_index)
    original_safe_target = policy._safe_target

    def force_transient_orientation_conflict(*, nominal_q, target_q, hand_target):
        if np.array_equal(nominal_q, committed_nominal) and np.array_equal(
            target_q, committed_target
        ):
            return original_safe_target(
                nominal_q=nominal_q,
                target_q=target_q,
                hand_target=hand_target,
            )
        raise _OrientationCorrectionConflict(
            required_rad=0.156779,
            limit_rad=config.max_orientation_correction_rad,
        )

    monkeypatch.setattr(policy, "_safe_target", force_transient_orientation_conflict)
    output = policy.action_for_sequence(2, points, valid, proprio)
    np.testing.assert_array_equal(
        output.exact_franka_target_q_rad,
        committed_target.astype(np.float32),
    )
    assert output.replay_frame_index == committed_template
    policy.commit_replay_proposal(2)
    diagnostics = policy.diagnostics_snapshot
    assert diagnostics["state"] == "adaptive_orientation_conflict_hold"
    assert diagnostics["orientation_conflict_hold"] is True
    assert diagnostics["orientation_conflict_required_rad"] == pytest.approx(
        0.156779
    )
    assert diagnostics["orientation_conflict_limit_rad"] == pytest.approx(
        0.12
    )
    assert diagnostics["target_step_rad"] == pytest.approx(0.0)
    np.testing.assert_array_equal(policy._committed_target_q, committed_target)
    np.testing.assert_array_equal(
        policy._committed_nominal_target_q, committed_nominal
    )
    np.testing.assert_array_equal(policy._committed_goal_q, committed_goal)
    assert policy._committed_adaptive_phase == "approach"
    assert diagnostics["committed_trace"][-1]["orientation_conflict_hold"] is True

    # The hold is not latched.  A fresh next observation can immediately
    # resume ordinary receding-horizon control once the correction is safe.
    monkeypatch.setattr(policy, "_safe_target", original_safe_target)
    resumed = policy.action_for_sequence(3, points, valid, proprio)
    assert resumed.replay_frame_index >= committed_template


def test_v5_closure_clearance_conflict_holds_arm_but_advances_hand(tmp_path, monkeypatch):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    catch = config.template_catch_index
    policy._committed_adaptive_phase = "closure"
    policy._committed_template_index = catch
    policy._committed_preshape_index = catch - 1
    policy._committed_closure_step = 0
    policy._committed_nominal_target_q = np.asarray(
        sequence.recorded_franka_target_q_rad[catch], dtype=np.float64
    ).copy()
    policy._committed_target_q = policy._committed_nominal_target_q.copy()
    policy._committed_correction = np.zeros(7, dtype=np.float64)
    policy._committed_goal_q = policy._reference_grasp_q.copy()
    policy._committed_tabletop_height_correction_m = 0.0
    policy._committed_live_fit_seen = True

    held_target = policy._committed_target_q.copy()
    original_safe_closure_target = policy._safe_closure_target
    safety_calls = 0

    def force_new_arm_target_below_table(
        *, nominal_q, target_q, hand_target, actual_fingertips_palm_m=None
    ):
        nonlocal safety_calls
        safety_calls += 1
        # Initial target plus ten ordinary correction-backoff attempts must
        # all fail.  Only the dedicated closure-arm hold may succeed.
        if safety_calls >= 12 and np.array_equal(target_q, held_target):
            return original_safe_closure_target(
                nominal_q=nominal_q,
                target_q=target_q,
                hand_target=hand_target,
                actual_fingertips_palm_m=actual_fingertips_palm_m,
            )
        raise _FingertipTableClearanceConflict(
            required_m=0.0142,
            limit_m=config.minimum_fingertip_clearance_m,
        )

    monkeypatch.setattr(
        policy, "_safe_closure_target", force_new_arm_target_below_table
    )
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    valid = np.ones((8, 128), dtype=np.float32)
    proprio = np.zeros((8, 96), dtype=np.float32)
    points = _synthetic_history_for_base_trajectory(
        index=0,
        start_center_base=reference,
        velocity_base=np.zeros(3),
        proprio_history=proprio,
        contract=contract,
    )
    output = policy.action_for_sequence(1, points, valid, proprio)
    np.testing.assert_array_equal(
        output.exact_franka_target_q_rad, held_target.astype(np.float32)
    )
    assert output.replay_frame_index == catch + 1
    np.testing.assert_array_equal(
        output.exact_rh56_angle_set_register_order,
        sequence.recorded_rh56_angle_set_register_order[catch + 1],
    )
    policy.commit_replay_proposal(1)
    diagnostics = policy.diagnostics_snapshot
    assert diagnostics["state"] == (
        "adaptive_closure_fingertip_clearance_hold"
    )
    assert diagnostics["fingertip_clearance_hold"] is True
    assert diagnostics["rejected_fingertip_clearance_m"] == pytest.approx(
        0.0142
    )
    assert diagnostics["fingertip_clearance_limit_m"] == pytest.approx(0.015)
    assert diagnostics["target_step_rad"] == pytest.approx(0.0)
    assert diagnostics["minimum_fingertip_clearance_m"] >= 0.015
    assert policy._committed_adaptive_phase == "closure"
    assert policy._committed_closure_step == 1


def test_v5_closure_finger_curl_raises_arm_instead_of_aborting(
    tmp_path, monkeypatch
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    catch = config.template_catch_index
    policy._committed_adaptive_phase = "closure"
    policy._committed_template_index = catch + 2
    policy._committed_preshape_index = catch - 1
    policy._committed_closure_step = 2
    policy._committed_nominal_target_q = np.asarray(
        sequence.recorded_franka_target_q_rad[catch + 2], dtype=np.float64
    ).copy()
    policy._committed_target_q = policy._committed_nominal_target_q.copy()
    policy._committed_correction = np.zeros(7, dtype=np.float64)
    policy._committed_goal_q = policy._reference_grasp_q.copy()
    policy._committed_tabletop_height_correction_m = 0.0
    policy._committed_live_fit_seen = True

    held_target = policy._committed_target_q.copy()
    held_palm = panda_T_base_policy_palm(
        held_target, contract.T_flange_policy_palm
    )
    safety_calls = 0

    def reject_until_pose_is_raised(
        *, nominal_q, target_q, hand_target, actual_fingertips_palm_m=None
    ):
        del nominal_q, hand_target, actual_fingertips_palm_m
        nonlocal safety_calls
        safety_calls += 1
        target = np.asarray(target_q, dtype=np.float64)
        palm = panda_T_base_policy_palm(
            target, contract.T_flange_policy_palm
        )
        raised = float(palm[2, 3] - held_palm[2, 3])
        if raised < 0.009:
            raise _FingertipTableClearanceConflict(
                required_m=0.006066,
                limit_m=config.minimum_fingertip_clearance_m,
            )
        return {
            "orientation_correction_rad": 0.0,
            "target_step_rad": float(np.max(np.abs(target - held_target))),
            "minimum_fingertip_clearance_m": (
                config.minimum_fingertip_clearance_m + 0.0005
            ),
            "commanded_hand_fingertip_clearance_m": (
                config.minimum_fingertip_clearance_m + 0.0005
            ),
            "closure_transition_fingertip_clearance_m": (
                config.minimum_fingertip_clearance_m + 0.001
            ),
            "palm_clearance_m": 0.18,
        }

    monkeypatch.setattr(
        policy, "_safe_closure_target", reject_until_pose_is_raised
    )
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    valid = np.ones((8, 128), dtype=np.float32)
    proprio = np.zeros((8, 96), dtype=np.float32)
    points = _synthetic_history_for_base_trajectory(
        index=0,
        start_center_base=reference,
        velocity_base=np.zeros(3),
        proprio_history=proprio,
        contract=contract,
    )
    output = policy.action_for_sequence(1, points, valid, proprio)
    raised_palm = panda_T_base_policy_palm(
        output.exact_franka_target_q_rad.astype(np.float64),
        contract.T_flange_policy_palm,
    )
    assert float(raised_palm[2, 3] - held_palm[2, 3]) >= 0.009
    assert output.replay_frame_index == catch + 3
    policy.commit_replay_proposal(1)
    diagnostics = policy.diagnostics_snapshot
    assert diagnostics["state"] == "adaptive_closure_clearance_lift"
    assert diagnostics["closure_clearance_lift_applied"] is True
    assert diagnostics["closure_clearance_lift_m"] == pytest.approx(0.009934)
    assert diagnostics["closure_clearance_before_lift_m"] == pytest.approx(
        0.006066
    )
    assert diagnostics["minimum_fingertip_clearance_m"] >= 0.015
    assert policy._committed_adaptive_phase == "closure"
    assert policy._committed_closure_step == 3
    assert safety_calls >= 13


def test_v5_approach_clearance_conflict_holds_safe_state_and_replans(
    tmp_path, monkeypatch
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    valid = np.ones((8, 128), dtype=np.float32)
    proprio = np.zeros((8, 96), dtype=np.float32)
    points = _synthetic_history_for_base_trajectory(
        index=0,
        start_center_base=reference,
        velocity_base=np.zeros(3),
        proprio_history=proprio,
        contract=contract,
    )
    policy.action_for_sequence(1, points, valid, proprio)
    policy.commit_replay_proposal(1)
    committed_target = policy._committed_target_q.copy()
    committed_template = int(policy._committed_template_index)
    original_safe_target = policy._safe_target

    def reject_new_approach_descent(*, nominal_q, target_q, hand_target):
        if np.array_equal(target_q, committed_target):
            return original_safe_target(
                nominal_q=nominal_q,
                target_q=target_q,
                hand_target=hand_target,
            )
        raise _FingertipTableClearanceConflict(
            required_m=-0.003183,
            limit_m=config.minimum_fingertip_clearance_m,
        )

    monkeypatch.setattr(policy, "_safe_target", reject_new_approach_descent)
    output = policy.action_for_sequence(2, points, valid, proprio)
    np.testing.assert_array_equal(
        output.exact_franka_target_q_rad,
        committed_target.astype(np.float32),
    )
    assert output.replay_frame_index == committed_template
    policy.commit_replay_proposal(2)
    diagnostics = policy.diagnostics_snapshot
    assert diagnostics["state"] == (
        "adaptive_approach_fingertip_clearance_hold"
    )
    assert diagnostics["approach_fingertip_clearance_hold"] is True
    assert diagnostics["fingertip_clearance_control_state_frozen"] is True
    assert diagnostics["rejected_fingertip_clearance_m"] == pytest.approx(
        -0.003183
    )
    assert diagnostics["minimum_fingertip_clearance_m"] >= 0.015
    assert policy._committed_adaptive_phase == "approach"

    monkeypatch.setattr(policy, "_safe_target", original_safe_target)
    resumed = policy.action_for_sequence(3, points, valid, proprio)
    assert resumed.replay_frame_index >= committed_template


def test_v5_closure_rejects_arm_pose_safe_only_for_future_curled_hand(tmp_path):
    """Regression for the measured 20260820-181832 table-contact stop.

    At the last committed closure tick the future curled-hand command appeared
    clear, while the still-open physical fingers were already below the
    commissioned table floor.  Closure safety must validate both geometries.
    """

    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    target_q = np.asarray(
        [
            -0.5876370072364807,
            0.5659258365631104,
            0.4757573902606964,
            -2.117982864379883,
            0.5592858791351318,
            3.2656137943267822,
            -0.13438650965690613,
        ],
        dtype=np.float64,
    )
    commanded_hand = np.asarray(
        [230, 71, 204, 207, 914, 4], dtype=np.int32
    )
    measured_hand = np.asarray(
        [294, 496, 675, 700, 981, 103], dtype=np.int32
    )
    measured_fingertips_palm = policy._fingertips.positions_base(
        angle_act_register_order=measured_hand,
        T_base_palm=np.eye(4, dtype=np.float64),
    )
    policy._committed_target_q = target_q.copy()
    commanded = policy._safe_target(
        nominal_q=target_q,
        target_q=target_q,
        hand_target=commanded_hand,
    )
    assert commanded["minimum_fingertip_clearance_m"] > (
        config.minimum_fingertip_clearance_m
    )
    with pytest.raises(_FingertipTableClearanceConflict) as captured:
        policy._safe_closure_target(
            nominal_q=target_q,
            target_q=target_q,
            hand_target=commanded_hand,
            actual_fingertips_palm_m=measured_fingertips_palm,
        )
    assert captured.value.required_m < config.minimum_fingertip_clearance_m


def test_v5_closure_orientation_conflict_allows_only_one_safe_hand_tick(
    tmp_path, monkeypatch
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    catch = config.template_catch_index
    policy._committed_adaptive_phase = "closure"
    policy._committed_template_index = catch
    policy._committed_preshape_index = catch - 1
    policy._committed_closure_step = 0
    policy._committed_nominal_target_q = policy._reference_approach_q.copy()
    policy._committed_target_q = policy._reference_approach_q.copy()
    policy._committed_correction = np.zeros(7, dtype=np.float64)
    policy._committed_goal_q = policy._reference_grasp_q.copy()
    policy._committed_tabletop_height_correction_m = 0.0
    policy._committed_live_fit_seen = True
    held_target = policy._committed_target_q.copy()
    safety_calls = 0

    def force_one_tick_orientation_hold(
        *, nominal_q, target_q, hand_target, actual_fingertips_palm_m=None
    ):
        del nominal_q, hand_target, actual_fingertips_palm_m
        nonlocal safety_calls
        safety_calls += 1
        if safety_calls == 12 and np.array_equal(target_q, held_target):
            return {
                "orientation_correction_rad": 0.110,
                "target_step_rad": 0.0,
                "minimum_fingertip_clearance_m": 0.020,
                "palm_clearance_m": 0.18,
            }
        raise _OrientationCorrectionConflict(
            required_rad=0.126805,
            limit_rad=config.max_orientation_correction_rad,
        )

    monkeypatch.setattr(
        policy, "_safe_closure_target", force_one_tick_orientation_hold
    )
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    valid = np.ones((8, 128), dtype=np.float32)
    proprio = np.zeros((8, 96), dtype=np.float32)
    points = _synthetic_history_for_base_trajectory(
        index=0,
        start_center_base=reference,
        velocity_base=np.zeros(3),
        proprio_history=proprio,
        contract=contract,
    )
    output = policy.action_for_sequence(1, points, valid, proprio)
    np.testing.assert_array_equal(
        output.exact_franka_target_q_rad, held_target.astype(np.float32)
    )
    assert output.replay_frame_index == catch + 1
    policy.commit_replay_proposal(1)
    diagnostics = policy.diagnostics_snapshot
    assert diagnostics["closure_orientation_conflict_hold"] is True
    assert diagnostics["orientation_conflict_required_rad"] == pytest.approx(
        0.126805
    )
    assert diagnostics["orientation_conflict_limit_rad"] == pytest.approx(0.12)
    with pytest.raises(_OrientationCorrectionConflict):
        policy.action_for_sequence(2, points, valid, proprio)


def test_v5_occluded_closure_completes_last_five_hand_ticks(tmp_path, monkeypatch):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, _contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    catch = config.template_catch_index
    end = config.closure_end_index
    policy._committed_adaptive_phase = "closure"
    policy._committed_template_index = end - 6
    policy._committed_closure_step = end - 6 - catch
    assert (
        policy.action_for_occluded_closure(
            1,
            observation_hold_reason=(
                "guarded_stale_palm_fail_closed:retained_pointcloud_frame=1"
            ),
        )
        is None
    )

    held = np.asarray(
        sequence.recorded_franka_target_q_rad[end - 5], dtype=np.float64
    ).copy()
    policy._committed_template_index = end - 5
    policy._committed_closure_step = end - 5 - catch
    policy._committed_nominal_target_q = held.copy()
    policy._committed_target_q = held.copy()
    policy._committed_correction = np.zeros(7, dtype=np.float64)
    policy._committed_goal_q = policy._reference_grasp_q.copy()
    policy._committed_tabletop_height_correction_m = 0.0
    policy._committed_live_fit_seen = True
    monkeypatch.setattr(
        policy,
        "_safe_target",
        lambda **_kwargs: {
            "orientation_correction_rad": 0.0,
            "target_step_rad": 0.0,
            "minimum_fingertip_clearance_m": 0.020,
            "palm_clearance_m": 0.18,
        },
    )

    for transaction_step in range(1, 6):
        output = policy.action_for_occluded_closure(
            transaction_step,
            observation_hold_reason=(
                "object_pointcloud_transient_invalid:"
                "status=invalid_motion_fallback_rejected"
                if transaction_step % 2
                else "guarded_stale_palm_fail_closed:"
                "retained_pointcloud_frame=1"
            ),
        )
        assert output is not None
        expected_index = end - 5 + transaction_step
        assert output.replay_frame_index == expected_index
        np.testing.assert_array_equal(
            output.exact_franka_target_q_rad, held.astype(np.float32)
        )
        np.testing.assert_array_equal(
            output.exact_rh56_angle_set_register_order,
            sequence.recorded_rh56_angle_set_register_order[expected_index],
        )
        policy.commit_replay_proposal(transaction_step)
        assert policy.diagnostics_snapshot["occluded_closure_hand_only"] is True
        assert policy.diagnostics_snapshot[
            "occluded_closure_maximum_ticks"
        ] == 5
        assert policy._committed_adaptive_phase == (
            "lift" if transaction_step == 5 else "closure"
        )

    assert policy._committed_adaptive_phase == "lift"
    assert policy._capture_q is not None
    assert policy._lift_q is not None
    assert (
        policy.action_for_occluded_closure(
            6,
            observation_hold_reason=(
                "guarded_stale_palm_fail_closed:retained_pointcloud_frame=1"
            ),
        )
        is None
    )

    sequence_id = 6
    while policy._committed_adaptive_phase == "lift":
        frozen = policy.action_for_frozen_post_closure(
            sequence_id,
            observation_hold_reason=(
                "object_pointcloud_transient_invalid:mask_px=0"
            ),
        )
        assert frozen is not None
        policy.commit_replay_proposal(sequence_id)
        sequence_id += 1
    assert policy._committed_adaptive_phase == "hold"
    final_hold = policy.action_for_frozen_post_closure(
        sequence_id,
        observation_hold_reason=(
            "guarded_stale_palm_fail_closed:retained_pointcloud_frame=1"
        ),
    )
    assert final_hold is not None
    policy.commit_replay_proposal(sequence_id)
    assert policy.replay_complete is True


def test_v5_degraded_fit_commit_completes_full_occluded_closure(
    tmp_path, monkeypatch
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, _contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    catch = config.template_catch_index
    end = config.closure_end_index
    held = policy._reference_approach_q.copy()
    policy._committed_adaptive_phase = "closure"
    policy._committed_template_index = catch
    policy._committed_preshape_index = catch - 1
    policy._committed_closure_step = 0
    policy._committed_nominal_target_q = held.copy()
    policy._committed_target_q = held.copy()
    policy._committed_correction = np.zeros(7, dtype=np.float64)
    policy._committed_goal_q = policy._reference_grasp_q.copy()
    policy._committed_tabletop_height_correction_m = 0.0
    policy._committed_live_fit_seen = True
    policy._committed_full_occluded_closure = True
    monkeypatch.setattr(
        policy,
        "_safe_closure_target",
        lambda **_kwargs: {
            "orientation_correction_rad": 0.0,
            "target_step_rad": 0.0,
            "minimum_fingertip_clearance_m": 0.020,
            "palm_clearance_m": 0.18,
        },
    )

    total = end - catch
    assert total > 5
    previous_target = held.astype(np.float32)
    for transaction_step in range(1, total + 1):
        output = policy.action_for_occluded_closure(
            transaction_step,
            observation_hold_reason=(
                "guarded_stale_palm_fail_closed:retained_pointcloud_frame=201"
            ),
        )
        assert output is not None
        assert output.replay_frame_index == catch + transaction_step
        assert float(
            np.max(
                np.abs(
                    output.exact_franka_target_q_rad - previous_target
                )
            )
        ) <= 0.025 + 1.0e-6
        previous_target = output.exact_franka_target_q_rad.copy()
        policy.commit_replay_proposal(transaction_step)
        diagnostics = policy.diagnostics_snapshot
        assert diagnostics["full_occluded_closure_committed"] is True
        assert diagnostics["occluded_closure_maximum_ticks"] == total
        assert diagnostics["arm_target_held"] is False
        assert diagnostics["frozen_intercept_closure_arm_advanced"] is True
        assert policy._committed_full_occluded_closure is True

    assert policy._committed_adaptive_phase == "lift"
    assert policy._capture_q is not None
    assert policy._lift_q is not None


def test_v5_current_volume_commit_finishes_occluded_closure_at_held_arm(
    tmp_path, monkeypatch
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, _contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    catch = config.template_catch_index
    end = config.closure_end_index
    held = policy._reference_approach_q.copy()
    policy._committed_adaptive_phase = "closure"
    policy._committed_template_index = catch
    policy._committed_preshape_index = catch - 1
    policy._committed_closure_step = 0
    policy._committed_nominal_target_q = held.copy()
    policy._committed_target_q = held.copy()
    policy._committed_correction = np.zeros(7, dtype=np.float64)
    policy._committed_goal_q = policy._reference_grasp_q.copy()
    policy._committed_tabletop_height_correction_m = 0.0
    policy._committed_live_fit_seen = True
    policy._committed_full_occluded_closure = True
    policy._committed_current_volume_capture = True
    monkeypatch.setattr(
        policy,
        "_safe_closure_target",
        lambda **_kwargs: {
            "orientation_correction_rad": 0.0,
            "target_step_rad": 0.0,
            "minimum_fingertip_clearance_m": 0.020,
            "palm_clearance_m": 0.18,
        },
    )

    for transaction_step in range(1, end - catch + 1):
        output = policy.action_for_occluded_closure(
            transaction_step,
            observation_hold_reason=(
                "guarded_stale_palm_fail_closed:"
                "retained_pointcloud_frame=227"
            ),
        )
        assert output is not None
        np.testing.assert_array_equal(
            output.exact_franka_target_q_rad, held.astype(np.float32)
        )
        policy.commit_replay_proposal(transaction_step)
        diagnostics = policy.diagnostics_snapshot
        assert diagnostics["current_volume_capture_committed"] is True
        assert diagnostics["arm_target_held"] is True
        assert diagnostics["frozen_intercept_closure_arm_advanced"] is False

    assert policy._committed_adaptive_phase == "lift"
    np.testing.assert_array_equal(policy._capture_q, held.astype(np.float32))
    assert policy._lift_q is not None


def test_v5_occluded_closure_preserves_table_clearance_with_small_lift(
    tmp_path, monkeypatch
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    catch = config.template_catch_index
    policy._committed_adaptive_phase = "closure"
    policy._committed_template_index = catch + 2
    policy._committed_closure_step = 2
    policy._committed_preshape_index = catch - 1
    policy._committed_nominal_target_q = policy._reference_approach_q.copy()
    policy._committed_target_q = policy._reference_approach_q.copy()
    policy._committed_goal_q = policy._reference_grasp_q.copy()
    policy._committed_full_occluded_closure = True
    policy._committed_current_volume_capture = True
    policy._committed_force_contact_capture = True
    policy._capture_q = policy._committed_target_q.copy()
    policy._lift_q = pose_preserving_joint_target(
        policy._capture_q,
        np.asarray([0.0, 0.0, config.grasp_lift_m]),
        T_flange_policy_palm=contract.T_flange_policy_palm,
        joint_limits_rad=contract.joint_limits_rad,
        joint_limit_margin_rad=config.joint_limit_margin_rad,
        damping=config.dls_damping,
        local_precision_refinement=True,
    )
    held_palm = panda_T_base_policy_palm(
        policy._committed_target_q, contract.T_flange_policy_palm
    )

    def require_lift(*, nominal_q, target_q, hand_target, **_kwargs):
        del nominal_q, hand_target
        palm = panda_T_base_policy_palm(
            np.asarray(target_q, dtype=np.float64),
            contract.T_flange_policy_palm,
        )
        if float(palm[2, 3] - held_palm[2, 3]) < 0.009:
            raise _FingertipTableClearanceConflict(
                required_m=0.006066,
                limit_m=config.minimum_fingertip_clearance_m,
            )
        return {
            "orientation_correction_rad": 0.0,
            "target_step_rad": float(
                np.max(
                    np.abs(
                        np.asarray(target_q, dtype=np.float64)
                        - policy._committed_target_q
                    )
                )
            ),
            "minimum_fingertip_clearance_m": 0.016,
            "commanded_hand_fingertip_clearance_m": 0.016,
            "closure_transition_fingertip_clearance_m": 0.017,
            "palm_clearance_m": 0.18,
        }

    monkeypatch.setattr(policy, "_safe_closure_target", require_lift)
    output = policy.action_for_occluded_closure(
        1,
        observation_hold_reason=(
            "guarded_stale_palm_fail_closed:requested_mode=guarded_v2"
        ),
    )
    assert output is not None
    assert output.replay_frame_index == catch + 3
    assert output.occluded_clearance_lift is True
    policy.commit_replay_proposal(1)
    diagnostics = policy.diagnostics_snapshot
    assert diagnostics["state"] == (
        "adaptive_occluded_closure_clearance_lift"
    )
    assert diagnostics["closure_clearance_lift_m"] == pytest.approx(0.009934)
    assert diagnostics["force_contact_capture_committed"] is True
    assert diagnostics["minimum_fingertip_clearance_m"] >= 0.015


def _capture_gate_at_measured_palm(policy, sequence, contract):
    measured_q = contract.q_home_rad.astype(np.float64)
    measured_palm = panda_T_base_policy_palm(
        measured_q, contract.T_flange_policy_palm
    )
    if policy._moving_capture_enabled:
        capture_center = (
            measured_palm[:3, 3]
            + measured_palm[:3, :3] @ policy._reference_object_in_palm_m
        )
        capture_center = np.asarray(capture_center, dtype=np.float64).copy()
        if not policy._mpc_capture_enabled:
            capture_center[2] -= policy.config.grasp_vertical_offset_m
        else:
            capture_center[2] -= policy.config.open_approach_lift_m
    else:
        capture_center = (
            measured_palm[:3, 3] + policy._reference_object_from_palm_m
        )
    diagnostics = {
        "object_center_base_m": capture_center
        - np.asarray([0.0, 0.060, 0.0]),
        "velocity_base_m_s": np.asarray([0.0, 0.300, 0.0]),
        "velocity_source": "live_fit",
    }
    return policy._adaptive_capture_gate(
        diagnostics=diagnostics,
        measured_q_rad=measured_q,
        goal_q_rad=policy._reference_grasp_q,
        hand_target=sequence.recorded_rh56_angle_set_register_order[53],
        preshape_ready=True,
    )


def test_v4_capture_uses_moving_hand_geometry_not_final_goal_arrival(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    moving, sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_high.zip"
    )
    legacy_sequence = replace(
        sequence,
        tabletop_online_planner=replace(
            sequence.tabletop_online_planner,
            version="tabletop_online_intercept_planner_v3",
        ),
    )
    checkpoint = load_checkpoint_safely(TABLETOP_CHECKPOINT.read_bytes())
    legacy = TransactionalTabletopOnlinePlannerPolicy(
        legacy_sequence,
        contract=contract,
        action_controller=ActionControllerParameters.from_metadata(
            checkpoint.metadata
        ),
        point_feature_dim=3,
        history_length=8,
        proprio_dim=96,
        selected_steps=legacy_sequence.action_count,
    )

    moving_accepted, moving_diagnostics = _capture_gate_at_measured_palm(
        moving, sequence, contract
    )
    legacy_accepted, legacy_diagnostics = _capture_gate_at_measured_palm(
        legacy, legacy_sequence, contract
    )
    assert moving_accepted is True
    assert moving_diagnostics["capture_arm_ready"] is False
    assert moving_diagnostics["capture_arm_error_rad"] > 0.80
    assert moving_diagnostics["capture_requires_final_goal_arrival"] is False
    assert moving_diagnostics["capture_current_geometry_safe"] is True
    assert legacy_accepted is False
    assert legacy_diagnostics["capture_requires_final_goal_arrival"] is True


def test_v4_capture_window_rotates_with_the_measured_wrist(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_low.zip"
    )
    hand = sequence.recorded_rh56_angle_set_register_order[53]
    centers = []
    for measured_q in (contract.q_home_rad, policy._reference_grasp_q):
        palm = panda_T_base_policy_palm(
            measured_q, contract.T_flange_policy_palm
        )
        expected = (
            palm[:3, 3]
            + palm[:3, :3] @ policy._reference_object_in_palm_m
        )
        expected = np.asarray(expected, dtype=np.float64).copy()
        expected[2] -= policy.config.grasp_vertical_offset_m
        accepted, diagnostics = policy._adaptive_capture_gate(
            diagnostics={
                "object_center_base_m": expected
                - np.asarray([0.0, 0.020, 0.0]),
                "velocity_base_m_s": np.asarray([0.0, 0.100, 0.0]),
                "velocity_source": "live_fit",
            },
            measured_q_rad=measured_q,
            goal_q_rad=policy._reference_grasp_q,
            hand_target=hand,
            preshape_ready=True,
        )
        assert accepted is True
        np.testing.assert_allclose(
            diagnostics["capture_center_base_m"], expected, atol=1.0e-10
        )
        centers.append(expected)
    assert np.linalg.norm(centers[1] - centers[0]) > 0.10


def test_v5_sphere_open_approach_is_lifted_without_biasing_z_gate(
    tmp_path,
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    goal, _desired = policy._goal_q_for_intercept(reference)
    goal_palm = panda_T_base_policy_palm(
        goal, contract.T_flange_policy_palm
    )
    unbiased_target = (
        reference
        - policy._reference_grasp_palm[:3, :3]
        @ policy._reference_object_in_palm_m
    )
    expected_target = np.asarray(unbiased_target, dtype=np.float64).copy()
    expected_target[2] += config.grasp_vertical_offset_m
    np.testing.assert_allclose(
        goal_palm[:3, 3], expected_target, atol=5.0e-4, rtol=0.0
    )
    assert config.grasp_vertical_offset_m == pytest.approx(-0.015)
    approach_q = policy._reference_approach_q
    palm = panda_T_base_policy_palm(
        approach_q, contract.T_flange_policy_palm
    )
    assert palm[2, 3] - policy._reference_grasp_palm[2, 3] == pytest.approx(
        config.open_approach_lift_m, abs=1.0e-3
    )

    expected = palm[:3, 3] + palm[:3, :3] @ policy._reference_object_in_palm_m
    expected = np.asarray(expected, dtype=np.float64).copy()
    expected[2] -= config.open_approach_lift_m
    accepted, diagnostics = policy._adaptive_capture_gate(
        diagnostics={
            "object_center_base_m": expected - np.asarray([0.0, 0.020, 0.0]),
            "velocity_base_m_s": np.asarray([0.0, 0.100, 0.0]),
            "velocity_source": "live_fit",
        },
        measured_q_rad=approach_q,
        goal_q_rad=goal,
        hand_target=sequence.recorded_rh56_angle_set_register_order[53],
        preshape_ready=True,
    )
    assert accepted is True
    assert diagnostics["capture_z_error_m"] == pytest.approx(0.0, abs=1.0e-9)


def test_v5_capture_gate_projects_arm_to_grasp_completion_pose(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    measured_q = policy._reference_approach_q.copy()
    goal_q = policy._reference_grasp_q.copy()
    horizon = policy._mpc_grasp_completion_horizon(
        phase="approach", closure_step=-1
    )
    reachable = config.arm_tracking_speed_rad_s * horizon
    projected_q = measured_q + np.clip(
        goal_q - measured_q, -reachable, reachable
    )
    projected_palm = panda_T_base_policy_palm(
        projected_q, contract.T_flange_policy_palm
    )
    projected_center = (
        projected_palm[:3, 3]
        + projected_palm[:3, :3] @ policy._reference_object_in_palm_m
    )
    accepted, diagnostics = policy._adaptive_capture_gate(
        diagnostics={
            "object_center_base_m": projected_center
            - np.asarray([0.0, 0.060, 0.0]),
            "velocity_base_m_s": np.asarray([0.0, 0.100, 0.0]),
            "velocity_source": "live_fit",
        },
        measured_q_rad=measured_q,
        goal_q_rad=goal_q,
        hand_target=(
            sequence.recorded_rh56_angle_set_register_order[
                config.template_catch_index - 1
            ]
        ),
        preshape_ready=True,
        arm_projection_horizon_s=horizon,
    )
    assert accepted is True
    assert diagnostics["capture_uses_projected_arm_pose"] is True
    assert diagnostics["capture_arm_projection_horizon_s"] == pytest.approx(
        horizon
    )
    np.testing.assert_allclose(
        diagnostics["capture_center_base_m"],
        projected_center,
        atol=1.0e-9,
    )
    assert diagnostics["capture_time_to_contact_s"] == pytest.approx(0.6)


def test_v5_capture_gate_rejects_mismatched_object_and_arm_horizons(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    policy._committed_live_fit_seen = True
    measured_q = policy._reference_approach_q.copy()
    goal_q = policy._reference_grasp_q.copy()
    horizon = policy._mpc_grasp_completion_horizon(
        phase="approach", closure_step=-1
    )
    reachable = config.arm_tracking_speed_rad_s * horizon
    projected_q = measured_q + np.clip(
        goal_q - measured_q, -reachable, reachable
    )
    projected_palm = panda_T_base_policy_palm(
        projected_q, contract.T_flange_policy_palm
    )
    projected_center = (
        projected_palm[:3, 3]
        + projected_palm[:3, :3] @ policy._reference_object_in_palm_m
    )
    accepted, diagnostics = policy._adaptive_capture_gate(
        diagnostics={
            # This recreates the 19:21 failure: the object reaches the
            # projected hand in 0.546 s, but the grasp-completion projection
            # is 0.82 s.  The former gate accepted these different instants.
            "object_center_base_m": projected_center
            - np.asarray([0.040, 0.106, 0.015]),
            "velocity_base_m_s": np.asarray([0.0, 0.194, 0.0]),
            "velocity_source": "live_fit",
            "fit_residual_m": 0.0137,
            "heading_cos": 0.99,
            "hold_last_intercept_goal": False,
        },
        measured_q_rad=measured_q,
        goal_q_rad=goal_q,
        hand_target=(
            sequence.recorded_rh56_angle_set_register_order[
                config.template_catch_index - 1
            ]
        ),
        preshape_ready=True,
        arm_projection_horizon_s=horizon,
    )
    assert diagnostics["capture_time_to_contact_s"] == pytest.approx(
        0.106 / 0.194
    )
    assert diagnostics["capture_temporal_alignment_used"] is True
    assert diagnostics["capture_along_track_m"] < -0.03
    assert accepted is False
    assert diagnostics["capture_temporal_alignment_committed"] is False


def _v5_measured_open_hand_capture_center(policy, contract, measured_q):
    palm = panda_T_base_policy_palm(
        measured_q, contract.T_flange_policy_palm
    )
    center = (
        palm[:3, 3]
        + palm[:3, :3] @ policy._reference_object_in_palm_m
    )
    center = np.asarray(center, dtype=np.float64).copy()
    center[2] -= policy.config.open_approach_lift_m
    return center


def test_v5_capture_gate_commits_fresh_object_inside_current_open_hand(
    tmp_path,
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    policy._committed_live_fit_seen = True
    measured_q = policy._reference_approach_q.copy()
    measured_center = _v5_measured_open_hand_capture_center(
        policy, contract, measured_q
    )
    horizon = policy._mpc_grasp_completion_horizon(
        phase="approach", closure_step=-1
    )
    accepted, diagnostics = policy._adaptive_capture_gate(
        diagnostics={
            # The ball is 50 mm down-track inside the measured open-hand
            # volume.  A full-closure-horizon extrapolation puts it well past
            # the held arm, so only the exact current-volume proof may admit.
            "object_center_base_m": (
                measured_center + np.asarray([0.0, 0.050, 0.0])
            ),
            "velocity_base_m_s": np.asarray([0.0, 0.120, 0.0]),
            "velocity_source": "live_fit",
        },
        measured_q_rad=measured_q,
        goal_q_rad=measured_q,
        hand_target=(
            sequence.recorded_rh56_angle_set_register_order[
                config.template_catch_index - 1
            ]
        ),
        preshape_ready=True,
        arm_projection_horizon_s=horizon,
    )
    assert accepted is True
    assert diagnostics["capture_current_volume_committed"] is True
    assert diagnostics["capture_temporal_alignment_committed"] is False
    assert diagnostics["capture_measured_along_track_m"] == pytest.approx(
        -0.050
    )
    assert diagnostics["capture_measured_cross_track_m"] == pytest.approx(
        0.0, abs=1.0e-9
    )


@pytest.mark.parametrize(
    ("offset", "velocity", "velocity_source", "preshape_ready"),
    (
        ((0.0, 0.061, 0.0), (0.0, 0.120, 0.0), "live_fit", True),
        ((0.061, 0.050, 0.0), (0.0, 0.120, 0.0), "live_fit", True),
        ((0.0, 0.050, 0.036), (0.0, 0.120, 0.0), "live_fit", True),
        ((0.0, 0.050, 0.0), (0.0, 0.120, 0.0), "scenario_prior", True),
        ((0.0, 0.050, 0.0), (0.0, 0.451, 0.0), "live_fit", True),
        ((0.0, 0.050, 0.0), (0.0, 0.120, 0.0), "live_fit", False),
    ),
)
def test_v5_current_open_hand_capture_keeps_every_boundary_authoritative(
    tmp_path, offset, velocity, velocity_source, preshape_ready
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    policy._committed_live_fit_seen = True
    measured_q = policy._reference_approach_q.copy()
    measured_center = _v5_measured_open_hand_capture_center(
        policy, contract, measured_q
    )
    accepted, diagnostics = policy._adaptive_capture_gate(
        diagnostics={
            "object_center_base_m": measured_center + np.asarray(offset),
            "velocity_base_m_s": np.asarray(velocity),
            "velocity_source": velocity_source,
        },
        measured_q_rad=measured_q,
        goal_q_rad=measured_q,
        hand_target=(
            sequence.recorded_rh56_angle_set_register_order[
                config.template_catch_index - 1
            ]
        ),
        preshape_ready=preshape_ready,
        arm_projection_horizon_s=policy._mpc_grasp_completion_horizon(
            phase="approach", closure_step=-1
        ),
    )
    assert accepted is False
    assert diagnostics["capture_current_volume_committed"] is False


def test_v5_fresh_near_hand_multifinger_contact_starts_frozen_closure(
    tmp_path, monkeypatch
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    catch = config.template_catch_index
    policy._committed_adaptive_phase = "approach"
    policy._committed_template_index = catch - 1
    policy._committed_preshape_index = catch - 1
    policy._committed_closure_step = -1
    policy._committed_live_fit_seen = True
    policy._committed_nominal_target_q = policy._reference_approach_q.copy()
    policy._committed_target_q = policy._reference_approach_q.copy()
    policy._committed_goal_q = policy._reference_grasp_q.copy()
    policy._last_diagnostics = {
        "state": "adaptive_predictive_approach",
        "adaptive_phase": "approach",
        "speed_m_s": 0.21,
        "capture_preshape_ready": True,
        "capture_current_geometry_safe": True,
        "capture_current_fingertip_clearance_m": 0.0195,
        "capture_measured_along_track_m": 0.027,
        "capture_measured_cross_track_m": 0.087,
        "capture_measured_z_error_m": 0.005,
    }

    def feedback(forces):
        return {
            "fresh": True,
            "forces_g": list(forces),
            "errors": [0, 0, 0, 0, 0, 0],
        }

    policy._last_committed_policy_sequence = 1
    policy.commit_rh56_feedback_after_dual_ack(
        1, feedback((13, 194, 5, 50, -5, -47))
    )
    assert policy._committed_force_contact_capture is False

    policy._last_committed_policy_sequence = 2
    policy._committed_diagnostics_trace = [{"sequence": 2}]
    policy.commit_rh56_feedback_after_dual_ack(
        2, feedback((293, 502, 289, 50, -5, -50))
    )
    assert policy._committed_adaptive_phase == "closure"
    assert policy._committed_full_occluded_closure is True
    assert policy._committed_current_volume_capture is True
    assert policy._committed_force_contact_capture is True
    diagnostics = policy.diagnostics_snapshot
    assert diagnostics["state"] == "adaptive_force_contact_capture_committed"
    assert diagnostics["force_contact_axes"] == [0, 1, 2]
    assert diagnostics["force_contact_planar_error_m"] == pytest.approx(
        np.hypot(0.027, 0.087)
    )

    monkeypatch.setattr(
        policy,
        "_safe_closure_target",
        lambda **_kwargs: {
            "orientation_correction_rad": 0.0,
            "target_step_rad": 0.0,
            "minimum_fingertip_clearance_m": 0.020,
            "commanded_hand_fingertip_clearance_m": 0.020,
            "closure_transition_fingertip_clearance_m": 0.020,
            "palm_clearance_m": 0.18,
        },
    )
    # The next observation can be fresh rather than occluded.  Contact is
    # committed after the preceding open-hand command, so closure_step=-1 is
    # a sealed transition marker and this fresh path must emit step zero
    # instead of rejecting it in the MPC horizon calculation.
    policy._committed_tabletop_height_correction_m = 0.0
    proprio = np.zeros((8, 96), dtype=np.float32)
    points = _synthetic_history_for_base_trajectory(
        index=0,
        start_center_base=np.asarray(
            config.reference_intercept_center_base_m, dtype=np.float64
        ),
        velocity_base=np.zeros(3),
        proprio_history=proprio,
        contract=contract,
    )
    fresh_output = policy.action_for_sequence(
        3, points, np.ones((8, 128), dtype=np.float32), proprio
    )
    assert fresh_output.replay_frame_index == catch
    assert policy._pending is not None
    assert policy._pending.closure_step == 0
    assert policy._pending.diagnostics["grasp_completion_horizon_s"] == (
        pytest.approx(
            (config.closure_end_index - catch + 1) * config.control_dt_s
            + config.contact_prediction_latency_s
        )
    )
    policy.discard_replay_proposal(3)

    output = policy.action_for_occluded_closure(
        3,
        observation_hold_reason=(
            "guarded_stale_palm_fail_closed:requested_mode=guarded_v2"
        ),
    )
    assert output is not None
    assert output.replay_frame_index == catch
    np.testing.assert_array_equal(
        output.exact_franka_target_q_rad,
        policy._committed_target_q.astype(np.float32),
    )
    policy.commit_replay_proposal(3)
    assert policy._committed_closure_step == 0
    assert policy._committed_force_contact_capture is True


@pytest.mark.parametrize(
    ("measured", "forces"),
    (
        ((0.027, 0.103, 0.005), (293, 502, 289, 50, -5, -50)),
        ((0.027, 0.087, 0.036), (293, 502, 289, 50, -5, -50)),
        ((0.027, 0.087, 0.005), (120, 502, 100, 50, -5, -50)),
    ),
)
def test_v5_force_contact_capture_keeps_geometry_and_multiaxis_bounds(
    tmp_path, measured, forces
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, _contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    policy._committed_adaptive_phase = "approach"
    policy._committed_template_index = config.template_catch_index - 1
    policy._committed_preshape_index = config.template_catch_index - 1
    policy._committed_closure_step = -1
    policy._committed_live_fit_seen = True
    policy._last_diagnostics = {
        "speed_m_s": 0.21,
        "capture_preshape_ready": True,
        "capture_current_geometry_safe": True,
        "capture_current_fingertip_clearance_m": 0.020,
        "capture_measured_along_track_m": measured[0],
        "capture_measured_cross_track_m": measured[1],
        "capture_measured_z_error_m": measured[2],
    }
    policy._last_committed_policy_sequence = 1
    baseline = {
        "fresh": True,
        "forces_g": [13, 194, 5, 50, -5, -47],
        "errors": [0] * 6,
    }
    policy.commit_rh56_feedback_after_dual_ack(1, baseline)
    policy._last_committed_policy_sequence = 2
    current = {
        "fresh": True,
        "forces_g": list(forces),
        "errors": [0] * 6,
    }
    policy.commit_rh56_feedback_after_dual_ack(2, current)
    assert policy._committed_force_contact_capture is False
    assert policy._committed_adaptive_phase == "approach"


@pytest.mark.parametrize(
    (
        "diagnostic_overrides",
        "cross_track_m",
        "expected_continuation",
        "expected_acceptance",
    ),
    (
        ({}, 0.062, True, True),
        ({"fit_residual_m": 0.026}, 0.062, False, False),
        ({"heading_cos": 0.94}, 0.062, False, False),
        ({"hold_last_intercept_goal": False}, 0.062, False, False),
        ({"velocity_source": "live_fit"}, 0.062, False, False),
        ({}, 0.066, True, False),
    ),
)
def test_v5_capture_gate_narrowly_continues_a_degraded_live_fit(
    tmp_path,
    diagnostic_overrides,
    cross_track_m,
    expected_continuation,
    expected_acceptance,
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    policy._committed_live_fit_seen = True
    measured_q = policy._reference_approach_q.copy()
    goal_q = policy._reference_grasp_q.copy()
    horizon = policy._mpc_grasp_completion_horizon(
        phase="approach", closure_step=-1
    )
    reachable = config.arm_tracking_speed_rad_s * horizon
    projected_q = measured_q + np.clip(
        goal_q - measured_q, -reachable, reachable
    )
    projected_palm = panda_T_base_policy_palm(
        projected_q, contract.T_flange_policy_palm
    )
    projected_center = (
        projected_palm[:3, 3]
        + projected_palm[:3, :3] @ policy._reference_object_in_palm_m
    )
    temporally_aligned_offset_m = horizon * 0.389 - 0.010
    diagnostics = {
        "object_center_base_m": projected_center
        - np.asarray(
            [cross_track_m, temporally_aligned_offset_m, 0.020]
        ),
        "velocity_base_m_s": np.asarray([0.0, 0.389, 0.0]),
        "velocity_source": "scenario_prior",
        "hold_last_intercept_goal": True,
        "fit_residual_m": 0.0212,
        "heading_cos": 0.997,
    }
    diagnostics.update(diagnostic_overrides)
    accepted, capture = policy._adaptive_capture_gate(
        diagnostics=diagnostics,
        measured_q_rad=measured_q,
        goal_q_rad=goal_q,
        hand_target=(
            sequence.recorded_rh56_angle_set_register_order[
                config.template_catch_index - 1
            ]
        ),
        preshape_ready=True,
        arm_projection_horizon_s=horizon,
    )
    assert accepted is expected_acceptance
    assert capture["capture_degraded_live_fit_continuation"] is (
        expected_continuation
    )
    assert capture["capture_cross_track_limit_m"] == pytest.approx(
        config.closure_cross_track_m
        + (0.005 if expected_continuation else 0.0)
    )
    assert capture["capture_temporal_alignment_committed"] is (
        expected_acceptance
    )


def test_v5_degraded_fit_capture_requires_prior_committed_live_fit(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    assert policy._committed_live_fit_seen is False
    measured_q = policy._reference_approach_q.copy()
    goal_q = policy._reference_grasp_q.copy()
    horizon = policy._mpc_grasp_completion_horizon(
        phase="approach", closure_step=-1
    )
    reachable = config.arm_tracking_speed_rad_s * horizon
    projected_q = measured_q + np.clip(
        goal_q - measured_q, -reachable, reachable
    )
    projected_palm = panda_T_base_policy_palm(
        projected_q, contract.T_flange_policy_palm
    )
    projected_center = (
        projected_palm[:3, 3]
        + projected_palm[:3, :3] @ policy._reference_object_in_palm_m
    )
    accepted, capture = policy._adaptive_capture_gate(
        diagnostics={
            "object_center_base_m": projected_center
            - np.asarray([0.062, 0.186, 0.020]),
            "velocity_base_m_s": np.asarray([0.0, 0.389, 0.0]),
            "velocity_source": "scenario_prior",
            "hold_last_intercept_goal": True,
            "fit_residual_m": 0.0212,
            "heading_cos": 0.997,
        },
        measured_q_rad=measured_q,
        goal_q_rad=goal_q,
        hand_target=(
            sequence.recorded_rh56_angle_set_register_order[
                config.template_catch_index - 1
            ]
        ),
        preshape_ready=True,
        arm_projection_horizon_s=horizon,
    )
    assert accepted is False
    assert capture["capture_degraded_live_fit_continuation"] is False
    assert capture["capture_cross_track_limit_m"] == pytest.approx(
        config.closure_cross_track_m
    )


def test_v5_closure_nominal_transition_respects_physical_arm_slew(
    tmp_path, monkeypatch
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    catch = config.template_catch_index
    policy._committed_adaptive_phase = "closure"
    policy._committed_template_index = catch
    policy._committed_closure_step = 0
    policy._committed_preshape_index = catch - 1
    previous_nominal = policy._reference_approach_q.copy()
    previous_nominal[0] += 0.080
    policy._committed_nominal_target_q = previous_nominal.copy()
    policy._committed_target_q = previous_nominal.copy()
    policy._committed_goal_q = policy._reference_grasp_q.copy()
    policy._committed_correction = np.zeros(7, dtype=np.float64)
    policy._committed_tabletop_height_correction_m = 0.0
    policy._committed_live_fit_seen = True
    monkeypatch.setattr(
        policy,
        "_safe_target",
        lambda **kwargs: {
            "orientation_correction_rad": 0.0,
            "target_step_rad": float(
                np.max(
                    np.abs(
                        np.asarray(kwargs["target_q"])
                        - policy._committed_target_q
                    )
                )
            ),
            "minimum_fingertip_clearance_m": 0.02,
            "palm_clearance_m": 0.18,
        },
    )
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    proprio = np.zeros((8, 96), dtype=np.float32)
    points = _synthetic_history_for_base_trajectory(
        index=0,
        start_center_base=reference,
        velocity_base=np.zeros(3),
        proprio_history=proprio,
        contract=contract,
    )
    policy.action_for_sequence(
        1, points, np.ones((8, 128), dtype=np.float32), proprio
    )
    assert policy._pending is not None
    limit = config.arm_tracking_speed_rad_s * config.control_dt_s
    assert np.max(
        np.abs(
            policy._pending.nominal_target_q_rad
            - previous_nominal
        )
    ) <= limit + 1.0e-9
    assert policy._pending.diagnostics["closure_nominal_slew_applied"] is True


def test_v5_closure_safety_backoff_retains_physical_slew_limit(
    tmp_path, monkeypatch
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    catch = config.template_catch_index
    policy._committed_adaptive_phase = "closure"
    policy._committed_template_index = catch
    policy._committed_preshape_index = catch - 1
    policy._committed_closure_step = 0
    policy._committed_nominal_target_q = np.asarray(
        sequence.recorded_franka_target_q_rad[catch], dtype=np.float64
    ).copy()
    policy._committed_target_q = policy._committed_nominal_target_q.copy()
    policy._committed_correction = np.zeros(7, dtype=np.float64)
    policy._committed_goal_q = policy._reference_grasp_q.copy()
    policy._committed_tabletop_height_correction_m = 0.0
    policy._committed_live_fit_seen = True
    previous_target = policy._committed_target_q.copy()
    safety_calls = 0

    def reject_initial_then_accept_backoff(
        *, nominal_q, target_q, hand_target, actual_fingertips_palm_m=None
    ):
        del nominal_q, hand_target, actual_fingertips_palm_m
        nonlocal safety_calls
        safety_calls += 1
        if safety_calls == 1:
            raise _FingertipTableClearanceConflict(
                required_m=0.014,
                limit_m=config.minimum_fingertip_clearance_m,
            )
        return {
            "target_step_rad": float(
                np.max(np.abs(np.asarray(target_q) - previous_target))
            ),
            "minimum_fingertip_clearance_m": 0.020,
            "palm_clearance_m": 0.18,
        }

    monkeypatch.setattr(
        policy, "_safe_closure_target", reject_initial_then_accept_backoff
    )
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    valid = np.ones((8, 128), dtype=np.float32)
    proprio = np.zeros((8, 96), dtype=np.float32)
    points = _synthetic_history_for_base_trajectory(
        index=0,
        start_center_base=reference,
        velocity_base=np.zeros(3),
        proprio_history=proprio,
        contract=contract,
    )
    output = policy.action_for_sequence(1, points, valid, proprio)
    limit = config.arm_tracking_speed_rad_s * config.control_dt_s
    assert safety_calls == 2
    assert np.max(
        np.abs(output.exact_franka_target_q_rad - previous_target)
    ) <= limit + 2.0e-7
    policy.commit_replay_proposal(1)
    assert policy.diagnostics_snapshot["target_step_rad"] <= limit + 2.0e-7


def test_v5_collision_replan_keeps_three_quarter_reachable_lead(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    collision, sequence, _contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    ordinary, _ordinary_sequence, _ordinary_contract = _online_policy(
        tmp_path / "sphere_posy_low.zip"
    )
    assert sequence.tabletop_online_planner is not None
    reference = np.asarray(
        sequence.tabletop_online_planner.reference_intercept_center_base_m,
        dtype=np.float64,
    )
    # Representative post-board position from the real sphere run.  V5 keeps
    # the 75% bounded lead while moving the grasp geometry into the sphere's
    # successful palm-frame capture volume.
    requested = reference + np.asarray([-0.120, 0.155, 0.0])

    collision_goal_calls = []
    collision_goal = collision._goal_q_for_intercept

    def counted_collision_goal(candidate):
        collision_goal_calls.append(np.asarray(candidate, dtype=np.float64).copy())
        return collision_goal(candidate)

    collision._goal_q_for_intercept = counted_collision_goal
    _goal, _desired, collision_diagnostics = (
        collision._adaptive_goal_for_intercept(requested)
    )
    _goal, _desired, ordinary_diagnostics = (
        ordinary._adaptive_goal_for_intercept(requested)
    )

    assert collision_diagnostics["joint_reachability_projection_scale"] == 0.75
    assert len(collision_goal_calls) == 1
    assert ordinary_diagnostics["joint_reachability_projection_scale"] in {
        0.5,
        1.0,
    }
    assert (
        collision_diagnostics["reserved_joint_correction_max_rad"]
        <= collision_diagnostics["reserved_joint_correction_limit_rad"]
    )


def test_v4_capture_fails_closed_when_current_hand_is_not_table_clear(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_high.zip"
    )

    class _UnsafeFingertips:
        @staticmethod
        def positions_base(*, angle_act_register_order, T_base_palm):
            del angle_act_register_order, T_base_palm
            return np.repeat(
                np.asarray([[0.0, 0.0, -1.0]], dtype=np.float64),
                5,
                axis=0,
            )

    policy._fingertips = _UnsafeFingertips()
    accepted, diagnostics = _capture_gate_at_measured_palm(
        policy, sequence, contract
    )
    assert accepted is False
    assert diagnostics["capture_current_geometry_safe"] is False


@pytest.mark.parametrize(
    ("bundle_name", "velocity_y_m_s", "nominal_arrival_tick"),
    (
        ("cylinder_posy_low.zip", 0.14, 30),
        ("cylinder_negy_low.zip", -0.14, 35),
        ("cylinder_posy_high.zip", 0.30, 30),
        ("cylinder_negy_high.zip", -0.30, 38),
    ),
)
def test_v4_tracks_continuously_moving_object_from_q_home_in_both_directions(
    tmp_path,
    bundle_name,
    velocity_y_m_s,
    nominal_arrival_tick,
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(tmp_path / bundle_name)
    config = sequence.tabletop_online_planner
    assert config is not None
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    velocity = np.asarray([0.0, velocity_y_m_s, 0.0], dtype=np.float64)
    start = reference - velocity * nominal_arrival_tick * config.control_dt_s

    def center_at(index):
        # Deliberately never stop the object at a fixed grasp pose.  This is
        # a dynamic interception test, not the former wait-for-arrival proxy.
        return start + velocity * max(0, index) * config.control_dt_s

    records = _run_adaptive_physical_rollout(
        policy=policy,
        sequence=sequence,
        contract=contract,
        center_at=center_at,
    )
    closure = next(
        item for item in records
        if item["diagnostics"]["adaptive_phase"] == "closure"
    )
    assert policy.replay_complete is True
    assert closure["runtime_index"] >= 17
    assert closure["diagnostics"]["capture_requires_final_goal_arrival"] is False
    assert closure["diagnostics"]["capture_current_geometry_safe"] is True
    assert closure["diagnostics"]["capture_gate_accepted"] is True
    assert closure["diagnostics"]["capture_cross_track_m"] <= (
        config.closure_cross_track_m + 1.0e-9
    )
    assert abs(closure["diagnostics"]["capture_z_error_m"]) <= (
        config.closure_z_tolerance_m + 1.0e-9
    )
    if abs(velocity_y_m_s) >= 0.20:
        assert closure["diagnostics"]["capture_arm_ready"] is False


def test_adaptive_v3_waits_for_physical_arm_then_closes_and_lifts(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_low.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    velocity = np.asarray([0.0, 0.14, 0.0], dtype=np.float64)
    arrival_tick = 48
    start = reference - velocity * arrival_tick * config.control_dt_s

    def center_at(index):
        bounded = int(np.clip(index, 0, arrival_tick))
        return start + velocity * bounded * config.control_dt_s

    records = _run_adaptive_physical_rollout(
        policy=policy,
        sequence=sequence,
        contract=contract,
        center_at=center_at,
    )
    phases = [item["diagnostics"]["adaptive_phase"] for item in records]
    assert "closure" in phases
    assert "lift" in phases
    assert phases[-1] == "hold"
    closure = next(
        item for item in records
        if item["diagnostics"]["adaptive_phase"] == "closure"
    )
    assert closure["diagnostics"]["capture_gate_accepted"] is True
    assert closure["diagnostics"]["capture_arm_ready"] is True
    assert closure["diagnostics"]["capture_time_to_contact_s"] <= (
        config.closure_trigger_ttc_s + 1.0e-9
    )
    assert max(
        float(item["diagnostics"]["target_step_rad"]) for item in records
    ) <= config.max_target_step_rad + 1.0e-9
    assert min(
        float(item["diagnostics"]["minimum_fingertip_clearance_m"])
        for item in records
    ) >= config.minimum_fingertip_clearance_m


def test_adaptive_v3_accepts_measured_0236_mps_without_joint_boundary_chase(
    tmp_path,
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_low.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    velocity = np.asarray([0.0, 0.236, 0.0], dtype=np.float64)
    arrival_tick = 44
    start = reference - velocity * arrival_tick * config.control_dt_s

    def center_at(index):
        bounded = int(np.clip(index, 0, arrival_tick))
        return start + velocity * bounded * config.control_dt_s

    records = _run_adaptive_physical_rollout(
        policy=policy,
        sequence=sequence,
        contract=contract,
        center_at=center_at,
    )
    closure = next(
        item for item in records
        if item["diagnostics"]["adaptive_phase"] == "closure"
    )
    assert closure["diagnostics"]["capture_gate_accepted"] is True
    assert any(
        item["diagnostics"].get("speed_outside_scenario_band") is True
        and item["diagnostics"].get("velocity_source") == "live_fit"
        for item in records
    )
    assert all(
        item["diagnostics"]["reserved_joint_correction_max_rad"]
        <= item["diagnostics"]["reserved_joint_correction_limit_rad"]
        + 1.0e-9
        for item in records
        if "reserved_joint_correction_max_rad" in item["diagnostics"]
    )


def test_adaptive_v4_rejects_physically_late_release_at_bounded_deadline(
    tmp_path,
):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_low.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    velocity = np.asarray([0.0, 0.236, 0.0], dtype=np.float64)
    arrival_tick = 14
    start = reference - velocity * arrival_tick * config.control_dt_s

    def center_at(index):
        return start + velocity * int(max(0, index)) * config.control_dt_s

    with pytest.raises(
        RuntimeError,
        match="capture opportunity was not established before the bounded lift deadline",
    ):
        _run_adaptive_physical_rollout(
            policy=policy,
            sequence=sequence,
            contract=contract,
            center_at=center_at,
        )
    assert policy._capture_q is None
    assert policy._lift_q is None


def test_adaptive_v3_refits_after_board_collision_and_reversal(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    policy, sequence, contract = _online_policy(
        tmp_path / "sphere_posy_board_collision.zip"
    )
    config = sequence.tabletop_online_planner
    assert config is not None and config.collision_replan is True
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    collision_tick = 10
    arrival_tick = 45
    before_velocity = np.asarray([0.0, 0.10, 0.0])
    after_velocity = np.asarray([0.0, -0.08, 0.0])
    collision_center = (
        reference - after_velocity * (arrival_tick - collision_tick) * 0.05
    )
    start = collision_center - before_velocity * collision_tick * 0.05

    def center_at(index):
        bounded = int(max(0, index))
        if bounded <= collision_tick:
            return start + before_velocity * bounded * 0.05
        post = min(bounded, arrival_tick) - collision_tick
        return collision_center + after_velocity * post * 0.05

    records = _run_adaptive_physical_rollout(
        policy=policy,
        sequence=sequence,
        contract=contract,
        center_at=center_at,
    )
    assert any(
        item["diagnostics"].get("velocity_base_m_s", [0.0, 0.0])[1]
        < -0.05
        and item["diagnostics"].get("velocity_source") == "live_fit"
        for item in records
    )
    closure = next(
        item for item in records
        if item["diagnostics"]["adaptive_phase"] == "closure"
    )
    assert closure["diagnostics"]["capture_gate_accepted"] is True


def test_adaptive_v3_discard_is_transactional(tmp_path):
    build_bundles(
        source_path=DEFAULT_SOURCE,
        deploy_bundle_path=DEFAULT_BUNDLE,
        output_directory=tmp_path,
    )
    staged, sequence, contract = _online_policy(
        tmp_path / "cylinder_posy_low.zip"
    )
    control, _, _ = _online_policy(tmp_path / "cylinder_posy_low.zip")
    config = sequence.tabletop_online_planner
    assert config is not None
    reference = np.asarray(
        config.reference_intercept_center_base_m, dtype=np.float64
    )
    proprio = np.zeros((8, 96), dtype=np.float32)
    valid = np.ones((8, 128), dtype=np.float32)

    def observation(y_shift):
        return _synthetic_history_for_base_center_function(
            index=0,
            center_at=lambda _index: reference
            + np.asarray([0.0, -0.20 + y_shift, 0.0]),
            proprio_history=proprio,
            contract=contract,
        )

    staged.action_for_sequence(1, observation(0.03), valid, proprio)
    staged.discard_replay_proposal(1)
    assert staged._committed_index == -1
    assert staged._committed_template_index == -1
    assert staged._committed_preshape_index == -1
    np.testing.assert_array_equal(
        staged._committed_target_q, contract.q_home_rad
    )

    retried = staged.action_for_sequence(1, observation(0.0), valid, proprio)
    expected = control.action_for_sequence(1, observation(0.0), valid, proprio)
    np.testing.assert_array_equal(
        retried.exact_franka_target_q_rad,
        expected.exact_franka_target_q_rad,
    )
    np.testing.assert_array_equal(
        retried.exact_rh56_angle_set_register_order,
        expected.exact_rh56_angle_set_register_order,
    )
