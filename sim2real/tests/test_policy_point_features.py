from pathlib import Path

import numpy as np
import pytest

from sim2real.deployment.bundle import CheckpointData, DeployBundle, load_checkpoint_safely
from sim2real.policy import (
    ActionControllerParameters,
    LEGACY_ACTION_CONTROLLER_CONTRACT_ID,
    QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
    RollingStudentPolicy,
)

BUNDLE = Path(__file__).resolve().parents[2] / "data/test_fixtures/sim2real/deploy.zip"
V61_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "data/checkpoints/thrown"
    / "thrown_v61_sixexpert_visualflight_perceptiondr025_cmp31p9_demo_candidate_20260815"
    / "thrown_v61_sixexpert_visualflight_perceptiondr025_cmp31p9_demo_candidate_20260815"
    / "checkpoint.pt"
)
V61_BUNDLE = V61_CHECKPOINT.parent
V61_ALIGNMENT = V61_BUNDLE.parent / "runtime_alignment/runtime_alignment"


def _legacy_schema_qd_g015_controller() -> dict[str, object]:
    return {
        "schema": "joint_target_action_adapter_v1",
        "control_dt_s": 0.05,
        "policy_frequency_hz": 20.0,
        "physics_hold_substeps": 6,
        "target_update_clock": "policy",
        "arm": {
            "dimensions": 7,
            "semantics": "incremental_joint_target",
            "incremental_reference": "shaper_q_d",
            "delta_scale_rad_per_policy_step": 0.15,
            "moving_average": 0.40,
            "effective_delta_scale_after_ema_rad_per_policy_step": 0.06,
            "max_target_delta_rad_per_policy_step": 0.0,
            "tracking_error_limit_rad": 0.0,
        },
        "hand": {
            "dimensions": 6,
            "semantics": "absolute_physical_motor_target",
            "moving_average": 0.737856,
            "max_target_delta_rad_per_policy_step": 0.30,
        },
    }


def _xyz_checkpoint() -> CheckpointData:
    base = load_checkpoint_safely(DeployBundle(BUNDLE).checkpoint_bytes())
    weights = {
        name: np.asarray(value).copy() for name, value in base.model_state_dict.items()
    }
    first = weights["point_encoder.0.weight"]
    weights["point_encoder.0.weight"] = np.concatenate(
        [first[:, :3], first[:, -1:]], axis=1
    )
    normalization = {
        name: np.asarray(value).copy() for name, value in base.normalization.items()
    }
    normalization["pointcloud_mean"] = normalization["pointcloud_mean"][..., :3]
    normalization["pointcloud_std"] = normalization["pointcloud_std"][..., :3]
    metadata = dict(base.metadata)
    metadata["point_features"] = "xyz"
    spec = dict(base.spec)
    spec["point_feature_dim"] = 3
    return CheckpointData(
        model_state_dict=weights,
        ppo_state_dict=base.ppo_state_dict,
        normalization=normalization,
        metadata=metadata,
        spec=spec,
        iteration=base.iteration,
    )


def test_xyz_checkpoint_runs_with_3d_point_history():
    policy = RollingStudentPolicy(_xyz_checkpoint())
    assert policy.point_feature_mode == "xyz"
    assert policy.point_feature_dim == 3
    assert policy.expected_model_parameter_count == 2_521_808
    points = np.zeros((4, 128, 3), dtype=np.float32)
    points[..., 2] = 0.1
    output = policy.act(
        points,
        np.ones((4, 128), dtype=np.float32),
        np.zeros((4, 67), dtype=np.float32),
    )
    assert output.action13.shape == (13,)
    assert np.all(np.isfinite(output.action13))


def test_generic_legacy_h16_xyz96_history_is_supported():
    checkpoint = load_checkpoint_safely(V61_CHECKPOINT.read_bytes())
    weights = {
        name: np.asarray(value).copy()
        for name, value in checkpoint.model_state_dict.items()
        if not name.startswith("analytic_future_contract_action_adapter.")
    }
    generic = CheckpointData(
        model_state_dict=weights,
        ppo_state_dict=checkpoint.ppo_state_dict,
        normalization=checkpoint.normalization,
        metadata=checkpoint.metadata,
        spec={
            **checkpoint.spec,
            "analytic_future_contract_action_adapter_enabled": False,
        },
        iteration=checkpoint.iteration,
    )
    policy = RollingStudentPolicy(generic)
    assert (policy.history_length, policy.proprio_dim) == (16, 96)
    output = policy.act(
        np.zeros((16, 128, 3), dtype=np.float32),
        np.ones((16, 128), dtype=np.float32),
        np.zeros((16, 96), dtype=np.float32),
    )
    assert output.action13.shape == (13,)
    assert np.all(np.isfinite(output.action13))


def test_v61_analytic_action_adapter_runs_with_frozen_exact_contract():
    checkpoint = load_checkpoint_safely(V61_CHECKPOINT.read_bytes())
    policy = RollingStudentPolicy(checkpoint)
    assert policy.analytic_future_contract == (
        "thrown_visual_ballistic_17d_20hz_v2"
    )
    assert policy.expected_model_parameter_count == 2_705_387

    points = np.zeros((16, 128, 3), dtype=np.float32)
    points[..., 2] = 0.1
    proprio = np.zeros((16, 96), dtype=np.float32)
    proprio[:, 29] = 1.0
    output = policy.act(
        points,
        np.ones((16, 128), dtype=np.float32),
        proprio,
    )
    assert output.action13.shape == (13,)
    assert np.all(np.isfinite(output.action13))
    assert np.max(np.abs(output.action13)) <= 1.0


@pytest.mark.parametrize(
    ("relative_checkpoint", "expected_contract", "expected_action"),
    [
        (
            "checkpoint.pt",
            "thrown_visual_ballistic_17d_20hz_v2",
            [
                1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
                0.28633296,
                -0.7848786,
                0.7365228,
                1.0,
                0.61643726,
                -1.0,
                0.10370347,
                0.42912298,
            ],
        ),
        (
            "checkpoints/base/side_negative_y.pt",
            "thrown_v35_ballistic_17d_v1",
            [
                1.0,
                0.41835642,
                0.90900505,
                0.44082576,
                0.4276895,
                1.0,
                -1.0,
                1.0,
                1.0,
                0.28046906,
                -1.0,
                0.10350168,
                0.9669457,
            ],
        ),
    ],
)
def test_v61_full_policy_action_matches_pytorch_reference(
    relative_checkpoint: str,
    expected_contract: str,
    expected_action: list[float],
):
    payload = np.load(V61_ALIGNMENT / "golden_vectors.npz", allow_pickle=False)
    checkpoint = load_checkpoint_safely((V61_BUNDLE / relative_checkpoint).read_bytes())
    policy = RollingStudentPolicy(checkpoint)
    assert policy.analytic_future_contract == expected_contract

    output = policy.act(
        payload["pointcloud_seq"][0],
        payload["valid_seq"][0],
        payload["proprio_seq"][0],
    )
    np.testing.assert_allclose(
        output.action13,
        np.asarray(expected_action, dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_point_feature_metadata_must_match_dimension():
    checkpoint = _xyz_checkpoint()
    broken = CheckpointData(
        model_state_dict=checkpoint.model_state_dict,
        ppo_state_dict=checkpoint.ppo_state_dict,
        normalization=checkpoint.normalization,
        metadata={**checkpoint.metadata, "point_features": "xyzrgb"},
        spec=checkpoint.spec,
        iteration=checkpoint.iteration,
    )
    with pytest.raises(ValueError, match="metadata mismatch for point_features"):
        RollingStudentPolicy(broken)


def test_legacy_schema_exact_qd_g015_signature_selects_qd_contract():
    controller = _legacy_schema_qd_g015_controller()
    parameters = ActionControllerParameters.from_metadata(
        {"action_controller": controller}
    )

    assert parameters.contract_id == QD_G015_ACTION_CONTROLLER_CONTRACT_ID
    assert parameters.arm_raw_gain_rad == pytest.approx(0.15)
    assert parameters.arm_target_filter_alpha == pytest.approx(0.40)
    assert parameters.maximum_arm_target_step_rad == 0.0
    assert parameters.initial_previous_action13 == (0.0,) * 13


def test_legacy_schema_current_shaper_qd_alias_selects_same_contract():
    controller = _legacy_schema_qd_g015_controller()
    arm = controller["arm"]
    assert isinstance(arm, dict)
    arm["incremental_reference"] = "current_shaper_q_d"

    parameters = ActionControllerParameters.from_metadata(
        {"action_controller": controller}
    )
    assert parameters.contract_id == QD_G015_ACTION_CONTROLLER_CONTRACT_ID


@pytest.mark.parametrize(
    ("field", "nearby_value"),
    [
        ("incremental_reference", "previous_target"),
        ("delta_scale_rad_per_policy_step", 0.150001),
        ("moving_average", 0.400001),
        (
            "effective_delta_scale_after_ema_rad_per_policy_step",
            0.060001,
        ),
        ("max_target_delta_rad_per_policy_step", 0.000001),
        ("tracking_error_limit_rad", 0.000001),
    ],
)
def test_legacy_schema_near_match_does_not_select_qd_contract(
    field: str, nearby_value: object
):
    controller = _legacy_schema_qd_g015_controller()
    arm = controller["arm"]
    assert isinstance(arm, dict)
    arm[field] = nearby_value

    assert (
        ActionControllerParameters._metadata_contract_id(
            {"action_controller": controller}, controller
        )
        == LEGACY_ACTION_CONTROLLER_CONTRACT_ID
    )


def test_original_legacy_action_controller_signature_remains_legacy():
    controller = _legacy_schema_qd_g015_controller()
    arm = controller["arm"]
    assert isinstance(arm, dict)
    arm.pop("incremental_reference")
    arm.pop("effective_delta_scale_after_ema_rad_per_policy_step")
    arm["delta_scale_rad_per_policy_step"] = 0.045
    arm["max_target_delta_rad_per_policy_step"] = 0.045
    arm["tracking_error_limit_rad"] = 0.05

    parameters = ActionControllerParameters.from_metadata(
        {"action_controller": controller}
    )
    assert parameters.contract_id == LEGACY_ACTION_CONTROLLER_CONTRACT_ID
    assert parameters.arm_raw_gain_rad == pytest.approx(0.045)
    assert parameters.arm_target_filter_alpha == pytest.approx(0.40)
    assert parameters.maximum_arm_target_step_rad == pytest.approx(0.045)


def test_qd_g015_v76_accepts_eight_frame_67d_and_zero_action_reset():
    checkpoint = _xyz_checkpoint()
    qd_checkpoint = CheckpointData(
        model_state_dict=checkpoint.model_state_dict,
        ppo_state_dict=checkpoint.ppo_state_dict,
        normalization=checkpoint.normalization,
        metadata={
            **checkpoint.metadata,
            "controller_contract": {
                "schema": QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
            },
        },
        spec={**checkpoint.spec, "history": 8, "proprio_dim": 67},
        iteration=checkpoint.iteration,
    )

    policy = RollingStudentPolicy(qd_checkpoint)

    assert policy.history_length == 8
    assert policy.proprio_dim == 67
    assert policy.action_controller.contract_id == QD_G015_ACTION_CONTROLLER_CONTRACT_ID
    assert policy.action_controller.arm_raw_gain_rad == pytest.approx(0.15)
    assert policy.action_controller.arm_target_filter_alpha == pytest.approx(0.40)
    assert policy.action_controller.maximum_arm_target_step_rad == 0.0
    np.testing.assert_array_equal(
        policy.initial_previous_action13,
        np.zeros(13, dtype=np.float32),
    )
    assert not policy.initial_previous_action13.flags.writeable

    output = policy.act(
        np.zeros((8, 128, 3), dtype=np.float32),
        np.ones((8, 128), dtype=np.float32),
        np.zeros((8, 67), dtype=np.float32),
    )
    assert output.action13.shape == (13,)
