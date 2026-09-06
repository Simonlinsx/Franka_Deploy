import io
import copy
from pathlib import Path
import zipfile

import numpy as np
import pytest

from sim2real.deployment.bundle import (
    CheckpointData,
    DeployBundle,
    load_checkpoint_safely,
)
from sim2real.deployment.safety import audit_hardware_readiness
from sim2real.policy import (
    QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
    RollingStudentPolicy,
)
from sim2real.deployment.verify import (
    _validate_qd_g015_checkpoint_contract,
    verify_v94_bundle,
    verify_v94_checkpoint_payload,
    verify_v94_checkpoint_override,
)

BUNDLE = Path(__file__).resolve().parents[2] / "data/test_fixtures/sim2real/deploy.zip"
PRETRAIN_CHECKPOINT = Path(__file__).resolve().parents[2] / (
    "data/test_fixtures/student_pretrain_epoch_0001.pt"
)
QD_G015_V75_CHECKPOINT = Path(__file__).resolve().parents[2] / (
    "data/checkpoints/q_d/shaper/student_pretrain_epoch_0050.pt"
)
LEGACY_V258_CHECKPOINT = Path(__file__).resolve().parents[2] / (
    "data/test_fixtures/"
    "inspire_v65_sphere_dynamic_fulldr_sft_ep150_preliminary_20260803/checkpoint.pt"
)


def test_bundle_hashes_checkpoint_and_golden_replay_pass():
    report = verify_v94_bundle(BUNDLE)
    assert report.checked_hashes == 16
    assert report.model_parameters == 2_522_192
    assert report.point_feature_mode == "xyzrgb"
    assert report.point_feature_dim == 6
    assert report.initial_action_max_abs_error < 2.0e-6
    assert report.closed_cpu_replay_max_abs_error < 2.0e-6
    assert report.closed_gpu_action_max_abs_error < 3.5e-4
    assert report.initial_target_max_abs_error == 0.0
    assert report.closed_target_max_abs_error == 0.0
    assert report.hardware_writes is False


def test_bundle_loader_rejects_missing_member():
    bundle = DeployBundle(BUNDLE)
    with pytest.raises(FileNotFoundError, match="missing"):
        bundle.read_bytes("not-present.bin")


def test_checkpoint_loader_rejects_unlisted_pickle_global():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("bad/data.pkl", b"cos\nsystem\n.")
    with pytest.raises(ValueError, match="unsafe checkpoint"):
        load_checkpoint_safely(stream.getvalue())


def test_student_pretrain_checkpoint_is_safe_and_v94_io_compatible():
    assert PRETRAIN_CHECKPOINT.is_file()
    checkpoint = load_checkpoint_safely(PRETRAIN_CHECKPOINT.read_bytes())
    assert checkpoint.ppo_state_dict == {}
    assert checkpoint.iteration == 1
    assert len(checkpoint.model_state_dict) == 52

    report = verify_v94_checkpoint_override(BUNDLE, PRETRAIN_CHECKPOINT)
    assert report.contract_compatible is True
    assert report.checkpoint_format == "student_pretrain"
    assert report.model_parameters == 2_522_192
    assert report.point_feature_mode == "xyzrgb"
    assert report.point_feature_dim == 6
    assert report.control_dt_s == pytest.approx(1.0 / 60.0)
    assert report.initial_observation_actions == 12
    assert report.closed_observation_actions == 11
    assert report.initial_action_max_abs <= 1.0
    assert report.closed_action_max_abs <= 1.0
    assert report.bundle_golden_action_equality_applies is False
    assert report.hardware_writes is False


def test_60hz_checkpoint_is_rejected_for_20hz_runtime_mode():
    payload = PRETRAIN_CHECKPOINT.read_bytes()
    with pytest.raises(ValueError, match="selected runtime mode"):
        verify_v94_checkpoint_payload(
            BUNDLE,
            payload,
            expected_control_dt_s=0.05,
        )


def test_qd_g015_v75_checkpoint_passes_contract_aware_20hz_smoke():
    assert QD_G015_V75_CHECKPOINT.is_file()
    report = verify_v94_checkpoint_override(
        BUNDLE,
        QD_G015_V75_CHECKPOINT,
        expected_control_dt_s=0.05,
    )

    assert report.contract_compatible is True
    assert report.checkpoint_format == "student_pretrain"
    assert report.checkpoint_iteration_or_epoch == 50
    assert report.model_parameters == 2_648_030
    assert report.point_feature_mode == "xyz"
    assert report.point_feature_dim == 3
    assert report.control_dt_s == pytest.approx(0.05)
    assert report.initial_observation_actions == 1
    assert report.closed_observation_actions == 1
    assert report.initial_action_max_abs <= 1.0
    assert report.closed_action_max_abs <= 1.0
    assert report.hardware_writes is False


def test_legacy_v258_checkpoint_branch_remains_compatible():
    assert LEGACY_V258_CHECKPOINT.is_file()
    report = verify_v94_checkpoint_override(
        BUNDLE,
        LEGACY_V258_CHECKPOINT,
        expected_control_dt_s=0.05,
    )
    assert report.contract_compatible is True
    assert report.model_parameters == 2_648_030
    assert report.point_feature_dim == 3
    assert report.initial_observation_actions == 1
    assert report.closed_observation_actions == 1


def test_qd_g015_verifier_rejects_reintroduced_explicit_target_delta():
    checkpoint = load_checkpoint_safely(QD_G015_V75_CHECKPOINT.read_bytes())
    policy = RollingStudentPolicy(checkpoint)
    assert (
        policy.action_controller.contract_id
        == QD_G015_ACTION_CONTROLLER_CONTRACT_ID
    )
    metadata = copy.deepcopy(checkpoint.metadata)
    metadata["action_controller"]["arm"][
        "max_target_delta_rad_per_policy_step"
    ] = 0.045
    mutated = CheckpointData(
        model_state_dict=checkpoint.model_state_dict,
        ppo_state_dict=checkpoint.ppo_state_dict,
        normalization=checkpoint.normalization,
        metadata=metadata,
        spec=checkpoint.spec,
        iteration=checkpoint.iteration,
    )

    with pytest.raises(ValueError, match="explicit target delta limiter"):
        _validate_qd_g015_checkpoint_contract(mutated, policy)


def test_hardware_readiness_fails_closed_with_concrete_blockers():
    readiness = audit_hardware_readiness()
    assert not readiness.ready
    assert readiness.offline_alignment_passed
    assert readiness.policy_nominal_max_velocity_rad_s == pytest.approx(0.18)
    assert readiness.commissioned_max_velocity_rad_s == pytest.approx(0.05)
    assert any(
        "rh56_six_axis_policy_motion_commissioned" in item
        for item in readiness.blockers
    )
    assert any("hold_gate" in item for item in readiness.blockers)
    assert any(
        "external_object_mask_depth_cleaning" in item for item in readiness.blockers
    )
    assert any("policy_reset_scene_height" in item for item in readiness.blockers)
    assert any("live_observation_action_replay" in item for item in readiness.blockers)
    assert any(
        "default-q motion is not authorized" in item for item in readiness.blockers
    )
    assert (
        np.max(np.abs(readiness.simulation_home_delta_from_recorded_real_rad)) < 1.0e-6
    )
