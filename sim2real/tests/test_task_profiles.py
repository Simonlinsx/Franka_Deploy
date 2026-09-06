from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim2real.perception import (
    _request_from_args as perception_request_from_args,
    build_parser as build_perception_parser,
)

from sim2real.deployment.bundle import DeployBundle
from sim2real.deployment.runner import (
    DEFAULT_BUNDLE,
    SupervisedV94RunError,
    build_deployment_request,
    build_parser,
)
from sim2real.observation.camera_profile import (
    RuntimeCameraProfileError,
    resolve_runtime_camera_contract,
    resolve_runtime_task_contract,
    task_profile_allows_robot_execution,
    task_profile_maximum_supervised_execute_steps,
    task_profile_policy_rgbd_resolution,
    task_profile_rollout_trigger,
)
from sim2real.tasks.launcher import (
    TaskLauncherError,
    build_task_command,
    main as task_launcher_main,
    materialize_task_config,
)
from sim2real.contracts.v94 import V94Contract
from sim2real.observation.model import PolicyRGBDResolutionAdapter


def _bundle_contract() -> V94Contract:
    return V94Contract.from_bundle(DeployBundle(DEFAULT_BUNDLE))


@pytest.mark.parametrize("task_name", ("tabletop", "thrown_object"))
def test_task_specific_help_is_a_successful_no_op(task_name, capsys) -> None:
    assert task_launcher_main([task_name, "--help"]) == 0
    assert f"sim2real.tasks {task_name}" in capsys.readouterr().out


def test_all_task_profiles_pin_policy_input_to_424x240() -> None:
    for task in (
        "tabletop",
        "thrown_object",
        "thrown_object_v60",
        "thrown_object_v61",
    ):
        config, metadata = materialize_task_config(task)
        assert config.is_file()
        assert metadata["policy_rgbd_resolution"] == "424x240"
        assert task_profile_policy_rgbd_resolution(config) == "424x240"


def test_tabletop_uses_exact_2x_policy_decimation() -> None:
    config, _metadata = materialize_task_config("tabletop")
    config_document = yaml.safe_load(config.read_text(encoding="utf-8"))
    contract = resolve_runtime_camera_contract(_bundle_contract(), config)
    adapter = PolicyRGBDResolutionAdapter(
        camera_K=contract.camera_K,
        source_image_size=(contract.camera_width, contract.camera_height),
        target_image_size=(424, 240),
    )

    assert contract.camera_serial == "337322072188"
    assert (contract.camera_width, contract.camera_height) == (848, 480)
    assert contract.camera_rate_hz == 30.0
    assert contract.depth_range_m == pytest.approx((0.25, 2.0))
    assert config_document["camera"][
        "max_color_depth_timestamp_skew_s"
    ] == pytest.approx(0.012)
    assert adapter.stride == 2
    np.testing.assert_allclose(adapter.camera_K, contract.camera_K / np.asarray(
        [[2.0, 1.0, 2.0], [1.0, 2.0, 2.0], [1.0, 1.0, 1.0]]
    ))


def test_thrown_task_uses_native_424x240_60hz_and_scaled_calibration() -> None:
    config, metadata = materialize_task_config("thrown_object")
    config_document = yaml.safe_load(config.read_text(encoding="utf-8"))
    contract = resolve_runtime_camera_contract(_bundle_contract(), config)
    adapter = PolicyRGBDResolutionAdapter(
        camera_K=contract.camera_K,
        source_image_size=(contract.camera_width, contract.camera_height),
        target_image_size=(424, 240),
    )

    assert metadata["commissioning_status"] == "accepted"
    assert metadata["robot_execution_enabled"] is True
    assert metadata["maximum_supervised_execute_steps"] == 20
    assert contract.camera_serial == "342222071785"
    assert (contract.camera_width, contract.camera_height) == (424, 240)
    assert contract.camera_rate_hz == 60.0
    assert contract.calibration_id == "eye-to-hand-5e3620b789d775b9"
    assert contract.depth_range_m == (0.25, 1.65)
    assert config_document["online_sam2"][
        "policy_current_semantic_mask_enabled"
    ] is True
    assert config_document["pointcloud"]["temporal_fallback"] == (
        "motion_compensated"
    )
    assert config_document["pointcloud"]["temporal_fallback_max_stale_s"] == (
        pytest.approx(0.25)
    )
    assert config_document["pointcloud"]["temporal_fallback_max_stale_steps"] == 5
    assert adapter.stride == 1
    trigger = task_profile_rollout_trigger(config)
    assert trigger is not None
    assert trigger.mode == "object_motion_or_entry"
    assert trigger.stable_max_centroid_speed_px_s == pytest.approx(80.0)
    assert trigger.trigger_min_centroid_speed_px_s == pytest.approx(120.0)
    tabletop_trigger = task_profile_rollout_trigger(
        materialize_task_config("tabletop")[0]
    )
    assert tabletop_trigger is not None
    assert tabletop_trigger.mode == "object_motion_only"
    assert tabletop_trigger.allow_absent_entry is False
    assert tabletop_trigger.display_name == "Object motion trigger"
    assert tabletop_trigger.stable_max_centroid_speed_px_s == pytest.approx(10.0)
    assert tabletop_trigger.trigger_min_centroid_speed_px_s == pytest.approx(25.0)
    assert tabletop_trigger.trigger_min_displacement_px == pytest.approx(0.3)
    release_gate = tabletop_trigger.tabletop_release_height_gate
    assert release_gate is None
    np.testing.assert_allclose(
        adapter.camera_K,
        np.asarray(
            [
                [302.3836364746094, 0.0, 210.95738220214844],
                [0.0, 302.2502746582031, 123.3145980834961],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def test_v60_thrown_rollout_uses_its_own_reset_camera_and_episode_cap() -> None:
    config, metadata = materialize_task_config("thrown_object_v60")
    camera = resolve_runtime_camera_contract(
        _bundle_contract().with_runtime_policy_rate_hz(20), config
    )
    resolved = resolve_runtime_task_contract(
        _bundle_contract().with_runtime_policy_rate_hz(20), config
    )

    assert metadata["commissioning_status"] == "accepted"
    assert metadata["robot_execution_enabled"] is True
    assert metadata["maximum_supervised_execute_steps"] == 72
    assert task_profile_allows_robot_execution(config) is True
    assert camera.camera_serial == "342222071785"
    assert (camera.camera_width, camera.camera_height) == (424, 240)
    assert camera.camera_rate_hz == pytest.approx(60.0)
    assert camera.depth_range_m == pytest.approx((0.25, 1.50))
    np.testing.assert_allclose(
        resolved.q_home_rad,
        [
            -2.544386888,
            -1.411713805,
            -0.049395346,
            -2.337750525,
            1.570796377,
            0.464266852,
            2.7,
        ],
        atol=1.0e-7,
        rtol=0.0,
    )
    assert resolved.joint_limits_rad[5, 0] == pytest.approx(0.4398)
    assert resolved.q_home_rad[5] > resolved.joint_limits_rad[5, 0] + 0.02
    command, _config, _metadata = build_task_command(
        "thrown_object_v60",
        "deploy",
        [
            "--checkpoint",
            "/tmp/v60.pt",
            "--profile",
            "/tmp/v60.json",
            "--steps",
            "72",
            "--execute",
        ],
    )
    assert "--execute" in command
    with pytest.raises(TaskLauncherError, match="permits only 1..72"):
        build_task_command(
            "thrown_object_v60",
            "deploy",
            [
                "--checkpoint",
                "/tmp/v60.pt",
                "--profile",
                "/tmp/v60.json",
                "--steps",
                "73",
                "--execute",
            ],
        )


def test_v61_task_materializes_camera_reset_and_forty_tick_cap() -> None:
    config, metadata = materialize_task_config("thrown_object_v61")
    camera = resolve_runtime_camera_contract(
        _bundle_contract().with_runtime_policy_rate_hz(20), config
    )
    resolved = resolve_runtime_task_contract(
        _bundle_contract().with_runtime_policy_rate_hz(20), config
    )

    assert metadata["commissioning_status"] == "supervised_40_tick_authorized"
    assert metadata["robot_execution_enabled"] is True
    assert metadata["maximum_supervised_execute_steps"] == 40
    assert task_profile_allows_robot_execution(config) is True
    assert camera.camera_serial == "342222071785"
    assert (camera.camera_width, camera.camera_height) == (424, 240)
    assert camera.camera_rate_hz == pytest.approx(60.0)
    assert camera.depth_range_m == pytest.approx((0.25, 2.0))
    np.testing.assert_allclose(
        resolved.q_home_rad,
        [
            0.6292979717254639,
            -0.8440930247306824,
            -0.008244000375270844,
            -2.096261978149414,
            -0.40424200892448425,
            1.8299169540405273,
            -1.7469099760055542,
        ],
        atol=1.0e-7,
        rtol=0.0,
    )
    command, _config, _metadata = build_task_command(
        "thrown_object_v61",
        "deploy",
        [
            "--checkpoint",
            "/tmp/v61.pt",
            "--profile",
            "/tmp/v61.json",
            "--steps",
            "40",
            "--execute",
        ],
    )
    assert command[command.index("--steps") + 1] == "40"
    with pytest.raises(TaskLauncherError, match="permits only 1..40"):
        build_task_command(
            "thrown_object_v61",
            "deploy",
            [
                "--checkpoint",
                "/tmp/v61.pt",
                "--profile",
                "/tmp/v61.json",
                "--steps",
                "41",
                "--execute",
            ],
        )


def test_thrown_perception_request_uses_v57_policy_rate() -> None:
    config, _metadata = materialize_task_config("thrown_object")
    args = build_perception_parser().parse_args(
        [
            "--pcd-config",
            str(config),
            "--policy-rgbd-resolution",
            "424x240",
            "--object-mask-mode",
            "guarded_v2",
            "--test-rollout-trigger",
            "--object-text",
            "red triangular object",
        ]
    )
    request = perception_request_from_args(args)
    assert request.policy_rate_hz == pytest.approx(20.0)


def test_task_launcher_owns_policy_resolution_mask_mode_and_config() -> None:
    command, config, _metadata = build_task_command(
        "thrown_object", "perception", ["--duration", "2"]
    )
    joined = " ".join(command)
    assert str(config) in command
    assert "--policy-rgbd-resolution 424x240" in joined
    assert "--object-mask-mode guarded_v2" in joined
    assert "--object-text red patterned ball" in joined

    for protected in (
        ("--pcd-config", "other.yaml"),
        ("--policy-rgbd-resolution", "848x480"),
        ("--object-mask-mode", "legacy"),
    ):
        with pytest.raises(TaskLauncherError, match="task-owned"):
            build_task_command("tabletop", "perception", list(protected))

    trigger_command, _config, _metadata = build_task_command(
        "thrown_object", "perception", ["--test-rollout-trigger"]
    )
    assert "--test-rollout-trigger" in trigger_command
    assert "--object-text" in trigger_command


def test_thrown_robot_execution_is_enabled_only_for_capped_first_motion() -> None:
    config, _metadata = materialize_task_config("thrown_object")
    assert task_profile_allows_robot_execution(config) is True
    assert task_profile_maximum_supervised_execute_steps(config) == 20
    command, _config, _metadata = build_task_command(
        "thrown_object",
        "deploy",
        [
            "--execute",
            "--steps",
            "20",
            "--checkpoint",
            "v57.pt",
            "--profile",
            "v57-profile.json",
        ],
    )
    assert "--execute" in command
    with pytest.raises(TaskLauncherError, match="permits only 1..20"):
        build_task_command(
            "thrown_object",
            "deploy",
            ["--execute", "--steps", "21"],
        )
    with pytest.raises(TaskLauncherError, match="explicit --steps"):
        build_task_command("thrown_object", "deploy", ["--execute"])

    args = build_parser().parse_args(
        [
            "--pcd-config",
            str(config),
            "--policy-rgbd-resolution",
            "424x240",
            "--policy-rate-hz",
            "20",
            "--checkpoint",
            str(
                Path(__file__).resolve().parents[2]
                / "data/checkpoints/thrown/test/student_pretrain_best_action.pt"
            ),
            "--object-text",
            "small patterned beanbag toy",
            "--steps",
            "20",
            "--run-id",
            "thrown-profile-direct-gate",
            "--execute",
            "--yes-i-am-supervising",
        ]
    )
    request = build_deployment_request(args)
    assert request.execute is True
    assert request.steps == 20

    args.steps = "21"
    with pytest.raises(SupervisedV94RunError, match="execution cap of 20"):
        build_deployment_request(args)


def test_thrown_deploy_requires_checkpoint_and_owns_v57_policy_rate() -> None:
    with pytest.raises(TaskLauncherError, match="requires an explicit V57-compatible"):
        build_task_command("thrown_object", "deploy", [])
    with pytest.raises(TaskLauncherError, match="policy-rate-hz is owned"):
        build_task_command(
            "thrown_object",
            "deploy",
            ["--policy-rate-hz", "60", "--checkpoint", "v57.pt"],
        )

    with pytest.raises(TaskLauncherError, match="task-specific --profile"):
        build_task_command(
            "thrown_object", "deploy", ["--checkpoint", "v57.pt"]
        )

    command, _config, metadata = build_task_command(
        "thrown_object",
        "deploy",
        ["--checkpoint", "v57.pt", "--profile", "v57-profile.json"],
    )
    index = command.index("--policy-rate-hz")
    assert command[index + 1] == "20"
    assert metadata["simulation_curriculum"] == "alpha_0_5"
    assert metadata["simulation_task_contract_control_hz"] == 20.0


def test_thrown_runtime_task_contract_replaces_only_the_v57_reset() -> None:
    config, _metadata = materialize_task_config("thrown_object")
    bundle = _bundle_contract().with_runtime_policy_rate_hz(20)
    camera_only = resolve_runtime_camera_contract(bundle, config)
    resolved = resolve_runtime_task_contract(bundle, config)

    np.testing.assert_array_equal(
        resolved.q_home_rad,
        np.asarray([0.0, -1.2, 0.0, -2.2, 0.0, 1.4, np.pi / 4], np.float32),
    )
    assert not np.array_equal(camera_only.q_home_rad, resolved.q_home_rad)
    np.testing.assert_array_equal(
        resolved.T_base_camera_optical, camera_only.T_base_camera_optical
    )
    np.testing.assert_array_equal(resolved.camera_K, camera_only.camera_K)


def test_tabletop_task_retains_robot_execution_latch() -> None:
    config, _metadata = materialize_task_config("tabletop")
    assert task_profile_allows_robot_execution(config) is True
    command, _config, _metadata = build_task_command(
        "tabletop", "deploy", ["--execute"]
    )
    assert "--execute" in command


def test_runtime_camera_profile_fails_closed_if_calibration_pin_is_changed(
    tmp_path: Path,
) -> None:
    config, _metadata = materialize_task_config("thrown_object")
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["task_profile"]["calibration_sha256"] = "0" * 64
    encoded = yaml.safe_dump(document, sort_keys=False).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    tampered = tmp_path / f"thrown_object-{digest[:16]}.yaml"
    tampered.write_bytes(encoded)

    with pytest.raises(RuntimeCameraProfileError, match="calibration bytes differ"):
        resolve_runtime_camera_contract(_bundle_contract(), tampered)
