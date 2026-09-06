from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim2real.deployment.bundle import DeployBundle
from sim2real.deployment.runner import DEFAULT_BUNDLE
from sim2real.observation.camera_profile import resolve_runtime_camera_contract
from sim2real.tasks.launcher import materialize_task_config
from sim2real.tasks.thrown_contract import (
    ThrownTaskContractError,
    apply_v57_task_contract,
    assess_v57_camera_visibility,
    load_v57_thrown_task_contract,
    resolve_v57_thrown_task_contract,
)
from sim2real.contracts.v94 import V94Contract


WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE = (
    WORKSPACE / "perception" / "v57_real_test_reset_and_throw_ranges.yaml"
)
SOURCE_SHA256 = "7d0646b1a1592895aeb0a7f591d65d771eb6b67cf14154d4aab1b54b0609f4c4"


def _camera_contract():
    config, _metadata = materialize_task_config("thrown_object")
    bundle = V94Contract.from_bundle(DeployBundle(DEFAULT_BUNDLE))
    return config, resolve_runtime_camera_contract(
        bundle.with_runtime_policy_rate_hz(20), config
    )


def test_v57_source_contract_is_exact_and_selected_alpha_half() -> None:
    task = load_v57_thrown_task_contract(
        SOURCE,
        selected_curriculum="alpha_0_5",
        expected_sha256=SOURCE_SHA256,
    )
    assert task.control_hz == 20.0
    assert task.rh56_speed_set == 600
    assert task.rh56_speed_register_order.tolist() == [600] * 6
    assert task.rh56_open_register_order.tolist() == [1000] * 6
    np.testing.assert_array_equal(
        task.franka_q_home_rad,
        np.asarray([0.0, -1.2, 0.0, -2.2, 0.0, 1.4, np.pi / 4], np.float32),
    )
    np.testing.assert_allclose(
        task.catch_reference_center_base_m,
        [0.1842689514, 0.0539417267, 0.5668298006],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        task.selected_curriculum.target_minimum_base_m,
        [0.0342689514, -0.0960582733, 0.4168298006],
    )
    np.testing.assert_allclose(
        task.selected_curriculum.target_maximum_base_m,
        [0.3342689514, 0.2039417267, 0.7168298006],
    )
    np.testing.assert_allclose(task.object_half_extent_base_m, [0.04, 0.035, 0.035])


def test_selected_alpha_half_is_visible_and_depth_covered() -> None:
    config, camera = _camera_contract()
    task = resolve_v57_thrown_task_contract(config)
    assert task is not None
    report = assess_v57_camera_visibility(task, camera)
    assert report.target_center_vertices_inside == 8
    assert report.target_center_box_fully_visible
    assert report.object_support_depth_covered
    assert report.target_center_camera_z_min_m == pytest.approx(1.1471540197)
    assert report.target_center_camera_z_max_m == pytest.approx(1.5050857367)
    assert report.object_support_camera_z_max_m == pytest.approx(1.54747236)


def test_v60_rollout_records_but_does_not_hide_reset_and_depth_mismatches() -> None:
    config, _metadata = materialize_task_config("thrown_object_v60")
    bundle = V94Contract.from_bundle(DeployBundle(DEFAULT_BUNDLE))
    camera = resolve_runtime_camera_contract(
        bundle.with_runtime_policy_rate_hz(20), config
    )
    task = resolve_v57_thrown_task_contract(config)
    assert task is not None
    report = assess_v57_camera_visibility(task, camera)

    assert report.target_center_box_fully_visible
    assert report.target_center_vertices_inside == 8
    assert not report.object_support_depth_covered
    assert report.object_support_camera_z_max_m == pytest.approx(1.6719709794)
    assert task.franka_q_home_rad[5] == pytest.approx(0.464266852)
    assert task.franka_q_home_rad[5] < bundle.joint_limits_rad[5, 0]

    diagnostic = apply_v57_task_contract(camera, config)
    np.testing.assert_array_equal(diagnostic.q_home_rad, task.franka_q_home_rad)


def test_v60_rollout_relaxations_cannot_be_removed_from_full_acceptance(
    tmp_path: Path,
) -> None:
    config, _metadata = materialize_task_config("thrown_object_v60")
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["task_profile"].pop("rollout_execution_limits")
    encoded = yaml.safe_dump(document, sort_keys=False, allow_unicode=True).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    tampered = tmp_path / f"thrown_object_v60-{digest[:16]}.yaml"
    tampered.write_bytes(encoded)
    bundle = V94Contract.from_bundle(DeployBundle(DEFAULT_BUNDLE))
    camera = resolve_runtime_camera_contract(
        bundle.with_runtime_policy_rate_hz(20), tampered
    )
    with pytest.raises(ThrownTaskContractError, match="outside joint limits"):
        apply_v57_task_contract(camera, tampered)


def test_full_alpha_one_is_not_falsely_commissioned_for_this_view() -> None:
    config, camera = _camera_contract()
    task = resolve_v57_thrown_task_contract(config)
    assert task is not None
    report = assess_v57_camera_visibility(task, camera, curriculum="alpha_1_0")
    assert report.target_center_vertices_inside == 6
    assert not report.target_center_box_fully_visible

    alpha_one = load_v57_thrown_task_contract(
        SOURCE,
        selected_curriculum="alpha_1_0",
        expected_sha256=SOURCE_SHA256,
    )
    # Simulate selecting alpha=1.0 without changing the sealed source bytes.
    original_resolver = resolve_v57_thrown_task_contract
    try:
        import sim2real.tasks.thrown_contract as module

        module.resolve_v57_thrown_task_contract = lambda _path: alpha_one
        with pytest.raises(ThrownTaskContractError, match="not fully visible"):
            apply_v57_task_contract(camera, config)
    finally:
        module.resolve_v57_thrown_task_contract = original_resolver


def test_wrong_v57_source_hash_fails_closed() -> None:
    with pytest.raises(ThrownTaskContractError, match="differ from SHA pin"):
        load_v57_thrown_task_contract(
            SOURCE,
            selected_curriculum="alpha_0_5",
            expected_sha256="0" * 64,
        )
