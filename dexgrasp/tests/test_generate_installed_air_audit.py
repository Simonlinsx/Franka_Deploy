from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from anydex_pipeline.control_config import load_control_config
from anydex_pipeline.hppfcl_installed_tool_backend import HppFclInstalledToolConfig
from anydex_pipeline.installed_tool_audit import (
    V7_T_EE_HAND,
    _array_sha256,
    build_joint_path,
)
from anydex_pipeline.snapshot import load_snapshot_npz


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "apps/generate_installed_air_audit.py"
CONFIG_PATH = ROOT / "configs/fr3_rh56_v7_commissioning.json"
SNAPSHOT_PATH = (
    ROOT / "runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz"
)
FILTERED_SCENE_PATH = (
    ROOT / "runs/live_scene_installed_filtered_candidate51_schema_v2prep_20260718.npz"
)
JOINT_PLAN_PATH = ROOT / "runs/candidate51_installed_air_joint_plan_20260718.json"


def _load_app():
    spec = importlib.util.spec_from_file_location(
        "test_generate_installed_air_audit_app", APP_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generator_exposes_only_air_audit_and_strict_ik_defaults():
    app = _load_app()
    parser = app.build_parser()
    destinations = {action.dest: action for action in parser._actions}
    assert "mode" not in destinations
    assert destinations["joint_plan"].required is True
    assert "pregrasp_q" not in destinations
    assert "grasp_q" not in destinations
    assert destinations["max_ik_position_error_m"].default == pytest.approx(0.0005)
    assert destinations["max_ik_rotation_error_rad"].default == pytest.approx(0.001)


def test_candidate_q6_must_be_inside_commissioned_coupled_range():
    app = _load_app()
    config, _ = load_control_config(CONFIG_PATH)
    targets = np.asarray([0, 358, 799, 911, 922, 646], dtype=np.float64)

    with pytest.raises(ValueError, match="coupled_closure_commissioned"):
        app._validate_candidate_q6_commissioning(config, targets)

    commissioned = copy.deepcopy(config)
    commissioned["inspire"]["six_axis_coupled_closure_commissioned"] = True
    with pytest.raises(ValueError, match="outside validated range"):
        app._validate_candidate_q6_commissioning(commissioned, targets)

    commissioned["inspire"]["thumb_rotate_validated_realtime_range"] = [600, 1000]
    with pytest.raises(ValueError, match="no exact commissioned"):
        app._validate_candidate_q6_commissioning(commissioned, targets)
    commissioned["inspire"]["commissioned_air_closure_targets"] = [
        [0, 358, 799, 911, 922, 646]
    ]
    app._validate_candidate_q6_commissioning(commissioned, targets)


def test_filtered_scene_q_source_is_explicit_operator_assertion(tmp_path):
    app = _load_app()
    evidence = tmp_path / "filter.evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "motion_authorized": False,
                "authoritative_for_unseen_camera_space": False,
                "unknown_space_policy_applied": False,
            }
        ),
        encoding="utf-8",
    )

    def write(path, source):
        np.savez_compressed(
            path,
            artifact_type=np.asarray("installed_filtered_live_scene"),
            filtered_scene_points=np.asarray([[0.1, 0.2, 0.3]]),
            reference_frame=np.asarray("robot_base"),
            scene_excludes_object=np.asarray(True),
            capture_q_rad=np.zeros(7),
            capture_q_source=np.asarray(source),
            captured_at_unix_s=np.asarray(1.0),
            calibration_id=np.asarray("calibration"),
            camera_serial=np.asarray("camera"),
            filter_evidence_path=np.asarray(str(evidence.resolve())),
            filter_evidence_sha256=np.asarray(app._sha256_file(evidence)),
        )

    valid = tmp_path / "valid.npz"
    write(valid, "cli_asserted")
    loaded = app._load_filtered_scene(valid)
    assert loaded[3] == "cli_asserted"

    forged = tmp_path / "forged.npz"
    write(forged, "automatic_franka_measurement")
    with pytest.raises(ValueError, match="cannot claim an automatic"):
        app._load_filtered_scene(forged)


def _joint_plan_arguments(app, path):
    config, config_path = load_control_config(CONFIG_PATH)
    snapshot = load_snapshot_npz(SNAPSHOT_PATH)
    index = 51
    canonical = np.asarray(snapshot.grasps.canonical_poses[index], dtype=np.float64)
    hand_pose = np.asarray(snapshot.grasps.hand_poses[index], dtype=np.float64)
    hand_targets = np.asarray(snapshot.grasps.hand_angles[index], dtype=np.float64)
    approach = canonical[:3, :3] @ np.asarray(
        snapshot.grasps.approach_axis_local, dtype=np.float64
    )
    approach /= np.linalg.norm(approach)
    planned_hand = hand_pose.copy()
    planned_hand[:3, 3] -= config["grasp"]["air_retreat_distance_m"] * approach
    final_pose = planned_hand @ np.linalg.inv(V7_T_EE_HAND)
    pregrasp_pose = final_pose.copy()
    pregrasp_pose[:3, 3] -= (
        config["grasp"]["air_pregrasp_distance_m"] * approach
    )
    with np.load(FILTERED_SCENE_PATH, allow_pickle=False) as archive:
        capture_q = np.asarray(archive["capture_q_rad"], dtype=np.float64)
    return dict(
        path=path,
        config_path=config_path,
        snapshot_path=SNAPSHOT_PATH.resolve(),
        urdf_path=HppFclInstalledToolConfig().fr3_urdf_path,
        candidate_index=index,
        canonical_pose=canonical,
        hand_pose=hand_pose,
        hand_targets=hand_targets,
        capture_q=capture_q,
        default_q=np.asarray(config["franka"]["default_q_rad"], dtype=np.float64),
        joint_limits=np.asarray(config["franka"]["joint_limits_rad"], dtype=np.float64),
        joint_limit_margin=float(config["franka"]["joint_limit_margin_rad"]),
        retreat_distance_m=float(config["grasp"]["air_retreat_distance_m"]),
        pregrasp_distance_m=float(config["grasp"]["air_pregrasp_distance_m"]),
        pregrasp_pose=pregrasp_pose,
        final_pose=final_pose,
    )


@pytest.mark.skipif(
    not all(path.is_file() for path in (SNAPSHOT_PATH, FILTERED_SCENE_PATH, JOINT_PLAN_PATH)),
    reason="candidate-51 offline integration artifacts are unavailable",
)
def test_joint_plan_manifest_is_bound_and_checksum_tamper_is_rejected(tmp_path):
    app = _load_app()
    payload = json.loads(JOINT_PLAN_PATH.read_text(encoding="utf-8"))
    if payload["inputs"]["config"]["sha256"] != app._sha256_file(CONFIG_PATH):
        pytest.skip(
            "the checked-in integration manifest predates the active control profile"
        )
    source, q_pre, q_final, max_step = app._load_joint_plan(
        **_joint_plan_arguments(app, JOINT_PLAN_PATH)
    )
    assert source == JOINT_PLAN_PATH.resolve()
    assert q_pre.shape == (7,) and q_final.shape == (7,)
    assert max_step == pytest.approx(0.005)

    payload["joint_plan"]["q_pregrasp_rad"][0] += 0.001
    tampered = tmp_path / "tampered_joint_plan.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="self checksum"):
        app._load_joint_plan(**_joint_plan_arguments(app, tampered))


@pytest.mark.skipif(
    not JOINT_PLAN_PATH.is_file(),
    reason="candidate-51 real joint-plan manifest is unavailable",
)
def test_real_manifest_and_collision_audit_use_identical_canonical_path_bytes():
    payload = json.loads(JOINT_PLAN_PATH.read_text(encoding="utf-8"))
    plan = payload["joint_plan"]
    maximum = float(plan["requested_max_joint_step_rad"])
    manifest_path, manifest_intervals = _load_app()._manifest_joint_path(
        tuple(
            np.asarray(plan[key], dtype=np.float64)
            for key in (
                "q_start_rad",
                "q_default_rad",
                "q_pregrasp_rad",
                "q_final_air_rad",
            )
        ),
        maximum,
    )
    audited_path, audited_segments = build_joint_path(
        plan["q_start_rad"],
        plan["q_default_rad"],
        plan["q_pregrasp_rad"],
        plan["q_final_air_rad"],
        max_joint_step_rad=maximum,
    )

    assert np.array_equal(audited_path, manifest_path)
    assert _array_sha256(audited_path) == plan["q_path_sha256"]
    assert tuple(
        item["end_index"] - item["start_index"] for item in audited_segments
    ) == manifest_intervals
