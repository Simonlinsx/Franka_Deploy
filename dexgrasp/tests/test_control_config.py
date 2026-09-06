from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from anydex_pipeline.control_config import (
    control_readiness,
    load_control_config,
    select_execution_candidate,
    validate_control_config,
    verify_adapter_assets,
)
from anydex_pipeline.snapshot import load_snapshot_npz


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
SUPERVISED_CONFIG = ROOT / "configs/fr3_rh56_v7_sim2real_supervised.json"


def test_checked_in_commissioning_config_is_valid_but_fail_closed():
    config, source = load_control_config(CONFIG)
    assert source == CONFIG.resolve()
    assets = verify_adapter_assets(config, source)
    assert assets.mesh_sha256 == "7fdd3dd06bd8dafed445f6a6910315edd3073bd1b9cd9015a951b6e415b947e1"
    assert assets.mesh_path.name == "V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.stl"
    readiness = control_readiness(config)
    assert not readiness.default_motion_ready
    assert not readiness.full_grasp_ready
    assert config["franka"]["expected_end_effector"] == {
        "mass_kg": 0.607,
        "F_x_Cee_m": [0.0, 0.0, 0.076],
        "inertia_kg_m2": [
            [0.00151, 0.0, 0.0],
            [0.0, 0.00169, 0.0],
            [0.0, 0.0, 0.000442],
        ],
    }
    np.testing.assert_allclose(config["franka"]["expected_F_T_EE"], np.eye(4))
    np.testing.assert_allclose(
        config["franka"]["default_q_rad"],
        [-0.1118436, -0.1207545, 0.0739457, -1.7431009, 0.046354, 1.6809169, 0.8117281],
        atol=0.0,
        rtol=0.0,
    )
    assert config["franka"]["default_q_provenance"] == {
        "source_kind": "user_reported_fci_stationary_readback",
        "recorded_date": "2026-07-18",
        "installed_configuration": "FR3 + V7 Bambu PLA adapter + Inspire RH56 right hand",
        "reason": "cable-friendly installed default; avoids an unnecessary q7 round trip through zero",
        "verification_scope": "reported stationary installed pose only; no path or collision authorization",
        "evidence_artifact": None,
        "motion_authorized": False,
    }
    assert config["tool"]["installed_on_franka_verified"] is True
    assert config["tool"]["assembled_yaw_verified"] is True
    assert config["tool"]["source_origin_datum_verified"] is True
    assert config["tool"]["mount_transform_commissioned"] is True
    np.testing.assert_allclose(
        config["tool"]["T_EE_hand"],
        [
            [0.0, np.sqrt(0.5), np.sqrt(0.5), 0.0],
            [0.0, -np.sqrt(0.5), np.sqrt(0.5), 0.0],
            [1.0, 0.0, 0.0, 0.010],
            [0.0, 0.0, 0.0, 1.0],
        ],
        atol=1e-12,
    )
    assert any("default_q" in item for item in readiness.default_motion_blockers)
    assert any("PLA adapter" in item for item in readiness.full_grasp_blockers)
    assert not any(
        "insertion offset" in item for item in readiness.full_grasp_blockers
    )


def test_sim2real_supervised_profile_validates_without_changing_legacy_profile():
    supervised, source = load_control_config(SUPERVISED_CONFIG)
    legacy, _ = load_control_config(CONFIG)

    assert source == SUPERVISED_CONFIG.resolve()
    assert supervised["inspire"]["thumb_rotate_validated_realtime_range"] == [0, 1000]
    assert supervised["inspire"]["air_closure_target_acceptance"] == (
        "operator_supervised_register_ranges_v1"
    )
    assert "air_closure_target_acceptance" not in legacy["inspire"]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("supervised_air_closure_scope", "loaded_lift", "scope"),
        (
            "supervised_air_closure_axis_ranges",
            [[0, 1000]] * 5 + [[415, 1000]],
            "q6 range",
        ),
        ("supervised_air_closure_axis_ranges", "bad", "six integer"),
    ],
)
def test_sim2real_supervised_profile_rejects_malformed_policy(key, value, message):
    config = json.loads(SUPERVISED_CONFIG.read_text(encoding="utf-8"))
    config["inspire"][key] = value
    with pytest.raises(ValueError, match=message):
        validate_control_config(config)


def test_adapter_mount_plane_is_10mm_while_collision_envelope_is_17p8mm():
    config, _ = load_control_config(CONFIG)
    tool = config["tool"]
    assert tool["fr3_face_to_rh56_seating_plane_m"] == pytest.approx(0.010)
    assert tool["adapter_total_axial_envelope_m"] == pytest.approx(0.0178)
    assert tool["adapter_disk_thickness_m"] + tool["adapter_spigot_protrusion_m"] == pytest.approx(
        tool["adapter_total_axial_envelope_m"]
    )


def test_config_rejects_using_total_envelope_as_mount_plane():
    config, _ = load_control_config(CONFIG)
    bad = copy.deepcopy(config)
    bad["tool"]["fr3_face_to_rh56_seating_plane_m"] = 0.0178
    with pytest.raises(ValueError, match="seating plane is the disk top"):
        validate_control_config(bad)


@pytest.mark.parametrize("margin", [0.0009, 0.0051])
def test_air_observed_scene_margin_has_a_narrow_profile_bound(margin):
    config, _ = load_control_config(CONFIG)
    bad = copy.deepcopy(config)
    bad["grasp"]["air_audit_observed_scene_margin_m"] = margin
    with pytest.raises(ValueError, match="air_audit_observed_scene_margin_m"):
        validate_control_config(bad)


@pytest.mark.parametrize("margin", [-0.001, 0.001, True])
def test_air_rh56_self_margin_is_locked_to_mesh_nonintersection(margin):
    config, _ = load_control_config(CONFIG)
    bad = copy.deepcopy(config)
    bad["grasp"]["air_audit_rh56_self_clearance_margin_m"] = margin
    with pytest.raises(ValueError, match="air_audit_rh56_self_clearance_margin_m"):
        validate_control_config(bad)


def test_current_real_snapshot_is_plan_only_and_not_executable():
    config, _ = load_control_config(CONFIG)
    snapshot = load_snapshot_npz(
        ROOT / "runs/d435_pink_cylinder_geometric_inspire_type4.npz"
    )
    readiness = control_readiness(config, snapshot)
    assert not readiness.full_grasp_ready
    assert any("diagnostic/geometric" in item for item in readiness.full_grasp_blockers)
    assert any("official provenance is incomplete" in item for item in readiness.full_grasp_blockers)
    assert any("collision result is not attached" in item for item in readiness.full_grasp_blockers)
    assert not any("T_EE_hand" in item for item in readiness.full_grasp_blockers)
    np.testing.assert_array_equal(
        snapshot.grasps.hand_angles[snapshot.grasps.selected_index],
        [695, 695, 695, 695, 830, 0],
    )


def test_end_effector_dynamics_are_all_or_nothing_and_physically_valid():
    config, _ = load_control_config(CONFIG)
    partial = copy.deepcopy(config)
    partial["franka"]["expected_end_effector"]["F_x_Cee_m"] = None
    with pytest.raises(ValueError, match="set together"):
        validate_control_config(partial)

    nonsymmetric = copy.deepcopy(config)
    nonsymmetric["franka"]["expected_end_effector"] = {
        "mass_kg": 0.75,
        "F_x_Cee_m": [0.0, 0.0, 0.08],
        "inertia_kg_m2": [[0.01, 0.1, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.002]],
    }
    with pytest.raises(ValueError, match="symmetric"):
        validate_control_config(nonsymmetric)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("franka", "default_path_collision_verified"),
        ("inspire", "six_axis_coupled_closure_commissioned"),
        ("tool", "mount_transform_commissioned"),
        ("tool", "installed_collision_model_verified"),
        ("grasp", "require_official_anydex_backend"),
        ("grasp", "extra_final_insertion_commissioned"),
    ],
)
def test_all_gate_booleans_reject_truthy_non_booleans(section, key):
    config, _ = load_control_config(CONFIG)
    bad = copy.deepcopy(config)
    bad[section][key] = 1
    with pytest.raises(ValueError, match="must be boolean"):
        validate_control_config(bad)


def test_nonzero_extra_insertion_requires_commissioning_but_zero_does_not():
    config, _ = load_control_config(CONFIG)
    zero = control_readiness(config)
    assert not any("insertion offset" in item for item in zero.full_grasp_blockers)

    nonzero = copy.deepcopy(config)
    nonzero["grasp"]["extra_final_insertion_m"] = 0.014
    readiness = control_readiness(nonzero)
    assert any("insertion offset" in item for item in readiness.full_grasp_blockers)


def test_configured_q6_selection_searches_all_top_k_candidates():
    config, _ = load_control_config(CONFIG)
    snapshot = load_snapshot_npz(
        ROOT / "runs/d435_pink_cylinder_geometric_inspire_type4.npz"
    )
    angles = np.asarray(snapshot.grasps.hand_angles).copy()
    angles[:, 5] = 0
    angles[1, 5] = 950
    checked = np.ones(snapshot.grasps.count, dtype=np.bool_)
    collision_free = np.ones(snapshot.grasps.count, dtype=np.bool_)
    grasps = replace(
        snapshot.grasps,
        hand_angles=angles,
        collision_checked=checked,
        collision_free=collision_free,
        selected_index=0,
    )

    assert select_execution_candidate(config, replace(snapshot, grasps=grasps)) == 0
