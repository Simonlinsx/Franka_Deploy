from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from anydex_pipeline.control_config import ControlReadiness, VerifiedAdapterAssets
from anydex_pipeline.pipeline_preview import load_pose_state_json
from anydex_pipeline.rh56_hand_path import rh56_feedback_envelope_policy


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "apps/execute_control_sequence.py"
CONFIG_PATH = ROOT / "configs/fr3_rh56_v7_commissioning.json"
SNAPSHOT_PATH = ROOT / "runs/d435_pink_cylinder_geometric_inspire_type4.npz"
OFFICIAL_SNAPSHOT_PATH = (
    ROOT / "runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz"
)
JOINT_PLAN_PATH = (
    ROOT
    / "runs/candidate51_installed_air_joint_plan_current_fr3_limits_20260726.json"
)


def _load_app():
    spec = importlib.util.spec_from_file_location(
        "test_execute_control_sequence_app", APP_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_audited_settle_artifact():
    manifest = json.loads(JOINT_PLAN_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(JOINT_PLAN_PATH.read_bytes()).hexdigest()
    return {
        "schema_version": 2,
        "mode": "air_grasp",
        "policies": {
            "hand_arrival_tolerance_units": 25,
            "hand_feedback_envelope": rh56_feedback_envelope_policy(25),
        },
        "bindings": {
            "control_profile": {
                "sha256": manifest["inputs"]["config"]["sha256"]
            },
            "snapshot": {
                "sha256": manifest["inputs"]["snapshot"]["sha256"]
            },
            "execution_plan": {
                "trajectory_contract": "joint_waypoint_polyline_v1",
                "pregrasp_pose_base_EE": manifest["air_geometry"][
                    "pregrasp_pose_base_EE"
                ],
                "grasp_pose_base_EE": manifest["air_geometry"][
                    "final_air_pose_base_EE"
                ],
                "contact_and_lift_forbidden": True,
            },
            "joint_path": {
                "sampling_algorithm": "numpy_linspace_float64_v1",
                "waypoints": [
                    {"name": "current", "q_rad": manifest["joint_plan"]["q_start_rad"]},
                    {"name": "default", "q_rad": manifest["joint_plan"]["q_default_rad"]},
                    {"name": "pregrasp", "q_rad": manifest["joint_plan"]["q_pregrasp_rad"]},
                    {"name": "grasp", "q_rad": manifest["joint_plan"]["q_final_air_rad"]},
                ]
            },
            "hand_execution_path": manifest["hand_execution_path"],
            "joint_plan_manifest": {
                "required": True,
                "path": str(JOINT_PLAN_PATH.resolve()),
                "sha256": digest,
                "sha256_after_audit": digest,
                "validation_error": "",
            },
        },
    }


def test_inspect_is_offline_and_never_loads_hardware_types(monkeypatch, capsys):
    app = _load_app()

    def forbidden():
        raise AssertionError("offline inspect attempted a hardware import")

    monkeypatch.setattr(app, "_load_hardware_types", forbidden)
    result = app.main(["inspect", "--config", str(CONFIG_PATH)])
    captured = capsys.readouterr()
    assert result == 0
    assert "OFFLINE INSPECT" in captured.out
    assert "no hardware driver was imported" in captured.out


def test_candidate_selection_is_an_in_memory_view_and_preserves_source_npz():
    app = _load_app()
    before = SNAPSHOT_PATH.read_bytes()
    source = app.load_snapshot_npz(SNAPSHOT_PATH)
    assert source.grasps.count > 1
    original = int(source.grasps.selected_index)

    selected = 1 if original != 1 else 0
    view = app._snapshot_with_selected_index(source, selected)

    assert int(source.grasps.selected_index) == original
    assert int(view.grasps.selected_index) == selected
    np.testing.assert_array_equal(
        view.grasps.canonical_poses, source.grasps.canonical_poses
    )
    np.testing.assert_array_equal(view.grasps.hand_angles, source.grasps.hand_angles)
    assert SNAPSHOT_PATH.read_bytes() == before


def test_inspect_selected_index_builds_that_candidate_without_writing_snapshot():
    app = _load_app()
    before = SNAPSHOT_PATH.read_bytes()
    inspection = app.inspect_offline(
        CONFIG_PATH,
        SNAPSHOT_PATH,
        collision_margin_m=0.005,
        collision_samples=21,
        selected_index=1,
    )
    assert inspection.snapshot is not None
    assert inspection.plan is not None
    assert inspection.snapshot.grasps.selected_index == 1
    assert inspection.plan.selected_index == 1
    assert SNAPSHOT_PATH.read_bytes() == before


def test_air_dry_run_labels_nominal_reference_and_exact_air_contract(
    monkeypatch, capsys
):
    app = _load_app()

    monkeypatch.setattr(
        app,
        "_load_hardware_types",
        lambda: (_ for _ in ()).throw(
            AssertionError("air dry-run attempted a hardware import")
        ),
    )
    result = app.main(
        [
            "air-grasp",
            "--config",
            str(CONFIG_PATH),
            "--snapshot",
            str(OFFICIAL_SNAPSHOT_PATH),
            "--selected-index",
            "51",
            "--trajectory-mode",
            "audited-joint",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert result == 3
    assert "[plan/nominal-contact-reference]" in captured.out
    assert "contact_xyz=[0.597035, 0.118036, 0.318216]" in captured.out
    assert "[plan/air-execution-contract]" in captured.out
    assert "pregrasp_xyz=[0.669015, 0.157798, 0.354795]" in captured.out
    assert "final_air_xyz=[0.661017, 0.15338, 0.35073]" in captured.out
    assert "[hardware] NOT IMPORTED" in captured.out


def test_selected_index_must_match_sidecar_selection(monkeypatch):
    app = _load_app()
    monkeypatch.setattr(app, "_artifact_selected_index", lambda _path: 2)
    with pytest.raises(ValueError, match="differs from installed-tool audit"):
        app.inspect_offline(
            CONFIG_PATH,
            SNAPSHOT_PATH,
            collision_margin_m=0.005,
            collision_samples=21,
            installed_audit_path=Path("audit.json"),
            selected_index=1,
        )


def test_installed_integrated_default_with_unverified_path_is_blocked_before_import(
    monkeypatch, capsys
):
    app = _load_app()
    calls = []

    def forbidden():
        calls.append("hardware-import")
        raise AssertionError("blocked default attempted a hardware import")

    monkeypatch.setattr(app, "_load_hardware_types", forbidden)
    result = app.main(
        [
            "default",
            "--config",
            str(CONFIG_PATH),
            "--confirm-workspace-clear",
            app.WORKSPACE_CLEAR_TOKEN,
            "--confirm-immediate-stop",
            app.IMMEDIATE_STOP_TOKEN,
            "--confirm-pla-low-speed",
            app.PLA_COMMISSIONING_TOKEN,
        ]
    )
    captured = capsys.readouterr()
    assert result == 3
    assert calls == []
    assert "path from the current pose to Franka default_q is not collision-verified" in captured.out
    assert "NOT IMPORTED; neither device was connected" in captured.out


def test_blocked_grasp_with_strong_tokens_never_loads_or_connects_hardware(
    monkeypatch, capsys
):
    app = _load_app()
    calls = []

    def forbidden():
        calls.append("hardware-import")
        raise AssertionError("blocked grasp attempted a hardware import")

    monkeypatch.setattr(app, "_load_hardware_types", forbidden)
    result = app.main(
        [
            "grasp",
            "--config",
            str(CONFIG_PATH),
            "--snapshot",
            str(SNAPSHOT_PATH),
            "--confirm-workspace-clear",
            app.WORKSPACE_CLEAR_TOKEN,
            "--confirm-immediate-stop",
            app.IMMEDIATE_STOP_TOKEN,
            "--confirm-full-grasp",
            app.FULL_GRASP_TOKEN,
            "--confirm-q6-preshape",
            app.Q6_PRESHAPE_TOKEN,
            "--confirm-installed-collision-model",
            app.COLLISION_MODEL_TOKEN,
        ]
    )
    captured = capsys.readouterr()
    assert result == 3
    assert calls == []
    assert "hardware execution gate" in captured.out
    assert "NOT IMPORTED; neither device was connected" in captured.out


def test_missing_exact_confirmation_blocks_otherwise_ready_default_before_import(
    monkeypatch, capsys
):
    app = _load_app()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["tool"]["installed_on_franka_verified"] = True
    config["franka"]["expected_F_T_EE"] = np.eye(4).tolist()
    config["franka"]["expected_end_effector"] = {
        "mass_kg": 0.6,
        "F_x_Cee_m": [0.0, 0.0, 0.03],
        "inertia_kg_m2": np.diag([0.001, 0.001, 0.001]).tolist(),
    }
    config["franka"]["default_path_collision_verified"] = True
    config["tool"]["assembled_yaw_rad"] = 0.0
    config["tool"]["assembled_yaw_verified"] = True
    config["tool"]["seating_to_hand_source_origin_m"] = [0.0, 0.0, 0.0]
    config["tool"]["source_origin_datum_verified"] = True
    config["tool"]["installed_collision_model_verified"] = True
    inspection = app.OfflineInspection(
        config=config,
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ("full is irrelevant",)),
        plan=None,
        adapter_audit=None,
    )
    monkeypatch.setattr(app, "inspect_offline", lambda *args, **kwargs: inspection)
    monkeypatch.setattr(
        app,
        "_load_hardware_types",
        lambda: (_ for _ in ()).throw(
            AssertionError("missing token attempted a hardware import")
        ),
    )

    result = app.main(
        [
            "default",
            "--confirm-workspace-clear",
            app.WORKSPACE_CLEAR_TOKEN,
            "--confirm-immediate-stop",
            app.IMMEDIATE_STOP_TOKEN,
            # Deliberately omit the exact PLA commissioning token.
        ]
    )
    captured = capsys.readouterr()
    assert result == 3
    assert "confirm-pla-low-speed must exactly equal" in captured.out


def test_commissioned_mount_inspect_builds_q6_staged_plan_and_adapter_audit(
    tmp_path, capsys
):
    app = _load_app()
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    config["tool"]["adapter_asset"] = str(
        (ROOT / "assets/adapter/V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.stl").resolve()
    )
    config["tool"]["adapter_provenance"] = str(
        (
            ROOT
            / "assets/adapter/V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.provenance.json"
        ).resolve()
    )
    config["tool"]["T_EE_hand"] = np.eye(4).tolist()
    config["tool"]["mount_transform_commissioned"] = True
    config["franka"]["expected_F_T_EE"] = np.eye(4).tolist()
    configured = tmp_path / "commissioned_for_offline_inspection.json"
    configured.write_text(json.dumps(config), encoding="utf-8")

    result = app.main(
        [
            "inspect",
            "--config",
            str(configured),
            "--snapshot",
            str(SNAPSHOT_PATH),
            "--collision-samples",
            "3",
        ]
    )
    captured = capsys.readouterr()
    assert result == 0, captured.err
    assert "THUMB_PRESHAPE" in captured.out
    assert "INSPIRE_CLOSE" in captured.out
    assert "[adapter-audit]" in captured.out
    assert "authoritative=False" in captured.out


def test_loaded_and_air_audit_modes_are_not_interchangeable():
    app = _load_app()
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    config["tool"]["installed_on_franka_verified"] = True
    base = dict(
        config=config,
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ()),
        plan=None,
        adapter_audit=None,
    )
    loaded = app.OfflineInspection(
        **base,
        installed_audit=app.InstalledAuditBinding(
            Path("loaded.json"), {"mode": "loaded_grasp"}, (), None
        ),
    )
    air = app.OfflineInspection(
        **base,
        installed_audit=app.InstalledAuditBinding(
            Path("air.json"), {"mode": "air_grasp"}, (), None
        ),
    )

    air_blockers = app._runtime_config_blockers(
        loaded, command="air-grasp", trajectory_mode="audited-joint"
    )
    loaded_blockers = app._runtime_config_blockers(
        air, command="grasp", trajectory_mode="audited-joint"
    )
    assert any("loaded_grasp cannot execute air-grasp" in item for item in air_blockers)
    assert any("air_grasp cannot execute grasp" in item for item in loaded_blockers)
    assert "loaded grasp execution refuses the installed PLA adapter" in loaded_blockers
    assert "loaded grasp execution refuses the installed PLA adapter" not in air_blockers


def test_null_cartesian_workspace_does_not_block_valid_audited_joint_settle():
    app = _load_app()
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    assert config["franka"]["cartesian_workspace_min_m"] is None
    assert config["franka"]["cartesian_workspace_max_m"] is None
    inspection = app.OfflineInspection(
        config=config,
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ()),
        plan=None,
        adapter_audit=None,
        installed_audit=app.InstalledAuditBinding(
            Path("air.json"), _valid_audited_settle_artifact(), (), None
        ),
    )

    blockers = app._runtime_config_blockers(
        inspection, command="air-grasp", trajectory_mode="audited-joint"
    )

    assert not any("Cartesian workspace bounds" in item for item in blockers)
    assert not any("q/FK/EEF binding failed" in item for item in blockers)


def test_audited_joint_settle_rejects_feedback_envelope_policy_tamper():
    app = _load_app()
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    artifact = _valid_audited_settle_artifact()
    artifact["policies"]["hand_feedback_envelope"]["sha256"] = "0" * 64
    inspection = app.OfflineInspection(
        config=config,
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ()),
        plan=None,
        adapter_audit=None,
        installed_audit=app.InstalledAuditBinding(
            Path("air.json"), artifact, (), None
        ),
    )

    blockers = app._audited_joint_settle_binding_blockers(inspection)

    assert any("feedback-envelope policy/hash differs" in item for item in blockers)


def test_main_removes_legacy_cartesian_readiness_blocker_for_audited_joint(
    monkeypatch, capsys
):
    app = _load_app()
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    inspection = app.OfflineInspection(
        config=config,
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness(
            (), ("Cartesian home/transit/grasp workspace is uncommissioned",)
        ),
        plan=SimpleNamespace(
            selected_index=51,
            execution_eligible=True,
            execution_blockers=(),
            stages=(
                SimpleNamespace(name=SimpleNamespace(value="INSPIRE_OPEN")),
            ),
            T_reference_EE_pregrasp=np.eye(4),
            T_reference_EE_grasp=np.eye(4),
        ),
        adapter_audit=None,
    )
    monkeypatch.setattr(app, "inspect_offline", lambda *args, **kwargs: inspection)
    monkeypatch.setattr(app, "_runtime_config_blockers", lambda *args, **kwargs: ())

    result = app.main(
        [
            "air-grasp",
            "--snapshot",
            str(SNAPSHOT_PATH),
            "--trajectory-mode",
            "audited-joint",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Cartesian home/transit/grasp workspace is uncommissioned" not in captured.out
    assert "dry-run completed" in captured.out


def test_dry_run_never_inspects_live_host_tcp_state(monkeypatch, capsys):
    app = _load_app()
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    inspection = app.OfflineInspection(
        config=config,
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ()),
        plan=None,
        adapter_audit=None,
    )
    monkeypatch.setattr(app, "inspect_offline", lambda *args, **kwargs: inspection)
    monkeypatch.setattr(app, "_runtime_config_blockers", lambda *args, **kwargs: ())

    def forbidden(_robot_ip):
        raise AssertionError("dry-run inspected live host TCP state")

    monkeypatch.setattr(app, "require_uncontended_franka_https_link", forbidden)

    result = app.main(["default", "--dry-run"])

    captured = capsys.readouterr()
    assert result == 0
    assert "dry-run completed" in captured.out


def test_formal_execution_blocks_desk_https_before_hardware_import(
    monkeypatch, capsys
):
    app = _load_app()
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    inspection = app.OfflineInspection(
        config=config,
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ()),
        plan=None,
        adapter_audit=None,
    )
    monkeypatch.setattr(app, "inspect_offline", lambda *args, **kwargs: inspection)
    monkeypatch.setattr(app, "_runtime_config_blockers", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        app,
        "require_uncontended_franka_https_link",
        lambda _robot_ip: (_ for _ in ()).throw(
            app.HostNetworkPreflightError(
                "motion blocked: found 37 ESTABLISHED TCP connection(s) "
                "to Franka Desk at 172.16.0.2:443"
            )
        ),
    )
    imports = []
    monkeypatch.setattr(
        app, "_load_hardware_types", lambda: imports.append("hardware") or {}
    )

    result = app.main(
        [
            "default",
            "--confirm-workspace-clear",
            app.WORKSPACE_CLEAR_TOKEN,
            "--confirm-immediate-stop",
            app.IMMEDIATE_STOP_TOKEN,
            "--confirm-pla-low-speed",
            app.PLA_COMMISSIONING_TOKEN,
        ]
    )

    captured = capsys.readouterr()
    assert result == 3
    assert imports == []
    assert "[host network preflight] failed" in captured.err
    assert "37 ESTABLISHED" in captured.err
    assert "NOT IMPORTED; neither device was connected" in captured.out


def test_formal_air_execution_requires_the_exact_commissioned_six_axis_target():
    app = _load_app()
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    target = (0, 358, 799, 911, 922, 646)
    config["inspire"]["thumb_rotate_validated_realtime_range"] = [646, 1000]
    config["inspire"]["six_axis_coupled_closure_commissioned"] = True
    config["inspire"]["commissioned_air_closure_targets"] = []
    plan = SimpleNamespace(
        stages=(
            SimpleNamespace(name=app.StageName.THUMB_PRESHAPE),
            SimpleNamespace(
                name=app.StageName.INSPIRE_CLOSE,
                inspire_angles=target,
            ),
        )
    )
    inspection = app.OfflineInspection(
        config=config,
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ()),
        plan=plan,
        adapter_audit=None,
        installed_audit=app.InstalledAuditBinding(
            Path("air.json"), _valid_audited_settle_artifact(), (), None
        ),
    )

    blockers = app._runtime_config_blockers(
        inspection, command="air-grasp", trajectory_mode="audited-joint"
    )
    exact_message = (
        "planned six-axis air-closure target has no exact commissioning evidence"
    )
    assert exact_message in blockers

    config["inspire"]["commissioned_air_closure_targets"] = [list(target)]
    blockers = app._runtime_config_blockers(
        inspection, command="air-grasp", trajectory_mode="audited-joint"
    )
    assert exact_message not in blockers

    config["inspire"]["commissioned_air_closure_targets"] = []
    config["inspire"]["air_closure_target_acceptance"] = (
        "operator_supervised_register_ranges_v1"
    )
    config["inspire"]["supervised_air_closure_scope"] = (
        "low_speed_no_contact_no_lift"
    )
    config["inspire"]["supervised_air_closure_axis_ranges"] = (
        [[0, 1000]] * 5 + [[646, 1000]]
    )
    blockers = app._runtime_config_blockers(
        inspection, command="air-grasp", trajectory_mode="audited-joint"
    )
    assert exact_message not in blockers


def test_missing_schema_v2_pose_binding_blocks_before_hardware_import(
    monkeypatch, capsys
):
    app = _load_app()
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    inspection = app.OfflineInspection(
        config=config,
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ()),
        plan=None,
        adapter_audit=None,
        installed_audit=app.InstalledAuditBinding(
            Path("air.json"),
            {
                "schema_version": 1,
                "mode": "air_grasp",
                "artifact_sha256": "0" * 64,
                "bindings": {"joint_path": {"sample_count": 4}},
                "decision": {"motion_authorized": True},
            },
            (),
            None,
        ),
    )
    imports = []
    monkeypatch.setattr(app, "inspect_offline", lambda *args, **kwargs: inspection)
    monkeypatch.setattr(
        app,
        "_load_hardware_types",
        lambda: imports.append("hardware") or {},
    )

    result = app.main(
        [
            "air-grasp",
            "--snapshot",
            str(SNAPSHOT_PATH),
            "--trajectory-mode",
            "audited-joint",
            "--confirm-workspace-clear",
            app.WORKSPACE_CLEAR_TOKEN,
            "--confirm-immediate-stop",
            app.IMMEDIATE_STOP_TOKEN,
            "--confirm-air-grasp",
            app.AIR_GRASP_TOKEN,
            "--confirm-q6-preshape",
            app.Q6_PRESHAPE_TOKEN,
            "--confirm-installed-collision-model",
            app.COLLISION_MODEL_TOKEN,
        ]
    )

    captured = capsys.readouterr()
    assert result == 3
    assert imports == []
    assert "requires schema-v2 audit evidence" in captured.out
    assert "NOT IMPORTED; neither device was connected" in captured.out


def test_loaded_grasp_confirmation_token_cannot_authorize_air_grasp():
    app = _load_app()
    args = SimpleNamespace(
        command="air-grasp",
        confirm_workspace_clear=app.WORKSPACE_CLEAR_TOKEN,
        confirm_immediate_stop=app.IMMEDIATE_STOP_TOKEN,
        confirm_air_grasp=app.FULL_GRASP_TOKEN,
        confirm_q6_preshape=app.Q6_PRESHAPE_TOKEN,
        confirm_installed_collision_model=app.COLLISION_MODEL_TOKEN,
    )
    blockers = app._confirmation_blockers(args)
    assert len(blockers) == 1
    assert app.AIR_GRASP_TOKEN in blockers[0]


def test_conditional_air_audit_requires_exact_runtime_workspace_token():
    app = _load_app()
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    inspection = app.OfflineInspection(
        config=config,
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ()),
        plan=None,
        adapter_audit=None,
        installed_audit=app.InstalledAuditBinding(
            Path("air.json"),
            {
                "mode": "air_grasp",
                "decision": {
                    "runtime_operator_workspace_clear_required": True,
                },
            },
            (),
            None,
        ),
    )
    missing = app._conditional_audit_blockers(
        SimpleNamespace(command="air-grasp", confirm_workspace_clear=""), inspection
    )
    assert any(app.WORKSPACE_CLEAR_TOKEN in item for item in missing)
    assert app._conditional_audit_blockers(
        SimpleNamespace(
            command="air-grasp",
            confirm_workspace_clear=app.WORKSPACE_CLEAR_TOKEN,
        ),
        inspection,
    ) == ()


def test_pregrasp_advisory_scene_still_requires_exact_runtime_workspace_token():
    app = _load_app()
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    inspection = app.OfflineInspection(
        config=config,
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ()),
        plan=None,
        adapter_audit=None,
        pregrasp_audit=app.PregraspOnlyAuditBinding(
            Path("prefix.json"),
            {
                "execution_scope": "open_hand_current_to_pregrasp_only",
                "decision": {"runtime_workspace_clear_required": True},
                # Deliberately ancient: age is advisory, the token is the
                # fresh per-run environment authority.
                "bindings": {"filtered_scene": {"captured_at_s": 1.0}},
            },
            (),
        ),
    )
    missing = app._conditional_audit_blockers(
        SimpleNamespace(command="pregrasp", confirm_workspace_clear=""),
        inspection,
    )
    assert any(app.WORKSPACE_CLEAR_TOKEN in item for item in missing)
    assert app._conditional_audit_blockers(
        SimpleNamespace(
            command="pregrasp",
            confirm_workspace_clear=app.WORKSPACE_CLEAR_TOKEN,
        ),
        inspection,
    ) == ()


def test_air_grasp_parser_exposes_audited_default_prefix_mode():
    app = _load_app()
    args = app.build_parser().parse_args(
        [
            "air-grasp",
            "--snapshot",
            str(SNAPSHOT_PATH),
            "--stop-after-default",
        ]
    )
    assert args.command == "air-grasp"
    assert args.stop_after_default is True


def test_pregrasp_parser_exposes_dedicated_audit_and_no_grasp_controls():
    app = _load_app()
    args = app.build_parser().parse_args(
        [
            "pregrasp",
            "--snapshot",
            str(SNAPSHOT_PATH),
            "--pregrasp-only-audit",
            "prefix.json",
        ]
    )
    assert args.command == "pregrasp"
    assert args.pregrasp_only_audit == Path("prefix.json")
    assert not hasattr(args, "hold_seconds")
    assert not hasattr(args, "confirm_q6_preshape")
    assert not hasattr(args, "confirm_air_grasp")


def test_pregrasp_without_dedicated_audit_is_blocked_before_hardware_import(
    monkeypatch, capsys
):
    app = _load_app()
    imported = []
    monkeypatch.setattr(
        app,
        "_load_hardware_types",
        lambda: imported.append(True) or {},
    )
    result = app.main(
        [
            "pregrasp",
            "--snapshot",
            str(SNAPSHOT_PATH),
            "--selected-index",
            "1",
            "--confirm-workspace-clear",
            app.WORKSPACE_CLEAR_TOKEN,
            "--confirm-immediate-stop",
            app.IMMEDIATE_STOP_TOKEN,
            "--confirm-pregrasp-only",
            app.PREGRASP_ONLY_TOKEN,
        ]
    )
    captured = capsys.readouterr()
    assert result == 3
    assert imported == []
    assert "dedicated passing pregrasp-only audit" in captured.out
    assert "NOT IMPORTED; neither device was connected" in captured.out


def _prefix_artifact_for_executor():
    q0 = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.8]
    q1 = [0.01, 0.0, 0.0, -1.57, 0.0, 1.57, 0.8]
    return {
        "schema_version": 2,
        "artifact_sha256": "a" * 64,
        "execution_scope": "open_hand_current_to_pregrasp_only",
        "motion_authorized": False,
        "decision": {
            "required_geometry_passed": True,
            "runtime_workspace_clear_required": True,
        },
        "bindings": {
            "joint_prefix": {
                "prefix_contract_sha256": "b" * 64,
                "samples_sha256": "c" * 64,
                "waypoints": [
                    {"name": "current", "q_rad": q0},
                    {"name": "default", "q_rad": q0},
                    {"name": "pregrasp", "q_rad": q1},
                ],
                "pregrasp_pose_base_EE": np.eye(4).tolist(),
                "max_joint_step_rad": 0.01,
                "max_q_tracking_error_rad": 0.002,
            },
            "filtered_scene": {
                "captured_at_s": 1.0,
                "capture_to_prefix_current_linf_rad": 0.25,
            },
        },
        "policies": {"max_scene_age_s": 10.0},
    }


def test_pregrasp_runtime_rejects_legacy_schema_before_hardware():
    app = _load_app()
    artifact = _prefix_artifact_for_executor()
    artifact["schema_version"] = 1
    inspection = app.OfflineInspection(
        config=copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8"))),
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ()),
        plan=None,
        adapter_audit=None,
        pregrasp_audit=app.PregraspOnlyAuditBinding(
            Path("legacy-prefix.json"), artifact, ()
        ),
    )

    blockers = app._runtime_config_blockers(
        inspection, command="pregrasp", trajectory_mode="audited-joint"
    )

    assert any("requires schema-v2 evidence" in item for item in blockers)


@pytest.mark.parametrize("interrupt", [False, True])
def test_pregrasp_executor_uses_only_prefix_api_and_cleans_up(
    monkeypatch, capsys, interrupt
):
    app = _load_app()
    artifact = _prefix_artifact_for_executor()
    log = []

    class Arm:
        last_control_loop_telemetry = SimpleNamespace(
            kind="joint",
            samples=321,
            max_read_to_write_us=432.125,
            read_to_write_overruns=0,
        )

        def stop(self):
            log.append("arm.stop")

    class Hand:
        def close(self):
            log.append("hand.close")

    class Sequence:
        def __init__(self, arm, hand, settle_tolerances):
            self.state = SimpleNamespace(value="disarmed")
            log.append("sequence.construct")

        def run_to_pregrasp_joint_waypoints(self, plan):
            log.append("sequence.pregrasp-only")
            assert not hasattr(plan, "hand_target6")
            if interrupt:
                raise KeyboardInterrupt()
            self.state = SimpleNamespace(value="pregrasp_verified")
            return self.state

        def abort(self, reason):
            log.append("sequence.abort")
            self.state = SimpleNamespace(value="stopped")
            return self.state

    class PrefixPlan:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    hardware = {
        "FrankaMotionLimits": object,
        "SettleTolerances": lambda **kwargs: kwargs,
        "StagedGraspSequence": Sequence,
        "AuditedPregraspSequencePlan": PrefixPlan,
    }
    inspection = app.OfflineInspection(
        config=copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8"))),
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ()),
        plan=None,
        adapter_audit=None,
        pregrasp_audit=app.PregraspOnlyAuditBinding(
            Path("prefix.json"), artifact, ()
        ),
    )
    monkeypatch.setattr(app, "_load_hardware_types", lambda: hardware)
    monkeypatch.setattr(app, "_build_franka_limits", lambda *a, **k: object())
    monkeypatch.setattr(app, "_connect_hardware", lambda *a, **k: (Arm(), Hand()))
    monkeypatch.setattr(app, "_verify_installed_end_effector", lambda *a: log.append("ee.verify"))
    monkeypatch.setattr(app, "_install_continuous_franka_gate", lambda *a: log.append("watchdog.install"))
    monkeypatch.setattr(app, "load_pregrasp_only_audit", lambda *a, **k: artifact)
    monkeypatch.setattr(app, "_verify_live_q_matches_pregrasp_audit", lambda *a: log.append("live-q.verify"))
    monkeypatch.setattr(
        app,
        "require_uncontended_franka_https_link",
        lambda _robot_ip: log.append("host-network.verify"),
    )

    result = app._execute_hardware(SimpleNamespace(command="pregrasp"), inspection)
    assert result == (130 if interrupt else 0)
    assert "sequence.pregrasp-only" in log
    assert "live-q.verify" in log
    assert log.index("live-q.verify") < log.index("host-network.verify")
    assert log.index("host-network.verify") < log.index("sequence.construct")
    assert "sequence.abort" in log
    assert log[-2:] == ["arm.stop", "hand.close"]
    timing_output = capsys.readouterr().out
    assert (
        "[Franka/FCI timing] kind=joint samples=321 "
        "max_read_to_write=432.125us over_500us=0"
    ) in timing_output


def test_air_execution_config_uses_short_air_pregrasp_distance():
    app = _load_app()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    loaded = app._execution_config(config, preshape_q6=True)
    air = app._execution_config(config, preshape_q6=True, air_grasp=True)
    assert loaded.pregrasp_distance_m == pytest.approx(0.10)
    assert air.pregrasp_distance_m == pytest.approx(0.01)


def test_live_q_replaces_offline_current_q_age_and_scene_is_rechecked(monkeypatch):
    app = _load_app()
    expected_q = np.asarray([0.01, 0.0, 0.0, -1.57, 0.0, 1.57, 0.8])
    state = SimpleNamespace(q=expected_q.copy())

    class Arm:
        robot = SimpleNamespace(read_once=lambda: state)

        @staticmethod
        def _validate_state(value, *, require_idle, enforce_success):
            assert value is state
            assert require_idle is True
            assert enforce_success is False

    artifact = {
        "bindings": {
            "joint_path": {
                "waypoints": [{"name": "current", "q_rad": expected_q.tolist()}],
                # Deliberately ancient: wall age is replaced by the fresh live-q match.
                "current_q_captured_at_s": 1.0,
            },
            "scene": {"captured_at_s": 99.0},
        },
        "policies": {
            "max_q_tracking_error_rad": 0.002,
            "max_scene_age_s": 2.0,
        },
    }
    monkeypatch.setattr(app.time, "time", lambda: 100.0)
    app._verify_live_q_matches_audit(Arm(), artifact)

    monkeypatch.setattr(app.time, "time", lambda: 102.0)
    with pytest.raises(RuntimeError, match="scene expired"):
        app._verify_live_q_matches_audit(Arm(), artifact)


def test_live_q_mismatch_fails_before_any_motion(monkeypatch):
    app = _load_app()
    expected = np.zeros(7)
    state = SimpleNamespace(q=np.asarray([0.003, 0, 0, 0, 0, 0, 0]))
    calls = []

    class Arm:
        robot = SimpleNamespace(read_once=lambda: state)

        @staticmethod
        def _validate_state(value, *, require_idle, enforce_success):
            calls.append("read-only-validated")

    artifact = {
        "bindings": {
            "joint_path": {
                "waypoints": [{"name": "current", "q_rad": expected.tolist()}],
            },
            "scene": {"captured_at_s": 99.0},
        },
        "policies": {
            "max_q_tracking_error_rad": 0.002,
            "max_scene_age_s": 2.0,
        },
    }
    monkeypatch.setattr(app.time, "time", lambda: 100.0)
    with pytest.raises(RuntimeError, match="moved since collision audit"):
        app._verify_live_q_matches_audit(Arm(), artifact)
    assert calls == ["read-only-validated"]


def test_pregrasp_live_q_binding_is_exact_and_scene_age_is_advisory(
    monkeypatch, capsys
):
    app = _load_app()
    expected = np.asarray([0.01, 0.0, 0.0, -1.57, 0.0, 1.57, 0.8])
    state = SimpleNamespace(q=expected.copy())
    calls = []

    class Arm:
        robot = SimpleNamespace(read_once=lambda: state)

        @staticmethod
        def _validate_state(value, *, require_idle, enforce_success):
            calls.append((require_idle, enforce_success))

    artifact = {
        "bindings": {
            "joint_prefix": {
                "waypoints": [{"name": "current", "q_rad": expected.tolist()}],
                "max_q_tracking_error_rad": 0.002,
            },
            # Ancient and captured at a different q by design: both values are
            # advisory provenance in arm-only mode.
            "filtered_scene": {
                "captured_at_s": 1.0,
                "capture_to_prefix_current_linf_rad": 0.25,
            },
        },
        "policies": {"max_scene_age_s": 2.0},
    }
    monkeypatch.setattr(app.time, "time", lambda: 10_000.0)
    app._verify_live_q_matches_pregrasp_audit(Arm(), artifact)
    assert calls == [(True, False)]
    output = capsys.readouterr().out
    assert "filtered-scene age=9999.0s" in output
    assert "scene/prefix q delta=0.2500000rad" in output
    assert "is not an execution gate" in output

    state.q = expected + np.asarray([0.003, 0, 0, 0, 0, 0, 0])
    with pytest.raises(RuntimeError, match="moved since pregrasp-only audit"):
        app._verify_live_q_matches_pregrasp_audit(Arm(), artifact)


def test_formal_connect_adopts_disabled_hand_before_arm_connect():
    app = _load_app()
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    log = []

    class Hand:
        @classmethod
        def connect(cls, **kwargs):
            log.append(("hand.connect",))
            return cls()

        def adopt_disabled_state_and_verify(self):
            log.append(("hand.adopt_disabled",))

        def close(self):
            log.append(("hand.close",))

    class Arm:
        @classmethod
        def connect(cls, ip, limits, *, enforce_realtime):
            log.append(("arm.connect", ip, enforce_realtime))
            return object()

    app._connect_hardware(
        config,
        {"RH56SequenceDriver": Hand, "FrankaSequenceDriver": Arm},
        object(),
    )

    assert [entry[0] for entry in log] == [
        "hand.connect",
        "hand.adopt_disabled",
        "arm.connect",
    ]
    assert log[-1][2] is True


def test_arm_connect_failure_occurs_only_after_hand_disable_and_closes_hand():
    app = _load_app()
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    log = []

    class Hand:
        @classmethod
        def connect(cls, **kwargs):
            log.append("hand.connect")
            return cls()

        def adopt_disabled_state_and_verify(self):
            log.append("hand.adopt_disabled")

        def close(self):
            log.append("hand.close")

    class Arm:
        @classmethod
        def connect(cls, ip, limits, *, enforce_realtime):
            log.append(("arm.connect", enforce_realtime))
            raise RuntimeError("FCI unavailable")

    with pytest.raises(RuntimeError, match="FCI unavailable"):
        app._connect_hardware(
            config,
            {"RH56SequenceDriver": Hand, "FrankaSequenceDriver": Arm},
            object(),
        )

    assert log == [
        "hand.connect",
        "hand.adopt_disabled",
        ("arm.connect", True),
        "hand.close",
    ]


def test_continuous_franka_gate_is_installed_and_runs_fresh_read_gate(monkeypatch):
    app = _load_app()
    calls = []

    class Hand:
        def install_external_safety_check(self, callback):
            calls.append("installed")
            self.callback = callback

    hand = Hand()
    arm = object()
    config = {"franka": {}}
    monkeypatch.setattr(
        app,
        "_verify_installed_end_effector",
        lambda actual_arm, actual_config: calls.append(
            ("franka.read_gate", actual_arm, actual_config)
        ),
    )

    app._install_continuous_franka_gate(hand, arm, config)
    hand.callback()

    assert calls == ["installed", ("franka.read_gate", arm, config)]


def test_grasp_lift_parser_exposes_independent_audit_and_boundary_output(tmp_path):
    app = _load_app()
    output = tmp_path / "current_pose.json"
    args = app.build_parser().parse_args(
        [
            "grasp-lift",
            "--snapshot",
            "snapshot.npz",
            "--installed-tool-audit",
            "loaded-grasp.json",
            "--loaded-lift-audit",
            "loaded-lift.json",
            "--pose-state-output",
            str(output),
            "--confirm-loaded-lift-round-trip",
            app.LOADED_LIFT_TOKEN,
            "--confirm-setdown-support",
            app.LOAD_SUPPORT_TOKEN,
            "--confirm-load-rated-tool",
            app.LOAD_RATED_TOKEN,
        ]
    )

    assert args.command == "grasp-lift"
    assert args.trajectory_mode == "audited-joint"
    assert args.loaded_lift_audit == Path("loaded-lift.json")
    assert args.pose_state_output == output
    assert args.confirm_loaded_lift_round_trip == app.LOADED_LIFT_TOKEN
    assert args.confirm_setdown_support == app.LOAD_SUPPORT_TOKEN
    assert args.confirm_load_rated_tool == app.LOAD_RATED_TOKEN


def test_checked_in_pla_grasp_lift_dry_run_stays_locked_before_hardware_import(
    tmp_path, monkeypatch, capsys
):
    app = _load_app()
    calls = []
    output = tmp_path / "must-not-be-created.json"

    def forbidden():
        calls.append("hardware-import")
        raise AssertionError("locked loaded lift attempted hardware import")

    monkeypatch.setattr(app, "_load_hardware_types", forbidden)
    result = app.main(
        [
            "grasp-lift",
            "--config",
            str(CONFIG_PATH),
            "--snapshot",
            str(SNAPSHOT_PATH),
            "--selected-index",
            "0",
            "--pose-state-output",
            str(output),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert result == 3
    assert calls == []
    assert not output.exists()
    assert "loaded grasp execution refuses the installed PLA adapter" in captured.out
    assert "a passing loaded-lift round-trip audit is required" in captured.out
    assert "[dry-run/stage 13]" in captured.out
    assert "NOT IMPORTED; neither device was connected" in captured.out


def test_loaded_time_law_must_match_runtime_limits_before_hardware():
    app = _load_app()
    inspection = app.inspect_offline(
        CONFIG_PATH,
        SNAPSHOT_PATH,
        collision_margin_m=0.005,
        collision_samples=21,
        selected_index=0,
    )
    fake_binding = SimpleNamespace(
        passed=True,
        blockers=(),
        artifact={"schema_version": 1},
        approved_max_joint_velocity_rad_s=0.05,
        approved_max_joint_acceleration_rad_s2=0.10,
        time_law_max_joint_velocity_rad_s=0.05,
        time_law_max_joint_acceleration_rad_s2=0.10,
        time_law_max_dynamic_segment_rad=0.20,
        # The checked-in driver requires 6 s; a one-second law must be rejected
        # by the offline gate, not after importing or connecting hardware.
        time_law_min_segment_duration_s=1.0,
    )
    inspection = app.replace(inspection, loaded_lift_audit=fake_binding)

    blockers = app._runtime_config_blockers(
        inspection, command="grasp-lift", trajectory_mode="audited-joint"
    )

    assert "loaded-lift time-law duration is below runtime minimum" in blockers


def test_waypoint_pose_state_publisher_is_atomic_and_preview_compatible(
    tmp_path, monkeypatch
):
    app = _load_app()
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (0.45, -0.05, 0.31)
    state = SimpleNamespace(
        q=np.asarray([0.1, -0.2, 0.3, -1.5, 0.2, 1.4, 0.7]),
        O_T_EE=pose.reshape(-1, order="F"),
    )

    class Arm:
        def __init__(self):
            self.reads = 0
            self.robot = SimpleNamespace(read_once=self.read_once)

        def read_once(self):
            self.reads += 1
            return state

        @staticmethod
        def _validate_state(value, *, require_idle, enforce_success):
            assert value is state
            assert require_idle is True
            assert enforce_success is False

    class Hand:
        def __init__(self):
            self.reads = 0

        def read_state_snapshot(self):
            self.reads += 1
            return SimpleNamespace(
                angle_targets=(700, 710, 720, 730, 800, 900),
                angles=(701, 709, 720, 732, 800, 900),
                positions=(1, 2, 3, 4, 5, 6),
                forces=(10, 11, 12, 13, 14, 15),
                currents=(20, 21, 22, 23, 24, 25),
                errors=(0, 0, 0, 0, 0, 0),
                statuses=(3, 3, 2, 3, 2, 2),
                temperatures=(30, 30, 31, 31, 29, 30),
                contact_axes=("pinky", "ring", "index"),
            )

    arm = Arm()
    hand = Hand()
    output = tmp_path / "nested" / "current_pose.json"
    monkeypatch.setattr(app.time, "time", lambda: 1234.5)
    observer = app._build_pose_state_observer(output, arm, hand)

    observer(
        "grasp",
        state.q.copy(),
        pose.copy(),
        (700, 710, 720, 730, 800, 900),
    )

    preview_state = load_pose_state_json(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert preview_state.reference_frame == "robot_base"
    assert preview_state.stage == "grasp"
    assert preview_state.source == "execute_control_sequence_waypoint_boundary"
    assert preview_state.sequence == 1
    np.testing.assert_array_equal(preview_state.T_reference_EE, pose)
    assert payload["tracking_error"]["q_linf_rad"] == pytest.approx(0.0)
    assert payload["tracking_error"]["position_m"] == pytest.approx(0.0)
    assert payload["tracking_error"]["rotation_rad"] == pytest.approx(0.0)
    assert payload["tracking_error"]["hand_abs_units"] == [1, 1, 0, 2, 0, 0]
    assert payload["tracking_error"]["hand_max_abs_units"] == 2
    assert payload["hand"]["contact_axes"] == ["pinky", "ring", "index"]
    assert arm.reads == 1
    assert hand.reads == 1
    assert list(output.parent.glob(".*.tmp")) == []

    observer("lift", state.q.copy(), pose.copy(), None)
    assert load_pose_state_json(output).sequence == 2
    assert arm.reads == 2
    assert hand.reads == 2


def test_loaded_hold_wait_rechecks_both_devices_instead_of_blind_sleep():
    app = _load_app()
    checks = []

    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            checks.append(("sleep", float(seconds)))
            self.now += float(seconds)

    class Sequence:
        def verify_loaded_lift_holding(self):
            checks.append(("verify",))

    clock = Clock()
    app._monitor_loaded_hold(
        Sequence(),
        0.45,
        poll_interval_s=0.20,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert [item[0] for item in checks] == [
        "verify",
        "sleep",
        "verify",
        "sleep",
        "verify",
        "sleep",
        "verify",
    ]
    assert clock.now == pytest.approx(0.45)


@pytest.mark.parametrize(
    ("duration", "expected_kinds", "expected_time"),
    [
        (0.0, ["verify"], 0.0),
        (
            0.45,
            ["verify", "sleep", "verify", "sleep", "verify", "sleep", "verify"],
            0.45,
        ),
    ],
)
def test_bounded_hold_wait_polls_real_clock_and_zero_still_verifies_once(
    duration, expected_kinds, expected_time
):
    app = _load_app()
    calls = []

    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            calls.append(("sleep", float(seconds)))
            self.now += float(seconds)

    class Sequence:
        def verify_bounded_holding(self):
            calls.append(("verify",))

    clock = Clock()
    app._monitor_bounded_hold(
        Sequence(),
        duration,
        poll_interval_s=0.20,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert [item[0] for item in calls] == expected_kinds
    assert clock.now == pytest.approx(expected_time)


@pytest.mark.parametrize("failure", [RuntimeError("hold drift"), KeyboardInterrupt()])
def test_bounded_hold_monitor_propagates_failure_without_an_extra_sleep(failure):
    app = _load_app()
    calls = []

    class Sequence:
        def verify_bounded_holding(self):
            calls.append("verify")
            raise failure

    with pytest.raises(type(failure)):
        app._monitor_bounded_hold(
            Sequence(),
            1.0,
            monotonic=lambda: 0.0,
            sleep=lambda _seconds: calls.append("sleep"),
        )

    assert calls == ["verify"]


@pytest.mark.parametrize(
    ("failure", "expected_result"),
    [(RuntimeError("hold drift"), 4), (KeyboardInterrupt(), 130)],
)
def test_bounded_hold_failure_or_ctrl_c_uses_executor_cleanup(
    monkeypatch, failure, expected_result
):
    app = _load_app()
    log = []
    artifact = {
        "mode": "loaded_grasp",
        "artifact_sha256": "a" * 64,
        "policies": {"max_q_tracking_error_rad": 0.002},
    }

    class Arm:
        last_control_loop_telemetry = None

        def stop(self):
            log.append("arm.stop")

    class Hand:
        def close(self):
            log.append("hand.close")

    class Sequence:
        def __init__(self, arm, hand, *, settle_tolerances):
            self.state = SimpleNamespace(value="disarmed")
            log.append("sequence.construct")

        def run_full_joint_waypoints(self, _plan):
            self.state = SimpleNamespace(value="holding")
            log.append("sequence.close")
            return self.state

        def verify_bounded_holding(self):
            log.append("sequence.verify-hold")
            raise failure

        def abort(self, _reason):
            log.append("sequence.abort")
            self.state = SimpleNamespace(value="stopped")
            return self.state

    hardware = {
        "FrankaMotionLimits": object,
        "SettleTolerances": lambda **kwargs: kwargs,
        "StagedGraspSequence": Sequence,
        "AuditedJointSequencePlan": object,
    }
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    inspection = app.OfflineInspection(
        config=config,
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ()),
        plan=object(),
        adapter_audit=None,
        installed_audit=app.InstalledAuditBinding(
            Path("grasp-audit.json"), artifact, (), None
        ),
    )
    args = SimpleNamespace(
        command="grasp",
        trajectory_mode="audited-joint",
        stop_after_default=False,
        hold_seconds=0.0,
        pose_state_output=None,
    )
    monkeypatch.setattr(app, "_load_hardware_types", lambda: hardware)
    monkeypatch.setattr(app, "_build_franka_limits", lambda *a, **k: object())
    monkeypatch.setattr(app, "_connect_hardware", lambda *a, **k: (Arm(), Hand()))
    monkeypatch.setattr(
        app, "_verify_installed_end_effector", lambda *a: log.append("ee.verify")
    )
    monkeypatch.setattr(
        app,
        "_install_continuous_franka_gate",
        lambda *a: log.append("watchdog.install"),
    )
    monkeypatch.setattr(app, "load_installed_tool_audit", lambda *a, **k: artifact)
    monkeypatch.setattr(
        app, "_verify_live_q_matches_audit", lambda *a: log.append("live-q.verify")
    )
    monkeypatch.setattr(
        app,
        "require_uncontended_franka_https_link",
        lambda _ip: log.append("host-network.verify"),
    )
    monkeypatch.setattr(app, "_build_audited_joint_plan", lambda *a: object())

    result = app._execute_hardware(args, inspection)

    assert result == expected_result
    assert log.count("sequence.verify-hold") == 1
    assert log.index("sequence.verify-hold") < log.index("sequence.abort")
    assert log[-3:] == ["sequence.abort", "arm.stop", "hand.close"]


def test_continuous_telemetry_cli_pair_is_rejected_before_native_or_hardware(
    monkeypatch, capsys
):
    app = _load_app()
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    inspection = app.OfflineInspection(
        config=config,
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ()),
        plan=None,
        adapter_audit=None,
    )
    monkeypatch.setattr(app, "inspect_offline", lambda *a, **k: inspection)
    monkeypatch.setattr(
        app,
        "_load_hardware_types",
        lambda: (_ for _ in ()).throw(
            AssertionError("unpaired telemetry option imported hardware")
        ),
    )

    result = app.main(
        ["default", "--dry-run", "--continuous-telemetry", "/tmp/missing.map"]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "must be supplied together" in captured.err
    assert "NOT IMPORTED" in captured.out


def test_continuous_telemetry_dry_run_validates_request_but_never_starts_native(
    monkeypatch, capsys, tmp_path
):
    app = _load_app()
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    inspection = app.OfflineInspection(
        config=config,
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ()),
        plan=None,
        adapter_audit=None,
    )
    request = SimpleNamespace(
        mapping_path=(tmp_path / "session.map").resolve(),
        manifest=SimpleNamespace(
            identity=SimpleNamespace(run_uuid="12345678-1234-5678-9234-567812345678")
        ),
    )
    calls = []
    monkeypatch.setattr(app, "inspect_offline", lambda *a, **k: inspection)
    monkeypatch.setattr(
        app,
        "_load_continuous_telemetry_request",
        lambda *a, **k: calls.append("manifest.validate") or request,
    )
    monkeypatch.setattr(app, "_runtime_config_blockers", lambda *a, **k: ())
    monkeypatch.setattr(
        app,
        "_start_continuous_telemetry",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("dry-run started native producer")
        ),
    )
    monkeypatch.setattr(
        app,
        "_load_hardware_types",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run imported hardware")),
    )

    result = app.main(
        [
            "default",
            "--dry-run",
            "--continuous-telemetry",
            str(tmp_path / "session.map"),
            "--telemetry-session-manifest",
            str(tmp_path / "session.json"),
            "--continuous-telemetry-python-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert calls == ["manifest.validate"]
    assert "native producer NOT IMPORTED" in captured.out
    assert "dry-run completed" in captured.out


def test_formal_telemetry_starts_after_final_live_gate_and_closes_after_devices(
    monkeypatch, capsys
):
    app = _load_app()
    log = []

    class Arm:
        last_control_loop_telemetry = None

        def stop(self):
            log.append("arm.stop")

    class Hand:
        def close(self):
            log.append("hand.close")

    class Runtime:
        def observe_transition(self, state):
            log.append(("telemetry.transition", state))

        def close(self):
            log.append("telemetry.close")

    class Sequence:
        def __init__(self, arm, hand, *, settle_tolerances, transition_observer):
            assert callable(transition_observer)
            self.state = SimpleNamespace(value="disarmed")
            log.append("sequence.construct")

        def run_to_default(self, _plan):
            log.append("sequence.default")
            self.state = SimpleNamespace(value="default_verified")
            return self.state

        def abort(self, _reason):
            log.append("sequence.abort")
            self.state = SimpleNamespace(value="stopped")
            return self.state

    class DefaultPlan:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    hardware = {
        "FrankaMotionLimits": object,
        "SettleTolerances": lambda **kwargs: kwargs,
        "StagedGraspSequence": Sequence,
        "DefaultSequencePlan": DefaultPlan,
    }
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    inspection = app.OfflineInspection(
        config=config,
        config_path=CONFIG_PATH,
        assets=VerifiedAdapterAssets(Path("mesh"), Path("provenance"), "0" * 64),
        snapshot=None,
        readiness=ControlReadiness((), ()),
        plan=None,
        adapter_audit=None,
    )
    request = SimpleNamespace(
        mapping_path=Path("/tmp/fake.map"),
        manifest=SimpleNamespace(
            identity=SimpleNamespace(run_uuid="12345678-1234-5678-9234-567812345678")
        ),
    )
    initial = object()
    monkeypatch.setattr(app, "_load_hardware_types", lambda: hardware)
    monkeypatch.setattr(app, "_build_franka_limits", lambda *a, **k: object())
    monkeypatch.setattr(app, "_connect_hardware", lambda *a, **k: (Arm(), Hand()))
    monkeypatch.setattr(
        app,
        "_verify_installed_end_effector",
        lambda *a: (initial, 1000, 2000),
    )
    monkeypatch.setattr(
        app,
        "_install_continuous_franka_gate",
        lambda *a: log.append("watchdog.install"),
    )
    monkeypatch.setattr(
        app,
        "require_uncontended_franka_https_link",
        lambda _ip: log.append("final-live-gate"),
    )

    def start(_request, **kwargs):
        assert kwargs["initial_validated_arm_state"] is initial
        assert kwargs["initial_arm_timestamp_unix_ns"] == 1000
        assert kwargs["initial_arm_timestamp_monotonic_ns"] == 2000
        log.append("telemetry.start")
        return Runtime()

    monkeypatch.setattr(app, "_start_continuous_telemetry", start)
    result = app._execute_hardware(
        SimpleNamespace(command="default", pose_state_output=None),
        inspection,
        telemetry_request=request,
    )

    assert result == 0
    assert log.index("final-live-gate") < log.index("telemetry.start")
    assert log.index("telemetry.start") < log.index("sequence.construct")
    assert log.index("arm.stop") < log.index("telemetry.close")
    assert log.index("hand.close") < log.index("telemetry.close")
    assert "success" in capsys.readouterr().out
