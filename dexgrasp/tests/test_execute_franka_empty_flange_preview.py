from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_app(name="test_execute_franka_empty_flange_preview_app"):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "apps" / "execute_franka_empty_flange_preview.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def app():
    return _load_app()


def _write_valid_artifacts(app, tmp_path, *, capture_s=1_000.0):
    scene_path = tmp_path / "live_scene.npz"
    rng = np.random.default_rng(7)
    points = rng.normal(size=(256, 3)).astype(np.float32)
    np.savez_compressed(
        scene_path,
        schema_name=np.asarray("dexgrasp_live_scene"),
        schema_version=np.asarray(1),
        scene_points=points,
        scene_colors=np.zeros_like(points),
        reference_frame=np.asarray("robot_base"),
        calibration_id=np.asarray("eye-to-hand-test"),
        camera_serial=np.asarray("337322072188"),
        capture_q_rad=app.DEFAULT_Q,
        capture_q_source=np.asarray("cli_asserted"),
        capture_completed_s=np.asarray(capture_s),
    )
    digest = app._sha256_file(scene_path)
    plan = {
        "schema_version": 2,
        "artifact_type": app.ARTIFACT_TYPE,
        "diagnostic_only": True,
        "installed_mount_calibration": False,
        "saved_pose_only": False,
        "q_start_rad": app.DEFAULT_Q.tolist(),
        "virtual_tcp": {
            "source": "upstream UR preview convention only",
            "F_T_TCP_z_m": 0.044,
            "must_not_copy_to_installed_profile": True,
        },
        "collision_scene_input": {
            "source_kind": "live_scene_npz",
            "scene_path": str(scene_path),
            "scene_sha256": digest,
            "source_schema_name": "dexgrasp_live_scene",
            "source_schema_version": 1,
            "saved_grasp_pose_retained": True,
            "snapshot_scene_ignored": True,
            "saved_pose_only": False,
            "stale_object_pose_override": False,
            "reference_frame": "robot_base",
            "calibration_id": "eye-to-hand-test",
            "camera_serial": "337322072188",
            "scene_points_raw": len(points),
            "capture_q_rad": app.DEFAULT_Q.tolist(),
            "capture_q_source": "cli_asserted",
            "capture_q_value_source": "live_scene_npz:capture_q_rad",
            "robot_return_filter_q_rad": app.DEFAULT_Q.tolist(),
            "robot_return_filter_q_source": "live_scene_npz:capture_q_rad",
            "capture_completed_s": capture_s,
        },
        "T_robot_base_flange_pregrasp": np.eye(4).tolist(),
        "T_robot_base_flange_grasp": np.eye(4).tolist(),
        "q_pregrasp_rad": (
            app.DEFAULT_Q + [0.10, 0.0, 0.0, -0.10, 0.0, 0.10, 0.0]
        ).tolist(),
        "q_grasp_rad": (
            app.DEFAULT_Q + [0.12, 0.02, 0.0, -0.12, 0.0, 0.12, 0.02]
        ).tolist(),
        "collision_audit": {
            "self_collision_free": True,
            "scene_capsule_clear": True,
            "live_scene_revalidated": True,
            "object_alignment_valid": True,
            "stale_object_pose_preview": False,
            "live_object_alignment": {
                "median_distance_m": 0.002,
                "p95_distance_m": 0.004,
                "maximum_distance_m": 0.006,
                "coverage_distance_m": 0.010,
                "coverage_fraction": 0.95,
                "median_max_m": 0.008,
                "p95_max_m": 0.015,
                "minimum_coverage": 0.80,
                "passed": True,
                "failures": [],
            },
            "authoritative_for_installed_tool": False,
            "scene_sha256": digest,
            "robot_return_filter_q_rad": app.DEFAULT_Q.tolist(),
            "robot_return_filter_q_source": "live_scene_npz:capture_q_rad",
            "scene_margin_m": 0.005,
            "minimum_sampled_signed_distance_m": 0.020,
            "path_samples": 201,
            "audited_at_s": capture_s + 1.0,
        },
    }
    plan_path = tmp_path / "preview.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path, scene_path, plan


def test_valid_plan_is_bound_to_fresh_live_scene(app, tmp_path):
    plan_path, scene_path, _ = _write_valid_artifacts(app, tmp_path)

    result = app.load_preview_plan(plan_path, now_s=1_005.0)

    assert result.artifact_path == plan_path
    assert result.scene_path == scene_path
    assert result.camera_serial == "337322072188"
    assert result.calibration_id == "eye-to-hand-test"
    np.testing.assert_allclose(result.q_capture_rad, app.DEFAULT_Q)


def test_legacy_v2_named_virtual_tcp_guard_remains_readable(app, tmp_path):
    plan_path, _, plan = _write_valid_artifacts(app, tmp_path)
    virtual_tcp = plan["virtual_tcp"]
    virtual_tcp["must_not_copy_to_installed_V2_profile"] = virtual_tcp.pop(
        "must_not_copy_to_installed_profile"
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = app.load_preview_plan(plan_path, now_s=1_005.0)

    assert result.artifact_path == plan_path


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value.update(schema_version=1), "schema_version"),
        (lambda value: value.update(diagnostic_only=False), "diagnostic_only"),
        (
            lambda value: value["collision_scene_input"].update(source_kind="snapshot"),
            "live_scene_npz",
        ),
        (
            lambda value: value["collision_audit"].update(
                live_scene_revalidated=False
            ),
            "live_scene_revalidated",
        ),
        (
            lambda value: value["collision_audit"].update(
                object_alignment_valid=False
            ),
            "object alignment",
        ),
    ],
)
def test_non_executable_artifact_flags_fail_closed(app, tmp_path, mutate, match):
    plan_path, _, document = _write_valid_artifacts(app, tmp_path)
    mutate(document)
    plan_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        app.load_preview_plan(plan_path, now_s=1_005.0)


def test_stale_capture_and_changed_scene_fail_closed(app, tmp_path):
    plan_path, scene_path, _ = _write_valid_artifacts(app, tmp_path)
    with pytest.raises(ValueError, match="stale"):
        app.load_preview_plan(plan_path, now_s=1_500.0)

    with scene_path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed after collision planning"):
        app.load_preview_plan(plan_path, now_s=1_005.0)


def _memory_plan(app, tmp_path):
    immutable = lambda value: np.asarray(value, dtype=np.float64)
    return app.PreviewPlan(
        artifact_path=tmp_path / "plan.json",
        scene_path=tmp_path / "scene.npz",
        scene_sha256="0" * 64,
        camera_serial="camera",
        calibration_id="calibration",
        capture_completed_s=1_000.0,
        audited_at_s=1_001.0,
        q_start_rad=immutable(app.DEFAULT_Q),
        q_capture_rad=immutable(app.DEFAULT_Q),
        q_pregrasp_rad=immutable(app.DEFAULT_Q + [0.1, 0, 0, -0.1, 0, 0.1, 0]),
        q_grasp_rad=immutable(app.DEFAULT_Q + [0.2, 0, 0, -0.2, 0, 0.2, 0]),
        scene_margin_m=0.005,
        minimum_clearance_m=0.02,
        saved_pose_only=False,
    )


class _FakeArm:
    def __init__(self, initial_q):
        self.state = SimpleNamespace(q=np.asarray(initial_q, dtype=np.float64))
        self.robot = SimpleNamespace(read_once=lambda: self.state)
        self.limits = SimpleNamespace(joint_arrival_tolerance_rad=0.015)
        self.moves = []
        self.validation_calls = []
        self.stop_count = 0

    def _validate_state(self, state, *, require_idle, enforce_success):
        self.validation_calls.append((require_idle, enforce_success))

    def move_joints(self, target):
        q = np.asarray(target, dtype=np.float64).copy()
        self.moves.append(q)
        self.state.q = q

    def stop(self):
        self.stop_count += 1


def test_executor_runs_reversible_low_risk_sequence_and_stops(app, tmp_path, monkeypatch):
    plan = _memory_plan(app, tmp_path)
    arm = _FakeArm(app.DEFAULT_Q)
    sleeps = []
    monkeypatch.setattr(app, "_connect_bare_franka", lambda _ip: arm)
    monkeypatch.setattr(app, "_sha256_file", lambda _path: plan.scene_sha256)
    monkeypatch.setattr(app.time, "time", lambda: 1_002.0)

    app.execute_preview(plan, "robot", sleep=sleeps.append)

    assert len(arm.moves) == 5
    np.testing.assert_allclose(arm.moves[0], plan.q_start_rad)
    np.testing.assert_allclose(arm.moves[1], plan.q_pregrasp_rad)
    np.testing.assert_allclose(arm.moves[2], plan.q_grasp_rad)
    np.testing.assert_allclose(arm.moves[3], plan.q_pregrasp_rad)
    np.testing.assert_allclose(arm.moves[4], plan.q_start_rad)
    assert sleeps == [1.0]
    assert arm.stop_count == 1


def test_live_q_mismatch_stops_before_any_motion(app, tmp_path, monkeypatch):
    plan = _memory_plan(app, tmp_path)
    arm = _FakeArm(app.DEFAULT_Q + [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(app, "_connect_bare_franka", lambda _ip: arm)
    monkeypatch.setattr(app, "_sha256_file", lambda _path: plan.scene_sha256)
    monkeypatch.setattr(app.time, "time", lambda: 1_002.0)

    with pytest.raises(RuntimeError, match="moved away"):
        app.execute_preview(plan, "robot", sleep=lambda _seconds: None)

    assert arm.moves == []
    assert arm.stop_count == 1


def test_missing_confirmations_never_reach_hardware(app, tmp_path, monkeypatch, capsys):
    plan = _memory_plan(app, tmp_path)
    calls = []
    monkeypatch.setattr(app, "load_preview_plan", lambda _path: plan)
    monkeypatch.setattr(app, "execute_preview", lambda *args: calls.append(args))

    assert app.main(["--plan", str(tmp_path / "plan.json")]) == 1
    assert calls == []
    assert "exact confirmations required" in capsys.readouterr().err


def test_exact_confirmations_are_required_before_hardware(app, tmp_path, monkeypatch):
    plan = _memory_plan(app, tmp_path)
    calls = []
    monkeypatch.setattr(app, "load_preview_plan", lambda _path: plan)
    monkeypatch.setattr(app, "execute_preview", lambda *args: calls.append(args))

    result = app.main(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--confirm-flange-empty",
            app.FLANGE_TOKEN,
            "--confirm-workspace-clear",
            app.CLEAR_TOKEN,
            "--confirm-stop-ready",
            app.STOP_TOKEN,
            "--confirm-live-scene-unchanged",
            app.SCENE_TOKEN,
            "--confirm-diagnostic-preview",
            app.PREVIEW_TOKEN,
        ]
    )

    assert result == 0
    assert calls == [(plan, "172.16.0.2")]


def test_bare_driver_configuration_requires_identity_and_zero_dynamics(
    app, monkeypatch
):
    captured = {}

    class Limits:
        def __init__(self, **kwargs):
            captured["limits"] = kwargs

    class Driver:
        @classmethod
        def connect(cls, *args, **kwargs):
            captured["connect"] = (args, kwargs)
            return "driver"

    fake_module = SimpleNamespace(
        FrankaMotionLimits=Limits, FrankaSequenceDriver=Driver
    )
    monkeypatch.setitem(
        sys.modules, "anydex_pipeline.franka_sequence_driver", fake_module
    )

    assert app._connect_bare_franka("robot") == "driver"
    values = captured["limits"]
    np.testing.assert_allclose(values["expected_F_T_EE"], np.eye(4))
    assert values["expected_m_ee_kg"] == 0.0
    np.testing.assert_allclose(values["expected_F_x_Cee_m"], np.zeros(3))
    np.testing.assert_allclose(values["expected_I_ee_kg_m2"], np.zeros((3, 3)))
    assert values["max_joint_speed_rad_s"] == 0.04
