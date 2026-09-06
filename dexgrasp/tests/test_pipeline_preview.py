from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from anydex_pipeline.control_plan import (
    AdapterGeometry,
    GraspExecutionPlan,
    PlannedStage,
    StageName,
)
from anydex_pipeline.pipeline_preview import (
    build_pipeline_preview_contract,
    load_pose_state_json,
    pose_error,
    pose_state_from_mapping,
)
from anydex_pipeline.continuous_telemetry import (
    NATIVE_TELEMETRY_ABI_SCHEMA_SHA256,
    TelemetryIdentity,
)
from anydex_pipeline.snapshot import (
    GraspCandidates,
    VisualizationSnapshot,
    save_snapshot_npz,
)
from anydex_pipeline.viewer_ready import viewer_ready_is_published


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"


def _pose(translation=(0.0, 0.0, 0.0), yaw_deg=0.0):
    yaw = np.deg2rad(yaw_deg)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def _plan():
    T_EE_hand = _pose([0.0, 0.0, 0.01], yaw_deg=-45.0)
    grasp = _pose([0.60, 0.10, 0.30], yaw_deg=20.0)
    pregrasp = grasp.copy()
    pregrasp[0, 3] -= 0.10
    stages = (
        PlannedStage(
            StageName.FRANKA_DEFAULT,
            "default",
            franka_q=np.asarray([0.0, 0.1, 0.2, -1.5, 0.0, 1.5, 0.5]),
        ),
        PlannedStage(
            StageName.INSPIRE_CLOSE,
            "close",
            inspire_angles=np.asarray([700, 710, 720, 730, 850, 940]),
        ),
    )
    return GraspExecutionPlan(
        reference_frame="robot_base",
        selected_index=3,
        T_reference_hand=grasp @ T_EE_hand,
        T_EE_hand=T_EE_hand,
        T_reference_EE_grasp=grasp,
        T_reference_EE_pregrasp=pregrasp,
        approach_reference=np.asarray([1.0, 0.0, 0.0]),
        stages=stages,
        adapter=AdapterGeometry(),
        diagnostic=True,
        official_model=False,
        calibrated=True,
        mount_transform_commissioned=True,
        execution_eligible=False,
        execution_blockers=("diagnostic",),
    )


def test_pose_error_is_translation_plus_so3_geodesic():
    actual = _pose([0.0, 0.0, 0.0], yaw_deg=0.0)
    target = _pose([0.03, 0.04, 0.0], yaw_deg=90.0)

    error = pose_error(actual, target)

    assert error.position_m == pytest.approx(0.05)
    assert error.position_mm == pytest.approx(50.0)
    assert error.rotation_rad == pytest.approx(np.pi / 2.0)
    assert error.rotation_deg == pytest.approx(90.0)


def test_pose_state_json_is_strict_and_stale_samples_are_rejected(tmp_path):
    path = tmp_path / "pose.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reference_frame": "robot_base",
                "timestamp_unix_s": 100.0,
                "T_reference_EE": _pose([0.5, 0.1, 0.3]).tolist(),
                "target": {
                    "T_reference_EE": _pose([0.6, 0.1, 0.35]).tolist(),
                },
                "stage": "moving_pregrasp",
                "source": "test",
                "sequence": 7,
                "hand": {
                    "angles": [900, 800, 700, 600, 500, 400],
                    "angle_targets": [-1, -1, 710, 610, 510, 410],
                },
            }
        ),
        encoding="utf-8",
    )

    state = load_pose_state_json(path, max_age_s=0.5, now_unix_s=100.4)
    assert state.sequence == 7
    assert state.stage == "moving_pregrasp"
    np.testing.assert_allclose(state.T_reference_EE[:3, 3], [0.5, 0.1, 0.3])
    np.testing.assert_allclose(
        state.target_T_reference_EE[:3, 3], [0.6, 0.1, 0.35]
    )
    assert state.hand_angles == (900, 800, 700, 600, 500, 400)
    assert state.hand_angle_targets == (-1, -1, 710, 610, 510, 410)

    with pytest.raises(ValueError, match="stale"):
        load_pose_state_json(path, max_age_s=0.5, now_unix_s=100.6)

    # A far-future timestamp must not be clamped to age zero and accepted
    # indefinitely.  A tiny scheduling skew remains tolerated.
    state = load_pose_state_json(path, max_age_s=0.5, now_unix_s=99.96)
    assert state.age_s(99.96) == 0.0
    with pytest.raises(ValueError, match="future"):
        load_pose_state_json(path, max_age_s=0.5, now_unix_s=99.94)


def test_pose_state_refuses_bad_frame_and_nonrigid_rotation():
    base = {
        "schema_version": 1,
        "reference_frame": "camera",
        "timestamp_unix_s": 1.0,
        "T_reference_EE": np.eye(4).tolist(),
    }
    with pytest.raises(ValueError, match="robot_base"):
        pose_state_from_mapping(base)

    base["reference_frame"] = "robot_base"
    base["T_reference_EE"][0][0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        pose_state_from_mapping(base)

    base["T_reference_EE"] = np.eye(4).tolist()
    base["target"] = []
    with pytest.raises(ValueError, match="target must be a JSON object"):
        pose_state_from_mapping(base)


@pytest.mark.parametrize(
    "hand, message",
    [
        ([], "hand must be a JSON object"),
        ({}, "missing angles"),
        ({"angles": [0, 1, 2, 3, 4]}, "six integers"),
        ({"angles": [0, 1, 2, 3, 4, True]}, "six integers"),
        ({"angles": [0, 1, 2, 3, 4, 5.0]}, "six integers"),
        ({"angles": [0, 1, 2, 3, 4, 1001]}, r"\[0,1000\]"),
        (
            {
                "angles": [0, 1, 2, 3, 4, 5],
                "angle_targets": [-2, 1, 2, 3, 4, 5],
            },
            r"\[-1,1000\]",
        ),
    ],
)
def test_pose_state_refuses_malformed_hand_register_feedback(hand, message):
    payload = {
        "schema_version": 1,
        "reference_frame": "robot_base",
        "timestamp_unix_s": 1.0,
        "T_reference_EE": np.eye(4).tolist(),
        "hand": hand,
    }
    with pytest.raises(ValueError, match=message):
        pose_state_from_mapping(payload)


def test_five_stage_lift_contract_is_preview_only_and_moves_vertical():
    plan = _plan()

    contract = build_pipeline_preview_contract(
        plan,
        lift_distance_m=0.05,
        adapter_material="PLA",
        low_speed_unloaded_only=True,
    )

    assert [stage.name for stage in contract.stages] == [
        "default",
        "pregrasp",
        "grasp",
        "close",
        "lift",
    ]
    assert contract.hardware_execution_allowed is False
    np.testing.assert_allclose(
        contract.T_reference_EE_lift[:3, 3]
        - contract.T_reference_EE_grasp[:3, 3],
        [0.0, 0.0, 0.05],
    )
    np.testing.assert_allclose(
        contract.lift_hand_pose,
        contract.T_reference_EE_lift @ contract.T_EE_hand,
    )
    assert any("PLA" in blocker for blocker in contract.hardware_blockers)
    assert any("loaded-lift audit" in blocker for blocker in contract.hardware_blockers)


def _diagnostic_snapshot() -> VisualizationSnapshot:
    hand = _pose([0.60, 0.10, 0.30], yaw_deg=20.0)
    grasps = GraspCandidates(
        canonical_poses=hand[None, ...],
        scores=np.asarray([0.9], dtype=np.float32),
        type_ids=np.asarray([4], dtype=np.int32),
        collision_free=np.asarray([False], dtype=np.bool_),
        collision_checked=np.asarray([False], dtype=np.bool_),
        selected_index=0,
        hand_poses=hand[None, ...],
        hand_angles=np.asarray([[700, 710, 720, 730, 850, 940]], dtype=np.float32),
        widths_m=np.asarray([0.06], dtype=np.float32),
        depths_m=np.asarray([0.02], dtype=np.float32),
    )
    return VisualizationSnapshot(
        scene_points=np.asarray([[0.4, 0.0, 0.1]], dtype=np.float32),
        scene_colors=np.asarray([[0.3, 0.3, 0.3]], dtype=np.float32),
        object_points=np.asarray([[0.6, 0.1, 0.3]], dtype=np.float32),
        object_colors=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        grasps=grasps,
        reference_frame="robot_base",
        T_reference_camera=np.eye(4),
        frame_id=1,
        timestamp_s=1.0,
        calibration_id="eye-to-hand-b722bce10485c8a3",
        camera_serial="337322072188",
        model_name="diagnostic geometric preview",
    )


def _load_app():
    path = ROOT / "apps/live_pipeline_preview.py"
    spec = importlib.util.spec_from_file_location("test_live_pipeline_preview_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_only_prints_all_stages_without_camera_or_hardware(tmp_path, capsys):
    snapshot_path = save_snapshot_npz(tmp_path / "snapshot.npz", _diagnostic_snapshot())
    app = _load_app()

    result = app.main(
        [
            str(snapshot_path),
            "--control-config",
            str(CONFIG),
            "--validate-only",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "PREVIEW ONLY" in captured.out
    assert "[preview-stage 1/5] default" in captured.out
    assert "[preview-stage 5/5] lift" in captured.out
    assert "hardware_execution_allowed=False" in captured.out
    assert "execution_mode=air" in captured.out
    assert "shared_with=planner,audit,executor" in captured.out
    assert "camera and Open3D were not imported" in captured.out
    assert "pylibfranka" not in sys.modules


def test_air_preview_uses_shared_retreat_contract_and_contact_is_explicit(tmp_path):
    snapshot_path = save_snapshot_npz(tmp_path / "snapshot.npz", _diagnostic_snapshot())
    app = _load_app()

    air_args = app.build_parser().parse_args(
        [str(snapshot_path), "--control-config", str(CONFIG), "--validate-only"]
    )
    contact_args = app.build_parser().parse_args(
        [
            str(snapshot_path),
            "--control-config",
            str(CONFIG),
            "--execution-mode",
            "contact",
            "--validate-only",
        ]
    )
    air = app.load_preview_inputs(air_args)
    contact = app.load_preview_inputs(contact_args)
    source = _diagnostic_snapshot()
    approach = source.grasps.canonical_poses[0, :3, :3] @ np.asarray(
        source.grasps.approach_axis_local, dtype=np.float64
    )
    # The diagnostic canonical pose has local +X approach and shares the same
    # rotation as its hand pose.  The configured air contract must retreat the
    # final target by 80 mm and then the pregrasp by another 10 mm.
    np.testing.assert_allclose(
        air.contract.T_reference_EE_grasp[:3, 3]
        - contact.contract.T_reference_EE_grasp[:3, 3],
        -0.08 * approach,
        atol=1.0e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        air.contract.stages[1].T_reference_EE[:3, 3]
        - air.contract.T_reference_EE_grasp[:3, 3],
        -0.01 * approach,
        atol=1.0e-10,
        rtol=0.0,
    )


def test_auto_error_target_prefers_exact_telemetry_then_stage_fallback():
    app = _load_app()
    contract = build_pipeline_preview_contract(
        _plan(),
        lift_distance_m=0.05,
        adapter_material="PLA",
        low_speed_unloaded_only=True,
    )
    exact = _pose([0.61, 0.11, 0.36], yaw_deg=5.0)
    state = pose_state_from_mapping(
        {
            "schema_version": 1,
            "reference_frame": "robot_base",
            "timestamp_unix_s": 1.0,
            "T_reference_EE": _pose().tolist(),
            "target": {"T_reference_EE": exact.tolist()},
            "stage": "lift_transit_0",
        }
    )
    label, target = app._active_target(contract, "auto", state)
    assert label == "telemetry"
    np.testing.assert_allclose(target, exact)

    state_without_exact = pose_state_from_mapping(
        {
            "schema_version": 1,
            "reference_frame": "robot_base",
            "timestamp_unix_s": 1.0,
            "T_reference_EE": _pose().tolist(),
            "stage": "lift_transit_0",
        }
    )
    label, target = app._active_target(contract, "auto", state_without_exact)
    assert label == "lift"
    np.testing.assert_allclose(target, contract.T_reference_EE_lift)

    setdown = replace(state_without_exact, stage="setdown_grasp")
    label, target = app._active_target(contract, "auto", setdown)
    assert label == "grasp"
    np.testing.assert_allclose(target, contract.T_reference_EE_grasp)

    moving_grasp = replace(state_without_exact, stage="moving_franka_grasp")
    label, target = app._active_target(contract, "auto", moving_grasp)
    assert label == "grasp"
    np.testing.assert_allclose(target, contract.T_reference_EE_grasp)

    moving_pregrasp = replace(state_without_exact, stage="moving_franka_pregrasp")
    label, target = app._active_target(contract, "auto", moving_pregrasp)
    assert label == "pregrasp"
    np.testing.assert_allclose(target, contract.stages[1].T_reference_EE)


class _FakeMesh:
    def __init__(self):
        self.transforms = []
        self.color = None

    def transform(self, transform):
        self.transforms.append(np.asarray(transform, dtype=np.float64).copy())

    def paint_uniform_color(self, color):
        self.color = tuple(color)

    def compute_vertex_normals(self):
        return None


class _FakeLine(_FakeMesh):
    pass


class _FakeVisualizer:
    def __init__(self):
        self.added = []
        self.removed = []
        self.updated = []

    def add_geometry(self, geometry, reset_bounding_box=False):
        self.added.append(geometry)

    def remove_geometry(self, geometry, reset_bounding_box=False):
        self.removed.append(geometry)

    def update_geometry(self, geometry):
        self.updated.append(geometry)


class _FakeMapper:
    def __init__(self):
        self.inputs = []

    def feedback_to_joint_positions_rad(self, values):
        self.inputs.append(tuple(values))
        return np.asarray(values * 2, dtype=np.float64)[:12] * 0.001


class _FakeHandModel:
    def __init__(self):
        self.links = (SimpleNamespace(name="link_a"), SimpleNamespace(name="link_b"))
        self.built_joints = []

    @staticmethod
    def _transforms(hand_pose, joints):
        first = np.asarray(hand_pose, dtype=np.float64).copy()
        second = np.asarray(hand_pose, dtype=np.float64).copy()
        first[0, 3] += float(joints[0])
        second[1, 3] += float(joints[1])
        return {"link_a": first, "link_b": second}

    def link_mesh_transforms(self, hand_pose, joints):
        return self._transforms(hand_pose, joints)

    def build_link_meshes(self, hand_pose, joints):
        self.built_joints.append(np.asarray(joints, dtype=np.float64).copy())
        return [_FakeMesh(), _FakeMesh()]


def _fake_o3d():
    return SimpleNamespace(
        geometry=SimpleNamespace(
            TriangleMesh=SimpleNamespace(
                create_coordinate_frame=lambda size: _FakeMesh()
            ),
            LineSet=_FakeLine,
        ),
        utility=SimpleNamespace(
            Vector2iVector=lambda value: value,
            Vector3dVector=lambda value: value,
        ),
    )


def test_run_viewer_ready_order_is_window_then_d435_then_mapping_wait(
    tmp_path, monkeypatch
):
    app = _load_app()
    events = []
    marker = tmp_path / "live.ready"

    class LiveVisualizer(_FakeVisualizer):
        def create_window(self, **_kwargs):
            events.append("window_created")
            return True

        def get_render_option(self):
            return SimpleNamespace(point_size=0.0)

        def poll_events(self):
            events.append("window_pumped")
            return True

        def update_renderer(self):
            events.append("window_rendered")

        def destroy_window(self):
            events.append("window_destroyed")

    visualizer = LiveVisualizer()
    o3d = _fake_o3d()
    o3d.visualization = SimpleNamespace(Visualizer=lambda: visualizer)
    monkeypatch.setitem(sys.modules, "open3d", o3d)

    scene = _FakeMesh()
    bundle = SimpleNamespace(
        scene_point_cloud=scene,
        geometry_list=lambda: [scene],
    )
    visualization = SimpleNamespace(
        VisualizationStyle=lambda **_kwargs: SimpleNamespace(
            scene_brightness=1.0, scene_color_floor=0.0
        ),
        build_open3d_geometries=lambda *_args, **_kwargs: bundle,
    )
    monkeypatch.setitem(
        sys.modules, "anydex_pipeline.visualization", visualization
    )

    class SceneSource:
        def __init__(self, *_args, **_kwargs):
            events.append("d435_started_and_warmed")

        def stop(self):
            events.append("d435_stopped")

    monkeypatch.setattr(app, "_RealSenseSceneSource", SceneSource)

    class NativeSource:
        def __init__(self, *_args, before_mapping_wait=None, **_kwargs):
            events.append("native_reader_validated")
            assert before_mapping_wait is not None
            before_mapping_wait()
            events.append("mapping_wait")
            raise RuntimeError("offline stop at mapping wait")

    monkeypatch.setattr(app, "_NativeTelemetrySource", NativeSource)
    publish = app.publish_viewer_ready

    def record_publish(path):
        events.append("ready_published")
        return publish(path)

    monkeypatch.setattr(app, "publish_viewer_ready", record_publish)

    args = app.build_parser().parse_args(
        [
            "snapshot.npz",
            "--source",
            "realsense",
            "--continuous-telemetry",
            "/tmp/offline-test.map",
            "--telemetry-session-manifest",
            "manifest.json",
            "--ready-file",
            str(marker),
            "--no-target-hand-mesh",
        ]
    )
    contract = build_pipeline_preview_contract(
        _plan(),
        lift_distance_m=0.05,
        adapter_material="PLA",
        low_speed_unloaded_only=True,
    )
    inputs = app.PreviewInputs(
        snapshot=_diagnostic_snapshot(),
        control_config={},
        contract=contract,
        execution_mode="air",
        telemetry_manifest=SimpleNamespace(identity=_telemetry_identity()),
    )

    with pytest.raises(RuntimeError, match="offline stop at mapping wait"):
        app.run_viewer(args, inputs)

    assert events.index("window_created") < events.index("d435_started_and_warmed")
    assert events.index("d435_started_and_warmed") < events.index("window_pumped")
    assert events[-5:] == [
        "native_reader_validated",
        "ready_published",
        "mapping_wait",
        "d435_stopped",
        "window_destroyed",
    ]
    assert stat.S_IMODE(marker.stat().st_mode) == 0o444


def test_current_hand_overlay_uses_actual_register_feedback_and_hides_when_missing():
    app = _load_app()
    mapper = _FakeMapper()
    model = _FakeHandModel()
    visualizer = _FakeVisualizer()
    overlay = app._CurrentPoseOverlay(
        _fake_o3d(),
        visualizer,
        T_EE_hand=np.eye(4),
        target_ee=np.eye(4),
        current_hand_model=model,
        current_hand_mapper=mapper,
        show_current_hand_mesh=True,
    )
    first = pose_state_from_mapping(
        {
            "schema_version": 1,
            "reference_frame": "robot_base",
            "timestamp_unix_s": 1.0,
            "T_reference_EE": _pose([0.4, 0.1, 0.3]).tolist(),
            "hand": {"angles": [900, 800, 700, 600, 500, 400]},
        }
    )
    overlay.update(first, target_ee=np.eye(4))

    assert mapper.inputs == [(900, 800, 700, 600, 500, 400)]
    np.testing.assert_allclose(
        model.built_joints[0],
        np.asarray([900, 800, 700, 600, 500, 400] * 2)[:12] * 0.001,
    )
    assert len(overlay.current_meshes) == 2
    assert all(mesh.color == (0.10, 0.95, 0.35) for mesh in overlay.current_meshes)

    second = replace(
        first,
        timestamp_unix_s=2.0,
        sequence=1,
        hand_angles=(850, 800, 700, 600, 500, 350),
    )
    overlay.update(second, target_ee=np.eye(4))
    assert mapper.inputs[-1] == (850, 800, 700, 600, 500, 350)
    assert all(mesh.transforms for mesh in overlay.current_meshes)

    no_hand = replace(second, timestamp_unix_s=3.0, sequence=2, hand_angles=None)
    prior_meshes = list(overlay.current_meshes)
    overlay.update(no_hand, target_ee=np.eye(4))
    assert overlay.current_meshes == []
    assert visualizer.removed == prior_meshes


def test_current_feedback_overlay_removes_all_stale_geometry_and_recovers():
    app = _load_app()
    mapper = _FakeMapper()
    model = _FakeHandModel()
    visualizer = _FakeVisualizer()
    overlay = app._CurrentPoseOverlay(
        _fake_o3d(),
        visualizer,
        T_EE_hand=np.eye(4),
        target_ee=np.eye(4),
        current_hand_model=model,
        current_hand_mapper=mapper,
        show_current_hand_mesh=True,
    )
    state = pose_state_from_mapping(
        {
            "schema_version": 1,
            "reference_frame": "robot_base",
            "timestamp_unix_s": 1.0,
            "T_reference_EE": _pose([0.4, 0.1, 0.3]).tolist(),
            "hand": {"angles": [900, 800, 700, 600, 500, 400]},
        }
    )
    overlay.update(state, target_ee=np.eye(4))

    old_frames = (
        overlay.current_ee_frame,
        overlay.current_hand_frame,
        overlay.error_line,
    )
    old_meshes = tuple(overlay.current_meshes)
    overlay.hide_current_feedback()

    assert all(geometry in visualizer.removed for geometry in old_frames)
    assert all(mesh in visualizer.removed for mesh in old_meshes)
    assert overlay.current_meshes == []
    assert overlay._frames_added is False
    np.testing.assert_allclose(overlay._last_ee, np.eye(4))
    assert overlay.current_ee_frame is not old_frames[0]
    assert overlay.current_hand_frame is not old_frames[1]
    assert overlay.error_line is not old_frames[2]

    recovered = replace(
        state,
        timestamp_unix_s=2.0,
        sequence=1,
        T_reference_EE=_pose([0.5, 0.2, 0.4]),
    )
    overlay.update(recovered, target_ee=np.eye(4))

    assert overlay._frames_added is True
    assert overlay.current_ee_frame in visualizer.added
    assert overlay.current_hand_frame in visualizer.added
    assert overlay.error_line in visualizer.added
    np.testing.assert_allclose(
        overlay.current_ee_frame.transforms[0], recovered.T_reference_EE
    )


def test_current_hand_mesh_requires_pose_telemetry_but_not_target_mesh():
    app = _load_app()
    args = app.build_parser().parse_args(["snapshot.npz", "--show-current-hand-mesh"])
    with pytest.raises(ValueError, match="requires --pose-state or --continuous"):
        app._validate_numeric_args(args)

    args = app.build_parser().parse_args(
        [
            "snapshot.npz",
            "--show-current-hand-mesh",
            "--no-target-hand-mesh",
            "--pose-state",
            "pose.json",
        ]
    )
    app._validate_numeric_args(args)


def test_continuous_telemetry_and_manifest_are_paired_and_legacy_is_exclusive():
    app = _load_app()
    parser = app.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "snapshot.npz",
                "--pose-state",
                "pose.json",
                "--continuous-telemetry",
                "/run/anydex.map",
            ]
        )

    args = parser.parse_args(
        ["snapshot.npz", "--continuous-telemetry", "/run/anydex.map"]
    )
    with pytest.raises(ValueError, match="required together"):
        app._validate_numeric_args(args)

    args = parser.parse_args(
        [
            "snapshot.npz",
            "--telemetry-session-manifest",
            "session.json",
        ]
    )
    with pytest.raises(ValueError, match="required together"):
        app._validate_numeric_args(args)

    args = parser.parse_args(
        [
            "snapshot.npz",
            "--continuous-telemetry",
            "/run/anydex.map",
            "--telemetry-session-manifest",
            "session.json",
            "--show-current-hand-mesh",
        ]
    )
    app._validate_numeric_args(args)
    assert args.telemetry_wait_seconds == pytest.approx(60.0)


def test_ready_file_contract_requires_live_realsense_continuous_mode(tmp_path):
    app = _load_app()
    parser = app.build_parser()
    marker = tmp_path / "viewer.ready"

    assert parser.parse_args(["snapshot.npz"]).ready_file is None

    args = parser.parse_args(["snapshot.npz", "--ready-file", str(marker)])
    with pytest.raises(ValueError, match="continuous-telemetry"):
        app._validate_numeric_args(args)

    args = parser.parse_args(
        [
            "snapshot.npz",
            "--source",
            "snapshot",
            "--continuous-telemetry",
            "/tmp/test.map",
            "--telemetry-session-manifest",
            "manifest.json",
            "--ready-file",
            str(marker),
        ]
    )
    with pytest.raises(ValueError, match="source realsense"):
        app._validate_numeric_args(args)

    args = parser.parse_args(
        [
            "snapshot.npz",
            "--continuous-telemetry",
            "/tmp/test.map",
            "--telemetry-session-manifest",
            "manifest.json",
            "--ready-file",
            str(marker),
            "--validate-only",
        ]
    )
    with pytest.raises(ValueError, match="validate-only"):
        app._validate_numeric_args(args)


def test_ready_file_is_o_excl_complete_and_exactly_0444(tmp_path):
    app = _load_app()
    marker = tmp_path / "viewer.ready"

    assert app.publish_viewer_ready(marker) == marker.resolve()
    assert marker.read_bytes() == b"ANYDEX_LIVE_PREVIEW_READY_V1\n"
    assert stat.S_IMODE(marker.stat().st_mode) == 0o444
    assert viewer_ready_is_published(marker)

    marker.chmod(0o644)
    assert not viewer_ready_is_published(marker)
    marker.chmod(0o444)
    assert viewer_ready_is_published(marker)

    with pytest.raises(FileExistsError):
        app.publish_viewer_ready(marker)
    assert marker.read_bytes() == b"ANYDEX_LIVE_PREVIEW_READY_V1\n"

    target = tmp_path / "target"
    target.write_text("do not replace\n", encoding="utf-8")
    symlink = tmp_path / "symlink.ready"
    os.symlink(target, symlink)
    with pytest.raises(FileExistsError):
        app.publish_viewer_ready(symlink)
    assert target.read_text(encoding="utf-8") == "do not replace\n"
    assert not viewer_ready_is_published(symlink)

    ready_alias = tmp_path / "ready-alias.ready"
    os.symlink(marker, ready_alias)
    assert not viewer_ready_is_published(ready_alias)

    missing_target = tmp_path / "missing-target.ready"
    dangling = tmp_path / "dangling.ready"
    os.symlink(missing_target, dangling)
    with pytest.raises(FileExistsError):
        app.publish_viewer_ready(dangling)
    assert not missing_target.exists()
    assert not viewer_ready_is_published(dangling)

    incomplete = tmp_path / "incomplete.ready"
    incomplete.write_bytes(b"ANYDEX_LIVE_PREVIEW_READY")
    incomplete.chmod(0o444)
    assert not viewer_ready_is_published(incomplete)


@pytest.mark.parametrize("value", ("-0.01", "nan", "inf"))
def test_telemetry_wait_seconds_must_be_finite_and_nonnegative(value):
    app = _load_app()
    args = app.build_parser().parse_args(
        ["snapshot.npz", "--telemetry-wait-seconds", value]
    )
    with pytest.raises(ValueError, match="telemetry-wait-seconds"):
        app._validate_numeric_args(args)


def _telemetry_identity():
    return TelemetryIdentity(
        run_uuid="12345678-1234-5678-9234-567812345678",
        execution_contract_sha256="a" * 64,
        source_snapshot_sha256="b" * 64,
        control_config_sha256="c" * 64,
        calibration_sha256="d" * 64,
        producer_build_sha256="e" * 64,
    )


def _telemetry_header(identity):
    return {
        "run_uuid": identity.run_uuid,
        "execution_contract_sha256": identity.execution_contract_sha256,
        "source_snapshot_sha256": identity.source_snapshot_sha256,
        "control_config_sha256": identity.control_config_sha256,
        "calibration_sha256": identity.calibration_sha256,
        "producer_build_sha256": identity.producer_build_sha256,
        "created_unix_ns": 1,
        "created_monotonic_ns": 1,
        "producer_name": "offline-test-producer",
        "robot_id": "fr3-test",
    }


def test_native_telemetry_source_lazy_opens_only_reviewed_read_only_api(
    tmp_path, monkeypatch
):
    app = _load_app()
    identity = _telemetry_identity()
    header = _telemetry_header(identity)
    calls = []

    class Reader:
        def header(self):
            calls.append(("header",))
            return dict(header)

        def read_arm(self, attempts):
            calls.append(("arm", attempts))
            return {"arm": True}

        def read_hand(self, attempts):
            calls.append(("hand", attempts))
            return {"hand": True}

    class ReaderType:
        @staticmethod
        def open_read_only(path):
            calls.append(("open_read_only", path))
            return Reader()

    native = SimpleNamespace(
        ABI_MAJOR=1,
        ABI_SCHEMA_SHA256=NATIVE_TELEMETRY_ABI_SCHEMA_SHA256,
        platform_is_supported_lock_free=lambda: True,
        TelemetryReader=ReaderType,
    )
    monkeypatch.setattr(app.importlib, "import_module", lambda name: native)
    sentinel = object()

    class Adapter:
        def __init__(self, expected_identity, **kwargs):
            assert expected_identity == identity

        def accept(self, actual_header, arm, hand):
            assert actual_header == header
            assert arm == {"arm": True}
            assert hand == {"hand": True}
            return sentinel

    monkeypatch.setattr(app, "NativeContinuousTelemetryAdapter", Adapter)
    manifest = SimpleNamespace(identity=identity)
    mapping = tmp_path / "read-only.map"

    source = app._NativeTelemetrySource(
        mapping,
        manifest,
        python_dir=None,
        arm_max_age_s=0.25,
        hand_max_age_s=0.75,
        read_attempts=6,
    )
    assert calls == [
        ("open_read_only", str(mapping.resolve())),
        ("header",),
    ]
    assert source.next() is sentinel
    assert calls[-3:] == [("header",), ("arm", 6), ("hand", 6)]
    source.close()
    assert source.reader is None


def test_native_telemetry_source_rejects_unreviewed_schema(tmp_path, monkeypatch):
    app = _load_app()
    native = SimpleNamespace(
        ABI_MAJOR=1,
        ABI_SCHEMA_SHA256="0" * 64,
        platform_is_supported_lock_free=lambda: True,
        TelemetryReader=SimpleNamespace(
            open_read_only=lambda _path: (_ for _ in ()).throw(
                AssertionError("mapping opened before ABI schema rejection")
            )
        ),
    )
    monkeypatch.setattr(app.importlib, "import_module", lambda name: native)
    manifest = SimpleNamespace(identity=_telemetry_identity())
    with pytest.raises(RuntimeError, match="schema digest"):
        app._NativeTelemetrySource(
            tmp_path / "must-not-open.map",
            manifest,
            python_dir=None,
            arm_max_age_s=0.25,
            hand_max_age_s=0.75,
            read_attempts=6,
        )


def test_native_telemetry_source_waits_only_for_missing_or_not_ready(
    tmp_path, monkeypatch
):
    app = _load_app()
    identity = _telemetry_identity()
    header = _telemetry_header(identity)
    calls = []
    sleeps = []

    class Reader:
        def header(self):
            return dict(header)

    outcomes = iter(
        (
            RuntimeError("open_read_only failed (code=3): open failed: errno=2"),
            RuntimeError(
                "open_read_only failed (code=7): telemetry mapping "
                "initialization is not committed"
            ),
            Reader(),
        )
    )

    def open_read_only(path):
        calls.append(path)
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    native = SimpleNamespace(
        ABI_MAJOR=1,
        ABI_SCHEMA_SHA256=NATIVE_TELEMETRY_ABI_SCHEMA_SHA256,
        platform_is_supported_lock_free=lambda: True,
        TelemetryReader=SimpleNamespace(open_read_only=open_read_only),
        initialize_mapping=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("read-only viewer attempted to initialize mapping")
        ),
    )
    monkeypatch.setattr(app.importlib, "import_module", lambda name: native)
    monkeypatch.setattr(app.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(app.time, "sleep", lambda seconds: sleeps.append(seconds))
    mapping = tmp_path / "producer-created-later.map"

    source = app._NativeTelemetrySource(
        mapping,
        SimpleNamespace(identity=identity),
        python_dir=None,
        arm_max_age_s=0.25,
        hand_max_age_s=0.75,
        read_attempts=6,
        wait_seconds=1.0,
    )

    assert calls == [str(mapping.resolve())] * 3
    assert sleeps == [pytest.approx(0.10), pytest.approx(0.10)]
    assert not mapping.exists()
    source.close()


@pytest.mark.parametrize(
    "message",
    (
        "open_read_only failed (code=3): open failed: errno=13",
        "open_read_only failed (code=8): telemetry ABI header or provenance is incompatible",
        "open_read_only failed (code=6): mmap failed: errno=12",
    ),
)
def test_native_telemetry_source_does_not_wait_on_fatal_open_errors(
    message, tmp_path, monkeypatch
):
    app = _load_app()
    calls = []

    def open_read_only(path):
        calls.append(path)
        raise RuntimeError(message)

    native = SimpleNamespace(
        ABI_MAJOR=1,
        ABI_SCHEMA_SHA256=NATIVE_TELEMETRY_ABI_SCHEMA_SHA256,
        platform_is_supported_lock_free=lambda: True,
        TelemetryReader=SimpleNamespace(open_read_only=open_read_only),
    )
    monkeypatch.setattr(app.importlib, "import_module", lambda name: native)
    monkeypatch.setattr(
        app.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(
            AssertionError("fatal telemetry error was retried")
        ),
    )
    mapping = tmp_path / "fatal.map"

    with pytest.raises(RuntimeError, match="open_read_only failed"):
        app._NativeTelemetrySource(
            mapping,
            SimpleNamespace(identity=_telemetry_identity()),
            python_dir=None,
            arm_max_age_s=0.25,
            hand_max_age_s=0.75,
            read_attempts=6,
            wait_seconds=60.0,
        )
    assert calls == [str(mapping.resolve())]
    assert not mapping.exists()


def test_native_telemetry_source_bridges_only_zero_byte_create_ftruncate_race(
    tmp_path, monkeypatch
):
    app = _load_app()
    identity = _telemetry_identity()
    header = _telemetry_header(identity)
    clock = {"now": 4.0}
    calls = []
    mapping = tmp_path / "initializing.map"
    mapping.write_bytes(b"")

    class Reader:
        def header(self):
            return dict(header)

    outcomes = iter(
        (
            RuntimeError(
                "open_read_only failed (code=8): telemetry ABI header or "
                "provenance is incompatible"
            ),
            RuntimeError(
                "open_read_only failed (code=7): telemetry mapping "
                "initialization is not committed"
            ),
            Reader(),
        )
    )

    def open_read_only(path):
        calls.append(path)
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    native = SimpleNamespace(
        ABI_MAJOR=1,
        ABI_SCHEMA_SHA256=NATIVE_TELEMETRY_ABI_SCHEMA_SHA256,
        platform_is_supported_lock_free=lambda: True,
        TelemetryReader=SimpleNamespace(open_read_only=open_read_only),
    )
    monkeypatch.setattr(app.importlib, "import_module", lambda name: native)
    monkeypatch.setattr(app.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        app.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )

    source = app._NativeTelemetrySource(
        mapping,
        SimpleNamespace(identity=identity),
        python_dir=None,
        arm_max_age_s=0.25,
        hand_max_age_s=0.75,
        read_attempts=6,
        wait_seconds=60.0,
    )
    assert len(calls) == 3
    assert clock["now"] == pytest.approx(4.2)
    source.close()


def test_native_telemetry_zero_byte_grace_is_bounded_to_half_second(
    tmp_path, monkeypatch
):
    app = _load_app()
    clock = {"now": 8.0}
    mapping = tmp_path / "stale-zero.map"
    mapping.write_bytes(b"")
    message = (
        "open_read_only failed (code=8): telemetry ABI header or "
        "provenance is incompatible"
    )

    native = SimpleNamespace(
        ABI_MAJOR=1,
        ABI_SCHEMA_SHA256=NATIVE_TELEMETRY_ABI_SCHEMA_SHA256,
        platform_is_supported_lock_free=lambda: True,
        TelemetryReader=SimpleNamespace(
            open_read_only=lambda _path: (_ for _ in ()).throw(
                RuntimeError(message)
            )
        ),
    )
    monkeypatch.setattr(app.importlib, "import_module", lambda name: native)
    monkeypatch.setattr(app.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        app.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )

    with pytest.raises(RuntimeError, match="code=8"):
        app._NativeTelemetrySource(
            mapping,
            SimpleNamespace(identity=_telemetry_identity()),
            python_dir=None,
            arm_max_age_s=0.25,
            hand_max_age_s=0.75,
            read_attempts=6,
            wait_seconds=60.0,
        )
    assert clock["now"] == pytest.approx(8.5)


def test_native_telemetry_source_missing_mapping_timeout_is_monotonic(
    tmp_path, monkeypatch
):
    app = _load_app()
    clock = {"now": 5.0}
    calls = []

    def open_read_only(path):
        calls.append(path)
        raise RuntimeError("open_read_only failed (code=3): open failed: errno=2")

    native = SimpleNamespace(
        ABI_MAJOR=1,
        ABI_SCHEMA_SHA256=NATIVE_TELEMETRY_ABI_SCHEMA_SHA256,
        platform_is_supported_lock_free=lambda: True,
        TelemetryReader=SimpleNamespace(open_read_only=open_read_only),
    )
    monkeypatch.setattr(app.importlib, "import_module", lambda name: native)
    monkeypatch.setattr(app.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        app.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )
    mapping = tmp_path / "never-created.map"

    with pytest.raises(RuntimeError, match=r"within 0\.250s"):
        app._NativeTelemetrySource(
            mapping,
            SimpleNamespace(identity=_telemetry_identity()),
            python_dir=None,
            arm_max_age_s=0.25,
            hand_max_age_s=0.75,
            read_attempts=6,
            wait_seconds=0.25,
        )
    assert len(calls) >= 3
    assert clock["now"] == pytest.approx(5.25)
    assert not mapping.exists()


def test_native_telemetry_wait_does_not_swallow_ctrl_c(tmp_path, monkeypatch):
    app = _load_app()

    def interrupted(_path):
        raise KeyboardInterrupt

    native = SimpleNamespace(
        ABI_MAJOR=1,
        ABI_SCHEMA_SHA256=NATIVE_TELEMETRY_ABI_SCHEMA_SHA256,
        platform_is_supported_lock_free=lambda: True,
        TelemetryReader=SimpleNamespace(open_read_only=interrupted),
    )
    monkeypatch.setattr(app.importlib, "import_module", lambda name: native)

    with pytest.raises(KeyboardInterrupt):
        app._NativeTelemetrySource(
            tmp_path / "ctrl-c.map",
            SimpleNamespace(identity=_telemetry_identity()),
            python_dir=None,
            arm_max_age_s=0.25,
            hand_max_age_s=0.75,
            read_attempts=6,
            wait_seconds=60.0,
        )


def test_native_reader_publishes_ready_once_immediately_before_mapping_wait(
    tmp_path, monkeypatch
):
    app = _load_app()
    events = []

    def open_read_only(_path):
        events.append("mapping_wait")
        raise KeyboardInterrupt

    native = SimpleNamespace(
        ABI_MAJOR=1,
        ABI_SCHEMA_SHA256=NATIVE_TELEMETRY_ABI_SCHEMA_SHA256,
        platform_is_supported_lock_free=lambda: True,
        TelemetryReader=SimpleNamespace(open_read_only=open_read_only),
    )
    monkeypatch.setattr(app.importlib, "import_module", lambda name: native)

    with pytest.raises(KeyboardInterrupt):
        app._NativeTelemetrySource(
            tmp_path / "not-created.map",
            SimpleNamespace(identity=_telemetry_identity()),
            python_dir=None,
            arm_max_age_s=0.25,
            hand_max_age_s=0.75,
            read_attempts=6,
            wait_seconds=60.0,
            before_mapping_wait=lambda: events.append("ready"),
        )

    assert events == ["ready", "mapping_wait"]

    events.clear()
    native.ABI_MAJOR = 2
    with pytest.raises(RuntimeError, match="ABI major"):
        app._NativeTelemetrySource(
            tmp_path / "not-created.map",
            SimpleNamespace(identity=_telemetry_identity()),
            python_dir=None,
            arm_max_age_s=0.25,
            hand_max_age_s=0.75,
            read_attempts=6,
            before_mapping_wait=lambda: events.append("ready"),
        )
    assert events == []


def test_continuous_validate_only_never_imports_native(monkeypatch):
    app = _load_app()
    args = [
        "snapshot.npz",
        "--continuous-telemetry",
        "/run/anydex.map",
        "--telemetry-session-manifest",
        "session.json",
        "--telemetry-wait-seconds",
        "60",
        "--validate-only",
    ]
    monkeypatch.setattr(app, "load_preview_inputs", lambda parsed: object())
    monkeypatch.setattr(app, "print_preview_contract", lambda inputs: None)

    def forbidden_import(name):
        raise AssertionError("validate-only imported native module: {}".format(name))

    monkeypatch.setattr(app.importlib, "import_module", forbidden_import)

    assert app.main(args) == 0
