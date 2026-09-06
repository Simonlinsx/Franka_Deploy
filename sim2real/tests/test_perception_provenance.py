from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace
import inspect

import numpy as np
import pytest

import sim2real.perception as perception
from sim2real.perception import (
    _current_projection_source_mask,
    _fail_closed_visualization_sample,
    _formal_publication_watchdog_expired,
    _frame_trace_record,
    _pointcloud_temporal_fallback_config,
    _sam2_service_provenance,
)
from sim2real.observation.model import MaskedRGBDProjector


def _projector() -> MaskedRGBDProjector:
    return MaskedRGBDProjector(
        camera_K=np.asarray([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]),
        T_base_camera_optical=np.eye(4),
        image_size=(4, 3),
        depth_range_m=(0.1, 2.0),
        num_points=4,
        minimum_valid_points=2,
    )


def _camera(*, frame_id: int, mask: np.ndarray, source: str) -> SimpleNamespace:
    return SimpleNamespace(
        frame_id=frame_id,
        timestamp_s=float(frame_id),
        mask=np.asarray(mask, dtype=bool),
        mask_valid=bool(np.any(mask)),
        requested_object_mask_mode="guarded",
        effective_object_mask_mode="guarded_v2",
        effective_mask_publication_mode="guarded_sam2_primary",
        effective_recovery_publication_mode="unified_three_evidence",
        provider_published_mask_source=source,
        provider_published_mask_message=f"published {source}",
        provider_online_sam2_status="tracked_async_pending",
        provider_timings_ms=np.arange(6, dtype=np.float64),
    )


def test_frame_trace_separates_current_provider_mask_from_stale_projector_mask():
    projector = _projector()
    color = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.ones((3, 4), dtype=np.float32)
    good = np.zeros((3, 4), dtype=bool)
    good[0, :2] = True
    projector.project(
        color_bgr=color,
        depth_m=depth,
        object_mask=good,
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=10.0,
        frame_id=10,
    )

    current_empty = np.zeros((3, 4), dtype=bool)
    stale = projector.project(
        color_bgr=color,
        depth_m=depth,
        object_mask=current_empty,
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=11.0,
        frame_id=11,
    )
    record = _frame_trace_record(
        camera=_camera(frame_id=11, mask=current_empty, source="none"),
        point_frame=stale,
        projector=projector,
    )

    assert record["schema_version"] == 2
    assert record["mask_provenance_schema"] == "provider_projector_split_v1"
    assert record["requested_object_mask_mode"] == "guarded"
    assert record["effective_object_mask_mode"] == "guarded_v2"
    assert record["effective_provider_mask_publication_mode"] == (
        "guarded_sam2_primary"
    )
    assert (
        record["effective_provider_recovery_publication_mode"]
        == "unified_three_evidence"
    )
    assert record["provider_published_mask_source"] == "none"
    assert record["provider_published_mask_message"] == "published none"
    assert record["provider_online_sam2_status"] == "tracked_async_pending"
    assert record["provider_published_mask_area_px"] == 0
    assert record["provider_published_mask_bbox_xyxy"] is None
    assert record["pointcloud_status"] == "stale_palm"
    assert record["stale_palm_policy"] == ("recoverable_fail_closed_no_policy_stage")
    assert record["policy_input_eligible"] is False
    assert (
        record["projector_effective_policy_mask_provenance"]
        == "retained_previous_fresh_projector_mask"
    )
    assert record["projector_effective_policy_mask_source_frame_id"] == 10
    assert record["projector_effective_policy_mask_area_px"] == 2
    assert record["projector_effective_policy_mask_bbox_xyxy"] == [0, 0, 1, 0]
    assert "mask_source" not in record
    assert "effective_policy_mask_area_px" not in record
    assert record["point_coordinate_frame"] == ("robot_base_via_identity_T_base_palm")
    assert record["palm_frame_geometry_validated"] is False


def test_perception_entry_point_never_imports_or_constructs_robot_readers():
    source = inspect.getsource(perception)
    assert "FrankaStateReader" not in source
    assert "T_base_policy_palm_from_franka" not in source
    assert "RH56" in source


def test_perception_starts_continuous_capture_before_switching_live_viewers():
    source = inspect.getsource(perception.run)
    capture_started = source.index("camera_thread.start()")
    formal_view_opened = source.index("visualizer.open()")
    grounding_view_closed = source.index("grounding_preview.close()")

    assert capture_started < formal_view_opened < grounding_view_closed


def test_throw_trigger_waits_through_fail_closed_publication_recovery():
    assert _formal_publication_watchdog_expired(
        rollout_trigger_enabled=False,
        last_fresh_publication_monotonic_s=10.0,
        now_monotonic_s=10.201,
    )
    assert not _formal_publication_watchdog_expired(
        rollout_trigger_enabled=True,
        last_fresh_publication_monotonic_s=10.0,
        now_monotonic_s=20.0,
    )
    assert not _formal_publication_watchdog_expired(
        rollout_trigger_enabled=False,
        last_fresh_publication_monotonic_s=None,
        now_monotonic_s=20.0,
    )


def test_throw_trigger_defaults_to_three_second_full_flight_capture():
    args = perception.build_parser().parse_args(
        ["--object-text", "red triangle", "--test-rollout-trigger"]
    )
    perception._validate_runtime_args(args)
    assert args.post_trigger_capture_s == 3.0

    args.post_trigger_capture_s = 10.01
    with pytest.raises(ValueError, match="post-trigger-capture-s"):
        perception._validate_runtime_args(args)


def test_full_flight_video_records_fail_closed_frames_without_old_mask_or_cloud():
    camera = SimpleNamespace(
        frame_id=12,
        timestamp_s=3.5,
        color_bgr=np.full((3, 4, 3), 17, dtype=np.uint8),
        mask=np.ones((3, 4), dtype=bool),
        requested_object_mask_mode="guarded_v2",
        effective_object_mask_mode="guarded_v2",
        effective_mask_publication_mode="guarded_sam2_primary",
        effective_recovery_publication_mode="unified_three_evidence",
        provider_published_mask_source="none",
        provider_published_mask_message="recovery pending",
        provider_online_sam2_status="recovery_pending_1",
    )

    sample = _fail_closed_visualization_sample(
        sequence=4,
        camera=camera,
        point_feature_dim=6,
    )

    assert sample.frame_id == 12
    assert np.array_equal(sample.color_bgr, camera.color_bgr)
    assert not np.any(sample.object_mask)
    assert sample.pointcloud_xyzrgb_palm.shape == (128, 6)
    assert not np.any(sample.pointcloud_xyzrgb_palm)
    assert not np.any(sample.pointcloud_valid)
    assert sample.provider_published_mask_valid is False
    assert sample.projector_effective_policy_mask_provenance == (
        "current_frame_fail_closed_empty_policy_input"
    )


def test_full_flight_projection_reuses_production_semantic_motion_fallback_contract():
    provider = SimpleNamespace(
        cfg={
            "pointcloud": {
                "temporal_fallback": "motion_compensated",
                "temporal_fallback_max_stale_s": 0.25,
                "temporal_fallback_max_stale_steps": 5,
                "temporal_fallback_max_image_speed_px_s": 2400.0,
                "unreviewed_option": "ignored",
            }
        }
    )
    assert _pointcloud_temporal_fallback_config(provider) == {
        "temporal_fallback": "motion_compensated",
        "temporal_fallback_max_stale_s": 0.25,
        "temporal_fallback_max_stale_steps": 5,
        "temporal_fallback_max_image_speed_px_s": 2400.0,
    }

    formal = np.zeros((3, 4), dtype=bool)
    semantic = np.zeros((3, 4), dtype=bool)
    semantic[1, 2:] = True
    camera = SimpleNamespace(
        mask=formal,
        mask_valid=False,
        policy_semantic_mask=semantic,
        policy_semantic_valid=True,
    )
    selected = _current_projection_source_mask(camera)
    assert np.array_equal(selected, semantic)

    camera.policy_semantic_valid = False
    assert not np.any(_current_projection_source_mask(camera))


def test_trigger_entrypoint_uses_compact_operator_console(
    monkeypatch, capsys, tmp_path
):
    def fake_run(_args):
        print("[RealSense] verbose transport details")
        print("[Object grounding SEARCHING] hold target")
        print("[Throw trigger ARMED] throw now")
        print("[Throw trigger DETECTED] frame=42")
        return {
            "result": "PASS",
            "run_id": "compact-trigger",
            "rollout_trigger": {"frame_id": 42},
            "record_video_path": "trigger.mp4",
        }

    report = tmp_path / "compact-trigger.json"
    monkeypatch.setattr(perception, "run", fake_run)
    monkeypatch.setattr(perception, "_save_compact_report", lambda _result: report)

    code = perception.main(
        [
            "--object-text",
            "red triangular beanbag toy",
            "--test-rollout-trigger",
            "--run-id",
            "compact-trigger",
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "Object grounding SEARCHING" in output
    assert "Throw trigger ARMED" in output
    assert "Throw trigger DETECTED" in output
    assert "RealSense" not in output
    assert "[Throw trigger test PASS]" in output
    assert str(report) in output


class _ImmediateFuture:
    def __init__(self, value):
        self.value = value

    def result(self, timeout=None):
        del timeout
        return self.value


class _ImmediateExecutor:
    def submit(self, function, *args, **kwargs):
        return _ImmediateFuture(function(*args, **kwargs))


def test_sam2_service_provenance_binds_health_to_local_model_bytes(tmp_path):
    checkpoint = tmp_path / "sam2.pt"
    model_config = tmp_path / "sam2.yaml"
    pcd_config = tmp_path / "d435.yaml"
    checkpoint.write_bytes(b"checkpoint bytes")
    model_config.write_bytes(b"model config bytes")
    pcd_config.write_bytes(b"camera: {}\n")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model_config_sha256 = hashlib.sha256(model_config.read_bytes()).hexdigest()

    health = {
        "service": "dynamic-pcd-online-sam2",
        "protocol_version": 1,
        "backend": "official-sam2-video-predictor",
        "loaded": True,
        "initialized": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "model_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "model_config_sha256": model_config_sha256,
        "resolved_model_config_path": str(model_config),
        "device": "cuda",
        "gpu_name": "test gpu",
        "image_size": 512,
        "amp_dtype": "bfloat16",
        "tf32": True,
        "fill_hole_area": 0,
        "safe_history_frames": 16,
        "reset_every_frames": 0,
        "generation": 2,
        "model_load_ms": 123.0,
    }

    class Manager:
        owns_process = True
        process = SimpleNamespace(pid=1234)

        def health(self, timeout_ms):
            assert timeout_ms == 1000
            return dict(health)

    provider = SimpleNamespace(
        online_sam2_cfg={
            "enabled": True,
            "service_addr": "tcp://127.0.0.1:5558",
            "checkpoint": str(checkpoint),
            "model_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "device": "cuda",
            "image_size": 512,
            "amp_dtype": "bfloat16",
            "reset_every_frames": 0,
            "mask_publication_mode": "semantic_sam2",
            "guarded_v2_semantic_primary": True,
        },
        online_sam2_manager=Manager(),
        _online_sam2_executor=_ImmediateExecutor(),
        _online_sam2_ready=True,
    )

    result = _sam2_service_provenance(provider, pcd_config_path=pcd_config)

    assert result["query_succeeded"] is True
    assert result["identity_valid_for_production"] is True
    assert result["manager_owns_process"] is True
    assert result["manager_process_pid"] == 1234
    assert result["health"]["checkpoint_sha256"] == checkpoint_sha256
    assert (
        result["configured"]["model_config_sha256_from_local_file"]
        == model_config_sha256
    )
    assert all(result["identity_checks"].values())

    health["checkpoint_sha256"] = "0" * 64
    mismatch = _sam2_service_provenance(provider, pcd_config_path=pcd_config)
    assert mismatch["identity_valid_for_production"] is False
    assert mismatch["identity_checks"]["checkpoint_hash_identity"] is False


def test_mocked_run_never_constructs_franka_or_rh56_interfaces(tmp_path, monkeypatch):
    """Exercise ``run`` while constructors are armed as hard-fail spies."""

    import sim2real.runtime.bounded_c2_runtime as bounded_runtime
    import sim2real.io as robot_io
    import sim2real.observation.live_preview as live_preview
    import robot_control.rh56.watchdog as rh56_owner

    calls = {
        "franka": 0,
        "inspire": 0,
        "rh56_transport": 0,
        "rh56_owner": 0,
    }

    def forbidden(name):
        class ForbiddenConstructor:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                calls[name] += 1
                raise AssertionError(f"forbidden constructor invoked: {name}")

        return ForbiddenConstructor

    monkeypatch.setattr(robot_io, "FrankaStateReader", forbidden("franka"))
    monkeypatch.setattr(robot_io, "InspireStateReader", forbidden("inspire"))
    monkeypatch.setattr(live_preview, "FrankaStateReader", forbidden("franka"))
    monkeypatch.setattr(live_preview, "InspireStateReader", forbidden("inspire"))
    monkeypatch.setattr(
        bounded_runtime,
        "LinuxRH56TransportFactory",
        forbidden("rh56_transport"),
    )
    monkeypatch.setattr(rh56_owner, "RH56WatchdogOwner", forbidden("rh56_owner"))
    monkeypatch.setattr(
        perception, "FrankaStateReader", forbidden("franka"), raising=False
    )
    monkeypatch.setattr(
        perception,
        "InspireStateReader",
        forbidden("inspire"),
        raising=False,
    )

    for name in ("bundle.zip", "profile.json", "pcd.yaml"):
        (tmp_path / name).write_bytes(b"test")
    save_directory = tmp_path / "snapshot"
    args = perception.build_parser().parse_args(
        [
            "--bundle",
            str(tmp_path / "bundle.zip"),
            "--profile",
            str(tmp_path / "profile.json"),
            "--pcd-config",
            str(tmp_path / "pcd.yaml"),
            "--object-roi",
            "0",
            "0",
            "2",
            "2",
            "--duration",
            "0.000000001",
            "--run-id",
            "mock-no-robot",
            "--save-directory",
            str(save_directory),
        ]
    )

    class FakeBundle:
        def __init__(self, path):
            self.path = path

        def verify(self):
            return None

    class FakeContract:
        camera_K = np.eye(3)
        camera_width = 2
        camera_height = 2
        T_base_camera_optical = np.eye(4)
        depth_range_m = (0.1, 2.0)

        @classmethod
        def from_bundle(cls, _bundle):
            return cls()

    class FakeGuard:
        def __init__(self, _threads):
            self.closed = False

        def start(self):
            return self

        def close(self):
            self.closed = True

    class FakeProvider:
        extractor = None

        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    provider = FakeProvider()
    selection = SimpleNamespace(
        effective_object_mask_mode="guarded_v2",
        effective_mask_publication_mode="guarded_sam2_primary",
        effective_recovery_publication_mode="unified_three_evidence",
    )

    mask = np.ones((2, 2), dtype=bool)
    camera = SimpleNamespace(
        frame_id=1,
        timestamp_s=1.0,
        color_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_m=np.ones((2, 2), dtype=np.float32),
        depth_raw=None,
        depth_scale_m_per_unit=0.001,
        mask=mask,
        mask_valid=True,
        requested_object_mask_mode="guarded_v2",
        effective_object_mask_mode="guarded_v2",
        effective_mask_publication_mode="guarded_sam2_primary",
        effective_recovery_publication_mode="unified_three_evidence",
        provider_published_mask_source="online_sam2_guarded_primary",
        provider_published_mask_message="mock exact current frame",
        provider_online_sam2_status="tracked_exact",
        provider_timings_ms=np.zeros(6, dtype=np.float64),
        object_mask_area_px=4,
        object_mask_bbox_xyxy=np.asarray([0, 0, 1, 1]),
    )

    class FakeLatest:
        def raise_if_failed(self):
            return None

        def get(self):
            return camera

        def wait_for_frame_change(self, frame_id, timeout_s):
            del frame_id, timeout_s
            return False

    class FakeThread:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def start(self):
            return None

        def join(self, timeout=None):
            del timeout

        def is_alive(self):
            return False

    class FakeAdapter:
        def __init__(self, **kwargs):
            del kwargs
            self.camera_K = np.eye(3)
            self.target_image_size = (2, 2)

        def adapt(self, *, color_bgr, depth_m, depth_raw, object_mask):
            return SimpleNamespace(
                color_bgr=color_bgr,
                depth_m=depth_m,
                depth_raw=depth_raw,
                object_mask=object_mask,
            )

        def mask_to_source_resolution(self, value):
            return np.asarray(value, dtype=bool)

    class FakeProjector:
        def __init__(self, **kwargs):
            del kwargs
            self.last_effective_object_mask = None
            self.last_effective_object_mask_provenance = None

        def project(self, **kwargs):
            self.last_effective_object_mask = np.asarray(
                kwargs["object_mask"], dtype=bool
            )
            self.last_effective_object_mask_provenance = SimpleNamespace(
                kind="current_frame_provider_mask",
                source_frame_id=int(kwargs["frame_id"]),
                source_captured_at_s=float(kwargs["captured_at_s"]),
                area_px=4,
                bbox_xyxy=(0, 0, 1, 1),
            )
            return SimpleNamespace(
                status="fresh",
                source_valid_points=4,
                xyzrgb_palm=np.zeros((128, 3), dtype=np.float32),
                valid=np.ones(128, dtype=np.float32),
                frame_id=int(kwargs["frame_id"]),
                captured_at_s=float(kwargs["captured_at_s"]),
            )

    class FakeVisualizer:
        def __init__(self, **kwargs):
            del kwargs

        def open(self):
            return None

        def try_publish(self, sample):
            del sample
            return True

        def close(self):
            return None

    def copy_namespace(value, **updates):
        payload = dict(vars(value))
        payload.update(updates)
        return SimpleNamespace(**payload)

    monkeypatch.setattr(perception, "DeployBundle", FakeBundle)
    monkeypatch.setattr(perception, "V94Contract", FakeContract)
    monkeypatch.setattr(
        perception,
        "_selected_point_feature_contract",
        lambda request: (3, "xyz", None, None),
    )
    monkeypatch.setattr(
        perception, "_preflight_current_object_roi", lambda request: request
    )
    monkeypatch.setattr(
        perception,
        "_initialize_provider",
        lambda *args, **kwargs: (provider, selection),
    )
    monkeypatch.setattr(
        perception,
        "_sam2_service_provenance",
        lambda *args, **kwargs: {"identity_valid_for_production": True},
    )
    monkeypatch.setattr(perception, "_ComputeThreadGuard", FakeGuard)
    monkeypatch.setattr(perception, "_Latest", FakeLatest)
    monkeypatch.setattr(
        perception,
        "threading",
        SimpleNamespace(Event=threading.Event, Thread=FakeThread),
    )
    monkeypatch.setattr(perception, "PolicyRGBDResolutionAdapter", FakeAdapter)
    monkeypatch.setattr(perception, "MaskedRGBDProjector", FakeProjector)
    monkeypatch.setattr(perception, "V94LiveVisualizer", FakeVisualizer)
    monkeypatch.setattr(perception, "replace", copy_namespace)

    result = perception.run(args)

    assert result["franka_interface_opened"] is False
    assert result["rh56_interface_opened"] is False
    assert result["robot_hardware_interfaces_opened"] is False
    assert result["robot_hardware_writes"] is False
    assert result["robot_commands"] is False
    assert provider.stopped is True
    assert calls == {
        "franka": 0,
        "inspire": 0,
        "rh56_transport": 0,
        "rh56_owner": 0,
    }
