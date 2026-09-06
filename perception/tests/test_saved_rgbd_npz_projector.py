import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dynamic_pcd.apps.validate_saved_rgbd_npz import (
    PINNED_LEGACY_REAL_CAPTURE_SHA256,
    _SAM2ServiceAudit,
    _config_matches_fixed_acceptance_contract,
    _finalize_acceptance_artifact,
    _load_fixed_acceptance_contract,
    _service_health_identity,
    _write_json_exclusive_atomic,
    evaluate_guarded_v2_provider_from_saved_rgbd_npz,
    evaluate_saved_rgbd_npz,
    main,
)
from dynamic_pcd.segmentation.sam2_video_protocol import SAM2VideoResult


def _archive(path: Path) -> Path:
    frames, height, width = 3, 48, 64
    rgb = np.zeros((frames, height, width, 3), dtype=np.uint8)
    rgb[:, 16:36, 22:42] = np.asarray([10, 20, 230], dtype=np.uint8)
    depth = np.full((frames, height, width), 1000, dtype=np.uint16)
    depth[:, 16:36, 22:42] = 800
    mask = np.zeros((frames, height, width), dtype=bool)
    mask[:, 16:36, 22:42] = True
    bbox = np.repeat(
        np.asarray([[22, 16, 42, 36]], dtype=np.int32), frames, axis=0
    )
    camera_k = np.asarray(
        [[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    np.savez_compressed(
        path,
        rgb=rgb,
        depth_raw=depth,
        object_mask=mask,
        object_mask_bbox_xyxy=bbox,
        camera_frame_id=np.arange(1, frames + 1, dtype=np.int64),
        camera_timestamp_s=10.0 + np.arange(frames) / 30.0,
        frame_camera_K=np.repeat(camera_k[None], frames, axis=0),
        frame_depth_scale_m_per_unit=np.full(frames, 0.001),
        T_base_camera_optical=np.eye(4, dtype=np.float64),
        depth_range_m=np.asarray([0.25, 1.2], dtype=np.float64),
        camera_serial=np.asarray("test-camera"),
        calibration_id=np.asarray("test-calibration"),
    )
    return path


def _guarded_config(path: Path) -> Path:
    config = {
        "camera": {
            "serial": None,
            "width": 64,
            "height": 48,
            "fps": 30,
            "z_min": 0.25,
            "z_max": 1.2,
            "runtime_frame_timeout_ms": 100,
        },
        "tracker": {
            "mode": "adaptive_color_depth",
            "min_area": 20,
            "morph_kernel": 1,
            "optical_flow_enabled": False,
            "publication_appendage_guard_enabled": False,
        },
        "pointcloud": {
            "remove_outliers": False,
            "erode_kernel": 1,
            "stride": 1,
            "workspace_min": None,
            "workspace_max": None,
        },
        "sam2": {"enabled": False, "require_for_bbox_init": False},
        "online_sam2": {
            "enabled": True,
            "guarded_v2_semantic_primary": True,
            "service_addr": "tcp://127.0.0.1:5558",
            "service_autostart": False,
            "request_timeout_s": 0.1,
            "initialization_timeout_s": 0.1,
            "tracked_frame_wait_timeout_s": 0.045,
            "frame_wait_timeout_s": 0.045,
            "semantic_frame_wait_timeout_s": 0.045,
            "trusted_min_valid_depth_ratio": 0.1,
            "trusted_appearance_probability_threshold": 0.0,
            "trusted_min_appearance_support_ratio": 0.0,
            "trusted_min_appearance_mean": 0.0,
        },
        "extrinsics": {
            "calibration_file": None,
            "require_calibration": False,
            "require_quality_pass": False,
            "strict_camera_serial": False,
            "calibrated": False,
            "T_base_camera": np.eye(4).tolist(),
        },
        "runtime": {"save_debug_masks": False},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


class _FakeTemporalSAM2:
    def __init__(self, *, invalid_frame_ids=()) -> None:
        self.mask = None
        self.calls = []
        self.invalid_frame_ids = {int(value) for value in invalid_frame_ids}

    @staticmethod
    def _result(frame_id: int, mask: np.ndarray) -> SAM2VideoResult:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        return SAM2VideoResult(
            mask=binary,
            valid=bool(np.any(binary)),
            frame_id=int(frame_id),
            message="fake exact temporal mask",
            timings_ms={"service_total": 0.25},
            internal_frame_idx=int(frame_id),
            mask_area=int(np.count_nonzero(binary)),
        )

    def start(self):
        self.calls.append(("start",))
        return {
            "service": "dynamic-pcd-online-sam2",
            "protocol_version": 1,
            "device": "fake",
            "image_size": 512,
            "vos_optimized": True,
            "vos_compile_mode": "max-autotune-no-cudagraphs",
            "vos_compile_cuda_graphs": False,
            "vos_component_compile_modes": {
                "image_encoder": "max-autotune-no-cudagraphs",
                "memory_encoder": "max-autotune-no-cudagraphs",
                "memory_attention": "max-autotune-no-cudagraphs",
                "sam_prompt_encoder": "max-autotune-no-cudagraphs",
                "sam_mask_decoder": "max-autotune-no-cudagraphs",
            },
            "vos_component_compile_dynamic": {
                "image_encoder": False,
                "memory_encoder": False,
                "memory_attention": True,
                "sam_prompt_encoder": False,
                "sam_mask_decoder": False,
            },
            "vos_memory_attention_rope_grid_hw": [32, 32],
            "vos_memory_attention_rope_expected_tokens": 1024,
            "vos_memory_attention_rope_cache_count": 8,
            "vos_memory_attention_rope_cache_token_counts": [1024] * 8,
            "vos_memory_attention_rope_caches_verified": True,
            "initialized": False,
            "compile_prewarm_required": True,
            "compile_prewarm_completed": True,
            "compile_prewarm_contract": (
                "initialize_box_track_reset_explicit_mask_track_reset_v1"
            ),
            "compile_prewarm_shape_hw": [48, 64],
            "compile_prewarm_ms": 1.0,
            "compile_prewarm_initialize_box_ms": 0.1,
            "compile_prewarm_track_ms": 0.1,
            "compile_prewarm_initialize_mask_ms": 0.1,
            "compile_prewarm_mask_track_ms": 0.1,
        }

    def initialize(self, _image_bgr, mask, frame_id, **_kwargs):
        self.mask = (np.asarray(mask) > 0).astype(np.uint8)
        self.calls.append(("initialize", int(frame_id)))
        return self._result(frame_id, self.mask)

    def track(self, _image_bgr, frame_id, **_kwargs):
        assert self.mask is not None
        self.calls.append(("track", int(frame_id)))
        if int(frame_id) in self.invalid_frame_ids:
            return self._result(frame_id, np.zeros_like(self.mask))
        return self._result(frame_id, self.mask)

    def reset(self, **_kwargs):
        self.calls.append(("reset",))
        return {"status": "ok"}

    def close(self):
        self.calls.append(("close",))


def test_saved_real_rgbd_mask_projects_to_128_fresh_points(tmp_path: Path) -> None:
    summary = evaluate_saved_rgbd_npz(
        _archive(tmp_path / "capture.npz"), point_feature_mode="xyzrgb"
    )
    assert summary["frames"] == 3
    assert summary["fresh_fraction"] == 1.0
    assert summary["status_counts"] == {"fresh": 3}
    assert summary["output_valid_points_min_p50_max"] == [128, 128.0, 128]
    assert summary["projector_full_128_fraction_including_initialization"] == 1.0
    assert summary["projector_mask_provenance"] == [
        "current_frame_projector_input_from_provider_mask"
    ]
    assert summary["hardware_interfaces_opened"] is False
    assert summary["prompt_grounding_evaluated"] is False
    assert len(summary["source_sha256"]) == 64


def test_guarded_v2_saved_rgbd_seeds_temporal_service_and_projects_every_frame(
    tmp_path: Path,
) -> None:
    manager = _FakeTemporalSAM2()

    def manager_factory(**_kwargs):
        return manager

    summary = evaluate_guarded_v2_provider_from_saved_rgbd_npz(
        _archive(tmp_path / "capture.npz"),
        config_path=_guarded_config(tmp_path / "config.yaml"),
        point_feature_mode="xyzrgb",
        online_sam2_manager_factory=manager_factory,
    )

    assert summary["mode"] == "guarded_v2_production_saved_rgbd"
    assert summary["requested_object_mask_mode"] == "guarded_v2"
    assert summary["effective_object_mask_mode"] == "guarded_v2"
    assert summary["effective_mask_publication_mode"] == (
        "guarded_sam2_primary"
    )
    assert summary["effective_recovery_publication_mode"] == (
        "unified_three_evidence"
    )
    assert summary["frame0_temporal_sam2_initialized"] is True
    assert summary["current_provider_valid_fraction"] == 1.0
    assert summary["projector_fresh_fraction_including_initialization"] == 1.0
    assert summary["projector_full_128_fraction_including_initialization"] == 1.0
    assert summary["projector_status_counts_including_initialization"] == {
        "fresh": 3
    }
    combined_timing = summary["provider_plus_projector_compute_ms"]
    assert combined_timing["sample_count"] == 2
    assert combined_timing["scope"] == (
        "post_seed_provider_step_plus_same_frame_128_point_projector"
    )
    assert combined_timing["max"] >= combined_timing["p95"]
    assert combined_timing["over_50ms_longest_run"] >= 0
    service = summary["sam2_temporal_service"]
    assert service["backend"] == "injected_test_fixture"
    assert service["frame0_initialize_call_count"] >= 1
    assert service["track_call_fraction"] == 1.0
    assert service["every_tracking_frame_called"] is True
    assert [call for call in manager.calls if call[0] == "track"] == [
        ("track", 2),
        ("track", 3),
    ]

    records = summary["frame_records"]
    assert [record["phase"] for record in records] == [
        "frame0_stored_mask_initialization",
        "guarded_v2_temporal_tracking",
        "guarded_v2_temporal_tracking",
    ]
    assert all(record["provider_valid"] for record in records)
    assert all(record["pcd_fresh"] for record in records)
    assert all(record["pcd_output_rows"] == 128 for record in records)
    assert all(record["pcd_output_valid_points"] == 128 for record in records)
    assert all(
        record["pcd_provenance_kind"]
        == "current_frame_projector_input_from_provider_mask"
        for record in records
    )
    assert all(
        record["pcd_provenance_source_frame_id"]
        == record["camera_frame_id"]
        for record in records
    )
    assert all(
        record["sam2_service_track_calls_for_frame"] == 1
        for record in records[1:]
    )
    assert summary["actual_local_sam2_service_evaluated"] is False
    assert summary["production_evidence_eligible"] is False
    assert (
        summary["sam2_temporal_service"]["health_identity_valid_for_production"]
        is False
    )
    assert summary["camera_interface_opened"] is False
    assert summary["realsense_interface_opened"] is False
    assert summary["franka_interface_opened"] is False
    assert summary["rh56_interface_opened"] is False
    assert summary["hardware_interfaces_opened"] is False


def test_guarded_v2_invalid_publication_records_stale_projector_provenance(
    tmp_path: Path,
) -> None:
    manager = _FakeTemporalSAM2(invalid_frame_ids={2})
    summary = evaluate_guarded_v2_provider_from_saved_rgbd_npz(
        _archive(tmp_path / "capture.npz"),
        config_path=_guarded_config(tmp_path / "config.yaml"),
        online_sam2_manager_factory=lambda **_kwargs: manager,
    )

    rejected = summary["frame_records"][1]
    assert rejected["camera_frame_id"] == 2
    assert rejected["provider_valid"] is False
    assert rejected["provider_mask_area_px"] == 0
    assert rejected["pcd_status"] == "stale_palm"
    assert rejected["pcd_fresh"] is False
    assert rejected["pcd_source_valid_points"] == 0
    assert rejected["pcd_provenance_kind"] == (
        "retained_previous_fresh_projector_mask"
    )
    assert rejected["pcd_provenance_source_frame_id"] == 1
    assert rejected["sam2_service_track_calls_for_frame"] == 1
    assert summary["sam2_temporal_service"]["track_call_fraction"] == 1.0


def test_saved_real_rgbd_validator_rejects_incomplete_archive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incomplete.npz"
    np.savez_compressed(path, rgb=np.zeros((1, 2, 3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="missing required fields"):
        evaluate_saved_rgbd_npz(path)


def test_summary_artifact_is_atomic_and_never_overwritten(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    _write_json_exclusive_atomic(path, {"result": "first"})
    assert path.read_text(encoding="utf-8").strip().endswith('"first"\n}')
    with pytest.raises(FileExistsError):
        _write_json_exclusive_atomic(path, {"result": "second"})
    assert '"first"' in path.read_text(encoding="utf-8")


def test_cli_writes_acceptance_checks_before_atomic_artifact(tmp_path: Path) -> None:
    output = tmp_path / "acceptance.json"
    result = main([str(_archive(tmp_path / "capture.npz")), "--output", str(output)])
    assert result == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["acceptance_profile"] == "diagnostic_saved_rgbd_v2"
    assert payload["production_defaults_used"] is False
    assert payload["accepted_geometry"] is True
    full_check = next(
        check
        for check in payload["checks"]
        if check["check"] == "projector_full_128_fraction"
    )
    assert full_check["actual"] == 1.0
    assert full_check["passed"] is True
    assert payload["accepted_pipeline_identity"] is False
    assert payload["accepted_end_to_end_cadence"] is False
    assert payload["accepted"] is False
    assert payload["thresholds"]
    assert payload["checks"]


def test_invalid_threshold_leaves_no_acceptance_artifact(tmp_path: Path) -> None:
    output = tmp_path / "must_not_exist.json"
    result = main(
        [
            str(tmp_path / "not-opened.npz"),
            "--minimum-fresh-fraction",
            "nan",
            "--output",
            str(output),
        ]
    )
    assert result == 2
    assert not output.exists()


def test_pinned_legacy_real_capture_provenance_and_cadence_are_explicit() -> None:
    archive_path = (
        Path(__file__).resolve().parents[2]
        / "dexgrasp"
        / "runs"
        / "real_mask_alignment_red_ball_20260808-190406.npz"
    )
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == (
        PINNED_LEGACY_REAL_CAPTURE_SHA256
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        assert int(np.asarray(archive["capture_schema_version"]).item()) == 1
        assert np.asarray(archive["rgb"]).shape[0] == 60
        assert str(np.asarray(archive["camera_serial"]).item()) == "337322072188"
        assert str(np.asarray(archive["calibration_id"]).item()).strip()
        assert bool(np.asarray(archive["hardware_writes"]).item()) is False
        assert bool(np.asarray(archive["robot_command_writes"]).item()) is False
        assert bool(np.asarray(archive["franka_interface_opened"]).item()) is False
        assert bool(np.asarray(archive["rh56_interface_opened"]).item()) is False
        retrieved_gaps = np.diff(
            np.asarray(archive["camera_retrieved_at_s"], dtype=np.float64)
        )
        sensor_gaps = np.diff(
            np.asarray(archive["camera_sensor_frame_number"], dtype=np.int64)
        )
        assert float(np.percentile(retrieved_gaps, 95)) <= 0.050
        assert float(np.max(retrieved_gaps)) <= 0.100
        assert float(np.mean(retrieved_gaps > 0.050)) <= 0.020
        assert int(np.max(sensor_gaps)) == 3


class _HealthOnlyService:
    def __init__(self, health, *, owns_process: bool) -> None:
        self._health = dict(health)
        self.owns_process = bool(owns_process)
        self.process = SimpleNamespace(pid=424242) if owns_process else None

    def start(self):
        return dict(self._health)


def _production_health_fixture(
    *,
    owns_process: bool,
    model_sha_mismatch: bool = False,
    vos_optimized_mismatch: bool = False,
    prewarm_mismatch: bool = False,
    compile_mode_mismatch: bool = False,
):
    contract = _load_fixed_acceptance_contract()
    cfg = {
        "camera": {"width": 848, "height": 480},
        "online_sam2": {
            "checkpoint": "/pinned/checkpoint.pt",
            "model_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "image_size": 512,
            "amp_dtype": "bfloat16",
            "vos_optimized": True,
            "vos_compile_mode": "max-autotune-no-cudagraphs",
        }
    }
    provenance = {
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "model_config_sha256": contract["model_config_sha256"],
    }
    health = {
        "service": "dynamic-pcd-online-sam2",
        "protocol_version": 1,
        "backend": "official-sam2-video-predictor",
        "loaded": True,
        "checkpoint": cfg["online_sam2"]["checkpoint"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "model_config": cfg["online_sam2"]["model_config"],
        "model_config_sha256": (
            "0" * 64
            if model_sha_mismatch
            else contract["model_config_sha256"]
        ),
        "device": "cuda:0",
        "gpu_name": "test-gpu",
        "image_size": 512,
        "amp_dtype": "bfloat16",
        "vos_optimized": not vos_optimized_mismatch,
        "vos_compile_mode": (
            "max-autotune"
            if compile_mode_mismatch
            else "max-autotune-no-cudagraphs"
        ),
        "vos_compile_cuda_graphs": bool(compile_mode_mismatch),
        "vos_component_compile_modes": (
            {
                "image_encoder": "max-autotune",
                "memory_encoder": "max-autotune",
                "memory_attention": "max-autotune",
                "sam_prompt_encoder": "max-autotune",
                "sam_mask_decoder": "max-autotune",
            }
            if compile_mode_mismatch
            else {
                "image_encoder": "max-autotune-no-cudagraphs",
                "memory_encoder": "max-autotune-no-cudagraphs",
                "memory_attention": "max-autotune-no-cudagraphs",
                "sam_prompt_encoder": "max-autotune-no-cudagraphs",
                "sam_mask_decoder": "max-autotune-no-cudagraphs",
            }
        ),
        "vos_component_compile_dynamic": {
            "image_encoder": False,
            "memory_encoder": False,
            "memory_attention": True,
            "sam_prompt_encoder": False,
            "sam_mask_decoder": False,
        },
        "vos_memory_attention_rope_grid_hw": [32, 32],
        "vos_memory_attention_rope_expected_tokens": 1024,
        "vos_memory_attention_rope_cache_count": 8,
        "vos_memory_attention_rope_cache_labels": [
            f"layer{layer}.{attention}"
            for layer in range(4)
            for attention in ("self_attention", "cross_attention")
        ],
        "vos_memory_attention_rope_cache_token_counts": [1024] * 8,
        "vos_memory_attention_rope_caches_verified": True,
        "initialized": False,
        "compile_prewarm_required": True,
        "compile_prewarm_completed": not prewarm_mismatch,
        "compile_prewarm_contract": (
            "initialize_box_track_reset_explicit_mask_track_reset_v1"
        ),
        "compile_prewarm_input_size_wh": [848, 480],
        "compile_prewarm_shape_hw": [480, 848],
        "compile_prewarm_ms": 123.0,
        "compile_prewarm_initialize_box_ms": 10.0,
        "compile_prewarm_track_ms": 11.0,
        "compile_prewarm_initialize_mask_ms": 12.0,
        "compile_prewarm_mask_track_ms": 13.0,
        "tf32": True,
        "fill_hole_area": 0,
    }
    audit = _SAM2ServiceAudit(
        lambda **_kwargs: _HealthOnlyService(
            health, owns_process=owns_process
        )
    )
    audit.manager_factory().start()
    return _service_health_identity(
        audit,
        cfg=cfg,
        config_provenance=provenance,
        acceptance_contract=contract,
        injected_fixture=False,
    )


def test_loopback_foreign_service_is_not_production_identity() -> None:
    _identity, checks, accepted = _production_health_fixture(owns_process=False)
    assert accepted is False
    by_name = {item["check"]: item["passed"] for item in checks}
    assert by_name[
        "validator_launcher_owned_child_no_foreign_service_reuse"
    ] is False


def test_model_config_sha_mismatch_is_not_production_identity() -> None:
    _identity, checks, accepted = _production_health_fixture(
        owns_process=True, model_sha_mismatch=True
    )
    assert accepted is False
    by_name = {item["check"]: item["passed"] for item in checks}
    assert by_name["model_config_identity"] is False


def test_vos_optimized_health_mismatch_is_not_production_identity() -> None:
    _identity, checks, accepted = _production_health_fixture(
        owns_process=True, vos_optimized_mismatch=True
    )
    assert accepted is False
    by_name = {item["check"]: item["passed"] for item in checks}
    assert by_name["vos_optimized_identity"] is False


def test_compile_prewarm_health_mismatch_is_not_production_identity() -> None:
    _identity, checks, accepted = _production_health_fixture(
        owns_process=True, prewarm_mismatch=True
    )
    assert accepted is False
    by_name = {item["check"]: item["passed"] for item in checks}
    assert by_name["optimized_compile_prewarm_identity"] is False


def test_cudagraph_compile_mode_is_not_production_identity() -> None:
    _identity, checks, accepted = _production_health_fixture(
        owns_process=True, compile_mode_mismatch=True
    )
    assert accepted is False
    by_name = {item["check"]: item["passed"] for item in checks}
    assert by_name["optimized_compile_mode_identity"] is False


def test_validator_owned_child_with_pinned_health_identity_passes() -> None:
    identity, checks, accepted = _production_health_fixture(owns_process=True)
    assert accepted is True
    assert all(item["passed"] for item in checks)
    assert identity["validator_launcher_ownership"]["child_pid"] == 424242


def test_default_config_mutation_fails_fixed_acceptance_contract(tmp_path: Path) -> None:
    contract = _load_fixed_acceptance_contract()
    assert contract["sha256"] == (
        "66d6186bdfe333419a9626d6c32def21cf9844984dc9d438bf275c5bde057f27"
    )
    assert contract["default_config_sha256"] == (
        "2e829bac00e5815e2c5c157672787a78201e822de9856c57f5131297bf8ff427"
    )
    assert contract["checkpoint_sha256"] == (
        "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
    )
    assert contract["model_config_sha256"] == (
        "f932eac1c6241e910031b2f000a81cd9f8a8d4896e2277ab5ffb721f378b188d"
    )
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text("camera: {serial: tampered}\n", encoding="utf-8")
    cfg = {"online_sam2": {"image_size": 512}}
    provenance = {
        "source_config_sha256": hashlib.sha256(mutated.read_bytes()).hexdigest(),
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "model_config_sha256": contract["model_config_sha256"],
    }
    assert not _config_matches_fixed_acceptance_contract(
        cfg=cfg,
        config_provenance=provenance,
        acceptance_contract=contract,
    )


def test_production_positive_acceptance_aggregate_requires_all_stages() -> None:
    summary = {
        "projector_fresh_fraction_including_initialization": 1.0,
        "projector_full_128_fraction_including_initialization": 1.0,
        "projector_current_frame_provenance_fraction": 1.0,
        "current_provider_valid_fraction": 0.98,
        "sam2_temporal_service": {"track_call_fraction": 1.0},
        "production_evidence_eligible": True,
        "provider_plus_projector_compute_ms": {
            "p50": 25.0,
            "p95": 30.0,
            "max": 40.0,
            "over_50ms_fraction": 0.0,
            "over_50ms_count": 0,
            "over_50ms_longest_run": 0,
            "sample_count": 59,
        },
        "capture_cadence_evidence": {
            "accepted_end_to_end_cadence": True
        },
    }
    result = _finalize_acceptance_artifact(
        summary,
        guarded_v2_mode=True,
        adaptive_mode=False,
        point_feature_mode="xyz",
        maximum_depth_deviation=0.055,
        fixed_sphere_radius_m=None,
        minimum_fresh_fraction=1.0,
        minimum_provider_valid_fraction=0.95,
        minimum_sam2_track_call_fraction=1.0,
        minimum_full_128_fraction=1.0,
    )
    assert result["production_defaults_used"] is True
    assert result["accepted_geometry"] is True
    assert result["accepted_pipeline_identity"] is True
    assert result["accepted_pipeline_timing"] is True
    assert result["accepted_end_to_end_cadence"] is True
    assert result["accepted"] is True


def test_production_acceptance_rejects_combined_provider_projector_stall() -> None:
    summary = {
        "projector_fresh_fraction_including_initialization": 1.0,
        "projector_full_128_fraction_including_initialization": 1.0,
        "projector_current_frame_provenance_fraction": 1.0,
        "current_provider_valid_fraction": 1.0,
        "sam2_temporal_service": {"track_call_fraction": 1.0},
        "production_evidence_eligible": True,
        "provider_plus_projector_compute_ms": {
            "p50": 30.0,
            "p95": 45.0,
            "max": 120.0,
            "over_50ms_fraction": 0.02,
            "over_50ms_count": 2,
            "over_50ms_longest_run": 2,
            "sample_count": 100,
        },
        "capture_cadence_evidence": {"accepted_end_to_end_cadence": True},
    }
    result = _finalize_acceptance_artifact(
        summary,
        guarded_v2_mode=True,
        adaptive_mode=False,
        point_feature_mode="xyz",
        maximum_depth_deviation=0.055,
        fixed_sphere_radius_m=None,
        minimum_fresh_fraction=1.0,
        minimum_provider_valid_fraction=0.95,
        minimum_sam2_track_call_fraction=1.0,
        minimum_full_128_fraction=1.0,
    )

    checks = {item["check"]: item for item in result["checks"]}
    assert result["accepted_pipeline_timing"] is False
    assert checks["provider_plus_projector_compute_max"]["passed"] is False
    assert (
        checks["provider_plus_projector_over_50ms_longest_run"]["passed"]
        is False
    )
    assert result["accepted"] is False
