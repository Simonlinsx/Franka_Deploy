import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dynamic_pcd.evaluation.guarded_v2_offline import (
    OfflineBenchmarkError,
    _load_timings,
    _sample_indices,
    _verify_production_sam2_readiness_provenance,
    evaluate_manifest,
)
from dynamic_pcd.segmentation.sam2_video_backend import (
    PRODUCTION_VOS_COMPONENT_COMPILE_MODES,
    PRODUCTION_VOS_COMPONENT_DYNAMIC,
    PRODUCTION_VOS_COMPILE_MODE,
    VOS_COMPILE_PREWARM_CONTRACT,
)


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 20.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _production_sam2_readiness_fixture() -> tuple[dict, dict]:
    checkpoint_sha = "3" * 64
    model_config_sha = "4" * 64
    case_summary = {
        "sam2_image_size": 512,
        "online_sam2_bbox_prewarm": {
            "enabled": True,
            "attempted": True,
            "passed": True,
            "seed_frame_id": 39,
            "required_stable_tracks": 2,
            "achieved_stable_tracks": 2,
            "max_rpc_ms": 12.0,
            "exact_reseed_rpc_ms": 10.0,
            "total_ms": 40.0,
            "status": "discarded_tracks_stable_then_exact_seed_restored",
        },
        "online_sam2_service_health": {
            "backend": "official-sam2-video-predictor",
            "loaded": True,
            "initialized": False,
            "checkpoint_sha256": checkpoint_sha,
            "model_config_sha256": model_config_sha,
            "image_size": 512,
            "vos_optimized": True,
            "vos_compile_mode": PRODUCTION_VOS_COMPILE_MODE,
            "vos_compile_cuda_graphs": False,
            "vos_component_compile_modes": dict(
                PRODUCTION_VOS_COMPONENT_COMPILE_MODES
            ),
            "vos_component_compile_dynamic": dict(
                PRODUCTION_VOS_COMPONENT_DYNAMIC
            ),
            "vos_memory_attention_rope_grid_hw": [32, 32],
            "vos_memory_attention_rope_expected_tokens": 1024,
            "vos_memory_attention_rope_cache_count": 8,
            "vos_memory_attention_rope_cache_token_counts": [1024] * 8,
            "vos_memory_attention_rope_caches_verified": True,
            "compile_prewarm_required": True,
            "compile_prewarm_completed": True,
            "compile_prewarm_contract": VOS_COMPILE_PREWARM_CONTRACT,
            "compile_prewarm_shape_hw": [480, 848],
            "compile_prewarm_initialize_box_ms": 11.0,
            "compile_prewarm_track_ms": 12.0,
            "compile_prewarm_initialize_mask_ms": 11.0,
            "compile_prewarm_mask_track_ms": 12.0,
        },
    }
    config_provenance = {
        "checkpoint_sha256": checkpoint_sha,
        "model_config_sha256": model_config_sha,
    }
    return case_summary, config_provenance


def test_production_sam2_readiness_requires_hot_exact_bbox_and_service_identity():
    case_summary, config_provenance = _production_sam2_readiness_fixture()
    kwargs = {
        "case_name": "fast_green_ball_entry",
        "config_provenance": config_provenance,
        "source_size_wh": [848, 480],
        "seed_source_frame": 39,
    }
    _verify_production_sam2_readiness_provenance(
        case_summary=case_summary, **kwargs
    )

    mutated = json.loads(json.dumps(case_summary))
    mutated["online_sam2_bbox_prewarm"]["passed"] = False
    with pytest.raises(OfflineBenchmarkError, match="bbox prewarm"):
        _verify_production_sam2_readiness_provenance(
            case_summary=mutated, **kwargs
        )

    mutated = json.loads(json.dumps(case_summary))
    mutated["online_sam2_service_health"]["compile_prewarm_completed"] = False
    with pytest.raises(OfflineBenchmarkError, match="service health"):
        _verify_production_sam2_readiness_provenance(
            case_summary=mutated, **kwargs
        )


def _strict_partial_replay_fixture(
    tmp_path: Path, *, incomplete_latency: bool = False, semantic_direct: bool = False
):
    name = "strict_partial"
    video = tmp_path / "source.mp4"
    frames = []
    for _ in range(4):
        frame = np.full((32, 48, 3), 20, dtype=np.uint8)
        cv2.circle(frame, (20, 18), 5, (20, 220, 20), -1)
        frames.append(frame)
    _write_video(video, frames, fps=20.0)
    root = tmp_path / "candidate"
    case_root = root / name
    masks_dir = case_root / "masks"
    masks_dir.mkdir(parents=True)
    masks = []
    for frame_index in range(4):
        path = masks_dir / f"{frame_index:06d}.png"
        mask = np.zeros((32, 48), dtype=np.uint8)
        cv2.circle(mask, (20, 18), 5, 255, -1)
        assert cv2.imwrite(str(path), mask)
        masks.append(
            {
                "frame_index": frame_index,
                "source_frame_index": frame_index,
                "path": f"masks/{frame_index:06d}.png",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    states = []
    for frame_index in range(4):
        evaluable = frame_index > 0
        if incomplete_latency and frame_index > 1:
            evaluable = False
        states.append(
            {
                "frame_index": frame_index,
                "source_frame_index": frame_index,
                "compute_processing_ms": float(10 + frame_index),
                "latency_evaluable": evaluable,
            }
        )
    states_path = case_root / "states.jsonl"
    states_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in states),
        encoding="utf-8",
    )
    artifact_content = {
        "masks": masks,
        "states": {
            "path": "states.jsonl",
            "sha256": hashlib.sha256(states_path.read_bytes()).hexdigest(),
        },
    }
    artifact = {
        **artifact_content,
        "aggregate_sha256": _canonical_sha256(artifact_content),
    }
    config_provenance = {
        "source_config_sha256": "1" * 64,
        "canonical_effective_config_sha256": "2" * 64,
        "checkpoint_sha256": "3" * 64,
        "model_config_sha256": "4" * 64,
    }
    source_sha = hashlib.sha256(video.read_bytes()).hexdigest()
    case_summary = {
        "schema": "guarded_v2_real_rgb_replay_summary_v2",
        "case": name,
        "source_video_sha256": source_sha,
        "source_video_metadata": {
            "fps_hz": 20.0,
            "frame_count": 4,
            "size_wh": [48, 32],
            "sha256": source_sha,
        },
        "sampled_frames": 4,
        "sampled_source_indices": [0, 1, 2, 3],
        "evaluation_fps_hz": 20.0,
        "seed_position": 0,
        "seed_source_frame": 0,
        "effective_config_provenance": config_provenance,
        "candidate_artifacts": artifact,
        "requested_object_mask_mode": (
            "diagnostic_semantic_sam2_direct" if semantic_direct else "guarded_v2"
        ),
        "diagnostic_only": True,
        "production_acceptance_eligible": False,
    }
    case_summary_path = case_root / "summary.json"
    case_summary_path.write_text(
        json.dumps(case_summary, sort_keys=True), encoding="utf-8"
    )
    chain = [
        {
            "case": name,
            "summary_path": f"{name}/summary.json",
            "summary_sha256": hashlib.sha256(
                case_summary_path.read_bytes()
            ).hexdigest(),
            "candidate_artifacts_sha256": artifact["aggregate_sha256"],
        }
    ]
    root_summary = {
        "schema": "guarded_v2_real_rgb_replay_summary_v2",
        "case_count": 1,
        "cases": [case_summary],
        "case_summary_chain": chain,
        "candidate_root_artifacts_sha256": _canonical_sha256(chain),
        "effective_config_provenance": config_provenance,
    }
    (root / "summary.json").write_text(
        json.dumps(root_summary, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "cases": [
            {
                "name": name,
                "video": str(video),
                "evaluation_fps_hz": 20.0,
                "candidate": {"kind": "directory", "path": str(masks_dir)},
                "target_proxy": {
                    "hsv_ranges": [[[35, 80, 40], [95, 255, 255]]],
                    "initial_seed_xy": [20, 18],
                    "min_area_px": 20,
                    "min_visible_area_px": 20,
                    "open_kernel": 1,
                    "close_kernel": 1,
                },
                "thresholds": {"latency_p95_ms_max": 20.0},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, root, case_root


def test_sample_indices_match_historical_30_to_20_hz_selection():
    assert _sample_indices(
        80, source_fps=30.0, start_frame=60, evaluation_fps=20.0
    ) == [60, 62, 63, 65, 66, 68, 69, 71, 72, 74, 75, 77, 78]


def test_strict_partial_replay_contract_is_diagnostic_not_production(tmp_path):
    manifest, root, _case_root = _strict_partial_replay_fixture(tmp_path)
    result = evaluate_manifest(
        manifest,
        tmp_path / "output",
        candidate_root=root,
        candidate_root_mode="diagnostic_partial",
    )
    assert result["diagnostic_checks_passed"] is True
    assert result["production_acceptance_eligible"] is False
    assert result["passed"] is False


def test_candidate_root_rejects_arbitrary_root_without_replay_summaries(tmp_path):
    manifest, _root, _case_root = _strict_partial_replay_fixture(tmp_path)
    arbitrary = tmp_path / "arbitrary"
    arbitrary.mkdir()
    with pytest.raises(OfflineBenchmarkError, match="replay summary"):
        evaluate_manifest(
            manifest,
            tmp_path / "output",
            candidate_root=arbitrary,
            candidate_root_mode="diagnostic_partial",
        )


def test_candidate_root_rejects_wrong_sampled_source_index(tmp_path):
    manifest, root, case_root = _strict_partial_replay_fixture(tmp_path)
    case_summary_path = case_root / "summary.json"
    case_summary = json.loads(case_summary_path.read_text(encoding="utf-8"))
    case_summary["sampled_source_indices"][1] = 99
    case_summary_path.write_text(json.dumps(case_summary), encoding="utf-8")
    root_summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    root_summary["cases"][0] = case_summary
    (root / "summary.json").write_text(json.dumps(root_summary), encoding="utf-8")
    with pytest.raises(OfflineBenchmarkError, match="sampled_source_indices"):
        evaluate_manifest(
            manifest,
            tmp_path / "output",
            candidate_root=root,
            candidate_root_mode="diagnostic_partial",
        )


def test_candidate_root_rejects_missing_middle_mask_plus_tail(tmp_path):
    manifest, root, case_root = _strict_partial_replay_fixture(tmp_path)
    middle = case_root / "masks" / "000001.png"
    tail = case_root / "masks" / "000004.png"
    tail.write_bytes(middle.read_bytes())
    middle.unlink()
    with pytest.raises(OfflineBenchmarkError, match="mask filenames"):
        evaluate_manifest(
            manifest,
            tmp_path / "output",
            candidate_root=root,
            candidate_root_mode="diagnostic_partial",
        )


def test_candidate_root_rejects_tampered_mask_bytes(tmp_path):
    manifest, root, case_root = _strict_partial_replay_fixture(tmp_path)
    tampered = np.zeros((32, 48), dtype=np.uint8)
    assert cv2.imwrite(str(case_root / "masks" / "000002.png"), tampered)
    with pytest.raises(OfflineBenchmarkError, match="digest/index/path"):
        evaluate_manifest(
            manifest,
            tmp_path / "output",
            candidate_root=root,
            candidate_root_mode="diagnostic_partial",
        )


def test_candidate_root_rejects_single_latency_sample(tmp_path):
    manifest, root, _case_root = _strict_partial_replay_fixture(
        tmp_path, incomplete_latency=True
    )
    with pytest.raises(OfflineBenchmarkError, match="latency_evaluable"):
        evaluate_manifest(
            manifest,
            tmp_path / "output",
            candidate_root=root,
            candidate_root_mode="diagnostic_partial",
        )


def _write_strict_bootstrap_timing_states(tmp_path: Path) -> tuple[Path, dict]:
    records = [
        {
            "frame_index": 0,
            "source_frame_index": 0,
            "compute_processing_ms": 10.0,
            "latency_evaluable": False,
        },
        {
            "frame_index": 1,
            "source_frame_index": 2,
            "compute_processing_ms": 11.0,
            "latency_evaluable": False,
            "guarded_v2_bootstrap": {
                "phase": "provisional",
                "commission_count": 0,
                "commissioned_frame_id": None,
            },
        },
        {
            "frame_index": 2,
            "source_frame_index": 3,
            "compute_processing_ms": 12.0,
            "latency_evaluable": False,
            "guarded_v2_bootstrap": {
                "phase": "provisional",
                "commission_count": 0,
                "commissioned_frame_id": None,
            },
        },
        {
            "frame_index": 3,
            "source_frame_index": 5,
            "compute_processing_ms": 13.0,
            "latency_evaluable": False,
            "guarded_v2_bootstrap": {
                "phase": "commissioned",
                "commission_count": 1,
                "commissioned_frame_id": 5,
            },
        },
        {
            "frame_index": 4,
            "source_frame_index": 6,
            "compute_processing_ms": 14.0,
            "latency_evaluable": True,
            "guarded_v2_bootstrap": {
                "phase": "commissioned",
                "commission_count": 1,
                "commissioned_frame_id": 5,
            },
        },
    ]
    path = tmp_path / "states.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    spec = {
        "states": str(path),
        "states_latency_field": "compute_processing_ms",
        "states_latency_evaluable_field": "latency_evaluable",
        "strict_states_contract": True,
        "expected_source_indices": [0, 2, 3, 5, 6],
        "seed_position": 1,
        "expected_guarded_v2_bootstrap_final": {
            "phase": "commissioned",
            "commission_count": 1,
            "commissioned_frame_id": 5,
        },
    }
    return path, spec


def test_strict_timing_contract_starts_after_bootstrap_commission(tmp_path):
    _path, spec = _write_strict_bootstrap_timing_states(tmp_path)

    timings = _load_timings(spec, tmp_path)

    assert timings == {4: 14.0}


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("premature_latency", "must remain false through bootstrap"),
        ("never_commissioned", "never reached one bootstrap commission"),
        ("regressed", "regressed from commissioned"),
        ("summary_mismatch", "does not match the replay summary"),
    ),
)
def test_strict_timing_contract_rejects_invalid_bootstrap_suffix(
    tmp_path, mutation, error
):
    path, spec = _write_strict_bootstrap_timing_states(tmp_path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "premature_latency":
        records[2]["latency_evaluable"] = True
    elif mutation == "never_commissioned":
        for record in records[3:]:
            record["guarded_v2_bootstrap"] = {
                "phase": "provisional",
                "commission_count": 0,
                "commissioned_frame_id": None,
            }
            record["latency_evaluable"] = False
    elif mutation == "regressed":
        records[4]["guarded_v2_bootstrap"] = {
            "phase": "provisional",
            "commission_count": 0,
            "commissioned_frame_id": None,
        }
        records[4]["latency_evaluable"] = False
    elif mutation == "summary_mismatch":
        spec["expected_guarded_v2_bootstrap_final"]["commissioned_frame_id"] = 3
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(OfflineBenchmarkError, match=error):
        _load_timings(spec, tmp_path)


def _write_strict_boundary_rearm_timing_states(tmp_path):
    records = [
        {
            "frame_index": 0,
            "source_frame_index": 92,
            "compute_processing_ms": 10.0,
            "latency_evaluable": False,
            "valid": True,
            "mask_source": "online_sam2_box_initialization",
            "guarded_v2_bootstrap": {
                "phase": "provisional",
                "commission_count": 0,
                "commissioned_frame_id": None,
                "boundary_rearm_count": 0,
                "seed_frame_id": 92,
                "status": "armed from provisional target seed",
            },
        },
        {
            "frame_index": 1,
            "source_frame_index": 93,
            "compute_processing_ms": 11.0,
            "latency_evaluable": False,
            "valid": True,
            "mask_source": "online_sam2_guarded_primary",
            "guarded_v2_bootstrap": {
                "phase": "expired",
                "commission_count": 0,
                "commissioned_frame_id": None,
                "boundary_rearm_count": 0,
                "seed_frame_id": 92,
                "status": (
                    "expired: raw candidate touches the effective image boundary"
                ),
            },
        },
        {
            "frame_index": 2,
            "source_frame_index": 95,
            "compute_processing_ms": 12.0,
            "latency_evaluable": False,
            "valid": False,
            "mask_source": "none",
            "guarded_v2_bootstrap": {
                "phase": "expired",
                "commission_count": 0,
                "commissioned_frame_id": None,
                "boundary_rearm_count": 0,
                "seed_frame_id": 92,
                "status": (
                    "expired: raw candidate touches the effective image boundary"
                ),
            },
        },
        {
            "frame_index": 3,
            "source_frame_index": 119,
            "compute_processing_ms": 13.0,
            "latency_evaluable": False,
            "valid": True,
            "mask_source": "recovery_committed",
            "guarded_v2_bootstrap": {
                "phase": "provisional",
                "commission_count": 0,
                "commissioned_frame_id": None,
                "boundary_rearm_count": 1,
                "seed_frame_id": 119,
                "status": (
                    "rearmed once from confirmed interior recovery after "
                    "boundary-clipped seed"
                ),
            },
        },
        {
            "frame_index": 4,
            "source_frame_index": 120,
            "compute_processing_ms": 14.0,
            "latency_evaluable": False,
            "valid": True,
            "mask_source": "online_sam2_guarded_primary",
            "guarded_v2_bootstrap": {
                "phase": "provisional",
                "commission_count": 0,
                "commissioned_frame_id": None,
                "boundary_rearm_count": 1,
                "seed_frame_id": 119,
                "status": "stable candidate 1/2 at frame 120",
            },
        },
        {
            "frame_index": 5,
            "source_frame_index": 122,
            "compute_processing_ms": 15.0,
            "latency_evaluable": False,
            "valid": True,
            "mask_source": "online_sam2_guarded_primary",
            "guarded_v2_bootstrap": {
                "phase": "commissioned",
                "commission_count": 1,
                "commissioned_frame_id": 122,
                "boundary_rearm_count": 1,
                "seed_frame_id": 119,
                "status": "commissioned",
            },
        },
        {
            "frame_index": 6,
            "source_frame_index": 123,
            "compute_processing_ms": 16.0,
            "latency_evaluable": True,
            "valid": True,
            "mask_source": "online_sam2_guarded_primary",
            "guarded_v2_bootstrap": {
                "phase": "commissioned",
                "commission_count": 1,
                "commissioned_frame_id": 122,
                "boundary_rearm_count": 1,
                "seed_frame_id": 119,
                "status": "commissioned",
            },
        },
    ]
    path = tmp_path / "states-boundary-rearm.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path, {
        "states": str(path),
        "states_latency_field": "compute_processing_ms",
        "states_latency_evaluable_field": "latency_evaluable",
        "strict_states_contract": True,
        "expected_source_indices": [92, 93, 95, 119, 120, 122, 123],
        "seed_position": 0,
        "expected_guarded_v2_bootstrap_final": {
            "phase": "commissioned",
            "commission_count": 1,
            "commissioned_frame_id": 122,
            "boundary_rearm_count": 1,
        },
    }


def test_strict_timing_accepts_boundary_expiry_then_one_shot_rearm(tmp_path):
    _path, spec = _write_strict_boundary_rearm_timing_states(tmp_path)

    timings = _load_timings(spec, tmp_path)

    assert timings == {6: 16.0}


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("wrong_expiry", "exact boundary-clipped seed"),
        ("invalid_rearm", "one-shot confirmed-interior recovery"),
        ("wrong_source", "one-shot confirmed-interior recovery"),
        ("wrong_seed", "one-shot confirmed-interior recovery"),
        ("second_rearm", "boundary_rearm_count must be integer 0 or 1"),
        ("expired_after_rearm", "exact boundary-clipped seed"),
    ),
)
def test_strict_timing_rejects_forged_boundary_rearm(tmp_path, mutation, error):
    path, spec = _write_strict_boundary_rearm_timing_states(tmp_path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "wrong_expiry":
        records[1]["guarded_v2_bootstrap"]["status"] = "expired: generic"
    elif mutation == "invalid_rearm":
        records[3]["valid"] = False
    elif mutation == "wrong_source":
        records[3]["mask_source"] = "online_sam2_guarded_primary"
    elif mutation == "wrong_seed":
        records[3]["guarded_v2_bootstrap"]["seed_frame_id"] = 118
    elif mutation == "second_rearm":
        records[4]["guarded_v2_bootstrap"]["boundary_rearm_count"] = 2
    elif mutation == "expired_after_rearm":
        records[4]["guarded_v2_bootstrap"].update(
            {
                "phase": "expired",
                "status": (
                    "expired: raw candidate touches the effective image boundary"
                ),
            }
        )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(OfflineBenchmarkError, match=error):
        _load_timings(spec, tmp_path)


def test_semantic_direct_partial_replay_cannot_report_production_pass(tmp_path):
    manifest, root, _case_root = _strict_partial_replay_fixture(
        tmp_path, semantic_direct=True
    )
    result = evaluate_manifest(
        manifest,
        tmp_path / "output",
        candidate_root=root,
        candidate_root_mode="diagnostic_partial",
    )
    assert result["diagnostic_checks_passed"] is True
    assert result["diagnostic_only"] is True
    assert result["passed"] is False


def test_offline_benchmark_reports_contamination_growth_jump_and_latency(tmp_path):
    video_path = tmp_path / "source.mp4"
    masks_path = tmp_path / "masks"
    masks_path.mkdir()
    states_path = tmp_path / "states.jsonl"
    frames: list[np.ndarray] = []
    state_lines: list[str] = []
    for frame_id in range(10):
        frame = np.full((64, 96, 3), 25, dtype=np.uint8)
        center = (24 + 2 * frame_id, 36)
        cv2.circle(frame, center, 9, (30, 220, 30), -1)
        frames.append(frame)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, 9, 255, -1)
        if frame_id == 6:
            # A neutral appendage mimics the RH56-finger contamination mode.
            mask[8:36, center[0] - 2 : center[0] + 3] = 255
        assert cv2.imwrite(str(masks_path / f"{frame_id:06d}.png"), mask)
        state_lines.append(
            json.dumps(
                {
                    "frame_index": frame_id,
                    "processing_ms": 400.0 if frame_id == 0 else 100.0,
                    "compute_processing_ms": 20 + frame_id,
                    "latency_evaluable": frame_id != 0,
                }
            )
        )
    _write_video(video_path, frames)
    states_path.write_text("\n".join(state_lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "cases": [
            {
                "name": "synthetic_hand_appendage",
                "video": str(video_path),
                "candidate": {
                    "name": "candidate",
                    "kind": "directory",
                    "path": str(masks_path),
                    "states": str(states_path),
                    "states_latency_field": "compute_processing_ms",
                },
                    "target_proxy": {
                    "hsv_ranges": [[[35, 80, 40], [95, 255, 255]]],
                    "initial_seed_xy": [24, 36],
                    "min_area_px": 80,
                    "min_visible_area_px": 80,
                    "open_kernel": 1,
                    "close_kernel": 1,
                },
                "contamination_proxy_dilation_px": 2,
                "thresholds": {
                    "target_coverage_min": 0.99,
                    "proxy_recall_p05_min": 0.90,
                    "contamination_p95_max": 0.05,
                    "bbox_growth_p95_max": 1.20,
                    "latency_p95_ms_max": 30.0,
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = evaluate_manifest(manifest_path, tmp_path / "output")

    case = result["cases"][0]
    assert case["target_coverage"]["value"] == 1.0
    # Lossy MP4 chroma edges make the colour proxy slightly larger/smaller
    # than the lossless PNG candidate, but their target overlap stays high.
    assert case["proxy_recall"]["p05"] > 0.85
    assert case["contamination_proxy"]["max"] > 0.10
    assert case["bbox_area_growth"]["max"] > 1.5
    assert case["latency_ms"]["p95"] < 30.0
    assert not result["passed"]
    assert (tmp_path / "output" / "REPORT.md").is_file()
    assert (
        tmp_path
        / "output"
        / "synthetic_hand_appendage"
        / "candidate"
        / "frames.csv"
    ).is_file()


def test_sparse_reviewed_ground_truth_reports_true_overlap_metrics(tmp_path):
    video_path = tmp_path / "source.mp4"
    masks_path = tmp_path / "masks"
    labels_path = tmp_path / "labels"
    masks_path.mkdir()
    labels_path.mkdir()
    frames: list[np.ndarray] = []
    sparse_records: list[dict[str, object]] = []
    for frame_id in range(5):
        frame = np.full((64, 96, 3), 20, dtype=np.uint8)
        center = (24 + 5 * frame_id, 36)
        cv2.circle(frame, center, 8, (20, 220, 20), -1)
        frames.append(frame)
        ground_truth = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(ground_truth, center, 8, 255, -1)
        label_path = labels_path / f"source_{frame_id:06d}.png"
        assert cv2.imwrite(str(label_path), ground_truth)
        sparse_records.append(
            {
                "source_frame": frame_id,
                "path": str(label_path),
                "sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
                "review_status": "human_confirmed",
                "label_method": "synthetic exact mask",
                "review_notes": "Lossless test fixture with visible target only.",
            }
        )
        candidate = ground_truth.copy()
        if frame_id == 4:
            candidate[30:43, center[0] + 8 : center[0] + 12] = 255
        assert cv2.imwrite(str(masks_path / f"{frame_id:06d}.png"), candidate)
    _write_video(video_path, frames)
    manifest = {
        "schema_version": 1,
        "cases": [
            {
                "name": "reviewed_sparse",
                "video": str(video_path),
                "candidate": {"kind": "directory", "path": str(masks_path)},
                "target_proxy": {
                    "hsv_ranges": [[[35, 80, 40], [95, 255, 255]]],
                    "initial_seed_xy": [24, 36],
                    "min_area_px": 20,
                    "min_visible_area_px": 20,
                    "open_kernel": 1,
                    "close_kernel": 1,
                    "max_center_jump_px": 10,
                },
                "reviewed_visible_source_intervals": [[0, 4]],
                "thresholds": {
                    "reviewed_visible_nonempty_coverage_min": 1.0,
                    # Deliberately impossible for the lossy HSV proxy.  Once
                    # enough human-reviewed masks exist this remains visible
                    # in the report, but it must not veto the true-mask gate.
                    "proxy_recall_p05_min": 1.01,
                },
                "sparse_ground_truth": {
                    "schema": "reviewed_visible_object_masks_v1",
                    "source_video": str(video_path),
                    "source_video_sha256": hashlib.sha256(
                        video_path.read_bytes()
                    ).hexdigest(),
                    "min_reviewed_frames": 5,
                    "records": sparse_records,
                    "thresholds": {
                        "sparse_gt_reviewed_frames_min": 5,
                        "sparse_gt_iou_p05_min": 0.75,
                        "sparse_gt_recall_p05_min": 0.95,
                        "sparse_gt_precision_p05_min": 0.75,
                        "sparse_gt_contamination_p95_max": 0.25,
                    },
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = evaluate_manifest(manifest_path, tmp_path / "output")

    case = result["cases"][0]
    assert case["sparse_gt_reviewed_frames"]["value"] == 5
    assert case["sparse_gt_iou"]["min"] < 1.0
    assert case["sparse_gt_iou"]["p50"] == 1.0
    assert case["sparse_gt_recall"]["min"] == 1.0
    assert case["sparse_gt_precision"]["min"] < 1.0
    assert case["sparse_ground_truth"]["records"][0]["sha256"]
    checks = {check["threshold"]: check for check in case["checks"]}
    assert checks["reviewed_visible_nonempty_coverage_min"]["passed"]
    assert checks["reviewed_visible_nonempty_coverage_min"]["gating"]
    assert not checks["proxy_recall_p05_min"]["passed"]
    assert not checks["proxy_recall_p05_min"]["gating"]
    assert case["dense_proxy_quality_role"] == (
        "diagnostic_non_gating_when_human_sparse_gt_is_sufficient"
    )
    assert result["passed"]
    assert (
        tmp_path
        / "output"
        / "reviewed_sparse"
        / "candidate"
        / "sparse_gt_source_000004.png"
    ).is_file()


def test_sparse_reviewed_ground_truth_rejects_sha_mismatch(tmp_path):
    video_path = tmp_path / "source.mp4"
    masks_path = tmp_path / "masks"
    labels_path = tmp_path / "labels"
    masks_path.mkdir()
    labels_path.mkdir()
    frame = np.full((48, 64, 3), 20, dtype=np.uint8)
    cv2.circle(frame, (28, 28), 7, (20, 220, 20), -1)
    _write_video(video_path, [frame] * 5)
    records = []
    for frame_id in range(5):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (28, 28), 7, 255, -1)
        candidate_path = masks_path / f"{frame_id:06d}.png"
        label_path = labels_path / f"source_{frame_id:06d}.png"
        assert cv2.imwrite(str(candidate_path), mask)
        assert cv2.imwrite(str(label_path), mask)
        records.append(
            {
                "source_frame": frame_id,
                "path": str(label_path),
                "sha256": "0" * 64,
                "review_status": "human_confirmed",
                "label_method": "synthetic",
                "review_notes": "Fixture deliberately has wrong digest.",
            }
        )
    manifest = {
        "schema_version": 1,
        "cases": [
            {
                "name": "bad_digest",
                "video": str(video_path),
                "candidate": {"kind": "directory", "path": str(masks_path)},
                "target_proxy": {
                    "hsv_ranges": [[[35, 80, 40], [95, 255, 255]]],
                    "min_area_px": 20,
                    "open_kernel": 1,
                    "close_kernel": 1,
                },
                "sparse_ground_truth": {
                    "schema": "reviewed_visible_object_masks_v1",
                    "source_video": str(video_path),
                    "source_video_sha256": hashlib.sha256(
                        video_path.read_bytes()
                    ).hexdigest(),
                    "records": records,
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        evaluate_manifest(manifest_path, tmp_path / "output")
    except RuntimeError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("wrong sparse ground-truth digest was accepted")


def test_motion_proxy_does_not_seed_from_first_static_same_colour_object(tmp_path):
    video_path = tmp_path / "source.mp4"
    masks_path = tmp_path / "masks"
    masks_path.mkdir()
    frames: list[np.ndarray] = []
    for frame_id in range(4):
        frame = np.full((64, 96, 3), 20, dtype=np.uint8)
        cv2.circle(frame, (80, 10), 8, (20, 220, 20), -1)
        center = (20 + 8 * frame_id, 42)
        cv2.circle(frame, center, 7, (20, 220, 20), -1)
        frames.append(frame)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, 7, 255, -1)
        assert cv2.imwrite(str(masks_path / f"{frame_id:06d}.png"), mask)
    _write_video(video_path, frames)
    manifest = {
        "schema_version": 1,
        "cases": [
            {
                "name": "moving_target",
                "video": str(video_path),
                "candidate": {
                    "kind": "directory",
                    "path": str(masks_path),
                },
                "target_proxy": {
                    "hsv_ranges": [[[35, 80, 40], [95, 255, 255]]],
                    "motion_threshold_bgr": 15,
                    "min_area_px": 20,
                    "min_visible_area_px": 20,
                    "open_kernel": 1,
                    "close_kernel": 1,
                },
                "thresholds": {"target_coverage_min": 0.0},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = evaluate_manifest(manifest_path, tmp_path / "output")

    case = result["cases"][0]
    assert case["visible_proxy_frames"] == 3
    assert case["target_coverage"]["value"] == 1.0


def test_known_entry_frame_keeps_earlier_same_colour_motion_target_absent(tmp_path):
    video_path = tmp_path / "source.mp4"
    masks_path = tmp_path / "masks"
    masks_path.mkdir()
    frames: list[np.ndarray] = []
    for frame_id in range(6):
        frame = np.full((64, 96, 3), 20, dtype=np.uint8)
        # An unrelated green region moves before the labelled ball enters.
        if frame_id < 3:
            cv2.circle(frame, (12 + frame_id * 5, 10), 7, (20, 220, 20), -1)
        else:
            cv2.circle(frame, (30 + frame_id * 3, 42), 7, (20, 220, 20), -1)
        frames.append(frame)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if frame_id >= 3:
            cv2.circle(mask, (30 + frame_id * 3, 42), 7, 255, -1)
        assert cv2.imwrite(str(masks_path / f"{frame_id:06d}.png"), mask)
    _write_video(video_path, frames)
    manifest = {
        "schema_version": 1,
        "cases": [
            {
                "name": "known_entry",
                "video": str(video_path),
                "target_first_visible_source_frame": 3,
                "candidate": {"kind": "directory", "path": str(masks_path)},
                "target_proxy": {
                    "hsv_ranges": [[[35, 80, 40], [95, 255, 255]]],
                    "initial_seed_xy": [39, 42],
                    "min_area_px": 20,
                    "min_visible_area_px": 20,
                    "open_kernel": 1,
                    "close_kernel": 1,
                },
                "thresholds": {"absent_false_positive_max": 0.0},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = evaluate_manifest(manifest_path, tmp_path / "output")

    case = result["cases"][0]
    assert case["target_first_visible_source_frame"] == 3
    assert case["visible_proxy_frames"] == 3
    assert case["absent_false_positive"]["absent_frames"] == 3
    assert case["absent_false_positive"]["value"] == 0.0
    assert result["passed"]


def test_candidate_root_latency_uses_post_seed_compute_only(tmp_path):
    video_path = tmp_path / "source.mp4"
    frames: list[np.ndarray] = []
    for _ in range(3):
        frame = np.full((48, 64, 3), 20, dtype=np.uint8)
        cv2.circle(frame, (28, 28), 7, (20, 220, 20), -1)
        frames.append(frame)
    _write_video(video_path, frames, fps=20.0)

    candidate_root = tmp_path / "candidate"
    case_root = candidate_root / "post_seed_compute"
    masks_path = case_root / "masks"
    masks_path.mkdir(parents=True)
    for frame_id in range(3):
        mask = np.zeros((48, 64), dtype=np.uint8)
        cv2.circle(mask, (28, 28), 7, 255, -1)
        assert cv2.imwrite(str(masks_path / f"{frame_id:06d}.png"), mask)
    (case_root / "states.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "frame_index": 0,
                        "processing_ms": 400.0,
                        "compute_processing_ms": 400.0,
                        "latency_evaluable": False,
                    }
                ),
                json.dumps(
                    {
                        "frame_index": 1,
                        "processing_ms": 999.0,
                        "compute_processing_ms": 10.0,
                        "latency_evaluable": True,
                    }
                ),
                # Old state streams have no evaluable field; absence must
                # remain backwards-compatible and therefore evaluable.
                json.dumps(
                    {
                        "frame_index": 2,
                        "processing_ms": 999.0,
                        "compute_processing_ms": 12.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "cases": [
            {
                "name": "post_seed_compute",
                "video": str(video_path),
                "evaluation_fps_hz": 20.0,
                "candidate": {
                    "kind": "directory",
                    "path": str(masks_path),
                },
                "target_proxy": {
                    "hsv_ranges": [[[35, 80, 40], [95, 255, 255]]],
                    "initial_seed_xy": [28, 28],
                    "min_area_px": 20,
                    "min_visible_area_px": 20,
                    "open_kernel": 1,
                    "close_kernel": 1,
                },
                "thresholds": {"latency_p95_ms_max": 20.0},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = evaluate_manifest(
        manifest_path,
        tmp_path / "output",
        candidate_root=candidate_root,
        candidate_root_mode="diagnostic_legacy",
    )

    latency = result["cases"][0]["latency_ms"]
    assert latency["min"] == 10.0
    assert latency["p50"] == 11.0
    assert latency["max"] == 12.0
    assert result["diagnostic_checks_passed"]
    assert not result["passed"]
    assert result["diagnostic_only"]


def test_reviewed_temporal_intervals_enforce_outage_recovery_latency_and_absence(
    tmp_path,
):
    video_path = tmp_path / "source.mp4"
    masks_path = tmp_path / "masks"
    masks_path.mkdir()
    states_path = tmp_path / "states.jsonl"
    frames: list[np.ndarray] = []
    # Candidate is deliberately empty for two ticks inside each visible
    # interval, emits once during a reviewed-absent interval, and has one
    # >50 ms compute sample.  Proxy visibility is irrelevant to these explicit
    # human-reviewed temporal gates.
    nonempty_frames = {1, 4, 5, 9}
    state_lines = []
    for frame_id in range(12):
        frame = np.full((48, 64, 3), 20, dtype=np.uint8)
        cv2.circle(frame, (28, 28), 7, (20, 220, 20), -1)
        frames.append(frame)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if frame_id in nonempty_frames:
            cv2.circle(mask, (28, 28), 7, 255, -1)
        assert cv2.imwrite(str(masks_path / f"{frame_id:06d}.png"), mask)
        state_lines.append(
            json.dumps(
                {
                    "frame_index": frame_id,
                    "compute_processing_ms": 55.0 if frame_id == 4 else 20.0,
                    "latency_evaluable": True,
                }
            )
        )
    _write_video(video_path, frames, fps=20.0)
    states_path.write_text("\n".join(state_lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "cases": [
            {
                "name": "reviewed_temporal",
                "video": str(video_path),
                "evaluation_fps_hz": 20.0,
                "candidate": {
                    "name": "candidate",
                    "kind": "directory",
                    "path": str(masks_path),
                    "states": str(states_path),
                    "states_latency_field": "compute_processing_ms",
                },
                "target_proxy": {
                    "hsv_ranges": [[[35, 80, 40], [95, 255, 255]]],
                    "initial_seed_xy": [28, 28],
                    "min_area_px": 20,
                    "min_visible_area_px": 20,
                    "open_kernel": 1,
                    "close_kernel": 1,
                },
                "reviewed_visible_source_intervals": [[1, 4], [7, 9]],
                "reviewed_reappearance_source_frames": [7],
                "reviewed_absent_source_intervals": [[0, 0], [5, 6], [10, 11]],
                "thresholds": {
                    "reviewed_visible_longest_invalid_run_max": 1,
                    "reviewed_reappearance_ticks_max": 3,
                    "latency_over_50ms_fraction_max": 0.01,
                    "latency_max_ms_max": 50.0,
                    "latency_over_50ms_longest_run_max": 0,
                    "reviewed_absent_false_positive_max": 0.0,
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = evaluate_manifest(manifest_path, tmp_path / "output")

    case = result["cases"][0]
    assert case["reviewed_visible_longest_invalid_run"]["value"] == 2
    assert case["reviewed_reappearance_ticks"]["max"] == 2
    assert case["reviewed_reappearance_ticks"]["resolved_event_count"] == 1
    assert case["reviewed_absent_false_positive"] == {
        "value": 0.2,
        "absent_frames": 5,
        "nonempty_frames": 1,
        "intervals": [[0, 0], [5, 6], [10, 11]],
    }
    assert case["latency_over_50ms_fraction"]["misses"] == 1
    assert abs(case["latency_over_50ms_fraction"]["value"] - 1.0 / 12.0) < 1e-12
    assert case["latency_over_50ms_longest_run"] == {
        "value": 1,
        "samples": 12,
        "deadline_ms": 50.0,
    }
    checks = {check["threshold"]: check for check in case["checks"]}
    assert not checks["reviewed_visible_longest_invalid_run_max"]["passed"]
    assert checks["reviewed_reappearance_ticks_max"]["passed"]
    assert not checks["latency_over_50ms_fraction_max"]["passed"]
    assert not checks["latency_max_ms_max"]["passed"]
    assert not checks["latency_over_50ms_longest_run_max"]["passed"]
    assert not checks["reviewed_absent_false_positive_max"]["passed"]
    assert not result["passed"]
    report = (tmp_path / "output" / "REPORT.md").read_text(encoding="utf-8")
    assert "Human-reviewed temporal gates" in report
    assert "reviewed_temporal" in report


def test_reappearance_requires_empty_preceding_reviewed_absence(tmp_path):
    video_path = tmp_path / "source.mp4"
    masks_path = tmp_path / "masks"
    masks_path.mkdir()
    frames = []
    for frame_id in range(4):
        frame = np.full((32, 48, 3), 20, dtype=np.uint8)
        cv2.circle(frame, (20, 18), 5, (20, 220, 20), -1)
        frames.append(frame)
        mask = np.zeros((32, 48), dtype=np.uint8)
        # A persistent hallucination remains non-empty through the explicitly
        # reviewed-absent frame immediately before reappearance.
        cv2.circle(mask, (20, 18), 5, 255, -1)
        assert cv2.imwrite(str(masks_path / f"{frame_id:06d}.png"), mask)
    _write_video(video_path, frames, fps=20.0)
    manifest = {
        "schema_version": 1,
        "cases": [
            {
                "name": "persistent_hallucination",
                "video": str(video_path),
                "evaluation_fps_hz": 20.0,
                "candidate": {
                    "name": "candidate",
                    "kind": "directory",
                    "path": str(masks_path),
                },
                "target_proxy": {
                    "hsv_ranges": [[[35, 80, 40], [95, 255, 255]]],
                    "initial_seed_xy": [20, 18],
                    "min_area_px": 20,
                    "min_visible_area_px": 20,
                    "open_kernel": 1,
                    "close_kernel": 1,
                },
                "reviewed_absent_source_intervals": [[1, 1]],
                "reviewed_visible_source_intervals": [[2, 3]],
                "reviewed_reappearance_source_frames": [2],
                "thresholds": {"reviewed_reappearance_ticks_max": 1},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = evaluate_manifest(manifest_path, tmp_path / "output")

    metric = result["cases"][0]["reviewed_reappearance_ticks"]
    assert metric["max"] is None
    assert metric["resolved_event_count"] == 0
    assert metric["events"][0]["preceding_reviewed_absent_was_empty"] is False
    assert not result["passed"]
