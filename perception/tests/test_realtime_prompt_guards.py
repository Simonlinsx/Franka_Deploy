import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dynamic_pcd.apps.realtime_masked_pcd import (
    boundary_entry_bbox_touches_image_edge,
    prompt_service_launcher_args,
    prompt_service_launcher_path,
    prompt_response_echo_error,
    select_yolo_world_boundary_refinement_candidate,
    select_yolo_world_entry_candidate,
    select_yolo_world_startup_confirmation_candidate,
    validate_prompt_service_health,
    validate_prompting_config,
)
from dynamic_pcd.config import load_config
from dynamic_pcd.segmentation.prompt_protocol import (
    PromptCandidate,
    PromptSegmentationResult,
)
from dynamic_pcd.segmentation.prompt_runtime import PromptReacquireRequest


def _prompt_config():
    return {
        "service_addr": "tcp://127.0.0.1:5557",
        "service_autostart": True,
        "service_device": "auto",
        "mask_backend": "sam1",
        "startup_timeout_s": 45.0,
        "request_timeout_s": 35.0,
        "box_threshold": 0.25,
        "text_threshold": 0.20,
        "mask_threshold": 0.50,
        "top_k": 5,
        "auto_reacquire": True,
        "global_reacquire": True,
        "replay_buffer_s": 2.0,
        "replay_max_frames": 12,
        "relocate_stale_mask": False,
        "stale_mask_after_frames": 3,
        "stale_mask_min_score": 0.70,
        "stale_mask_min_score_margin": 0.04,
        "reacquire_lost_frames": 8,
        "reacquire_cooldown_s": 2.0,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("startup_timeout_s", float("nan")),
        ("request_timeout_s", 0.0),
        ("reacquire_cooldown_s", float("inf")),
        ("replay_buffer_s", 0.0),
        ("box_threshold", -0.1),
        ("text_threshold", 1.1),
        ("mask_threshold", float("nan")),
        ("top_k", 1.5),
        ("reacquire_lost_frames", 0),
        ("replay_max_frames", 1),
        ("auto_reacquire", "false"),
        ("service_device", "gpu"),
        ("mask_backend", "sam3"),
    ],
)
def test_prompting_config_rejects_invalid_numeric_and_boolean_values(name, value):
    config = _prompt_config()
    config[name] = value
    with pytest.raises(ValueError, match=name):
        validate_prompting_config(config)


def test_prompting_config_accepts_default_values():
    validate_prompting_config(_prompt_config())


def test_prompt_service_launcher_args_forward_normalized_model_options():
    config = _prompt_config()
    config["service_device"] = " CUDA:1 "
    config["mask_backend"] = "SAM2"
    assert prompt_service_launcher_args(config) == (
        "--device",
        "cuda:1",
        "--mask-backend",
        "sam2",
    )


def test_yolo_world_launcher_is_explicit_and_preloads_requested_text():
    config = _prompt_config()
    config.update(
        {
            "detector_backend": "yolo_world",
            "service_device": "cuda",
            "yolo_world_weights": "/tmp/yolov8s-worldv2.pt",
            "yolo_world_confidence": 0.03,
            "yolo_world_image_size": 640,
            "search_interval_s": 0.05,
        }
    )
    validate_prompting_config(config)
    assert prompt_service_launcher_path(config).name == (
        "run_yolo_world_prompt_service.sh"
    )
    assert prompt_service_launcher_args(config, prompt=" green ball ") == (
        "--device",
        "cuda",
        "--weights",
        "/tmp/yolov8s-worldv2.pt",
        "--confidence",
        "0.03",
        "--image-size",
        "640",
        "--preload-prompt",
        "green ball",
    )
    validate_prompt_service_health(
        config,
        {"detector_backend": "yolo_world", "mask_backend": "bbox_only"},
    )
    with pytest.raises(RuntimeError, match="backend mismatch"):
        validate_prompt_service_health(
            config,
            {"detector_backend": "native", "mask_backend": "sam1"},
        )


def test_yolo_world_accepts_calibrated_search_reference_roi():
    config = _prompt_config()
    config.update(
        {
            "detector_backend": "yolo_world",
            "service_device": "cuda",
            "yolo_world_weights": "/tmp/yolov8s-worldv2.pt",
            "yolo_world_confidence": 0.001,
            "yolo_world_image_size": 640,
            "search_interval_s": 0.05,
            "search_reference_roi_xyxy": [380, 95, 545, 405],
        }
    )
    validate_prompting_config(config)
    config["search_reference_roi_xyxy"] = [545, 95, 380, 405]
    with pytest.raises(ValueError, match="search_reference_roi_xyxy"):
        validate_prompting_config(config)


def test_deployed_text_search_has_no_fixed_reference_roi():
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "d435_default.yaml"
    )
    config = load_config(str(config_path))
    assert config["prompting"]["search_reference_roi_xyxy"] is None


def _bbox_only_result(candidates):
    height, width = 240, 424
    first = candidates[0] if candidates else None
    mask = np.zeros((height, width), dtype=np.uint8)
    if first is not None:
        x1, y1, x2, y2 = np.asarray(first.bbox_xyxy, dtype=np.int32)
        mask[y1:y2, x1:x2] = 1
    return PromptSegmentationResult(
        mask=mask,
        bbox_xyxy=(
            None
            if first is None
            else np.asarray(first.bbox_xyxy, dtype=np.int32)
        ),
        score=0.0 if first is None else first.detector_score,
        valid=first is not None,
        prompt="small patterned beanbag toy",
        selected_index=None if first is None else first.detector_index,
        candidates=list(candidates),
        frame_metadata={"prompt_output_kind": "bbox_only"},
    )


def _candidate(box, score, index):
    return PromptCandidate(
        bbox_xyxy=np.asarray(box, dtype=np.float32),
        detector_score=float(score),
        label="small patterned beanbag toy",
        rank_score=float(score),
        detector_index=int(index),
    )


def test_yolo_entry_filter_prefers_moving_compact_candidate_over_static_large_top():
    previous = np.zeros((240, 424), dtype=np.uint8)
    image = np.zeros((240, 424, 3), dtype=np.uint8)
    image[35:55, 65:85] = 220
    large = _candidate([300, 10, 423, 239], 0.40, 0)
    moving = _candidate([65, 35, 85, 55], 0.004, 1)
    selected, reason = select_yolo_world_entry_candidate(
        _bbox_only_result([large, moving]), image, previous
    )
    assert selected is moving
    assert "motion=1.000" in reason


def test_yolo_entry_filter_rejects_weak_static_compact_false_positive():
    previous = np.zeros((240, 424), dtype=np.uint8)
    image = np.zeros((240, 424, 3), dtype=np.uint8)
    weak = _candidate([65, 35, 85, 55], 0.004, 1)
    selected, reason = select_yolo_world_entry_candidate(
        _bbox_only_result([weak]), image, previous
    )
    assert selected is None
    assert "weak_static" in reason


def test_yolo_entry_filter_allows_strong_static_compact_semantics():
    image = np.zeros((240, 424, 3), dtype=np.uint8)
    strong = _candidate([65, 35, 85, 55], 0.20, 1)
    selected, reason = select_yolo_world_entry_candidate(
        _bbox_only_result([strong]), image, None
    )
    assert selected is strong
    assert "det=0.2000" in reason


def _boundary_refinement_config():
    return {
        "entry_candidate_filter_enabled": True,
        "entry_candidate_min_extent_ratio": 0.03,
        "entry_candidate_max_extent_ratio": 0.15,
        "entry_candidate_max_aspect_ratio": 2.5,
        "entry_candidate_motion_difference": 10,
        "entry_candidate_min_motion_fraction": 0.15,
        "entry_candidate_strong_static_score": 0.05,
        "boundary_entry_refinement_enabled": True,
        "boundary_entry_refinement_max_ticks": 2,
        "boundary_entry_refinement_max_center_step_ratio": 0.25,
        "boundary_entry_refinement_min_area_ratio": 0.35,
        "boundary_entry_refinement_max_area_ratio": 4.0,
        "boundary_entry_refinement_min_inward_step_px": 1.0,
        "boundary_entry_refinement_min_area_growth_ratio": 1.05,
        "startup_semantic_confirmation_enabled": True,
        "startup_semantic_confirmation_min_elapsed_s": 0.0,
        "startup_semantic_confirmation_max_ticks": 10,
        "startup_semantic_confirmation_search_every_ticks": 2,
        "startup_semantic_confirmation_min_detector_score": 0.10,
        "startup_semantic_confirmation_max_center_speed_px_s": 600.0,
        "startup_semantic_confirmation_min_area_ratio": 0.35,
        "startup_semantic_confirmation_max_area_ratio": 4.0,
        "startup_semantic_confirmation_interior_margin_px": 1,
    }


def test_boundary_entry_refinement_accepts_continuous_bottom_entry():
    previous = np.zeros((240, 424), dtype=np.uint8)
    image = np.zeros((240, 424, 3), dtype=np.uint8)
    image[220:240, 253:272] = 220
    candidate = _candidate([254, 221, 271, 240], 0.0015, 0)
    decision, reason = select_yolo_world_boundary_refinement_candidate(
        _bbox_only_result([candidate]),
        image,
        previous,
        [272, 227, 289, 240],
        _boundary_refinement_config(),
    )
    assert decision is not None
    assert decision.candidate is candidate
    assert decision.bbox_xyxy.tolist() == [254, 221, 271, 240]
    assert "same boundary entry" in reason


def test_boundary_entry_refinement_requires_boundary_seed():
    image = np.full((240, 424, 3), 255, dtype=np.uint8)
    candidate = _candidate([60, 65, 80, 85], 0.2, 0)
    decision, reason = select_yolo_world_boundary_refinement_candidate(
        _bbox_only_result([candidate]),
        image,
        None,
        [55, 60, 75, 80],
        _boundary_refinement_config(),
    )
    assert decision is None
    assert "not image-boundary clipped" in reason


def test_boundary_entry_refinement_rejects_remote_compact_distractor():
    previous = np.zeros((240, 424), dtype=np.uint8)
    image = np.zeros((240, 424, 3), dtype=np.uint8)
    image[30:50, 300:320] = 220
    candidate = _candidate([300, 30, 320, 50], 0.2, 0)
    decision, reason = select_yolo_world_boundary_refinement_candidate(
        _bbox_only_result([candidate]),
        image,
        previous,
        [55, 226, 77, 240],
        _boundary_refinement_config(),
    )
    assert decision is None
    assert "center jump" in reason


def test_boundary_entry_refinement_rejects_no_inward_or_growth_progress():
    previous = np.zeros((240, 424), dtype=np.uint8)
    image = np.zeros((240, 424, 3), dtype=np.uint8)
    image[229:240, 56:75] = 220
    candidate = _candidate([56, 229, 75, 240], 0.2, 0)
    decision, reason = select_yolo_world_boundary_refinement_candidate(
        _bbox_only_result([candidate]),
        image,
        previous,
        [55, 226, 77, 240],
        _boundary_refinement_config(),
    )
    assert decision is None
    assert "neither inward nor area progress" in reason


def test_boundary_entry_edge_detection_is_exact_and_clipped():
    shape = (240, 424, 3)
    assert boundary_entry_bbox_touches_image_edge([10, 220, 30, 240], shape)
    assert boundary_entry_bbox_touches_image_edge([-2, 20, 10, 40], shape)
    assert not boundary_entry_bbox_touches_image_edge([10, 20, 30, 40], shape)


def test_startup_semantic_confirmation_accepts_strong_interior_same_entry():
    previous = np.zeros((240, 424), dtype=np.uint8)
    image = np.zeros((240, 424, 3), dtype=np.uint8)
    image[207:228, 176:191] = 220
    candidate = _candidate([176, 207, 191, 228], 0.1157, 0)
    decision, reason = select_yolo_world_startup_confirmation_candidate(
        _bbox_only_result([candidate]),
        image,
        previous,
        [272, 227, 289, 240],
        1.00,
        _boundary_refinement_config(),
    )
    assert decision is not None
    assert decision.candidate is candidate
    assert decision.bbox_xyxy.tolist() == [176, 207, 191, 228]
    assert "strong interior startup confirmation" in reason


def test_startup_semantic_confirmation_rejects_weak_interior_candidate():
    previous = np.zeros((240, 424), dtype=np.uint8)
    image = np.zeros((240, 424, 3), dtype=np.uint8)
    image[207:228, 176:191] = 220
    decision, reason = select_yolo_world_startup_confirmation_candidate(
        _bbox_only_result([_candidate([176, 207, 191, 228], 0.099, 0)]),
        image,
        previous,
        [272, 227, 289, 240],
        1.00,
        _boundary_refinement_config(),
    )
    assert decision is None
    assert "score below limit" in reason


def test_startup_semantic_confirmation_rejects_early_candidate():
    previous = np.zeros((240, 424), dtype=np.uint8)
    image = np.zeros((240, 424, 3), dtype=np.uint8)
    image[207:228, 176:191] = 220
    config = _boundary_refinement_config()
    config["startup_semantic_confirmation_min_elapsed_s"] = 1.0
    decision, reason = select_yolo_world_startup_confirmation_candidate(
        _bbox_only_result([_candidate([176, 207, 191, 228], 0.2, 0)]),
        image,
        previous,
        [272, 227, 289, 240],
        0.95,
        config,
    )
    assert decision is None
    assert "before the stable confirmation window" in reason


def test_startup_semantic_confirmation_rejects_nonboundary_seed():
    image = np.full((240, 424, 3), 255, dtype=np.uint8)
    decision, reason = select_yolo_world_startup_confirmation_candidate(
        _bbox_only_result([_candidate([60, 65, 80, 85], 0.2, 0)]),
        image,
        None,
        [55, 60, 75, 80],
        1.00,
        _boundary_refinement_config(),
    )
    assert decision is None
    assert "requires an image-boundary entry" in reason


def test_startup_semantic_confirmation_rejects_boundary_candidate():
    previous = np.zeros((240, 424), dtype=np.uint8)
    image = np.zeros((240, 424, 3), dtype=np.uint8)
    image[220:240, 250:270] = 220
    decision, reason = select_yolo_world_startup_confirmation_candidate(
        _bbox_only_result([_candidate([250, 220, 270, 240], 0.2, 0)]),
        image,
        previous,
        [272, 227, 289, 240],
        1.00,
        _boundary_refinement_config(),
    )
    assert decision is None
    assert "not interior" in reason


def test_startup_semantic_confirmation_rejects_remote_fast_distractor():
    previous = np.zeros((240, 424), dtype=np.uint8)
    image = np.zeros((240, 424, 3), dtype=np.uint8)
    image[20:40, 20:40] = 220
    config = _boundary_refinement_config()
    config["startup_semantic_confirmation_max_center_speed_px_s"] = 100.0
    decision, reason = select_yolo_world_startup_confirmation_candidate(
        _bbox_only_result([_candidate([20, 20, 40, 40], 0.2, 0)]),
        image,
        previous,
        [272, 227, 289, 240],
        1.00,
        config,
    )
    assert decision is None
    assert "center speed exceeds limit" in reason


@pytest.mark.parametrize(
    "box",
    (
        [20, 20, 150, 90],
        [20, 20, 100, 25],
        [20, 20, 22, 40],
    ),
)
def test_yolo_entry_filter_rejects_noncompact_geometry(box):
    image = np.full((240, 424, 3), 255, dtype=np.uint8)
    selected, _ = select_yolo_world_entry_candidate(
        _bbox_only_result([_candidate(box, 0.9, 0)]),
        image,
        np.zeros((240, 424), dtype=np.uint8),
    )
    assert selected is None


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "CUDA:0"])
def test_prompting_config_accepts_supported_service_devices(device):
    config = _prompt_config()
    config["service_device"] = device
    validate_prompting_config(config)


def test_prompting_config_requires_mapping():
    with pytest.raises(ValueError, match="mapping"):
        validate_prompting_config(5)


def _request_and_result():
    request = PromptReacquireRequest(
        image_bgr=np.zeros((10, 12, 3), dtype=np.uint8),
        prompt="pink cylinder",
        reference_bbox_xyxy=np.asarray([1, 2, 8, 9], dtype=np.float32),
        frame_id=17,
        frame_timestamp=12.25,
        generation=4,
        reason="automatic_lost",
        request_id="prompt-g4-f17-test",
    )
    result = PromptSegmentationResult(
        mask=np.ones((10, 12), dtype=np.uint8),
        bbox_xyxy=np.asarray([1, 2, 8, 9], dtype=np.int32),
        score=0.9,
        valid=True,
        prompt=request.prompt,
        request_id=request.request_id,
        frame_id=request.frame_id,
        frame_timestamp=request.frame_timestamp,
        frame_metadata={
            "tracker_generation": request.generation,
            "reason": request.reason,
        },
    )
    return request, result


def test_prompt_response_echo_guard_accepts_exact_source_metadata():
    request, result = _request_and_result()
    assert prompt_response_echo_error(request, result, request.prompt) is None


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("request_id", "another-request", "request_id"),
        ("prompt", "blue mug", "prompt"),
        ("frame_id", 18, "frame_id"),
        ("frame_timestamp", 12.5, "frame_timestamp"),
    ],
)
def test_prompt_response_echo_guard_rejects_wrong_source_fields(
    field, value, expected
):
    request, result = _request_and_result()
    setattr(result, field, value)
    assert expected in prompt_response_echo_error(request, result, request.prompt)


def test_prompt_response_echo_guard_rejects_wrong_generation_and_reason():
    request, result = _request_and_result()
    result.frame_metadata["tracker_generation"] = 5
    assert "tracker_generation" in prompt_response_echo_error(
        request, result, request.prompt
    )

    request, result = _request_and_result()
    result.frame_metadata["reason"] = "manual_key"
    assert "reason" in prompt_response_echo_error(request, result, request.prompt)


def test_prompt_cli_rejects_incompatible_explicit_tracker_mode():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dynamic_pcd.apps.realtime_masked_pcd",
            "--prompt",
            "pink cylinder",
            "--mode",
            "roi_depth",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--prompt requires --mode adaptive_color_depth" in completed.stderr
