import json
from types import SimpleNamespace

import numpy as np
import pytest

from dynamic_pcd.apps.prompt_segmentation_service import (
    PromptSegmentationService,
    build_arg_parser,
    resolve_service_device,
)
from dynamic_pcd.segmentation.grounded_sam import (
    GroundedSAMBackend,
    PromptModelDependencyError,
    TextDetection,
    bbox_iou,
    non_maximum_suppression,
    rank_detections,
)
from dynamic_pcd.segmentation.prompt_client import ZMQPromptSegmentationClient
from dynamic_pcd.segmentation.prompt_protocol import (
    PromptSegmentationResult,
    decode_json,
    decode_segment_request,
    make_result_parts,
    make_segment_request,
)


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections
        self.prompts = []
        self.load_count = 0

    def load(self):
        self.load_count += 1

    def detect(self, image_rgb, prompt, box_threshold, text_threshold):
        self.prompts.append(prompt)
        return self.detections


class FakeMaskPredictor:
    def __init__(self, shape_hw, scores=None):
        self.shape_hw = shape_hw
        self.scores = list(scores or [0.95])
        self.image_count = 0
        self.boxes = []
        self.load_count = 0

    def load(self):
        self.load_count += 1

    def set_image(self, image_rgb):
        self.image_count += 1

    def predict(self, bbox_xyxy):
        self.boxes.append(np.asarray(bbox_xyxy).copy())
        h, w = self.shape_hw
        x1, y1, x2, y2 = np.rint(bbox_xyxy).astype(int)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 1
        score = self.scores[min(len(self.boxes) - 1, len(self.scores) - 1)]
        return mask[None], np.array([score], dtype=np.float32)


def _fake_torch(cuda_available, device_count=1):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            device_count=lambda: device_count,
        )
    )


def test_prompt_service_device_auto_selects_cuda_or_cpu():
    assert (
        resolve_service_device(
            "auto", torch_module=_fake_torch(cuda_available=True)
        )
        == "cuda"
    )
    assert (
        resolve_service_device(
            "auto", torch_module=_fake_torch(cuda_available=False)
        )
        == "cpu"
    )


def test_prompt_service_explicit_unavailable_cuda_fails_clearly():
    with pytest.raises(PromptModelDependencyError, match="explicit device.*CUDA"):
        resolve_service_device(
            "cuda", torch_module=_fake_torch(cuda_available=False)
        )


def test_prompt_service_validates_cuda_index_and_defaults_to_auto():
    assert build_arg_parser().parse_args([]).device == "auto"
    assert (
        resolve_service_device(
            "CUDA:1",
            torch_module=_fake_torch(cuda_available=True, device_count=2),
        )
        == "cuda:1"
    )
    with pytest.raises(PromptModelDependencyError, match="does not exist"):
        resolve_service_device(
            "cuda:2",
            torch_module=_fake_torch(cuda_available=True, device_count=2),
        )


def test_prompt_wire_round_trip_keeps_request_and_frame_metadata():
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    parts = make_segment_request(
        image_bgr=image,
        prompt="striped ceramic mug",
        request_id="req-17",
        box_threshold=0.31,
        text_threshold=0.22,
        mask_threshold=0.51,
        reference_bbox_xyxy=[1, 0, 4, 3],
        top_k=7,
        frame_id=123,
        frame_timestamp=42.25,
        frame_metadata={"camera": "left", "generation": 8},
    )
    header, decoded_image = decode_segment_request(parts)
    assert np.array_equal(decoded_image, image)
    assert header["prompt"] == "striped ceramic mug"
    assert header["request_id"] == "req-17"
    assert header["frame_id"] == 123
    assert header["frame_timestamp"] == 42.25
    assert header["frame_metadata"]["generation"] == 8

    result = PromptSegmentationResult(
        mask=np.ones((4, 5), dtype=np.uint8),
        bbox_xyxy=np.array([0, 0, 5, 4]),
        score=0.8,
        valid=True,
        prompt=header["prompt"],
        request_id=header["request_id"],
        frame_id=header["frame_id"],
        frame_timestamp=header["frame_timestamp"],
        frame_metadata=header["frame_metadata"],
    )
    response_header, response_mask = make_result_parts(result)
    decoded_header = decode_json(response_header)
    decoded_result = PromptSegmentationResult.from_wire(
        decoded_header, response_mask
    )
    assert decoded_result.request_id == "req-17"
    assert decoded_result.frame_id == 123
    assert decoded_result.frame_metadata == {"camera": "left", "generation": 8}
    assert np.array_equal(decoded_result.mask, result.mask)


def test_previous_bbox_ranks_matching_candidate_above_higher_text_score():
    detections = [
        TextDetection(np.array([5, 5, 25, 25]), 0.92, "object", index=0),
        TextDetection(np.array([70, 60, 90, 90]), 0.72, "object", index=1),
    ]
    ranked = rank_detections(
        detections,
        image_shape_hw=(100, 100),
        reference_bbox_xyxy=[69, 59, 91, 91],
        reference_iou_weight=1.0,
        reference_center_weight=0.25,
    )
    assert ranked[0].detector_index == 1
    assert ranked[0].reference_iou > 0.8
    assert ranked[0].reference_center_distance < ranked[1].reference_center_distance


def test_nms_keeps_distinct_candidates_and_suppresses_duplicate():
    detections = [
        TextDetection(np.array([10, 10, 40, 40]), 0.9, "item", index=0),
        TextDetection(np.array([11, 11, 41, 41]), 0.8, "item", index=1),
        TextDetection(np.array([60, 10, 90, 40]), 0.7, "item", index=2),
    ]
    kept = non_maximum_suppression(detections, iou_threshold=0.7)
    assert [item.index for item in kept] == [0, 2]
    assert bbox_iou(detections[0].bbox_xyxy, detections[1].bbox_xyxy) > 0.8


def test_grounded_sam_backend_is_prompt_generic_and_returns_candidates():
    detector = FakeDetector(
        [
            TextDetection(np.array([5, 5, 25, 30]), 0.91, "first", index=0),
            TextDetection(np.array([45, 10, 70, 35]), 0.78, "second", index=1),
        ]
    )
    masker = FakeMaskPredictor((50, 80))
    backend = GroundedSAMBackend(
        detector=detector,
        mask_predictor=masker,
        min_mask_area=10,
        max_sam_candidates=2,
    )
    image = np.zeros((50, 80, 3), dtype=np.uint8)
    result = backend.segment(
        image,
        prompt="translucent container with blue lid",
        reference_bbox_xyxy=[44, 9, 71, 36],
        request_id="async-4",
        frame_id=404,
        frame_timestamp=12.5,
        frame_metadata={"capture": "recovery"},
    )
    assert result.valid
    assert detector.prompts == ["translucent container with blue lid"]
    assert len(result.candidates) == 2
    assert result.candidates[0].detector_index == 1
    assert result.selected_index == 0
    assert result.request_id == "async-4"
    assert result.frame_id == 404
    assert result.frame_metadata == {"capture": "recovery"}
    assert masker.image_count == 1
    assert np.array_equal(result.bbox_xyxy, np.array([45, 10, 70, 35]))


def test_service_handler_returns_result_without_loading_real_models():
    detector = FakeDetector(
        [TextDetection(np.array([2, 3, 10, 12]), 0.9, "thing", index=0)]
    )
    masker = FakeMaskPredictor((16, 20))
    backend = GroundedSAMBackend(detector=detector, mask_predictor=masker)
    service = PromptSegmentationService(backend, model_load_ms=1234.0)
    image = np.zeros((16, 20, 3), dtype=np.uint8)
    request = make_segment_request(
        image,
        prompt="arbitrary user description",
        request_id="rpc-1",
        box_threshold=0.3,
        text_threshold=0.25,
        mask_threshold=0.5,
        reference_bbox_xyxy=None,
        top_k=3,
        frame_id=99,
    )
    response = service.handle_request(request)
    header = decode_json(response[0])
    result = PromptSegmentationResult.from_wire(header, response[1])
    assert header["status"] == "ok"
    assert result.valid
    assert result.request_id == "rpc-1"
    assert result.frame_id == 99
    assert result.timings_ms["service_inference"] >= 0.0

    health_request = (
        json.dumps(
            {"version": 1, "op": "health", "request_id": "health-1"}
        ).encode(),
    )
    health = decode_json(service.handle_request(health_request)[0])
    assert health["model_load_ms"] == 1234.0
    assert health["request_count"] == 1


def test_client_api_accepts_previous_bbox_and_caller_request_id():
    class FakeTransportClient(ZMQPromptSegmentationClient):
        def __init__(self):
            pass

        def _round_trip(self, parts, timeout_ms):
            header, image = decode_segment_request(parts)
            assert header["reference_bbox_xyxy"] == [1.0, 2.0, 8.0, 9.0]
            assert header["frame_id"] == 55
            result = PromptSegmentationResult(
                mask=np.ones(image.shape[:2], dtype=np.uint8),
                bbox_xyxy=np.array([1, 2, 8, 9]),
                score=0.9,
                valid=True,
                prompt=header["prompt"],
                request_id=header["request_id"],
                frame_id=header["frame_id"],
            )
            return make_result_parts(result)

    client = FakeTransportClient()
    result = client.segment(
        np.zeros((10, 12, 3), dtype=np.uint8),
        "any object prompt",
        previous_bbox_xyxy=[1, 2, 8, 9],
        request_id="caller-owned-id",
        frame_id=55,
    )
    assert result.request_id == "caller-owned-id"
    assert result.frame_id == 55
