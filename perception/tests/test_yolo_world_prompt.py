from types import SimpleNamespace

import numpy as np

from dynamic_pcd.segmentation.yolo_world_prompt import YOLOWorldPromptBackend


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeModel:
    def __init__(self, boxes, scores):
        self.boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self.scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        self.classes = []
        self.calls = []

    def set_classes(self, classes):
        self.classes.append(tuple(classes))

    def predict(self, image, **kwargs):
        self.calls.append((image.copy(), dict(kwargs)))
        result = SimpleNamespace(
            boxes=SimpleNamespace(
                xyxy=_Tensor(self.boxes),
                conf=_Tensor(self.scores),
            )
        )
        return [result]


def test_yolo_world_result_is_explicit_bbox_only_not_policy_mask():
    model = _FakeModel([[8.2, 5.1, 19.8, 16.9]], [0.72])
    backend = YOLOWorldPromptBackend(
        weights="unused.pt",
        device="0",
        confidence=0.03,
        preload_prompt="green ball",
        model=model,
    )
    backend.load()
    result = backend.segment(
        image_bgr=np.zeros((24, 32, 3), dtype=np.uint8),
        prompt="green ball",
        box_threshold=0.03,
        text_threshold=0.20,
        mask_threshold=0.50,
        reference_bbox_xyxy=None,
        top_k=5,
        request_id="request-1",
        frame_id=7,
        frame_timestamp=1.25,
        frame_metadata={"tracker_generation": 0},
    )

    assert result.valid
    np.testing.assert_array_equal(result.bbox_xyxy, [8, 5, 20, 17])
    assert int(result.mask.sum()) == 12 * 12
    assert result.frame_metadata["prompt_output_kind"] == "bbox_only"
    assert result.frame_metadata["must_initialize_sam2_from_bbox"] is True
    assert model.classes == [("green ball",)]
    assert len(model.calls) == 1


def test_yolo_world_no_detection_is_fail_closed():
    backend = YOLOWorldPromptBackend(
        weights="unused.pt",
        model=_FakeModel([], []),
    )
    result = backend.segment(
        image_bgr=np.zeros((20, 30, 3), dtype=np.uint8),
        prompt="ball",
        box_threshold=0.03,
        text_threshold=0.20,
        mask_threshold=0.50,
        reference_bbox_xyxy=None,
        top_k=5,
        request_id="request-2",
        frame_id=8,
        frame_timestamp=1.30,
        frame_metadata={},
    )

    assert not result.valid
    assert result.bbox_xyxy is None
    assert int(result.mask.sum()) == 0


def test_explicit_color_and_reference_region_out_rank_raw_false_positive():
    image = np.zeros((100, 180, 3), dtype=np.uint8)
    image[35:70, 75:110] = (45, 55, 230)
    model = _FakeModel(
        [[135, 20, 175, 80], [72, 32, 114, 74]],
        [0.80, 0.01],
    )
    backend = YOLOWorldPromptBackend(weights="unused.pt", model=model)
    result = backend.segment(
        image_bgr=image,
        prompt="red cube",
        box_threshold=0.001,
        text_threshold=0.20,
        mask_threshold=0.50,
        reference_bbox_xyxy=[50, 10, 125, 90],
        top_k=5,
        request_id="request-color",
        frame_id=9,
        frame_timestamp=1.35,
        frame_metadata={},
    )

    np.testing.assert_array_equal(result.bbox_xyxy, [72, 32, 114, 74])
    assert result.candidates[0].detector_score < result.candidates[1].detector_score


def test_global_blue_ball_search_does_not_apply_removed_fixed_region_prior():
    """Reproduce the moved-ball deployment candidates from 2026-08-05.

    With no explicit instance hint, the genuine colour-consistent 0.587 ball
    must outrank the 0.003 background proposal.  The old deployed fixed ROI
    reversed these two candidates solely because their centres fell on
    opposite sides of an image-space boundary.
    """

    image = np.zeros((480, 848, 3), dtype=np.uint8)
    image[249:288, 324:363] = (230, 40, 35)
    model = _FakeModel(
        [[527, 120, 553, 143], [324, 249, 363, 288]],
        [0.003, 0.587],
    )
    backend = YOLOWorldPromptBackend(
        weights="unused.pt",
        confidence=0.001,
        model=model,
    )

    result = backend.segment(
        image_bgr=image,
        prompt="blue ball",
        box_threshold=0.001,
        text_threshold=0.20,
        mask_threshold=0.50,
        reference_bbox_xyxy=None,
        top_k=20,
        request_id="moved-blue-ball-global-search",
        frame_id=1,
        frame_timestamp=1.0,
        frame_metadata={},
    )

    np.testing.assert_array_equal(result.bbox_xyxy, [324, 249, 363, 288])
    assert np.isclose(result.candidates[0].detector_score, 0.587)
    assert result.candidates[0].reference_iou is None
