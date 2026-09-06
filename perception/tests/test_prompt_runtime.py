import threading
import time

import numpy as np
import pytest

from dynamic_pcd.segmentation.prompt_protocol import PromptSegmentationResult
from dynamic_pcd.segmentation.prompt_runtime import (
    AsyncPromptReacquirer,
    PromptRetrySchedule,
    relocate_prompt_mask_to_current_frame,
)


def _result(request_id, frame_id, frame_timestamp, metadata, shape):
    return PromptSegmentationResult(
        mask=np.ones(shape, dtype=np.uint8),
        bbox_xyxy=np.asarray([1, 2, 8, 9], dtype=np.int32),
        score=0.8,
        valid=True,
        prompt="arbitrary object",
        request_id=request_id,
        frame_id=frame_id,
        frame_timestamp=frame_timestamp,
        frame_metadata=metadata,
    )


def test_stale_prompt_mask_is_relocated_to_current_object_position():
    source = np.zeros((80, 140, 3), dtype=np.uint8)
    current = np.zeros_like(source)
    source_bbox = (12, 24, 36, 58)
    current_bbox = (96, 30, 120, 64)
    color = (210, 80, 235)
    source[24:58, 12:36] = color
    current[30:64, 96:120] = color
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    mask[24:58, 12:36] = 1

    relocated = relocate_prompt_mask_to_current_frame(
        source,
        current,
        mask,
        min_score=0.70,
        min_score_margin=0.04,
    )

    assert relocated is not None
    np.testing.assert_array_equal(relocated.bbox_xyxy, current_bbox)
    assert relocated.mask[30:64, 96:120].all()
    assert int(relocated.mask.sum()) == 24 * 34


def test_stale_prompt_mask_relocation_rejects_two_equal_category_instances():
    source = np.zeros((80, 160, 3), dtype=np.uint8)
    current = np.zeros_like(source)
    color = (230, 100, 25)
    source[22:56, 12:36] = color
    current[22:56, 64:88] = color
    current[22:56, 116:140] = color
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    mask[22:56, 12:36] = 1

    relocated = relocate_prompt_mask_to_current_frame(
        source,
        current,
        mask,
        min_score=0.70,
        min_score_margin=0.04,
    )

    assert relocated is None


def test_async_prompt_worker_preserves_generation_and_never_blocks_submitter():
    entered = threading.Event()
    release = threading.Event()

    class FakeClient:
        def segment(self, image, **kwargs):
            entered.set()
            assert release.wait(timeout=1.0)
            return _result(
                kwargs["request_id"],
                kwargs["frame_id"],
                kwargs["frame_timestamp"],
                kwargs["frame_metadata"],
                image.shape[:2],
            )

        def close(self):
            pass

    worker = AsyncPromptReacquirer(
        "inproc://fake", client_factory=lambda: FakeClient()
    )
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    started = time.perf_counter()
    request_id = worker.submit(
        image,
        prompt="arbitrary object",
        reference_bbox_xyxy=[1, 2, 8, 9],
        frame_id=17,
        frame_timestamp=1.25,
        generation=4,
        reason="lost",
    )
    assert time.perf_counter() - started < 0.05
    assert request_id is not None
    assert entered.wait(timeout=1.0)
    assert worker.busy
    assert (
        worker.submit(
            image,
            "another prompt",
            None,
            18,
            1.30,
            5,
            "duplicate",
        )
        is None
    )

    release.set()
    response = None
    deadline = time.monotonic() + 1.0
    while response is None and time.monotonic() < deadline:
        response = worker.poll()
        time.sleep(0.005)
    worker.close()

    assert response is not None
    assert response.error is None
    assert response.request.generation == 4
    assert response.result.request_id == request_id
    assert response.result.frame_metadata["tracker_generation"] == 4
    assert response.result.frame_metadata["reason"] == "lost"


def test_async_prompt_worker_surfaces_model_error_without_dying():
    class FailingClient:
        def segment(self, _image, **_kwargs):
            raise RuntimeError("model failed")

        def close(self):
            pass

    worker = AsyncPromptReacquirer(
        "inproc://fake", client_factory=lambda: FailingClient()
    )
    request_id = worker.submit(
        np.zeros((10, 12, 3), dtype=np.uint8),
        "mug",
        None,
        1,
        0.1,
        1,
        "initial",
    )
    assert request_id is not None

    response = None
    deadline = time.monotonic() + 1.0
    while response is None and time.monotonic() < deadline:
        response = worker.poll()
        time.sleep(0.005)
    worker.close()

    assert response is not None
    assert response.result is None
    assert isinstance(response.error, RuntimeError)
    assert "model failed" in str(response.error)


def test_unpolled_response_keeps_single_request_slot_busy():
    class FakeClient:
        def segment(self, image, **kwargs):
            return _result(
                kwargs["request_id"],
                kwargs["frame_id"],
                kwargs["frame_timestamp"],
                kwargs["frame_metadata"],
                image.shape[:2],
            )

        def close(self):
            pass

    worker = AsyncPromptReacquirer(
        "inproc://fake", client_factory=lambda: FakeClient()
    )
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    first_id = worker.submit(image, "first", None, 1, 0.1, 1, "lost")
    assert first_id is not None

    deadline = time.monotonic() + 1.0
    completed = False
    while time.monotonic() < deadline:
        with worker._condition:
            completed = worker._response is not None and not worker._busy
        if completed:
            break
        time.sleep(0.002)
    assert completed
    assert worker.busy
    assert worker.submit(image, "second", None, 2, 0.2, 1, "lost") is None

    response = worker.poll()
    assert response is not None
    assert response.request.request_id == first_id
    assert not worker.busy
    assert worker.close()


def test_close_is_bounded_and_can_finish_after_blocking_rpc_releases():
    entered = threading.Event()
    release = threading.Event()

    class BlockingClient:
        def segment(self, _image, **_kwargs):
            entered.set()
            release.wait(timeout=2.0)
            return None

        def close(self):
            pass

    worker = AsyncPromptReacquirer(
        "inproc://fake", client_factory=lambda: BlockingClient()
    )
    assert worker.submit(
        np.zeros((10, 12, 3), dtype=np.uint8),
        "mug",
        None,
        1,
        0.1,
        1,
        "lost",
    )
    assert entered.wait(timeout=1.0)

    started = time.perf_counter()
    assert not worker.close(join_timeout_s=0.02)
    assert time.perf_counter() - started < 0.20

    release.set()
    assert worker.close(join_timeout_s=1.0)


def test_slow_failure_can_retry_immediately_from_request_start_cooldown():
    retry = PromptRetrySchedule(cooldown_s=2.0)

    assert retry.ready(now=10.0)
    assert retry.mark_submitted(now=10.0) == 1
    retry.request_retry("no_valid_candidate")

    # The model response took 13 seconds.  We do not add another cooldown
    # after receiving it, so the camera loop may submit its newest frame now.
    assert retry.cooldown_remaining_s(now=23.0) == 0.0
    assert retry.ready(now=23.0)
    assert retry.retry_pending
    assert retry.pending_reason == "no_valid_candidate"
    assert retry.mark_submitted(now=23.0) == 2
    assert not retry.retry_pending


def test_fast_failures_are_throttled_without_resetting_pending_retry():
    retry = PromptRetrySchedule(cooldown_s=2.0)
    retry.mark_submitted(now=10.0)
    retry.request_retry("service_error")

    assert retry.cooldown_remaining_s(now=10.2) == pytest.approx(1.8)
    assert not retry.ready(now=10.2)
    assert retry.retry_pending
    assert not retry.ready(now=11.999)
    assert retry.ready(now=12.0)


def test_retry_success_clears_sequence_but_retains_start_rate_limit():
    retry = PromptRetrySchedule(cooldown_s=2.0)
    retry.mark_submitted(now=10.0)
    retry.request_retry("current_frame_gate_rejected")

    retry.clear()

    assert not retry.retry_pending
    assert retry.attempt == 0
    assert not retry.ready(now=10.5)
    assert retry.ready(now=12.0)


def test_retry_uses_latest_frame_and_worker_never_has_two_requests():
    calls = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    class RecordingClient:
        def segment(self, image, **kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                calls.append(
                    (
                        kwargs["frame_id"],
                        int(image[0, 0, 0]),
                        dict(kwargs["frame_metadata"]),
                    )
                )
                return _result(
                    kwargs["request_id"],
                    kwargs["frame_id"],
                    kwargs["frame_timestamp"],
                    kwargs["frame_metadata"],
                    image.shape[:2],
                )
            finally:
                with lock:
                    active -= 1

        def close(self):
            pass

    worker = AsyncPromptReacquirer(
        "inproc://fake", client_factory=lambda: RecordingClient()
    )
    retry = PromptRetrySchedule(cooldown_s=2.0)

    first = np.full((10, 12, 3), 17, dtype=np.uint8)
    assert worker.submit(first, "mug", None, 17, 1.7, 4, "automatic_lost")
    retry.mark_submitted(now=10.0)

    first_response = None
    deadline = time.monotonic() + 1.0
    while first_response is None and time.monotonic() < deadline:
        first_response = worker.poll()
        time.sleep(0.002)
    assert first_response is not None

    retry.request_retry("no_valid_candidate")
    assert retry.ready(now=23.0)
    newest = np.full((10, 12, 3), 99, dtype=np.uint8)
    assert worker.submit(
        newest,
        "mug",
        None,
        99,
        9.9,
        4,
        "automatic_retry",
    )
    retry.mark_submitted(now=23.0)

    second_response = None
    deadline = time.monotonic() + 1.0
    while second_response is None and time.monotonic() < deadline:
        second_response = worker.poll()
        time.sleep(0.002)
    assert second_response is not None
    assert worker.close()

    assert calls == [
        (17, 17, {"tracker_generation": 4, "reason": "automatic_lost"}),
        (99, 99, {"tracker_generation": 4, "reason": "automatic_retry"}),
    ]
    assert max_active == 1
