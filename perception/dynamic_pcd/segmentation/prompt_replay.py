from __future__ import annotations

import copy
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from dynamic_pcd.segmentation.adaptive_color_depth_tracker import (
    AdaptiveColorDepthTracker,
)
from dynamic_pcd.types import RGBDFrame


@dataclass(frozen=True)
class RGBDReplayWindow:
    """A frame-stamped, uniformly sampled source-to-head replay window."""

    frames: Tuple[RGBDFrame, ...]
    source_frame_id: int
    head_frame_id: int
    available_frame_count: int

    @property
    def sampled_frame_ids(self) -> Tuple[int, ...]:
        return tuple(int(frame.frame_id) for frame in self.frames)


@dataclass(frozen=True)
class PromptMaskReplayResult:
    """Fail-closed result of propagating one semantic source mask."""

    valid: bool
    mask: np.ndarray
    bbox_xyxy: Optional[np.ndarray]
    score: float
    source_frame_id: int
    target_frame_id: int
    sampled_frame_ids: Tuple[int, ...]
    elapsed_ms: float
    message: str
    failure_code: str = ""


class RecentRGBDFrameBuffer:
    """Bounded recent RGB-D frames keyed by strictly increasing frame IDs.

    Camera arrays are copied on insertion.  This is important for backends
    which recycle capture buffers after the next frame.  Time-based eviction
    provides the requested replay horizon, while ``capacity_frames`` is a hard
    memory bound if wall-clock timestamps stop advancing.
    """

    def __init__(self, retention_s: float = 2.0, capacity_frames: int = 64):
        retention = float(retention_s)
        if not np.isfinite(retention) or retention <= 0.0:
            raise ValueError("retention_s must be a finite positive number")
        capacity = int(capacity_frames)
        if isinstance(capacity_frames, bool) or capacity < 2:
            raise ValueError("capacity_frames must be an integer >= 2")
        self.retention_s = retention
        self.capacity_frames = capacity
        self._frames: Deque[RGBDFrame] = deque()

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def frame_ids(self) -> Tuple[int, ...]:
        return tuple(int(frame.frame_id) for frame in self._frames)

    @property
    def head(self) -> Optional[RGBDFrame]:
        return None if not self._frames else self._frames[-1]

    def get(self, frame_id: int) -> Optional[RGBDFrame]:
        wanted = int(frame_id)
        for frame in self._frames:
            if int(frame.frame_id) == wanted:
                return frame
        return None

    def append(self, frame: RGBDFrame) -> None:
        frame_id = int(frame.frame_id)
        timestamp = float(frame.timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("RGB-D frame timestamp must be finite")
        if self._frames and frame_id <= int(self._frames[-1].frame_id):
            raise ValueError(
                "RGB-D frame IDs must be strictly increasing: "
                f"received {frame_id} after {self._frames[-1].frame_id}"
            )
        self._frames.append(_copy_rgbd_frame(frame))
        while len(self._frames) > self.capacity_frames:
            self._frames.popleft()
        newest_timestamp = float(self._frames[-1].timestamp)
        while (
            len(self._frames) > 1
            and newest_timestamp - float(self._frames[0].timestamp)
            > self.retention_s
        ):
            self._frames.popleft()

    def replay_window(
        self, source_frame_id: int, max_frames: int = 12
    ) -> Optional[RGBDReplayWindow]:
        """Return source through current head, uniformly sampled and ordered.

        The exact source and exact current head are always retained.  ``None``
        means the semantic result's source was evicted or was never captured;
        callers must retry on a new frame rather than applying stale pixels.
        """

        maximum = int(max_frames)
        if isinstance(max_frames, bool) or maximum < 2:
            raise ValueError("max_frames must be an integer >= 2")
        source_id = int(source_frame_id)
        frames = tuple(self._frames)
        source_index = next(
            (
                index
                for index, frame in enumerate(frames)
                if int(frame.frame_id) == source_id
            ),
            None,
        )
        if source_index is None:
            return None
        available = frames[source_index:]
        if not available:
            return None
        selected = _uniformly_sample_frames(available, maximum)
        _validate_strict_frame_order(selected)
        return RGBDReplayWindow(
            frames=selected,
            source_frame_id=source_id,
            head_frame_id=int(available[-1].frame_id),
            available_frame_count=len(available),
        )


def replay_prompt_mask(
    window: RGBDReplayWindow,
    source_mask: np.ndarray,
    tracker_cfg: Mapping[str, object],
    *,
    tracker_factory: Callable[[Dict[str, object]], object] = (
        AdaptiveColorDepthTracker
    ),
) -> PromptMaskReplayResult:
    """Propagate a source-frame prompt mask to the exact replay-window head.

    A fresh generic tracker is initialized from semantic evidence on its own
    source RGB-D frame.  Every sampled update must be valid.  The first invalid
    update terminates replay and returns an all-zero head-shaped mask, even if a
    later buffered frame contains a plausible object again.  This prevents a
    complete occlusion inside the semantic request interval from being bridged
    by stale pixels or an unrelated recovery hypothesis.
    """

    started = time.perf_counter()
    frames = tuple(window.frames)
    if not frames:
        raise ValueError("replay window must contain at least one RGB-D frame")
    _validate_strict_frame_order(frames)
    if int(frames[0].frame_id) != int(window.source_frame_id):
        raise ValueError("replay window does not start at source_frame_id")
    if int(frames[-1].frame_id) != int(window.head_frame_id):
        raise ValueError("replay window does not end at head_frame_id")

    source = frames[0]
    head = frames[-1]
    mask_u8 = (np.asarray(source_mask) > 0).astype(np.uint8)
    if mask_u8.shape != source.depth_raw.shape or int(mask_u8.sum()) == 0:
        return _invalid_replay_result(
            window,
            head,
            started,
            failure_code="invalid_source_mask",
            message="source semantic mask is empty or has the wrong shape",
        )
    for frame in frames:
        if frame.depth_raw.shape != source.depth_raw.shape:
            return _invalid_replay_result(
                window,
                head,
                started,
                failure_code="frame_shape_changed",
                message=(
                    f"RGB-D shape changed at frame {frame.frame_id}: "
                    f"{frame.depth_raw.shape} != {source.depth_raw.shape}"
                ),
            )

    tracker = tracker_factory(dict(tracker_cfg))
    try:
        result = tracker.initialize(source, mask=mask_u8)
    except Exception as exc:
        return _invalid_replay_result(
            window,
            head,
            started,
            failure_code="tracker_initialize_error",
            message=f"temporary tracker initialization failed: {exc}",
        )
    if not bool(getattr(result, "valid", False)):
        return _invalid_replay_result(
            window,
            head,
            started,
            failure_code="tracker_initialize_invalid",
            message=(
                "temporary tracker rejected source mask: "
                f"{getattr(result, 'message', 'invalid')}"
            ),
        )

    for replay_frame in frames[1:]:
        try:
            result = tracker.update(replay_frame)
        except Exception as exc:
            return _invalid_replay_result(
                window,
                head,
                started,
                failure_code="tracker_update_error",
                message=(
                    f"temporary tracker failed at frame {replay_frame.frame_id}: "
                    f"{exc}"
                ),
            )
        if not bool(getattr(result, "valid", False)):
            return _invalid_replay_result(
                window,
                head,
                started,
                failure_code="tracker_update_invalid",
                message=(
                    f"temporary tracker became invalid at frame "
                    f"{replay_frame.frame_id}: "
                    f"{getattr(result, 'message', 'invalid')}"
                ),
            )

    result_mask = (np.asarray(result.mask) > 0).astype(np.uint8)
    if result_mask.shape != head.depth_raw.shape or int(result_mask.sum()) == 0:
        return _invalid_replay_result(
            window,
            head,
            started,
            failure_code="invalid_head_mask",
            message="temporary tracker returned an empty or wrong-shaped head mask",
        )
    bbox = np.asarray(result.bbox_xyxy, dtype=np.int32).reshape(4).copy()
    return PromptMaskReplayResult(
        valid=True,
        mask=result_mask,
        bbox_xyxy=bbox,
        score=float(getattr(result, "score", 0.0)),
        source_frame_id=int(window.source_frame_id),
        target_frame_id=int(window.head_frame_id),
        sampled_frame_ids=window.sampled_frame_ids,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        message=(
            f"temporary adaptive replay reached exact head frame "
            f"{window.head_frame_id} using {len(frames)}/"
            f"{window.available_frame_count} buffered frames"
        ),
    )


def _copy_rgbd_frame(frame: RGBDFrame) -> RGBDFrame:
    return RGBDFrame(
        color_bgr=np.ascontiguousarray(frame.color_bgr).copy(),
        depth_raw=np.ascontiguousarray(frame.depth_raw).copy(),
        depth_scale=float(frame.depth_scale),
        intrinsics=copy.deepcopy(frame.intrinsics),
        timestamp=float(frame.timestamp),
        frame_id=int(frame.frame_id),
    )


def _uniformly_sample_frames(
    frames: Sequence[RGBDFrame], max_frames: int
) -> Tuple[RGBDFrame, ...]:
    count = len(frames)
    if count <= max_frames:
        return tuple(frames)
    # count > max_frames >= 2 makes the rounded positions strictly increasing.
    positions = tuple(
        int(round(index * (count - 1) / float(max_frames - 1)))
        for index in range(max_frames)
    )
    if positions[0] != 0 or positions[-1] != count - 1:
        raise AssertionError("uniform sampling lost a replay endpoint")
    if any(right <= left for left, right in zip(positions, positions[1:])):
        raise AssertionError("uniform sampling produced duplicate positions")
    return tuple(frames[position] for position in positions)


def _validate_strict_frame_order(frames: Sequence[RGBDFrame]) -> None:
    frame_ids = tuple(int(frame.frame_id) for frame in frames)
    if any(right <= left for left, right in zip(frame_ids, frame_ids[1:])):
        raise ValueError(
            f"replay frame IDs must be strictly increasing, received {frame_ids}"
        )


def _invalid_replay_result(
    window: RGBDReplayWindow,
    head: RGBDFrame,
    started: float,
    *,
    failure_code: str,
    message: str,
) -> PromptMaskReplayResult:
    return PromptMaskReplayResult(
        valid=False,
        mask=np.zeros(head.depth_raw.shape, dtype=np.uint8),
        bbox_xyxy=None,
        score=0.0,
        source_frame_id=int(window.source_frame_id),
        target_frame_id=int(window.head_frame_id),
        sampled_frame_ids=window.sampled_frame_ids,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        message=str(message),
        failure_code=str(failure_code),
    )
