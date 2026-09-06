from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from dynamic_pcd.segmentation.prompt_client import ZMQPromptSegmentationClient
from dynamic_pcd.segmentation.prompt_protocol import PromptSegmentationResult


@dataclass
class PromptReacquireRequest:
    image_bgr: np.ndarray
    prompt: str
    reference_bbox_xyxy: Optional[np.ndarray]
    frame_id: int
    frame_timestamp: float
    generation: int
    reason: str
    request_id: str


@dataclass
class PromptReacquireResponse:
    request: PromptReacquireRequest
    result: Optional[PromptSegmentationResult]
    error: Optional[BaseException]
    elapsed_s: float


@dataclass
class RelocatedPromptMask:
    """A source-frame semantic mask translated onto the newest RGB frame."""

    mask: np.ndarray
    bbox_xyxy: np.ndarray
    score: float
    score_margin: float


def relocate_prompt_mask_to_current_frame(
    source_image_bgr: np.ndarray,
    current_image_bgr: np.ndarray,
    source_mask: np.ndarray,
    *,
    min_score: float = 0.70,
    min_score_margin: float = 0.04,
) -> Optional[RelocatedPromptMask]:
    """Relocate a stale prompt mask by masked colour-template matching.

    Grounded-SAM on CPU may finish hundreds of camera frames after submission.
    Applying its source-frame pixels directly to the newest RGB-D frame samples
    unrelated colour/depth whenever the object moved.  This helper estimates a
    translation first; the adaptive tracker's appearance, depth and size gates
    still make the final accept/reject decision.

    A near-tied second peak is rejected instead of choosing between visually
    indistinguishable category instances.
    """

    source = np.asarray(source_image_bgr)
    current = np.asarray(current_image_bgr)
    mask = (np.asarray(source_mask) > 0).astype(np.uint8)
    if (
        source.dtype != np.uint8
        or current.dtype != np.uint8
        or source.ndim != 3
        or current.ndim != 3
        or source.shape[2] != 3
        or current.shape[2] != 3
        or mask.shape != source.shape[:2]
        or source.shape[:2] != current.shape[:2]
        or int(mask.sum()) == 0
    ):
        return None

    ys, xs = np.nonzero(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    template = np.ascontiguousarray(source[y1:y2, x1:x2])
    template_mask = np.ascontiguousarray(mask[y1:y2, x1:x2] * 255)
    th, tw = template.shape[:2]
    h, w = current.shape[:2]
    if th < 3 or tw < 3 or th > h or tw > w:
        return None

    # Lab retains chromatic separation for low-texture objects better than a
    # grayscale template. TM_CCORR_NORMED supports an explicit object mask.
    source_lab = cv2.cvtColor(template, cv2.COLOR_BGR2LAB).astype(np.float32)
    current_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_lab[..., 0] /= 255.0
    current_lab[..., 0] /= 255.0
    source_lab[..., 1:] = (source_lab[..., 1:] - 128.0) / 127.0
    current_lab[..., 1:] = (current_lab[..., 1:] - 128.0) / 127.0
    try:
        scores = cv2.matchTemplate(
            current_lab,
            source_lab,
            cv2.TM_CCORR_NORMED,
            mask=template_mask,
        )
    except cv2.error:
        scores = cv2.matchTemplate(
            current_lab,
            source_lab,
            cv2.TM_CCORR_NORMED,
        )
    if scores.size == 0 or not np.isfinite(scores).any():
        return None
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, best_score, _, best_location = cv2.minMaxLoc(scores)

    # Suppress the same physical peak before measuring ambiguity. Adjacent
    # translations of one object otherwise look like a false second instance.
    second_scores = scores.copy()
    bx, by = int(best_location[0]), int(best_location[1])
    radius_x = max(2, tw // 2)
    radius_y = max(2, th // 2)
    sx1 = max(0, bx - radius_x)
    sx2 = min(second_scores.shape[1], bx + radius_x + 1)
    sy1 = max(0, by - radius_y)
    sy2 = min(second_scores.shape[0], by + radius_y + 1)
    second_scores[sy1:sy2, sx1:sx2] = -1.0
    second_score = (
        float(np.max(second_scores)) if second_scores.size else -1.0
    )
    margin = float(best_score - second_score)
    if float(best_score) < float(min_score) or margin < float(min_score_margin):
        return None

    relocated = np.zeros((h, w), dtype=np.uint8)
    relocated[by : by + th, bx : bx + tw] = template_mask > 0
    return RelocatedPromptMask(
        mask=relocated,
        bbox_xyxy=np.asarray([bx, by, bx + tw, by + th], dtype=np.int32),
        score=float(best_score),
        score_margin=margin,
    )


class PromptRetrySchedule:
    """Main-thread retry timing for the one-slot semantic worker.

    The cooldown is measured between *request start times*.  Consequently a
    slow failed inference can be retried on the newest camera frame as soon as
    its response is consumed, while a service that fails immediately remains
    rate-limited.  This class deliberately does not own a timer or thread: the
    camera loop supplies the newest frame and performs the next submission.
    """

    def __init__(self, cooldown_s: float):
        cooldown_s = float(cooldown_s)
        if not np.isfinite(cooldown_s) or cooldown_s <= 0.0:
            raise ValueError("cooldown_s must be a finite positive number")
        self.cooldown_s = cooldown_s
        self._last_submit_t = float("-inf")
        self._pending_reason: Optional[str] = None
        self._attempt = 0

    @property
    def retry_pending(self) -> bool:
        return self._pending_reason is not None

    @property
    def pending_reason(self) -> Optional[str]:
        return self._pending_reason

    @property
    def attempt(self) -> int:
        return self._attempt

    def cooldown_remaining_s(self, now: float) -> float:
        now = float(now)
        if not np.isfinite(now):
            raise ValueError("now must be finite")
        if self._last_submit_t == float("-inf"):
            return 0.0
        elapsed_s = max(0.0, now - self._last_submit_t)
        return max(0.0, self.cooldown_s - elapsed_s)

    def ready(self, now: float, force: bool = False) -> bool:
        return bool(force) or self.cooldown_remaining_s(now) <= 0.0

    def request_retry(self, reason: str) -> None:
        reason = str(reason).strip()
        if not reason:
            raise ValueError("retry reason must be non-empty")
        # Only one semantic response can arrive at a time.  Keeping one
        # coalesced pending reason is therefore sufficient and prevents an
        # unbounded queue when the camera continues at 20-30 Hz.
        self._pending_reason = reason

    def mark_submitted(self, now: float) -> int:
        now = float(now)
        if not np.isfinite(now):
            raise ValueError("now must be finite")
        self._last_submit_t = now
        self._pending_reason = None
        self._attempt += 1
        return self._attempt

    def clear(self) -> None:
        """End the current retry sequence without weakening rate limiting."""

        self._pending_reason = None
        self._attempt = 0


class PromptServiceManager:
    """Connect to, or optionally launch, the persistent prompt service."""

    def __init__(
        self,
        addr: str = "tcp://127.0.0.1:5557",
        autostart: bool = True,
        startup_timeout_s: float = 45.0,
        request_timeout_ms: int = 30000,
        launcher_path: Optional[str] = None,
        launcher_args: Optional[Sequence[str]] = None,
    ):
        self.addr = str(addr)
        self.autostart = bool(autostart)
        self.startup_timeout_s = max(1.0, float(startup_timeout_s))
        self.request_timeout_ms = max(1, int(request_timeout_ms))
        repo_root = Path(__file__).resolve().parents[2]
        self.launcher_path = Path(
            launcher_path
            if launcher_path is not None
            else repo_root / "scripts" / "run_prompt_segmentation_service.sh"
        ).expanduser()
        self.launcher_args = [str(value) for value in (launcher_args or ())]
        self.client: Optional[ZMQPromptSegmentationClient] = None
        self.process: Optional[subprocess.Popen] = None
        self.owns_process = False

    def start(self) -> dict:
        if self.client is not None:
            return self.client.health(timeout_ms=1000)
        self.client = ZMQPromptSegmentationClient(
            self.addr, timeout_ms=self.request_timeout_ms
        )
        try:
            health = self.client.health(timeout_ms=300)
            print(
                f"[Prompt] using existing service at {self.addr}: "
                f"detector={health.get('detector_backend')} "
                f"mask={health.get('mask_backend')} "
                f"device={health.get('device')}"
            )
            return health
        except Exception as first_error:
            if not self.autostart:
                self.client.close()
                self.client = None
                raise RuntimeError(
                    f"prompt service at {self.addr} is unavailable and "
                    "autostart is disabled"
                ) from first_error

        if not self.launcher_path.is_file():
            self.close()
            raise RuntimeError(
                f"prompt service launcher does not exist: {self.launcher_path}"
            )
        command = [str(self.launcher_path), "--addr", self.addr]
        command.extend(self.launcher_args)
        self.process = subprocess.Popen(
            command,
            cwd=str(self.launcher_path.parent.parent),
        )
        self.owns_process = True
        print(
            f"[Prompt] starting persistent service pid={self.process.pid}; "
            f"waiting up to {self.startup_timeout_s:.0f}s for models"
        )

        deadline = time.monotonic() + self.startup_timeout_s
        last_error: Optional[BaseException] = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                code = self.process.returncode
                self.close()
                raise RuntimeError(
                    f"prompt service exited during startup with code {code}"
                )
            try:
                health = self.client.health(timeout_ms=500)
                print(
                    "[Prompt] service ready: "
                    f"detector={health.get('detector_backend')} "
                    f"mask={health.get('mask_backend')} "
                    f"device={health.get('device')} "
                    f"load={float(health.get('model_load_ms', 0.0)):.1f}ms"
                )
                return health
            except Exception as exc:
                last_error = exc
                time.sleep(0.10)
        self.close()
        raise TimeoutError(
            f"prompt service at {self.addr} did not become ready within "
            f"{self.startup_timeout_s:.1f}s"
        ) from last_error

    def segment(self, *args, **kwargs) -> PromptSegmentationResult:
        if self.client is None:
            raise RuntimeError("prompt service manager is not started")
        kwargs.setdefault("timeout_ms", self.request_timeout_ms)
        return self.client.segment(*args, **kwargs)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.owns_process and self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2.0)
            print("[Prompt] owned service stopped")
        self.process = None
        self.owns_process = False


class AsyncPromptReacquirer:
    """One-request-at-a-time background prompt worker.

    The ZMQ client/socket is constructed inside the worker thread.  The camera
    loop therefore never shares a ZMQ socket across threads and never waits for
    CPU-heavy GroundingDINO/SAM inference.
    """

    def __init__(
        self,
        addr: str,
        timeout_ms: int = 30000,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        mask_threshold: float = 0.50,
        top_k: int = 5,
        client_factory: Optional[Callable[[], ZMQPromptSegmentationClient]] = None,
    ):
        self.addr = str(addr)
        self.timeout_ms = max(1, int(timeout_ms))
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.mask_threshold = float(mask_threshold)
        self.top_k = max(1, int(top_k))
        self.client_factory = client_factory or (
            lambda: ZMQPromptSegmentationClient(
                self.addr, timeout_ms=self.timeout_ms
            )
        )
        self._condition = threading.Condition()
        self._pending: Optional[PromptReacquireRequest] = None
        self._response: Optional[PromptReacquireResponse] = None
        self._busy = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="prompt-reacquire",
            daemon=True,
        )
        self._thread.start()

    @property
    def busy(self) -> bool:
        with self._condition:
            # A completed response still belongs to the sole in-flight slot
            # until the camera thread consumes it.  Accepting another request
            # here would let the worker overwrite an unpolled response.
            return (
                self._busy
                or self._pending is not None
                or self._response is not None
            )

    def submit(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        reference_bbox_xyxy: Optional[Sequence[float]],
        frame_id: int,
        frame_timestamp: float,
        generation: int,
        reason: str,
    ) -> Optional[str]:
        image = np.asarray(image_bgr)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image_bgr must be uint8 HxWx3")
        with self._condition:
            if (
                self._closed
                or self._busy
                or self._pending is not None
                or self._response is not None
            ):
                return None
            request_id = (
                f"prompt-g{int(generation)}-f{int(frame_id)}-"
                f"{uuid.uuid4().hex[:8]}"
            )
            self._pending = PromptReacquireRequest(
                image_bgr=np.ascontiguousarray(image).copy(),
                prompt=str(prompt),
                reference_bbox_xyxy=(
                    None
                    if reference_bbox_xyxy is None
                    else np.asarray(reference_bbox_xyxy, dtype=np.float32).copy()
                ),
                frame_id=int(frame_id),
                frame_timestamp=float(frame_timestamp),
                generation=int(generation),
                reason=str(reason),
                request_id=request_id,
            )
            self._condition.notify_all()
            return request_id

    def poll(self) -> Optional[PromptReacquireResponse]:
        with self._condition:
            response = self._response
            self._response = None
            return response

    def _run(self) -> None:
        client: Optional[ZMQPromptSegmentationClient] = None
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._closed:
                        self._condition.wait(timeout=0.25)
                    if self._closed:
                        return
                    request = self._pending
                    self._pending = None
                    self._busy = True
                assert request is not None
                started = time.monotonic()
                result = None
                error: Optional[BaseException] = None
                try:
                    if client is None:
                        client = self.client_factory()
                    result = client.segment(
                        request.image_bgr,
                        prompt=request.prompt,
                        previous_bbox_xyxy=request.reference_bbox_xyxy,
                        request_id=request.request_id,
                        frame_id=request.frame_id,
                        frame_timestamp=request.frame_timestamp,
                        frame_metadata={
                            "tracker_generation": request.generation,
                            "reason": request.reason,
                        },
                        box_threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                        mask_threshold=self.mask_threshold,
                        top_k=self.top_k,
                        timeout_ms=self.timeout_ms,
                    )
                except BaseException as exc:
                    error = exc
                    if client is not None:
                        client.close()
                        client = None
                response = PromptReacquireResponse(
                    request=request,
                    result=result,
                    error=error,
                    elapsed_s=time.monotonic() - started,
                )
                with self._condition:
                    # ``close`` may have been requested while the external
                    # model was running.  Do not retain a late response after
                    # shutdown, and never replace an unconsumed response.
                    if not self._closed and self._response is None:
                        self._response = response
                    self._busy = False
                    self._condition.notify_all()
                    if self._closed:
                        return
        finally:
            if client is not None:
                client.close()

    def request_close(self) -> None:
        """Request worker shutdown without waiting for a model RPC to finish."""

        with self._condition:
            self._closed = True
            self._pending = None
            self._response = None
            self._condition.notify_all()

    def close(self, join_timeout_s: float = 1.0) -> bool:
        """Request shutdown and wait for at most ``join_timeout_s``.

        ZMQ sockets are owned exclusively by the worker thread, so forcibly
        closing its socket from this thread would violate ZeroMQ's threading
        contract.  The bounded join keeps application shutdown responsive; the
        return value tells the caller whether the worker actually exited.  An
        application that owns the external service can request close, stop that
        service to interrupt the RPC, and call this method again.
        """

        self.request_close()
        self._thread.join(timeout=max(0.0, float(join_timeout_s)))
        stopped = not self._thread.is_alive()
        if not stopped:
            print(
                "[Prompt][WARN] semantic worker is still finishing an external "
                "request; continuing bounded shutdown"
            )
        return stopped
