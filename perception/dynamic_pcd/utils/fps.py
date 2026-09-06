from __future__ import annotations

import time
from collections import deque
from math import floor, isfinite
from typing import Callable, Deque, Optional, Tuple


class FPSMeter:
    """Simple moving-window FPS and latency meter."""

    def __init__(self, window: int = 60):
        self.window = int(window)
        self.times = deque(maxlen=self.window)
        self.last_t = None
        self.ema_fps = 0.0
        self.ema_ms = 0.0

    def tick(self, dt: float | None = None):
        now = time.perf_counter()
        if dt is None:
            if self.last_t is None:
                self.last_t = now
                return 0.0, 0.0
            dt = max(1e-9, now - self.last_t)
            self.last_t = now
        fps = 1.0 / max(1e-9, dt)
        ms = dt * 1000.0
        self.times.append(dt)
        if self.ema_fps <= 0:
            self.ema_fps = fps
            self.ema_ms = ms
        else:
            self.ema_fps = 0.9 * self.ema_fps + 0.1 * fps
            self.ema_ms = 0.9 * self.ema_ms + 0.1 * ms
        return fps, ms

    @property
    def avg_fps(self) -> float:
        if not self.times:
            return 0.0
        return 1.0 / max(1e-9, sum(self.times) / len(self.times))

    @property
    def avg_ms(self) -> float:
        if not self.times:
            return 0.0
        return 1000.0 * sum(self.times) / len(self.times)

    @property
    def sample_count(self) -> int:
        return len(self.times)


class DeadlineRateGate:
    """Rate-limit work without the common near-equal-rate half-speed bug.

    A gate implemented as ``now - last_run >= period`` and then
    ``last_run = now`` loses the fractional remainder on every run.  When a
    nominal 30 Hz camera arrives a fraction of a millisecond before a 30 Hz
    publishing period, that implementation can reject every other frame and
    publish at roughly 15 Hz.

    This gate instead advances an absolute deadline by whole periods.  It
    therefore preserves phase/remainder across calls, catches up after a late
    frame without bursting, and uses a monotonic clock by default.
    """

    def __init__(
        self,
        rate_hz: float,
        clock: Callable[[], float] = time.perf_counter,
    ):
        rate_hz = float(rate_hz)
        if not isfinite(rate_hz) or not rate_hz > 0.0:
            raise ValueError(f"rate_hz must be > 0, got {rate_hz}")
        self.rate_hz = rate_hz
        self.period_s = 1.0 / rate_hz
        self.clock = clock
        self.next_deadline_s: Optional[float] = None

    def ready(self, now_s: Optional[float] = None) -> bool:
        """Return whether one unit of work is due at ``now_s``.

        The first call is immediately ready.  If several periods elapsed, one
        call is admitted and the missed slots are discarded; callers never
        receive a burst of stale work.
        """

        now = float(self.clock() if now_s is None else now_s)
        if self.next_deadline_s is None:
            self.next_deadline_s = now + self.period_s
            return True
        if now < self.next_deadline_s:
            return False

        elapsed_periods = floor((now - self.next_deadline_s) / self.period_s)
        self.next_deadline_s += (elapsed_periods + 1) * self.period_s
        return True

    def reset(self) -> None:
        """Forget accumulated phase; the next call will be admitted."""

        self.next_deadline_s = None


class PacketRateMonitor:
    """Measure attempted and *valid object-PCD* publication separately.

    Counting every packet can report a healthy rate while every packet is an
    invalid/lost notification.  This monitor keeps a time window of publication
    events and exposes an immediate LOST/NO_VALID state when the newest packet
    has no current object payload.
    """

    def __init__(
        self,
        window_s: float = 2.0,
        lost_timeout_s: float = 0.25,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.window_s = float(window_s)
        self.lost_timeout_s = float(lost_timeout_s)
        if not isfinite(self.window_s) or self.window_s <= 0.0:
            raise ValueError("window_s must be positive")
        if not isfinite(self.lost_timeout_s) or self.lost_timeout_s <= 0.0:
            raise ValueError("lost_timeout_s must be positive")
        self.clock = clock
        self.events: Deque[Tuple[float, bool]] = deque()
        self.last_valid_t: Optional[float] = None

    def tick(self, valid: bool, now_s: Optional[float] = None) -> None:
        now = float(self.clock() if now_s is None else now_s)
        is_valid = bool(valid)
        self.events.append((now, is_valid))
        if is_valid:
            self.last_valid_t = now
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def _rates(self, now_s: Optional[float] = None) -> Tuple[float, float, float]:
        now = float(self.clock() if now_s is None else now_s)
        self._prune(now)
        if len(self.events) < 2:
            ratio = float(self.events[0][1]) if self.events else 0.0
            return 0.0, 0.0, ratio
        span = max(1e-9, self.events[-1][0] - self.events[0][0])
        valid_count = sum(1 for _, valid in self.events if valid)
        publish_hz = (len(self.events) - 1) / span
        valid_hz = max(0, valid_count - 1) / span
        return publish_hz, valid_hz, valid_count / len(self.events)

    @property
    def publish_hz(self) -> float:
        return self._rates()[0]

    @property
    def valid_hz(self) -> float:
        return self._rates()[1]

    @property
    def valid_ratio(self) -> float:
        return self._rates()[2]

    @property
    def sample_count(self) -> int:
        self._prune(float(self.clock()))
        return len(self.events)

    def age_since_valid(self, now_s: Optional[float] = None) -> float:
        now = float(self.clock() if now_s is None else now_s)
        if self.last_valid_t is None:
            return float("inf")
        return max(0.0, now - self.last_valid_t)

    def status(
        self,
        min_valid_hz: float,
        warmup_events: int = 30,
        now_s: Optional[float] = None,
    ) -> str:
        now = float(self.clock() if now_s is None else now_s)
        self._prune(now)
        if self.last_valid_t is None:
            return "NO_VALID"
        if self.age_since_valid(now) > self.lost_timeout_s:
            return "NO_VALID"
        if not self.events or not self.events[-1][1]:
            return "LOST"
        if len(self.events) < max(1, int(warmup_events)):
            return "WARMUP"
        return "PASS" if self._rates(now)[1] >= float(min_valid_hz) else "LOW"
