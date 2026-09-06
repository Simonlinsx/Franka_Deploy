"""Compact, colored terminal output for supervised deployment.

The deployment audit remains authoritative.  This module only controls what
is mirrored to the operator terminal; it does not alter runtime decisions or
hardware communication.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import os
import sys
import threading
from typing import Iterator, TextIO


_RESET = "\033[0m"
_BRIGHT_CYAN = "\033[1;96m"
_BRIGHT_GREEN = "\033[1;92m"
_BRIGHT_YELLOW = "\033[1;93m"
_BRIGHT_RED = "\033[1;91m"

_KEY_STAGE_MARKERS = (
    "[Deployment preflight PASS]",
    "[Automatic reset]",
    "[Automatic reset PASS]",
    "[Object grounding START]",
    "[Object grounding SEARCHING]",
    "[Object grounding RETRY]",
    "[Object grounding TRACKED]",
    "[Object grounding VERIFYING]",
    "[Object grounding PASS]",
    "[Object ROI selector START]",
    "[Object ROI selector PASS]",
    "[Object ROI preflight PASS]",
    "[Object session handoff PASS]",
    "[Perception PREPARING]",
    "[Perception UI READY]",
    "[Rollout START]",
    "[Throw trigger START]",
    "[Throw trigger ARMED]",
    "[Throw trigger DETECTED]",
    "[Rollout PASS]",
    "[V94 video] saved=",
    "[V94 visualization] SAVED",
    "[RH56 force log]",
    "[Throw trigger test PASS]",
    "[Throw trigger test FAILED]",
)


def _use_color(stream: TextIO) -> bool:
    return bool(
        os.environ.get("NO_COLOR") is None
        and hasattr(stream, "isatty")
        and stream.isatty()
    )


def _line_style(line: str) -> str:
    lowered = line.lower()
    if (
        "failed" in lowered
        or "refused" in lowered
        or "[error]" in lowered
        or "stop unconfirmed" in lowered
    ):
        return _BRIGHT_RED
    if (
        "[warn" in lowered
        or "warning" in lowered
        or "retry]" in lowered
        or "searching]" in lowered
        or "preparing]" in lowered
    ):
        return _BRIGHT_YELLOW
    if (
        "pass]" in lowered
        or "saved=" in lowered
        or "] saved" in lowered
        or "tracked]" in lowered
        or "armed]" in lowered
        or "detected]" in lowered
        or "ready]" in lowered
    ):
        return _BRIGHT_GREEN
    return _BRIGHT_CYAN


def _is_key_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if any(marker in stripped for marker in _KEY_STAGE_MARKERS):
        return True
    if stripped.startswith("Select a ROI") or stripped.startswith(
        "Cancel the selection"
    ):
        return True
    return bool(
        "failed" in lowered
        or "refused" in lowered
        or "[error]" in lowered
        or "[warn" in lowered
        or "stop unconfirmed" in lowered
    )


class _CompactStream(io.TextIOBase):
    """Line-buffered filtering proxy safe for Python worker-thread prints."""

    def __init__(self, target: TextIO) -> None:
        self._target = target
        self._buffer = ""
        self._lock = threading.Lock()
        self._color = _use_color(target)

    @property
    def encoding(self):  # pragma: no cover - delegated terminal metadata
        return getattr(self._target, "encoding", "utf-8")

    def isatty(self) -> bool:
        return bool(getattr(self._target, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._target.fileno()

    def writable(self) -> bool:
        return True

    def _emit(self, line: str) -> None:
        if not _is_key_line(line):
            return
        if self._color:
            self._target.write(f"{_line_style(line)}{line}{_RESET}")
        else:
            self._target.write(line)

    def write(self, text: str) -> int:
        data = str(text)
        with self._lock:
            self._buffer += data
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._emit(line + "\n")
        return len(data)

    def flush(self) -> None:
        with self._lock:
            # Do not force a partial line through the filter: several existing
            # stage messages are assembled with print(..., end="").
            self._target.flush()

    def close_pending(self) -> None:
        with self._lock:
            if self._buffer:
                self._emit(self._buffer + "\n")
                self._buffer = ""
            self._target.flush()


@contextmanager
def compact_deployment_console(enabled: bool = True) -> Iterator[None]:
    """Hide diagnostic chatter while retaining stage/failure lines."""

    if not enabled:
        yield
        return
    stdout_proxy = _CompactStream(sys.stdout)
    stderr_proxy = _CompactStream(sys.stderr)
    try:
        with redirect_stdout(stdout_proxy), redirect_stderr(stderr_proxy):
            yield
    finally:
        stdout_proxy.close_pending()
        stderr_proxy.close_pending()


def emit_operator_line(message: str, *, error: bool = False) -> None:
    """Print one colored operator-facing line outside the filter context."""

    target = sys.stderr if error else sys.stdout
    line = str(message)
    if _use_color(target):
        target.write(f"{_line_style(line)}{line}{_RESET}\n")
    else:
        target.write(line + "\n")
    target.flush()


__all__ = ["compact_deployment_console", "emit_operator_line"]
