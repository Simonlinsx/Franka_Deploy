"""One-shot readiness marker for the perception-only live viewer.

The marker is intentionally tiny.  Its only meaning is that the viewer has
finished its local window/camera/native-reader setup and is about to wait for
the executor-created telemetry mapping.  It is not motion authorization.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat


VIEWER_READY_BYTES = b"ANYDEX_LIVE_PREVIEW_READY_V1\n"


def normalize_unfollowed_leaf(path: Path) -> Path:
    """Return an absolute path without dereferencing its final component.

    ``Path.resolve()`` follows the final symlink even when it is dangling.  A
    caller that subsequently uses ``O_EXCL``/``O_NOFOLLOW`` would therefore
    operate on the symlink target instead of rejecting the link itself.  Only
    the parent is canonicalized here; the leaf is preserved for the guarded
    ``open(2)`` call.
    """

    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    if raw.name in ("", ".", ".."):
        raise ValueError("readiness path must name one file")
    return raw.parent.resolve() / raw.name


def publish_viewer_ready(path: Path) -> Path:
    """Create one immutable marker without replacing any existing path."""

    target = normalize_unfollowed_leaf(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(target), flags, 0o444)
    try:
        offset = 0
        while offset < len(VIEWER_READY_BYTES):
            written = os.write(descriptor, VIEWER_READY_BYTES[offset:])
            if written <= 0:
                raise OSError("short write while publishing viewer readiness")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    try:
        directory = os.open(
            str(target.parent),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
    except OSError:
        directory = None
    if directory is not None:
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return target


def viewer_ready_is_published(path: Path) -> bool:
    """Return true only for the complete reviewed marker and exact mode."""

    try:
        target = normalize_unfollowed_leaf(path)
    except (OSError, ValueError):
        return False
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(target), flags)
    except OSError:
        return False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            return False
        if stat.S_IMODE(metadata.st_mode) != 0o444:
            return False
        content = b""
        while len(content) <= len(VIEWER_READY_BYTES):
            chunk = os.read(descriptor, len(VIEWER_READY_BYTES) + 1 - len(content))
            if not chunk:
                break
            content += chunk
        return content == VIEWER_READY_BYTES
    finally:
        os.close(descriptor)
