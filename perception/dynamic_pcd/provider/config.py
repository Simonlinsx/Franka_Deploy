"""Validated provider-local runtime configuration helpers."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Dict

import numpy as np


DEFAULT_RUNTIME_FRAME_TIMEOUT_MS = 1000
MAX_RUNTIME_FRAME_TIMEOUT_MS = 1000


def validated_runtime_frame_timeout_ms(camera_cfg: Dict[str, Any]) -> int:
    """Return the bounded timeout used only by the steady capture pipeline.

    ROI acquisition intentionally keeps ``RealSenseCamera.get_frame``'s
    existing one-second default. Once the provider enters its formal
    frame-by-frame path, the deployed configuration can use a shorter timeout
    so a missing RGB-D frame is reported before an actuator heartbeat expires.
    """

    value = camera_cfg.get(
        "runtime_frame_timeout_ms", DEFAULT_RUNTIME_FRAME_TIMEOUT_MS
    )
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(
            "camera.runtime_frame_timeout_ms must be an integer in "
            f"1..{MAX_RUNTIME_FRAME_TIMEOUT_MS}"
        )
    timeout_ms = int(value)
    if not 1 <= timeout_ms <= MAX_RUNTIME_FRAME_TIMEOUT_MS:
        raise ValueError(
            "camera.runtime_frame_timeout_ms must be an integer in "
            f"1..{MAX_RUNTIME_FRAME_TIMEOUT_MS}"
        )
    return timeout_ms


__all__ = [
    "DEFAULT_RUNTIME_FRAME_TIMEOUT_MS",
    "MAX_RUNTIME_FRAME_TIMEOUT_MS",
    "validated_runtime_frame_timeout_ms",
]
