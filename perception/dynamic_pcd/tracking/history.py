from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class HistoryState:
    pcd_history: np.ndarray
    center: np.ndarray
    velocity: np.ndarray
    valid: bool
    message: str = ""


class PCDHistoryBuffer:
    """Maintain fixed-length point cloud history and object velocity."""

    def __init__(self, history_len: int = 4, max_center_jump: float = 0.20):
        self.history_len = int(history_len)
        self.max_center_jump = float(max_center_jump)
        self.points: Deque[np.ndarray] = deque(maxlen=self.history_len)
        self.centers: Deque[np.ndarray] = deque(maxlen=self.history_len)
        self.timestamps: Deque[float] = deque(maxlen=self.history_len)
        self.last_valid_center: Optional[np.ndarray] = None

    def reset(self):
        self.points.clear()
        self.centers.clear()
        self.timestamps.clear()
        self.last_valid_center = None

    def update(self, policy_points: np.ndarray, center: np.ndarray, timestamp: float) -> HistoryState:
        center = center.astype(np.float32)
        if self.last_valid_center is not None:
            jump = float(np.linalg.norm(center - self.last_valid_center))
            if jump > self.max_center_jump:
                return self._state(valid=False, message=f"center jump too large: {jump:.3f}m")

        self.points.append(policy_points.astype(np.float32))
        self.centers.append(center)
        self.timestamps.append(float(timestamp))
        self.last_valid_center = center.copy()

        return self._state(valid=len(self.points) >= 1, message="ok")

    def _state(self, valid: bool, message: str = "") -> HistoryState:
        if len(self.points) == 0:
            return HistoryState(
                pcd_history=np.zeros((0, 0, 3), dtype=np.float32),
                center=np.zeros(3, dtype=np.float32),
                velocity=np.zeros(3, dtype=np.float32),
                valid=False,
                message="empty history",
            )

        # Pad by repeating oldest point cloud until history_len.
        pts = list(self.points)
        while len(pts) < self.history_len:
            pts.insert(0, pts[0])
        pcd_history = np.stack(pts[-self.history_len :], axis=0).astype(np.float32)

        center = self.centers[-1].astype(np.float32)
        if len(self.centers) >= 2:
            c0 = self.centers[-2]
            c1 = self.centers[-1]
            t0 = self.timestamps[-2]
            t1 = self.timestamps[-1]
            dt = max(1e-3, float(t1 - t0))
            vel = ((c1 - c0) / dt).astype(np.float32)
        else:
            vel = np.zeros(3, dtype=np.float32)

        return HistoryState(
            pcd_history=pcd_history,
            center=center,
            velocity=vel,
            valid=valid,
            message=message,
        )
