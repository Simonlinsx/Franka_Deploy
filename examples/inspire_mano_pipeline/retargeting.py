from __future__ import annotations

from collections import deque
import os
from pathlib import Path
from typing import Deque, Optional, Sequence

import numpy as np

from .calibration import PipelineCalibration
from .model import DEX_JOINTS, HARDWARE_JOINTS, ManoDetection, RetargetOutput


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEX_ROOT = Path(
    os.environ.get("DEX_ROOT", str(WORKSPACE_ROOT / "third_party/dex-retargeting"))
).expanduser()


class TemporalQposFilter:
    """Short median filter followed by an EMA for geometric MANO qpos."""

    def __init__(
        self,
        median_window: int,
        ema_alpha: float,
        active_indices: Optional[Sequence[int]] = None,
    ) -> None:
        if median_window < 1 or median_window % 2 == 0:
            raise ValueError("median_window must be a positive odd integer")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        self.median_window = int(median_window)
        self.ema_alpha = float(ema_alpha)
        self.active_indices = (
            tuple(range(len(HARDWARE_JOINTS)))
            if active_indices is None
            else tuple(int(index) for index in active_indices)
        )
        if (
            len(set(self.active_indices)) != len(self.active_indices)
            or any(index < 0 or index >= len(HARDWARE_JOINTS) for index in self.active_indices)
        ):
            raise ValueError("active_indices must be unique RH56 qpos indices")
        self._history: Deque[np.ndarray] = deque(maxlen=self.median_window)
        self._state: Optional[np.ndarray] = None

    def apply(self, qpos: np.ndarray) -> np.ndarray:
        value = np.asarray(qpos, dtype=np.float32)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("temporal filter expects six finite qpos values")
        if not self.active_indices:
            return value.copy()
        if not self._history:
            # Seed the full window so a single early outlier cannot dominate an
            # otherwise stable initial pose.
            for _ in range(self.median_window):
                self._history.append(value.copy())
        else:
            self._history.append(value.copy())
        median = np.median(np.stack(tuple(self._history)), axis=0).astype(np.float32)
        if self._state is None:
            self._state = median
        else:
            self._state = (
                self.ema_alpha * median + (1.0 - self.ema_alpha) * self._state
            ).astype(np.float32)
        result = value.copy()
        result[list(self.active_indices)] = self._state[list(self.active_indices)]
        return result

    def reset(self) -> None:
        self._history.clear()
        self._state = None


class EndpointHysteresisFilter:
    """Latch calibrated open/closed qpos until the pose clearly exits an endpoint."""

    def __init__(
        self,
        calibration: PipelineCalibration,
        active_axes: Optional[Sequence[str]] = None,
    ) -> None:
        self.calibration = calibration
        self.active_axes = (
            set(HARDWARE_JOINTS) if active_axes is None else set(active_axes)
        )
        if not self.active_axes.issubset(HARDWARE_JOINTS):
            raise ValueError("active_axes contains an unknown RH56 axis")
        self._states: list[Optional[str]] = [None] * len(HARDWARE_JOINTS)

    def apply(self, qpos: np.ndarray) -> np.ndarray:
        value = np.asarray(qpos, dtype=np.float32)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("endpoint filter expects six finite qpos values")
        result = value.copy()
        for index, name in enumerate(HARDWARE_JOINTS):
            if name not in self.active_axes:
                continue
            axis = self.calibration.axes[name]
            current = float(value[index])
            state = self._states[index]
            if state == "open" and current > axis.open_exit_q:
                state = None
            elif state == "closed" and current < axis.closed_exit_q:
                state = None
            if state is None:
                if current <= axis.open_enter_q:
                    state = "open"
                elif current >= axis.closed_enter_q:
                    state = "closed"
            self._states[index] = state
            if state == "open":
                result[index] = axis.q_open
            elif state == "closed":
                result[index] = axis.q_closed
        return result

    def reset(self) -> None:
        self._states = [None] * len(HARDWARE_JOINTS)


class DexInspireRetargeter:
    """Official dexsuite vector optimizer for the six active Inspire joints."""

    def __init__(
        self,
        calibration: PipelineCalibration,
        dex_root: Path = DEFAULT_DEX_ROOT,
        config_path: Optional[Path] = None,
    ) -> None:
        try:
            from dex_retargeting.retargeting_config import RetargetingConfig
        except ImportError as exc:
            raise RuntimeError(
                "dex-retargeting is an optional dependency; run "
                "examples/setup_inspire_mano_env.sh or provide DEX_ROOT"
            ) from exc

        self.calibration = calibration
        dex_root = Path(dex_root).expanduser().resolve()
        if config_path is None:
            config_path = (
                dex_root
                / "src/dex_retargeting/configs/teleop/inspire_hand_right.yml"
            )
        robot_dir = dex_root / "assets/robots/hands"
        if not Path(config_path).is_file():
            raise FileNotFoundError(f"dex-retargeting config not found: {config_path}")
        if not robot_dir.is_dir():
            raise FileNotFoundError(f"dex-retargeting robot assets not found: {robot_dir}")
        RetargetingConfig.set_default_urdf_dir(robot_dir)
        config = RetargetingConfig.load_from_file(config_path)
        self._retargeting = config.build()
        self._human_indices = np.asarray(
            self._retargeting.optimizer.target_link_human_indices, dtype=np.int64
        )
        joint_names = list(self._retargeting.joint_names)
        self._output_indices = np.asarray(
            [joint_names.index(name) for name in DEX_JOINTS], dtype=np.int64
        )
        self.name = "dex-vector"

    def retarget(self, detection: ManoDetection) -> RetargetOutput:
        joints = _validate_joints(detection.keypoints_3d_canonical)
        origin = self._human_indices[0]
        task = self._human_indices[1]
        reference = joints[task] - joints[origin]
        robot_qpos = np.asarray(
            self._retargeting.retarget(reference), dtype=np.float32
        )
        qpos = robot_qpos[self._output_indices]
        lower = np.asarray(
            [self.calibration.axes[name].q_open for name in HARDWARE_JOINTS],
            dtype=np.float32,
        )
        upper = np.asarray(
            [self.calibration.axes[name].q_closed for name in HARDWARE_JOINTS],
            dtype=np.float32,
        )
        qpos = np.clip(qpos, lower, upper)
        return RetargetOutput(
            qpos=qpos,
            hardware_targets=self.calibration.map_qpos(qpos),
            backend=self.name,
        )

    def reset(self) -> None:
        self._retargeting.reset()
        if self._retargeting.filter is not None:
            self._retargeting.filter.reset()


class GeometricInspireRetargeter:
    """Dependency-light fallback based on palm-local MANO geometry."""

    _THUMB_BEND_OPEN_RADIANS = float(np.deg2rad(35.0))
    _THUMB_BEND_CLOSED_RADIANS = float(np.deg2rad(47.0))

    _CHAINS = (
        (17, 18, 19, 20),
        (13, 14, 15, 16),
        (9, 10, 11, 12),
        (5, 6, 7, 8),
        (1, 2, 3, 4),
    )

    def __init__(self, calibration: PipelineCalibration) -> None:
        self.calibration = calibration
        self.name = "geometric"
        self._filter = TemporalQposFilter(
            calibration.temporal_median_window,
            calibration.temporal_ema_alpha,
            active_indices=tuple(
                HARDWARE_JOINTS.index(name)
                for name in calibration.temporal_filter_axes
            ),
        )
        self._endpoint_filter = EndpointHysteresisFilter(
            calibration,
            active_axes=calibration.temporal_filter_axes,
        )

    @staticmethod
    def _curl(joints: np.ndarray, chain: tuple[int, ...]) -> float:
        points = joints[np.asarray(chain)]
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        chain_length = float(segment_lengths.sum())
        if chain_length < 1e-6:
            raise ValueError("degenerate MANO finger chain")
        straightness = float(np.linalg.norm(points[-1] - points[0]) / chain_length)
        return float(np.clip((0.96 - straightness) / (0.96 - 0.30), 0.0, 1.0))

    @classmethod
    def _thumb_bend(cls, joints: np.ndarray) -> float:
        """Return functional thumb flexion from its palm-local proximal angle.

        MANO's thumb chain remains nearly straight while a real thumb closes, so
        chain straightness is a poor bend signal.  Instead, express the first
        thumb segment in an orthonormal palm frame.  The resulting angle uses
        only normalized relative vectors and dot products, making it invariant
        to rigid transforms and uniform scale.
        """

        across = joints[5] - joints[17]
        across_norm = float(np.linalg.norm(across))
        if across_norm < 1e-6:
            raise ValueError("degenerate MANO palm width axis")
        across = across / across_norm

        longitudinal = joints[9] - joints[0]
        longitudinal = longitudinal - float(np.dot(longitudinal, across)) * across
        longitudinal_norm = float(np.linalg.norm(longitudinal))
        if longitudinal_norm < 1e-6:
            raise ValueError("degenerate MANO palm longitudinal axis")
        longitudinal = longitudinal / longitudinal_norm

        thumb_proximal = joints[2] - joints[1]
        thumb_proximal_norm = float(np.linalg.norm(thumb_proximal))
        if thumb_proximal_norm < 1e-6:
            raise ValueError("degenerate MANO thumb proximal segment")
        thumb_proximal = thumb_proximal / thumb_proximal_norm

        bend_angle = float(
            np.arctan2(
                np.dot(thumb_proximal, longitudinal),
                np.dot(thumb_proximal, across),
            )
        )
        fraction = (
            bend_angle - cls._THUMB_BEND_OPEN_RADIANS
        ) / (cls._THUMB_BEND_CLOSED_RADIANS - cls._THUMB_BEND_OPEN_RADIANS)
        return float(np.clip(fraction, 0.0, 1.0))

    def retarget(self, detection: ManoDetection) -> RetargetOutput:
        joints = _validate_joints(detection.keypoints_3d_canonical)
        curls = [self._curl(joints, chain) for chain in self._CHAINS[:4]]
        thumb_bend = self._thumb_bend(joints)
        palm_width = float(np.linalg.norm(joints[5] - joints[17]))
        thumb_to_index = float(np.linalg.norm(joints[4] - joints[5])) / palm_width
        opposition = float(np.clip((1.55 - thumb_to_index) / 1.15, 0.0, 1.0))
        q_ranges = np.asarray([1.47, 1.47, 1.47, 1.47, 0.60, 1.308])
        raw_qpos = np.asarray(
            [*curls, thumb_bend, opposition], dtype=np.float32
        ) * q_ranges
        qpos = self._endpoint_filter.apply(self._filter.apply(raw_qpos))
        return RetargetOutput(
            qpos=qpos,
            hardware_targets=self.calibration.map_qpos(qpos),
            backend=self.name,
            raw_qpos=raw_qpos,
            raw_hardware_targets=self.calibration.map_qpos(raw_qpos),
        )

    def reset(self) -> None:
        self._filter.reset()
        self._endpoint_filter.reset()


def _validate_joints(joints: np.ndarray) -> np.ndarray:
    value = np.asarray(joints, dtype=np.float32)
    if value.shape != (21, 3):
        raise ValueError(f"expected MANO joints with shape (21, 3), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("MANO joints contain NaN or Inf")
    palm_width = float(np.linalg.norm(value[5] - value[17]))
    if not 0.02 <= palm_width <= 0.20:
        raise ValueError(f"implausible MANO palm width: {palm_width:.4f} m")
    return value
