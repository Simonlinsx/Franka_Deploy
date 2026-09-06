"""Bounded, fail-soft recording of exact accepted policy I/O.

The control path only validates and copies accepted tick data into memory.
Stacking, normalization, compression, and every filesystem write are deferred
to :meth:`PolicyIORecorder.save`, which the deployment runtime calls only after
it has attempted to stop both devices.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Mapping, Optional

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
POLICY_IO_SCHEMA_VERSION = 1
_ACTION_DIM = 13
_ARM_DIM = 7
_HAND_DIM = 6
_REGISTER_TO_POLICY = np.asarray([5, 4, 3, 2, 1, 0], dtype=np.int64)


def default_policy_io_path(run_id: object) -> Path:
    """Return the standard non-overwriting policy-I/O artifact path."""

    value = str(run_id).strip()
    if not value or value in {".", ".."}:
        raise ValueError("run_id must be non-empty")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError("run_id must be one path-safe component")
    return WORKSPACE_ROOT / "dexgrasp" / "runs" / f"{value}_policy_io.npz"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("metadata floats must be finite")
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"metadata contains unsupported type {type(value).__name__}")


def _normalizer(
    value: object,
    *,
    dimensions: int,
    name: str,
    positive: bool = False,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    expected = (1,) * (dimensions - 1) + (result.shape[-1],) if result.ndim else ()
    if result.ndim != dimensions or result.shape != expected:
        raise ValueError(f"{name} must have singleton leading axes and rank {dimensions}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return result.copy()


def _vector(
    value: object,
    shape: tuple[int, ...],
    name: str,
    *,
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result.copy()


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    try:
        if float(value) != float(numeric):
            raise ValueError(f"{name} must be an integer")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if numeric < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return numeric


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_vector(
    value: Optional[object], shape: tuple[int, ...], name: str
) -> Optional[np.ndarray]:
    return None if value is None else _vector(value, shape, name)


@dataclass(frozen=True)
class _RawPolicyTick:
    logical_policy_step: int
    hardware_sequence: int
    startup_non_actuated: bool
    pointcloud_history_metric: np.ndarray
    pointcloud_valid_history: np.ndarray
    proprio_history_raw: np.ndarray
    previous_policy_action13: np.ndarray
    previous_ledger_action13: np.ndarray
    raw_model_action13: np.ndarray
    sent_action13: np.ndarray
    camera_frame_id: int
    pointcloud_source_frame_id: int
    pointcloud_status: str
    source_valid_points: int
    observation_realtime_s: float
    pointcloud_captured_realtime_s: float
    franka_state_captured_monotonic_s: float
    produced_monotonic_s: float
    measured_franka_q_rad: np.ndarray
    shaper_q_d_rad: Optional[np.ndarray]
    franka_target_q_rad: Optional[np.ndarray]
    rh56_target_register_order: Optional[np.ndarray]
    hardware_command_valid: bool
    hold_arm_target: bool


class PolicyIORecorder:
    """Collect exact accepted model histories without perturbing execution.

    ``record_tick`` is deliberately fail-soft: malformed diagnostic input or a
    full buffer latches a diagnostic string and returns ``False``.  It never
    performs I/O and never raises into the robot-control transaction.
    """

    def __init__(
        self,
        output_path: object,
        *,
        maximum_records: object,
        pointcloud_mean: object,
        pointcloud_std: object,
        proprio_mean: object,
        proprio_std: object,
        q_hand_close_rad: object,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        if self.output_path.suffix.lower() != ".npz":
            raise ValueError("policy-I/O output path must end in .npz")
        if self.output_path.exists():
            raise FileExistsError(
                f"policy-I/O output already exists: {self.output_path}"
            )
        self.maximum_records = _integer(
            maximum_records, "maximum_records", minimum=1
        )
        self.pointcloud_mean = _normalizer(
            pointcloud_mean,
            dimensions=4,
            name="pointcloud_mean",
        )
        self.pointcloud_std = _normalizer(
            pointcloud_std,
            dimensions=4,
            name="pointcloud_std",
            positive=True,
        )
        if self.pointcloud_mean.shape != self.pointcloud_std.shape:
            raise ValueError("pointcloud normalizer shapes differ")
        self.proprio_mean = _normalizer(
            proprio_mean,
            dimensions=3,
            name="proprio_mean",
        )
        self.proprio_std = _normalizer(
            proprio_std,
            dimensions=3,
            name="proprio_std",
            positive=True,
        )
        if self.proprio_mean.shape != self.proprio_std.shape:
            raise ValueError("proprio normalizer shapes differ")
        self.q_hand_close_rad = _vector(
            q_hand_close_rad, (_HAND_DIM,), "q_hand_close_rad"
        )
        if np.any(self.q_hand_close_rad <= 0.0):
            raise ValueError("q_hand_close_rad must be strictly positive")

        normalized_metadata = _jsonable({} if metadata is None else metadata)
        if not isinstance(normalized_metadata, dict):
            raise TypeError("metadata must be a mapping")
        self.metadata = normalized_metadata
        self._metadata_json = json.dumps(
            normalized_metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        configured_history = normalized_metadata.get(
            "history_length", normalized_metadata.get("history")
        )
        self._history_length: Optional[int] = None
        if configured_history is not None:
            self._history_length = _integer(
                configured_history, "metadata.history_length", minimum=1
            )
        configured_points = normalized_metadata.get("num_object_points", 128)
        self._num_object_points = _integer(
            configured_points, "metadata.num_object_points", minimum=1
        )

        self._records: list[_RawPolicyTick] = []
        self._record_error: Optional[str] = None
        self._dropped_records = 0
        self._saved = False
        self._saved_bytes: Optional[int] = None
        self._lock = threading.Lock()

    @property
    def record_error(self) -> Optional[str]:
        with self._lock:
            return self._record_error

    def _latch_record_error(self, exc: object) -> None:
        if self._record_error is None:
            if isinstance(exc, BaseException):
                self._record_error = f"{type(exc).__name__}: {exc}"
            else:
                self._record_error = str(exc).strip() or "unspecified recorder error"

    def record_tick(
        self,
        *,
        logical_policy_step: object,
        hardware_sequence: object,
        startup_non_actuated: object,
        pointcloud_history_metric: object,
        pointcloud_valid_history: object,
        proprio_history_raw: object,
        previous_policy_action13: object,
        previous_ledger_action13: object,
        raw_model_action13: object,
        sent_action13: object,
        camera_frame_id: object,
        pointcloud_source_frame_id: object,
        pointcloud_status: object,
        source_valid_points: object,
        observation_realtime_s: object,
        pointcloud_captured_realtime_s: object,
        franka_state_captured_monotonic_s: object,
        produced_monotonic_s: object,
        measured_franka_q_rad: object,
        shaper_q_d_rad: Optional[object],
        franka_target_q_rad: Optional[object],
        rh56_target_register_order: Optional[object],
        hardware_command_valid: object,
        hold_arm_target: object,
    ) -> bool:
        """Copy one already accepted logical tick into the bounded RAM buffer."""

        with self._lock:
            if self._record_error is not None or self._saved:
                self._dropped_records += 1
                return False
            if len(self._records) >= self.maximum_records:
                self._dropped_records += 1
                self._latch_record_error(
                    f"record capacity {self.maximum_records} was exceeded"
                )
                return False
            try:
                points = np.asarray(
                    pointcloud_history_metric, dtype=np.float32
                )
                validity = np.asarray(
                    pointcloud_valid_history, dtype=np.float32
                )
                proprio = np.asarray(proprio_history_raw, dtype=np.float32)
                expected_point_dim = int(self.pointcloud_mean.shape[-1])
                expected_proprio_dim = int(self.proprio_mean.shape[-1])
                if (
                    points.ndim != 3
                    or points.shape[1:] != (
                        self._num_object_points,
                        expected_point_dim,
                    )
                ):
                    raise ValueError(
                        "pointcloud_history_metric must have shape "
                        f"[H,{self._num_object_points},{expected_point_dim}]"
                    )
                history = int(points.shape[0])
                if self._history_length is None:
                    self._history_length = history
                if history != self._history_length:
                    raise ValueError(
                        "pointcloud history length differs from recorder contract"
                    )
                if validity.shape != (history, self._num_object_points):
                    raise ValueError(
                        "pointcloud_valid_history shape differs from point history"
                    )
                if proprio.shape != (history, expected_proprio_dim):
                    raise ValueError(
                        "proprio_history_raw shape differs from normalizer contract"
                    )
                if not (
                    np.all(np.isfinite(points))
                    and np.all(np.isfinite(validity))
                    and np.all(np.isfinite(proprio))
                ):
                    raise ValueError("policy histories must be finite")
                if np.any(validity < 0.0) or np.any(validity > 1.0):
                    raise ValueError("pointcloud validity must lie in [0,1]")

                status = str(pointcloud_status)
                if not status:
                    raise ValueError("pointcloud_status must be non-empty")
                rh56_target: Optional[np.ndarray] = None
                if rh56_target_register_order is not None:
                    raw_registers = _vector(
                        rh56_target_register_order,
                        (_HAND_DIM,),
                        "rh56_target_register_order",
                        dtype=np.dtype(np.float64),
                    )
                    if not np.all(raw_registers == np.rint(raw_registers)):
                        raise ValueError(
                            "rh56_target_register_order must contain integers"
                        )
                    if np.any(raw_registers < 0.0) or np.any(raw_registers > 1000.0):
                        raise ValueError(
                            "rh56_target_register_order must lie in [0,1000]"
                        )
                    rh56_target = raw_registers.astype(np.int64)

                record = _RawPolicyTick(
                    logical_policy_step=_integer(
                        logical_policy_step, "logical_policy_step", minimum=0
                    ),
                    hardware_sequence=_integer(
                        hardware_sequence, "hardware_sequence", minimum=-1
                    ),
                    startup_non_actuated=bool(startup_non_actuated),
                    pointcloud_history_metric=points.copy(),
                    pointcloud_valid_history=validity.copy(),
                    proprio_history_raw=proprio.copy(),
                    previous_policy_action13=_vector(
                        previous_policy_action13,
                        (_ACTION_DIM,),
                        "previous_policy_action13",
                    ),
                    previous_ledger_action13=_vector(
                        previous_ledger_action13,
                        (_ACTION_DIM,),
                        "previous_ledger_action13",
                    ),
                    raw_model_action13=_vector(
                        raw_model_action13,
                        (_ACTION_DIM,),
                        "raw_model_action13",
                    ),
                    sent_action13=_vector(
                        sent_action13, (_ACTION_DIM,), "sent_action13"
                    ),
                    camera_frame_id=_integer(
                        camera_frame_id, "camera_frame_id", minimum=0
                    ),
                    pointcloud_source_frame_id=_integer(
                        pointcloud_source_frame_id,
                        "pointcloud_source_frame_id",
                        minimum=0,
                    ),
                    pointcloud_status=status,
                    source_valid_points=_integer(
                        source_valid_points, "source_valid_points", minimum=0
                    ),
                    observation_realtime_s=_finite(
                        observation_realtime_s, "observation_realtime_s"
                    ),
                    pointcloud_captured_realtime_s=_finite(
                        pointcloud_captured_realtime_s,
                        "pointcloud_captured_realtime_s",
                    ),
                    franka_state_captured_monotonic_s=_finite(
                        franka_state_captured_monotonic_s,
                        "franka_state_captured_monotonic_s",
                    ),
                    produced_monotonic_s=_finite(
                        produced_monotonic_s, "produced_monotonic_s"
                    ),
                    measured_franka_q_rad=_vector(
                        measured_franka_q_rad,
                        (_ARM_DIM,),
                        "measured_franka_q_rad",
                    ),
                    shaper_q_d_rad=_optional_vector(
                        shaper_q_d_rad, (_ARM_DIM,), "shaper_q_d_rad"
                    ),
                    franka_target_q_rad=_optional_vector(
                        franka_target_q_rad,
                        (_ARM_DIM,),
                        "franka_target_q_rad",
                    ),
                    rh56_target_register_order=rh56_target,
                    hardware_command_valid=bool(hardware_command_valid),
                    hold_arm_target=bool(hold_arm_target),
                )
                self._records.append(record)
                return True
            except Exception as exc:
                self._dropped_records += 1
                self._latch_record_error(exc)
                return False

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return self._stats_locked()

    def _stats_locked(self) -> dict[str, Any]:
        startup = sum(int(record.startup_non_actuated) for record in self._records)
        return {
            "path": str(self.output_path),
            "schema_version": POLICY_IO_SCHEMA_VERSION,
            "maximum_records": self.maximum_records,
            # Short names are consumed by the runtime audit.  The explicit
            # names below remain useful when the recorder is used directly.
            "records": len(self._records),
            "error": self._record_error,
            "recorded_ticks": len(self._records),
            "startup_non_actuated_ticks": startup,
            "hardware_command_ticks": sum(
                int(record.hardware_command_valid) for record in self._records
            ),
            "dropped_records": self._dropped_records,
            "record_error": self._record_error,
            "saved": self._saved,
            "saved_bytes": self._saved_bytes,
        }

    def _payload_locked(self) -> dict[str, np.ndarray]:
        records = tuple(self._records)
        count = len(records)
        history = 0 if self._history_length is None else self._history_length
        point_dim = int(self.pointcloud_mean.shape[-1])
        proprio_dim = int(self.proprio_mean.shape[-1])

        def stack(name: str, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
            if not records:
                return np.empty((0,) + shape, dtype=dtype)
            return np.stack(
                [np.asarray(getattr(record, name), dtype=dtype) for record in records]
            )

        points = stack(
            "pointcloud_history_metric",
            (history, self._num_object_points, point_dim),
            np.dtype(np.float32),
        )
        valid = stack(
            "pointcloud_valid_history",
            (history, self._num_object_points),
            np.dtype(np.float32),
        )
        proprio = stack(
            "proprio_history_raw",
            (history, proprio_dim),
            np.dtype(np.float32),
        )
        normalized_points = (
            (points - self.pointcloud_mean) / self.pointcloud_std
        ).astype(np.float32, copy=False)
        normalized_proprio = (
            (proprio - self.proprio_mean) / self.proprio_std
        ).astype(np.float32, copy=False)

        def optional_stack(name: str) -> np.ndarray:
            result = np.full((count, _ARM_DIM), np.nan, dtype=np.float32)
            for index, record in enumerate(records):
                value = getattr(record, name)
                if value is not None:
                    result[index] = value
            return result

        rh56_targets = np.full((count, _HAND_DIM), -1, dtype=np.int64)
        for index, record in enumerate(records):
            if record.rh56_target_register_order is not None:
                rh56_targets[index] = record.rh56_target_register_order
        close_fraction = np.full((count, _HAND_DIM), np.nan, dtype=np.float32)
        commanded = np.all(rh56_targets >= 0, axis=1)
        if np.any(commanded):
            policy_order = rh56_targets[commanded][:, _REGISTER_TO_POLICY]
            close_fraction[commanded] = (
                np.float32(1.0)
                - policy_order.astype(np.float32) / np.float32(1000.0)
            )

        produced = np.asarray(
            [record.produced_monotonic_s for record in records], dtype=np.float64
        )
        time_s = produced.copy()
        if count:
            time_s -= produced[0]
        status_width = max(
            1, max((len(record.pointcloud_status) for record in records), default=1)
        )
        exact_previous = (
            proprio[:, -1, 54:67].copy()
            if history > 0 and proprio_dim >= 67
            else np.empty((count, _ACTION_DIM), dtype=np.float32)
        )

        payload: dict[str, np.ndarray] = {
            # Names shared with the simulator's policy_io.npz.
            "policy_step": np.asarray(
                [record.logical_policy_step for record in records], dtype=np.int64
            ),
            "time_s": time_s,
            "input_pointcloud_history_metric": points,
            "input_pointcloud_history_pre_temporal_alignment": points,
            "input_pointcloud_history_normalized": normalized_points,
            "input_pointcloud_valid_history": valid,
            "input_proprio_history_raw": proprio,
            "input_proprio_history_normalized": normalized_proprio,
            "input_previous_executed_action": exact_previous,
            "output_model_action": stack(
                "raw_model_action13", (_ACTION_DIM,), np.dtype(np.float32)
            ),
            "output_policy_action_after_sample_bias_clamp": stack(
                "raw_model_action13", (_ACTION_DIM,), np.dtype(np.float32)
            ),
            "output_action_sent_to_env": stack(
                "sent_action13", (_ACTION_DIM,), np.dtype(np.float32)
            ),
            "rh56_angle_set_register_order": rh56_targets,
            "rh56_close_fraction_policy_order": close_fraction,
            "constant__normalization_pointcloud_mean": self.pointcloud_mean.copy(),
            "constant__normalization_pointcloud_std": self.pointcloud_std.copy(),
            "constant__normalization_proprio_mean": self.proprio_mean.copy(),
            "constant__normalization_proprio_std": self.proprio_std.copy(),
            "constant__rh56_semantic_close_rad_policy_order": (
                self.q_hand_close_rad.copy()
            ),
            # Deployment-only provenance.  These names intentionally cannot be
            # mistaken for simulator ground truth.
            "real_hardware_sequence": np.asarray(
                [record.hardware_sequence for record in records], dtype=np.int64
            ),
            "real_startup_non_actuated": np.asarray(
                [record.startup_non_actuated for record in records], dtype=np.bool_
            ),
            "real_hardware_command_valid": np.asarray(
                [record.hardware_command_valid for record in records], dtype=np.bool_
            ),
            "real_hold_arm_target": np.asarray(
                [record.hold_arm_target for record in records], dtype=np.bool_
            ),
            "real_camera_frame_id": np.asarray(
                [record.camera_frame_id for record in records], dtype=np.int64
            ),
            "real_pointcloud_source_frame_id": np.asarray(
                [record.pointcloud_source_frame_id for record in records],
                dtype=np.int64,
            ),
            "real_pointcloud_status": np.asarray(
                [record.pointcloud_status for record in records],
                dtype=f"<U{status_width}",
            ),
            "real_source_valid_points": np.asarray(
                [record.source_valid_points for record in records], dtype=np.int64
            ),
            "real_observation_realtime_s": np.asarray(
                [record.observation_realtime_s for record in records],
                dtype=np.float64,
            ),
            "real_pointcloud_captured_realtime_s": np.asarray(
                [record.pointcloud_captured_realtime_s for record in records],
                dtype=np.float64,
            ),
            "real_franka_state_captured_monotonic_s": np.asarray(
                [record.franka_state_captured_monotonic_s for record in records],
                dtype=np.float64,
            ),
            "real_produced_monotonic_s": produced,
            "real_measured_franka_q_rad": stack(
                "measured_franka_q_rad", (_ARM_DIM,), np.dtype(np.float32)
            ),
            "real_shaper_q_d_rad": optional_stack("shaper_q_d_rad"),
            "real_franka_target_q_rad": optional_stack("franka_target_q_rad"),
            "real_previous_policy_action13": stack(
                "previous_policy_action13", (_ACTION_DIM,), np.dtype(np.float32)
            ),
            "real_previous_ledger_executed_action13": stack(
                "previous_ledger_action13", (_ACTION_DIM,), np.dtype(np.float32)
            ),
            "constant__policy_io_schema_version": np.asarray(
                POLICY_IO_SCHEMA_VERSION, dtype=np.int64
            ),
            "constant__metadata_json": np.asarray(self._metadata_json),
            "constant__checkpoint_sha256": np.asarray(
                str(self.metadata.get("checkpoint_sha256", ""))
            ),
            "constant__recording_contract": np.asarray(
                "accepted_ticks_raw_copy_in_control_normalize_and_write_after_stop_v1"
            ),
        }
        for name, array in payload.items():
            if np.asarray(array).dtype == object:
                raise TypeError(f"policy-I/O field {name} cannot use object dtype")
        return payload

    def save(self) -> dict[str, Any]:
        """Normalize and atomically publish the buffered archive once."""

        with self._lock:
            if self._saved:
                return self._stats_locked()
            if self.output_path.exists():
                raise FileExistsError(
                    f"policy-I/O output already exists: {self.output_path}"
                )
            payload = self._payload_locked()
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=self.output_path.stem + ".",
                suffix=".npz",
                dir=self.output_path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            try:
                np.savez_compressed(temporary, **payload)
                # Hard-link publication is atomic and, unlike Path.replace,
                # cannot overwrite a file created by another process.
                os.link(temporary, self.output_path)
                temporary.unlink()
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            self._saved = True
            self._saved_bytes = self.output_path.stat().st_size
            return self._stats_locked()


__all__ = [
    "POLICY_IO_SCHEMA_VERSION",
    "PolicyIORecorder",
    "default_policy_io_path",
]
