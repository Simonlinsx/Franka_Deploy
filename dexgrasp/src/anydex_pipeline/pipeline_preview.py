"""Hardware-free contracts for the live grasp-pipeline preview.

The preview intentionally consumes pose telemetry through a small JSON file
instead of opening a second Franka connection.  A control process may publish
that file atomically at a non-realtime boundary; this module only validates and
reads it.  Nothing in this module imports libfranka, a serial transport,
RealSense, or Open3D.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .control_plan import GraspExecutionPlan, validate_rigid_transform


PathLike = Union[str, Path]
POSE_STATE_SCHEMA_VERSION = 1
MAX_POSE_STATE_BYTES = 64 * 1024
# Permit only a tiny scheduler/filesystem ordering skew.  A timestamp farther
# into the future could otherwise be clamped to age zero forever and masquerade
# as fresh telemetry.
MAX_POSE_STATE_FUTURE_S = 0.05


@dataclass(frozen=True)
class PoseState:
    """One read-only robot/hand sample published outside the preview.

    ``hand_angles`` is the six-value RH56 ``ANGLE_ACT`` feedback vector in
    official register order (little, ring, middle, index, thumb bend, thumb
    rotate).  It is optional so older EEF-only publishers remain compatible.
    When present it is deliberately kept as integer register feedback; this
    module does not invent a six-to-twelve-joint interpolation.
    """

    reference_frame: str
    timestamp_unix_s: float
    T_reference_EE: np.ndarray
    target_T_reference_EE: Optional[np.ndarray] = None
    stage: str = "unknown"
    source: str = "external_read_only"
    sequence: int = 0
    hand_angles: Optional[Tuple[int, ...]] = None
    hand_angle_targets: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.reference_frame != "robot_base":
            raise ValueError("pose state reference_frame must be 'robot_base'")
        timestamp = float(self.timestamp_unix_s)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("pose state timestamp_unix_s must be finite and non-negative")
        if not str(self.stage).strip():
            raise ValueError("pose state stage must not be empty")
        if not str(self.source).strip():
            raise ValueError("pose state source must not be empty")
        if isinstance(self.sequence, (bool, np.bool_)) or not isinstance(
            self.sequence, (int, np.integer)
        ):
            raise ValueError("pose state sequence must be an integer")
        if int(self.sequence) < 0:
            raise ValueError("pose state sequence must be non-negative")
        object.__setattr__(self, "timestamp_unix_s", timestamp)
        object.__setattr__(
            self,
            "T_reference_EE",
            validate_rigid_transform(self.T_reference_EE, "T_reference_EE"),
        )
        if self.target_T_reference_EE is not None:
            object.__setattr__(
                self,
                "target_T_reference_EE",
                validate_rigid_transform(
                    self.target_T_reference_EE,
                    "pose state target.T_reference_EE",
                ),
            )
        object.__setattr__(self, "stage", str(self.stage))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "sequence", int(self.sequence))
        if self.hand_angles is not None:
            object.__setattr__(
                self,
                "hand_angles",
                _six_integer_registers(
                    self.hand_angles,
                    "pose state hand.angles",
                    minimum=0,
                ),
            )
        if self.hand_angle_targets is not None:
            object.__setattr__(
                self,
                "hand_angle_targets",
                _six_integer_registers(
                    self.hand_angle_targets,
                    "pose state hand.angle_targets",
                    minimum=-1,
                ),
            )

    def age_s(self, now_unix_s: Optional[float] = None) -> float:
        now = time.time() if now_unix_s is None else float(now_unix_s)
        if not np.isfinite(now):
            raise ValueError("now_unix_s must be finite")
        return max(0.0, now - self.timestamp_unix_s)


@dataclass(frozen=True)
class PoseError:
    """Translation and SO(3) geodesic error from actual to target."""

    position_m: float
    rotation_rad: float

    @property
    def position_mm(self) -> float:
        return 1000.0 * self.position_m

    @property
    def rotation_deg(self) -> float:
        return float(np.degrees(self.rotation_rad))


def _six_integer_registers(
    value: Any,
    name: str,
    *,
    minimum: int,
) -> Tuple[int, ...]:
    """Validate six JSON integer registers without accepting bool/float repair."""

    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ValueError("{} must contain six integers".format(name))
    result = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, np.integer)
        ):
            raise ValueError("{} must contain six integers".format(name))
        numeric = int(item)
        if numeric < int(minimum) or numeric > 1000:
            raise ValueError(
                "{} values must be in [{},1000]".format(name, int(minimum))
            )
        result.append(numeric)
    return tuple(result)


@dataclass(frozen=True)
class PipelinePreviewStage:
    """One high-level stage shown by the preview; never an execution permit."""

    name: str
    target_q: Optional[np.ndarray] = None
    T_reference_EE: Optional[np.ndarray] = None
    inspire_target6: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.name not in ("default", "pregrasp", "grasp", "close", "lift"):
            raise ValueError("unknown pipeline preview stage {!r}".format(self.name))
        populated = sum(
            value is not None
            for value in (self.target_q, self.T_reference_EE, self.inspire_target6)
        )
        if populated < 1:
            raise ValueError("preview stage must contain at least one target")
        if self.target_q is not None:
            q = np.asarray(self.target_q, dtype=np.float64)
            if q.shape != (7,) or not np.all(np.isfinite(q)):
                raise ValueError("preview target_q must contain seven finite values")
            object.__setattr__(self, "target_q", q.copy())
        if self.T_reference_EE is not None:
            object.__setattr__(
                self,
                "T_reference_EE",
                validate_rigid_transform(
                    self.T_reference_EE, "preview stage T_reference_EE"
                ),
            )
        if self.inspire_target6 is not None:
            values = np.asarray(self.inspire_target6, dtype=np.float64)
            if (
                values.shape != (6,)
                or not np.all(np.isfinite(values))
                or np.any(values < 0.0)
                or np.any(values > 1000.0)
                or not np.array_equal(values, np.rint(values))
            ):
                raise ValueError(
                    "preview inspire_target6 must contain six integers in [0, 1000]"
                )
            object.__setattr__(
                self, "inspire_target6", tuple(int(item) for item in values)
            )


@dataclass(frozen=True)
class PipelinePreviewContract:
    """Five-stage default→pregrasp→grasp→close→lift preview contract.

    ``hardware_execution_allowed`` is deliberately fixed to ``False``.  A
    visual preview cannot replace a loaded-tool collision audit, payload
    verification, or material approval.
    """

    selected_index: int
    lift_distance_m: float
    T_EE_hand: np.ndarray
    target_hand_pose: np.ndarray
    lift_hand_pose: np.ndarray
    stages: Tuple[PipelinePreviewStage, ...]
    hardware_blockers: Tuple[str, ...]
    hardware_execution_allowed: bool = False

    def __post_init__(self) -> None:
        distance = float(self.lift_distance_m)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("lift_distance_m must be finite and positive")
        if self.selected_index < 0:
            raise ValueError("selected_index must be non-negative")
        if self.hardware_execution_allowed is not False:
            raise ValueError("a preview contract can never authorize hardware execution")
        stages = tuple(self.stages)
        if tuple(stage.name for stage in stages) != (
            "default",
            "pregrasp",
            "grasp",
            "close",
            "lift",
        ):
            raise ValueError(
                "preview stages must be exactly default, pregrasp, grasp, close, lift"
            )
        blockers = tuple(str(item).strip() for item in self.hardware_blockers)
        if not blockers or any(not item for item in blockers):
            raise ValueError("preview contract must retain explicit hardware blockers")
        object.__setattr__(self, "lift_distance_m", distance)
        object.__setattr__(
            self, "T_EE_hand", validate_rigid_transform(self.T_EE_hand, "T_EE_hand")
        )
        object.__setattr__(
            self,
            "target_hand_pose",
            validate_rigid_transform(self.target_hand_pose, "target_hand_pose"),
        )
        object.__setattr__(
            self,
            "lift_hand_pose",
            validate_rigid_transform(self.lift_hand_pose, "lift_hand_pose"),
        )
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "hardware_blockers", blockers)

    @property
    def T_reference_EE_grasp(self) -> np.ndarray:
        return self.stages[2].T_reference_EE.copy()

    @property
    def T_reference_EE_lift(self) -> np.ndarray:
        return self.stages[4].T_reference_EE.copy()


def pose_error(T_actual: np.ndarray, T_target: np.ndarray) -> PoseError:
    """Return Euclidean translation and geodesic SO(3) orientation error."""

    actual = validate_rigid_transform(T_actual, "T_actual")
    target = validate_rigid_transform(T_target, "T_target")
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    relative = actual[:3, :3].T @ target[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return PoseError(position_m=position, rotation_rad=float(np.arccos(cosine)))


def pose_state_from_mapping(payload: Mapping[str, Any]) -> PoseState:
    """Validate a decoded pose-state object without repairing bad values."""

    if not isinstance(payload, Mapping):
        raise ValueError("pose state must be a JSON object")
    schema = payload.get("schema_version")
    if isinstance(schema, bool) or schema != POSE_STATE_SCHEMA_VERSION:
        raise ValueError(
            "pose state schema_version must be {}".format(POSE_STATE_SCHEMA_VERSION)
        )
    required = {"reference_frame", "timestamp_unix_s", "T_reference_EE"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError("pose state is missing {}".format(", ".join(missing)))
    target_pose = None
    if "target" in payload:
        target = payload["target"]
        if not isinstance(target, Mapping):
            raise ValueError("pose state target must be a JSON object")
        if "T_reference_EE" in target:
            target_pose = np.asarray(target["T_reference_EE"], dtype=np.float64)
    hand_angles = None
    hand_angle_targets = None
    if "hand" in payload:
        hand = payload["hand"]
        if not isinstance(hand, Mapping):
            raise ValueError("pose state hand must be a JSON object")
        if "angles" not in hand:
            raise ValueError("pose state hand is missing angles")
        hand_angles = _six_integer_registers(
            hand["angles"], "pose state hand.angles", minimum=0
        )
        if "angle_targets" in hand:
            hand_angle_targets = _six_integer_registers(
                hand["angle_targets"],
                "pose state hand.angle_targets",
                minimum=-1,
            )
    return PoseState(
        reference_frame=str(payload["reference_frame"]),
        timestamp_unix_s=float(payload["timestamp_unix_s"]),
        T_reference_EE=np.asarray(payload["T_reference_EE"], dtype=np.float64),
        target_T_reference_EE=target_pose,
        stage=str(payload.get("stage", "unknown")),
        source=str(payload.get("source", "external_read_only")),
        sequence=payload.get("sequence", 0),
        hand_angles=hand_angles,
        hand_angle_targets=hand_angle_targets,
    )


def load_pose_state_json(
    path: PathLike,
    *,
    max_age_s: Optional[float] = None,
    now_unix_s: Optional[float] = None,
) -> PoseState:
    """Load one strict, bounded JSON sample and optionally reject stale data."""

    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    if len(raw) > MAX_POSE_STATE_BYTES:
        raise ValueError("pose state exceeds {} bytes".format(MAX_POSE_STATE_BYTES))
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pose state is not valid UTF-8 JSON: {}".format(exc)) from exc
    state = pose_state_from_mapping(payload)
    if max_age_s is not None:
        maximum = float(max_age_s)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("max_age_s must be finite and positive")
        now = time.time() if now_unix_s is None else float(now_unix_s)
        if not np.isfinite(now):
            raise ValueError("now_unix_s must be finite")
        delta = now - state.timestamp_unix_s
        if delta < -MAX_POSE_STATE_FUTURE_S:
            raise ValueError(
                "pose state timestamp is in the future: skew={:.3f}s > {:.3f}s".format(
                    -delta, MAX_POSE_STATE_FUTURE_S
                )
            )
        age = max(0.0, delta)
        if age > maximum:
            raise ValueError(
                "pose state is stale: age={:.3f}s > {:.3f}s".format(age, maximum)
            )
    return state


def build_pipeline_preview_contract(
    plan: GraspExecutionPlan,
    *,
    lift_distance_m: float,
    adapter_material: str,
    low_speed_unloaded_only: bool,
) -> PipelinePreviewContract:
    """Build the five-stage visual contract without granting motion authority."""

    if not isinstance(plan, GraspExecutionPlan):
        raise TypeError("plan must be a GraspExecutionPlan")
    distance = float(lift_distance_m)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("lift_distance_m must be finite and positive")
    if not isinstance(low_speed_unloaded_only, (bool, np.bool_)):
        raise ValueError("low_speed_unloaded_only must be boolean")

    lift_pose = np.array(plan.T_reference_EE_grasp, dtype=np.float64, copy=True)
    # The reference frame is robot_base; +Z is the explicit vertical lift
    # direction.  This is only a displayed proposal until an audited joint
    # path binds the same endpoint and swept volume.
    lift_pose[:3, 3] += np.asarray([0.0, 0.0, distance], dtype=np.float64)
    target_hand = plan.T_reference_EE_grasp @ plan.T_EE_hand
    lift_hand = lift_pose @ plan.T_EE_hand

    close_stage = next(stage for stage in plan.stages if stage.name.value == "INSPIRE_CLOSE")
    default_stage = next(stage for stage in plan.stages if stage.name.value == "FRANKA_DEFAULT")
    blockers = [
        "preview-only lift is not bound to an installed-tool loaded-lift audit",
        "lift endpoint and continuous swept path have no execution-side joint binding",
        "grasp retention/contact and loaded payload have not been verified",
    ]
    if str(adapter_material).strip().upper() == "PLA" or bool(low_speed_unloaded_only):
        blockers.append("installed PLA adapter is approved only for low-speed unloaded checks")

    stages = (
        PipelinePreviewStage("default", target_q=default_stage.franka_q),
        PipelinePreviewStage("pregrasp", T_reference_EE=plan.T_reference_EE_pregrasp),
        PipelinePreviewStage("grasp", T_reference_EE=plan.T_reference_EE_grasp),
        PipelinePreviewStage(
            "close",
            T_reference_EE=plan.T_reference_EE_grasp,
            inspire_target6=tuple(int(round(item)) for item in close_stage.inspire_angles),
        ),
        PipelinePreviewStage("lift", T_reference_EE=lift_pose),
    )
    return PipelinePreviewContract(
        selected_index=int(plan.selected_index),
        lift_distance_m=distance,
        T_EE_hand=plan.T_EE_hand,
        target_hand_pose=target_hand,
        lift_hand_pose=lift_hand,
        stages=stages,
        hardware_blockers=tuple(blockers),
    )


__all__ = [
    "MAX_POSE_STATE_BYTES",
    "POSE_STATE_SCHEMA_VERSION",
    "PipelinePreviewContract",
    "PipelinePreviewStage",
    "PoseError",
    "PoseState",
    "build_pipeline_preview_contract",
    "load_pose_state_json",
    "pose_error",
    "pose_state_from_mapping",
]
