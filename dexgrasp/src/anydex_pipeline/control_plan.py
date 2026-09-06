"""Hardware-free planning for a staged Franka/Inspire grasp.

This module deliberately has no robot, serial, camera, GUI, or file-system
imports.  It only turns one validated :class:`VisualizationSnapshot` into an
immutable plan which a separately reviewed executor may consume.

Every pose follows the ``T_A_B`` convention: ``T_A_B`` maps column-vector
coordinates expressed in frame B into frame A.  In particular, the explicitly
commissioned hand mounting transform is ``T_EE_hand`` and therefore

``T_reference_EE = T_reference_hand @ inverse(T_EE_hand)``.

The adapter's 10 mm hand seating plane and its 17.8 mm total material envelope
are intentionally represented separately.  The 7.8 mm spigot enters the hand
socket; treating all 17.8 mm as a flange-to-hand offset would be a frame error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence, Tuple

import numpy as np

from .snapshot import (
    VisualizationSnapshot,
    has_complete_official_provenance,
    validate_snapshot,
)


_DEFAULT_ADAPTER_MESH = "V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.stl"


def validate_rigid_transform(value: np.ndarray, name: str = "transform") -> np.ndarray:
    """Return a validated copy of one finite, proper rigid transform.

    The check is intentionally stricter than merely accepting a homogeneous
    matrix: the bottom row must be exact within numerical tolerance, the
    rotation must be orthonormal, and reflections are rejected.
    """

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or infinity")
    if not np.allclose(
        matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-8, rtol=0.0
    ):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0
    ):
        raise ValueError(f"{name} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} rotation determinant must be +1, got {determinant}")
    return _readonly_copy(matrix)


def inverse_rigid_transform(T_A_B: np.ndarray) -> np.ndarray:
    """Return ``T_B_A`` for a validated ``T_A_B`` without a generic inverse."""

    transform = validate_rigid_transform(T_A_B, "T_A_B")
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -(inverse[:3, :3] @ transform[:3, 3])
    return validate_rigid_transform(inverse, "T_B_A")


@dataclass(frozen=True)
class CollisionCylinder:
    """A solid conservative cylinder, with local +Z as its long axis."""

    name: str
    radius_m: float
    length_m: float
    T_adapter_primitive: np.ndarray

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("collision primitive name must not be empty")
        _positive_finite(self.radius_m, "radius_m")
        _positive_finite(self.length_m, "length_m")
        object.__setattr__(
            self,
            "T_adapter_primitive",
            validate_rigid_transform(
                self.T_adapter_primitive, "T_adapter_primitive"
            ),
        )


@dataclass(frozen=True)
class AdapterMeshMetadata:
    """Expected geometry for an optional visualization mesh.

    The source name is provenance only; this pure module never reads it.  The
    enclosing bounds remain useful when no mesh loader is available and make
    it explicit that collision checking must not rely on triangle winding or
    internal holes in the printed part.
    """

    source_name: str
    expected_extents_m: Tuple[float, float, float]
    length_unit: str = "m"
    collision_authoritative: bool = False

    def __post_init__(self) -> None:
        if not str(self.source_name).strip():
            raise ValueError("adapter mesh source_name must not be empty")
        extents = tuple(float(value) for value in self.expected_extents_m)
        if len(extents) != 3 or not all(np.isfinite(extents)):
            raise ValueError("expected_extents_m must contain three finite values")
        if not all(value > 0.0 for value in extents):
            raise ValueError("expected_extents_m must be positive")
        if self.length_unit != "m":
            raise ValueError("adapter mesh length_unit must be 'm'")
        if not isinstance(self.collision_authoritative, (bool, np.bool_)):
            raise ValueError("adapter mesh collision_authoritative must be boolean")
        object.__setattr__(self, "expected_extents_m", extents)
        object.__setattr__(
            self, "collision_authoritative", bool(self.collision_authoritative)
        )


@dataclass(frozen=True)
class AdapterGeometry:
    """Confirmed FR3-to-RH56 adapter geometry in metres.

    Frame ``adapter`` is at the FR3 mounting plane, centered on the flange,
    with +Z pointing toward the Inspire hand.  Holes are deliberately ignored
    by the collision primitives, making the represented occupied volume
    conservative.
    """

    disk_diameter_m: float = 0.070
    disk_thickness_m: float = 0.010
    spigot_diameter_m: float = 0.0376
    spigot_protrusion_m: float = 0.0078
    total_height_m: float = 0.0178
    mount_plane_offset_m: float = 0.010
    mesh_source_name: str = _DEFAULT_ADAPTER_MESH

    def __post_init__(self) -> None:
        for name in (
            "disk_diameter_m",
            "disk_thickness_m",
            "spigot_diameter_m",
            "spigot_protrusion_m",
            "total_height_m",
            "mount_plane_offset_m",
        ):
            _positive_finite(getattr(self, name), name)
        if self.spigot_diameter_m > self.disk_diameter_m:
            raise ValueError("spigot_diameter_m must not exceed disk_diameter_m")
        expected_total = self.disk_thickness_m + self.spigot_protrusion_m
        if not np.isclose(
            self.total_height_m, expected_total, atol=1e-9, rtol=0.0
        ):
            raise ValueError(
                "total_height_m must equal disk_thickness_m + "
                "spigot_protrusion_m"
            )
        if not np.isclose(
            self.mount_plane_offset_m,
            self.disk_thickness_m,
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError(
                "mount_plane_offset_m is the disk top/seating plane and must "
                "equal disk_thickness_m"
            )
        if not str(self.mesh_source_name).strip():
            raise ValueError("mesh_source_name must not be empty")

    @property
    def collision_primitives(self) -> Tuple[CollisionCylinder, ...]:
        """Solid disk/spigot primitives plus one full enclosing cylinder."""

        disk = CollisionCylinder(
            name="adapter_disk_solid",
            radius_m=0.5 * self.disk_diameter_m,
            length_m=self.disk_thickness_m,
            T_adapter_primitive=_translation_z(0.5 * self.disk_thickness_m),
        )
        spigot = CollisionCylinder(
            name="adapter_spigot_solid",
            radius_m=0.5 * self.spigot_diameter_m,
            length_m=self.spigot_protrusion_m,
            T_adapter_primitive=_translation_z(
                self.disk_thickness_m + 0.5 * self.spigot_protrusion_m
            ),
        )
        envelope = CollisionCylinder(
            name="adapter_full_envelope",
            radius_m=0.5 * self.disk_diameter_m,
            length_m=self.total_height_m,
            T_adapter_primitive=_translation_z(0.5 * self.total_height_m),
        )
        return (disk, spigot, envelope)

    @property
    def mesh_metadata(self) -> AdapterMeshMetadata:
        return AdapterMeshMetadata(
            source_name=self.mesh_source_name,
            expected_extents_m=(
                self.disk_diameter_m,
                self.disk_diameter_m,
                self.total_height_m,
            ),
            collision_authoritative=False,
        )


@dataclass(frozen=True)
class ExecutionConfig:
    """Hardware-independent targets and distances for one staged grasp."""

    default_q: np.ndarray = field(
        default_factory=lambda: np.asarray(
            [0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, 0.0],
            dtype=np.float64,
        )
    )
    inspire_open_angles: np.ndarray = field(
        default_factory=lambda: np.full(6, 1000.0, dtype=np.float64)
    )
    pregrasp_distance_m: float = 0.10
    final_insertion_m: float = 0.0
    enable_thumb_preshape: bool = False

    def __post_init__(self) -> None:
        default_q = _finite_vector(self.default_q, 7, "default_q")
        open_angles = _finite_vector(
            self.inspire_open_angles, 6, "inspire_open_angles"
        )
        if np.any(open_angles < 0.0) or np.any(open_angles > 1000.0):
            raise ValueError("inspire_open_angles must be within [0, 1000]")
        _positive_finite(self.pregrasp_distance_m, "pregrasp_distance_m")
        insertion = float(self.final_insertion_m)
        if not np.isfinite(insertion) or insertion < 0.0:
            raise ValueError("final_insertion_m must be finite and non-negative")
        if not isinstance(self.enable_thumb_preshape, (bool, np.bool_)):
            raise ValueError("enable_thumb_preshape must be boolean")
        object.__setattr__(self, "default_q", default_q)
        object.__setattr__(self, "inspire_open_angles", open_angles)
        object.__setattr__(
            self, "enable_thumb_preshape", bool(self.enable_thumb_preshape)
        )
        object.__setattr__(self, "final_insertion_m", insertion)


class StageName(str, Enum):
    FRANKA_DEFAULT = "FRANKA_DEFAULT"
    INSPIRE_OPEN = "INSPIRE_OPEN"
    FRANKA_PREGRASP = "FRANKA_PREGRASP"
    FRANKA_GRASP = "FRANKA_GRASP"
    EEF_SETTLE_GATE = "EEF_SETTLE_GATE"
    THUMB_PRESHAPE = "THUMB_PRESHAPE"
    INSPIRE_CLOSE = "INSPIRE_CLOSE"


@dataclass(frozen=True)
class PlannedStage:
    """One command or verification gate in an ordered control plan."""

    name: StageName
    description: str
    franka_q: Optional[np.ndarray] = None
    T_reference_EE: Optional[np.ndarray] = None
    inspire_angles: Optional[np.ndarray] = None
    is_verification_gate: bool = False

    def __post_init__(self) -> None:
        try:
            stage_name = StageName(self.name)
        except ValueError as exc:
            raise ValueError(f"unknown control stage {self.name!r}") from exc
        if not str(self.description).strip():
            raise ValueError("planned stage description must not be empty")
        object.__setattr__(self, "name", stage_name)
        if self.franka_q is not None:
            object.__setattr__(
                self, "franka_q", _finite_vector(self.franka_q, 7, "franka_q")
            )
        if self.T_reference_EE is not None:
            object.__setattr__(
                self,
                "T_reference_EE",
                validate_rigid_transform(self.T_reference_EE, "T_reference_EE"),
            )
        if self.inspire_angles is not None:
            angles = _finite_vector(self.inspire_angles, 6, "inspire_angles")
            if np.any(angles < 0.0) or np.any(angles > 1000.0):
                raise ValueError("inspire_angles must be within [0, 1000]")
            object.__setattr__(self, "inspire_angles", angles)
        if not isinstance(self.is_verification_gate, (bool, np.bool_)):
            raise ValueError("is_verification_gate must be boolean")
        object.__setattr__(
            self, "is_verification_gate", bool(self.is_verification_gate)
        )


@dataclass(frozen=True)
class GraspExecutionPlan:
    """A complete immutable plan; it does not imply permission to execute."""

    reference_frame: str
    selected_index: int
    T_reference_hand: np.ndarray
    T_EE_hand: np.ndarray
    T_reference_EE_grasp: np.ndarray
    T_reference_EE_pregrasp: np.ndarray
    approach_reference: np.ndarray
    stages: Tuple[PlannedStage, ...]
    adapter: AdapterGeometry
    diagnostic: bool
    official_model: bool
    calibrated: bool
    mount_transform_commissioned: bool
    execution_eligible: bool
    execution_blockers: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.reference_frame != "robot_base":
            raise ValueError("control plan reference_frame must be 'robot_base'")
        if self.selected_index < 0:
            raise ValueError("selected_index must be non-negative")
        for name in (
            "T_reference_hand",
            "T_EE_hand",
            "T_reference_EE_grasp",
            "T_reference_EE_pregrasp",
        ):
            object.__setattr__(
                self, name, validate_rigid_transform(getattr(self, name), name)
            )
        approach = _finite_vector(
            self.approach_reference, 3, "approach_reference"
        )
        if not np.isclose(
            np.linalg.norm(approach), 1.0, atol=1e-6, rtol=0.0
        ):
            raise ValueError("approach_reference must have unit norm")
        object.__setattr__(self, "approach_reference", approach)
        stages = tuple(self.stages)
        if not stages:
            raise ValueError("control plan must contain stages")
        object.__setattr__(self, "stages", stages)
        for name in (
            "diagnostic",
            "official_model",
            "calibrated",
            "mount_transform_commissioned",
            "execution_eligible",
        ):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
            object.__setattr__(self, name, bool(value))
        blockers = tuple(str(item) for item in self.execution_blockers)
        if self.execution_eligible and blockers:
            raise ValueError("execution_eligible plan cannot contain blockers")
        if self.diagnostic and self.execution_eligible:
            raise ValueError("a diagnostic plan can never be execution eligible")
        object.__setattr__(self, "execution_blockers", blockers)


def build_control_plan(
    snapshot: VisualizationSnapshot,
    *,
    T_EE_hand: Optional[np.ndarray],
    mount_transform_commissioned: bool,
    config: Optional[ExecutionConfig] = None,
    adapter: Optional[AdapterGeometry] = None,
    allow_diagnostic: bool = False,
    selected_index: Optional[int] = None,
) -> GraspExecutionPlan:
    """Build a staged plan from the selected hand pose in ``snapshot``.

    ``allow_diagnostic`` authorizes *planning and visualization only*.  It can
    never make a geometric/diagnostic snapshot execution eligible.
    """

    validate_snapshot(snapshot)
    if snapshot.reference_frame != "robot_base":
        raise ValueError(
            "control planning requires snapshot.reference_frame='robot_base'"
        )
    grasps = snapshot.grasps
    if selected_index is None:
        selected = int(grasps.selected_index)
    else:
        if isinstance(selected_index, (bool, np.bool_)) or not isinstance(
            selected_index, (int, np.integer)
        ):
            raise ValueError("selected_index override must be an integer")
        selected = int(selected_index)
    if selected < 0 or selected >= grasps.count or grasps.count == 0:
        raise ValueError("snapshot has no selected grasp")
    if grasps.hand_poses is None:
        raise ValueError("selected grasp has no T_reference_hand pose")
    if grasps.hand_angles is None:
        raise ValueError("selected grasp has no six-axis Inspire close target")
    close_angles_all = np.asarray(grasps.hand_angles, dtype=np.float64)
    if close_angles_all.shape[1] != 6:
        raise ValueError("control planning requires exactly six Inspire axes")
    close_angles = _finite_vector(
        close_angles_all[selected], 6, "selected hand_angles"
    )
    if np.any(close_angles < 0.0) or np.any(close_angles > 1000.0):
        raise ValueError("selected hand_angles must be within [0, 1000]")

    diagnostic = is_diagnostic_snapshot(snapshot)
    if diagnostic and not allow_diagnostic:
        raise ValueError(
            "diagnostic/geometric snapshot refused; set allow_diagnostic=True "
            "for planning-only visualization"
        )
    if T_EE_hand is None:
        raise ValueError("T_EE_hand is required and must be explicitly commissioned")
    if not isinstance(mount_transform_commissioned, (bool, np.bool_)):
        raise ValueError("mount_transform_commissioned must be boolean")
    if not bool(mount_transform_commissioned):
        raise ValueError(
            "mount_transform_commissioned=True is required; adapter dimensions "
            "alone do not determine T_EE_hand"
        )
    if not isinstance(allow_diagnostic, (bool, np.bool_)):
        raise ValueError("allow_diagnostic must be boolean")

    active_config = config if config is not None else ExecutionConfig()
    active_adapter = adapter if adapter is not None else AdapterGeometry()
    if not isinstance(active_config, ExecutionConfig):
        raise TypeError("config must be an ExecutionConfig")
    if not isinstance(active_adapter, AdapterGeometry):
        raise TypeError("adapter must be an AdapterGeometry")

    T_reference_hand = validate_rigid_transform(
        np.asarray(grasps.hand_poses)[selected], "T_reference_hand"
    )
    T_EE_hand_valid = validate_rigid_transform(T_EE_hand, "T_EE_hand")
    T_reference_EE_grasp = validate_rigid_transform(
        T_reference_hand @ inverse_rigid_transform(T_EE_hand_valid),
        "T_reference_EE_grasp",
    )

    canonical_pose = validate_rigid_transform(
        np.asarray(grasps.canonical_poses)[selected], "T_reference_grasp"
    )
    approach_local = _finite_vector(
        grasps.approach_axis_local, 3, "approach_axis_local"
    )
    approach_reference = canonical_pose[:3, :3] @ approach_local
    approach_reference /= np.linalg.norm(approach_reference)
    approach_reference = _readonly_copy(approach_reference)

    # The snapshot hand pose already includes its predicted AnyDex depth.  The
    # upstream UR demo added a separate empirical 14 mm insertion during robot
    # execution; it is not a mount calibration.  Keep that term explicit and
    # zero by default until it is separately commissioned for this FR3 setup.
    if active_config.final_insertion_m:
        inserted = np.array(T_reference_EE_grasp, copy=True)
        inserted[:3, 3] += (
            approach_reference * active_config.final_insertion_m
        )
        T_reference_EE_grasp = validate_rigid_transform(
            inserted, "T_reference_EE_grasp_with_insertion"
        )

    T_reference_EE_pregrasp = np.array(T_reference_EE_grasp, copy=True)
    T_reference_EE_pregrasp[:3, 3] -= (
        approach_reference * active_config.pregrasp_distance_m
    )
    T_reference_EE_pregrasp = validate_rigid_transform(
        T_reference_EE_pregrasp, "T_reference_EE_pregrasp"
    )

    stages = [
        PlannedStage(
            StageName.INSPIRE_OPEN,
            "Command all six Inspire axes to the fully open target.",
            inspire_angles=active_config.inspire_open_angles,
        ),
        PlannedStage(
            StageName.FRANKA_DEFAULT,
            "After open-hand verification, move Franka to the reviewed "
            "collision-free default joint pose.",
            franka_q=active_config.default_q,
        ),
        PlannedStage(
            StageName.FRANKA_PREGRASP,
            "Move the EEF to the pregrasp pose opposite the canonical approach.",
            T_reference_EE=T_reference_EE_pregrasp,
        ),
        PlannedStage(
            StageName.FRANKA_GRASP,
            "Approach from pregrasp to the planned EEF grasp pose.",
            T_reference_EE=T_reference_EE_grasp,
        ),
        PlannedStage(
            StageName.EEF_SETTLE_GATE,
            "Verify EEF pose arrival and settling before any closing command.",
            T_reference_EE=T_reference_EE_grasp,
            is_verification_gate=True,
        ),
    ]
    if active_config.enable_thumb_preshape:
        thumb_preshape = np.array(active_config.inspire_open_angles, copy=True)
        thumb_preshape[5] = close_angles[5]
        stages.append(
            PlannedStage(
                StageName.THUMB_PRESHAPE,
                "After the EEF settle gate, preshape only the thumb rotation axis.",
                inspire_angles=thumb_preshape,
            )
        )
    stages.append(
        PlannedStage(
            StageName.INSPIRE_CLOSE,
            "After the EEF settle gate, command the selected six-axis grasp target.",
            inspire_angles=close_angles,
        )
    )

    official = is_official_snapshot(snapshot)
    calibrated = bool(str(snapshot.calibration_id).strip())
    blockers = []
    if diagnostic:
        blockers.append("snapshot is diagnostic/geometric and is planning-only")
    elif not official:
        blockers.append("snapshot is not identified as official AnyDexGrasp output")
    if not calibrated:
        blockers.append("snapshot has no calibration_id")
    if not bool(np.asarray(grasps.collision_checked)[selected]):
        blockers.append("selected grasp has no completed collision check")
    elif not bool(np.asarray(grasps.collision_free)[selected]):
        blockers.append("selected grasp is not marked collision-free")

    return GraspExecutionPlan(
        reference_frame=snapshot.reference_frame,
        selected_index=selected,
        T_reference_hand=T_reference_hand,
        T_EE_hand=T_EE_hand_valid,
        T_reference_EE_grasp=T_reference_EE_grasp,
        T_reference_EE_pregrasp=T_reference_EE_pregrasp,
        approach_reference=approach_reference,
        stages=tuple(stages),
        adapter=active_adapter,
        diagnostic=diagnostic,
        official_model=official,
        calibrated=calibrated,
        mount_transform_commissioned=True,
        execution_eligible=not blockers,
        execution_blockers=tuple(blockers),
    )


def plan_selected_grasp(*args, **kwargs) -> GraspExecutionPlan:
    """Compatibility spelling for :func:`build_control_plan`."""

    return build_control_plan(*args, **kwargs)


def is_diagnostic_snapshot(snapshot: VisualizationSnapshot) -> bool:
    """Conservatively recognize geometric/commissioning model provenance."""

    model = str(snapshot.model_name).strip().lower()
    markers = (
        "diagnostic",
        "geometric",
        "commissioning",
        "pca/obb",
        "not anydexgrasp",
        "not decision-model output",
    )
    return any(marker in model for marker in markers)


def is_official_snapshot(snapshot: VisualizationSnapshot) -> bool:
    """Return whether model identity and all released artifacts are proven."""

    model = str(snapshot.model_name).strip().lower()
    return bool(
        model
        and "anydexgrasp" in model
        and not is_diagnostic_snapshot(snapshot)
        and has_complete_official_provenance(snapshot)
    )


def eligible_candidate_indices(
    snapshot: VisualizationSnapshot,
    *,
    thumb_rotate_range: Optional[Sequence[float]] = None,
    require_collision_checked: bool = True,
    require_collision_free: bool = True,
) -> Tuple[int, ...]:
    """Rank eligible top-k candidates by score without trusting index zero.

    This is deliberately a pure selection API.  A caller can pass the returned
    index to ``build_control_plan(..., selected_index=index)`` after attaching
    its collision result and commissioned q6 range.
    """

    validate_snapshot(snapshot)
    for name, value in (
        ("require_collision_checked", require_collision_checked),
        ("require_collision_free", require_collision_free),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
    if require_collision_free and not require_collision_checked:
        raise ValueError(
            "require_collision_free=True also requires collision_checked"
        )
    q6_range = None
    if thumb_rotate_range is not None:
        q6_range = _finite_vector(thumb_rotate_range, 2, "thumb_rotate_range")
        if not 0.0 <= q6_range[0] <= q6_range[1] <= 1000.0:
            raise ValueError("thumb_rotate_range must lie within [0, 1000]")

    grasps = snapshot.grasps
    if grasps.hand_poses is None or grasps.hand_angles is None:
        return ()
    angles = np.asarray(grasps.hand_angles, dtype=np.float64)
    if angles.ndim != 2 or angles.shape != (grasps.count, 6):
        return ()
    checked = np.asarray(grasps.collision_checked, dtype=np.bool_)
    collision_free = np.asarray(grasps.collision_free, dtype=np.bool_)
    eligible = []
    for index in range(grasps.count):
        if require_collision_checked and not bool(checked[index]):
            continue
        if require_collision_free and not bool(collision_free[index]):
            continue
        if q6_range is not None and not (
            q6_range[0] <= angles[index, 5] <= q6_range[1]
        ):
            continue
        eligible.append(index)
    scores = np.asarray(grasps.scores, dtype=np.float64)
    return tuple(sorted(eligible, key=lambda index: (-scores[index], index)))


def select_eligible_candidate_index(
    snapshot: VisualizationSnapshot,
    **kwargs,
) -> int:
    """Return the best eligible top-k index, or ``-1`` if none qualifies."""

    eligible = eligible_candidate_indices(snapshot, **kwargs)
    return eligible[0] if eligible else -1


def _translation_z(z_m: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[2, 3] = float(z_m)
    return transform


def _finite_vector(value: Sequence[float], length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return _readonly_copy(array)


def _positive_finite(value: float, name: str) -> None:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    copied = np.array(value, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied


__all__ = [
    "AdapterGeometry",
    "AdapterMeshMetadata",
    "CollisionCylinder",
    "ExecutionConfig",
    "GraspExecutionPlan",
    "PlannedStage",
    "StageName",
    "build_control_plan",
    "eligible_candidate_indices",
    "inverse_rigid_transform",
    "is_diagnostic_snapshot",
    "is_official_snapshot",
    "plan_selected_grasp",
    "select_eligible_candidate_index",
    "validate_rigid_transform",
]
