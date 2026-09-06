"""Portable, pickle-free snapshots for dexterous-grasp visualization.

The snapshot contract deliberately keeps all display geometry in one named
reference frame.  Canonical grasp poses and hand/wrist poses are distinct:

* ``canonical_poses`` describe the contact-frame pose used by GraspNet-style
  proposals.  For AnyDexGrasp the canonical local ``+X`` axis is approach.
* ``hand_poses`` are optional, hand-specific wrist poses used for a hand mesh.

All transforms use the ``T_A_B`` convention: they map column-vector points
from frame B into frame A.  NPZ archives contain only numeric or Unicode
arrays and are always loaded with ``allow_pickle=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Optional, Union

import numpy as np


SCHEMA_VERSION = 2
LENGTH_UNIT = "m"
POSE_CONVENTION = "T_reference_grasp_maps_grasp_local_to_reference"


@dataclass(frozen=True)
class GraspCandidates:
    """Canonical grasp candidates plus optional hand-specific information."""

    canonical_poses: np.ndarray
    scores: np.ndarray
    type_ids: Optional[np.ndarray] = None
    collision_free: Optional[np.ndarray] = None
    collision_checked: Optional[np.ndarray] = None
    selected_index: int = -1
    approach_axis_local: np.ndarray = field(
        default_factory=lambda: np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    )
    hand_poses: Optional[np.ndarray] = None
    hand_angles: Optional[np.ndarray] = None
    widths_m: Optional[np.ndarray] = None
    depths_m: Optional[np.ndarray] = None
    source_indices: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        poses = np.asarray(self.canonical_poses)
        count = int(poses.shape[0]) if poses.ndim >= 1 else 0
        if self.type_ids is None:
            object.__setattr__(
                self, "type_ids", np.full(count, -1, dtype=np.int32)
            )
        if self.collision_free is None:
            object.__setattr__(
                self, "collision_free", np.zeros(count, dtype=np.bool_)
            )
        if self.collision_checked is None:
            object.__setattr__(
                self, "collision_checked", np.zeros(count, dtype=np.bool_)
            )

    @property
    def count(self) -> int:
        poses = np.asarray(self.canonical_poses)
        return int(poses.shape[0]) if poses.ndim >= 1 else 0


@dataclass(frozen=True)
class VisualizationSnapshot:
    """One synchronized scene/object frame and its grasp candidates."""

    scene_points: np.ndarray
    scene_colors: np.ndarray
    object_points: np.ndarray
    object_colors: np.ndarray
    grasps: GraspCandidates
    reference_frame: str
    T_reference_camera: np.ndarray
    frame_id: int
    timestamp_s: float
    calibration_id: str = ""
    camera_serial: str = ""
    scene_excludes_object: bool = True
    inference_points: Optional[np.ndarray] = None
    model_name: str = ""
    checkpoint_sha256: str = ""
    representation_checkpoint_sha256: str = ""
    decision_checkpoint_sha256s: tuple[str, ...] = ()
    official_source_commit: str = ""

    def __post_init__(self) -> None:
        # ``checkpoint_sha256`` was the v1 spelling.  Preserve it as a strict
        # alias while making the representation checkpoint explicit in v2.
        legacy = self.checkpoint_sha256
        representation = self.representation_checkpoint_sha256
        if not representation and legacy:
            object.__setattr__(self, "representation_checkpoint_sha256", legacy)
        elif representation and not legacy:
            object.__setattr__(self, "checkpoint_sha256", representation)
        object.__setattr__(
            self,
            "decision_checkpoint_sha256s",
            tuple(self.decision_checkpoint_sha256s),
        )


PathLike = Union[str, Path]


_V1_REQUIRED_KEYS = {
    "schema_version",
    "length_unit",
    "pose_convention",
    "reference_frame",
    "frame_id",
    "timestamp_s",
    "calibration_id",
    "camera_serial",
    "scene_excludes_object",
    "T_reference_camera",
    "scene_points",
    "scene_colors",
    "object_points",
    "object_colors",
    "canonical_grasp_poses",
    "grasp_scores",
    "grasp_type_ids",
    "grasp_collision_free",
    "selected_grasp_index",
    "approach_axis_local",
    "hand_poses",
    "hand_angles",
    "grasp_widths_m",
    "grasp_depths_m",
    "grasp_source_indices",
    "inference_points",
    "model_name",
    "checkpoint_sha256",
}

_V2_REQUIRED_KEYS = _V1_REQUIRED_KEYS | {
    "grasp_collision_checked",
    "representation_checkpoint_sha256",
    "decision_checkpoint_sha256s",
    "official_source_commit",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_ID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def validate_snapshot(snapshot: VisualizationSnapshot) -> None:
    """Raise ``ValueError`` when a snapshot violates the in-memory contract."""

    if not isinstance(snapshot, VisualizationSnapshot):
        raise TypeError("snapshot must be a VisualizationSnapshot")

    _validate_nonempty_text(snapshot.reference_frame, "reference_frame")
    _validate_text(snapshot.calibration_id, "calibration_id")
    _validate_text(snapshot.camera_serial, "camera_serial")
    _validate_text(snapshot.model_name, "model_name")
    _validate_text(snapshot.checkpoint_sha256, "checkpoint_sha256")
    _validate_text(
        snapshot.representation_checkpoint_sha256,
        "representation_checkpoint_sha256",
    )
    _validate_optional_sha256(snapshot.checkpoint_sha256, "checkpoint_sha256")
    _validate_optional_sha256(
        snapshot.representation_checkpoint_sha256,
        "representation_checkpoint_sha256",
    )
    if (
        snapshot.checkpoint_sha256
        and snapshot.representation_checkpoint_sha256
        and snapshot.checkpoint_sha256
        != snapshot.representation_checkpoint_sha256
    ):
        raise ValueError(
            "checkpoint_sha256 must equal representation_checkpoint_sha256"
        )
    decision_hashes = snapshot.decision_checkpoint_sha256s
    if not isinstance(decision_hashes, tuple):
        raise ValueError("decision_checkpoint_sha256s must be a tuple")
    if len(decision_hashes) not in (0, 8):
        raise ValueError(
            "decision_checkpoint_sha256s must be empty or contain exactly 8 hashes"
        )
    for index, digest in enumerate(decision_hashes):
        _validate_sha256(digest, f"decision_checkpoint_sha256s[{index}]")
    _validate_text(snapshot.official_source_commit, "official_source_commit")
    if snapshot.official_source_commit and not _GIT_OBJECT_ID_RE.fullmatch(
        snapshot.official_source_commit
    ):
        raise ValueError(
            "official_source_commit must be a lowercase 40- or 64-hex Git object ID"
        )

    if isinstance(snapshot.frame_id, (bool, np.bool_)) or not isinstance(
        snapshot.frame_id, (int, np.integer)
    ):
        raise ValueError("frame_id must be an integer")
    if int(snapshot.frame_id) < 0:
        raise ValueError("frame_id must be non-negative")
    timestamp = float(snapshot.timestamp_s)
    if not np.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError("timestamp_s must be finite and non-negative")
    if not isinstance(snapshot.scene_excludes_object, (bool, np.bool_)):
        raise ValueError("scene_excludes_object must be boolean")

    _validate_rigid_transform(snapshot.T_reference_camera, "T_reference_camera")
    scene_points = _validate_points(snapshot.scene_points, "scene_points")
    object_points = _validate_points(snapshot.object_points, "object_points")
    _validate_colors(snapshot.scene_colors, len(scene_points), "scene_colors")
    _validate_colors(snapshot.object_colors, len(object_points), "object_colors")
    if snapshot.inference_points is not None:
        _validate_points(snapshot.inference_points, "inference_points")

    _validate_grasps(snapshot.grasps)


def save_snapshot_npz(path: PathLike, snapshot: VisualizationSnapshot) -> Path:
    """Validate and save a compressed v2 NPZ without pickle-backed values."""

    validate_snapshot(snapshot)
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    grasps = snapshot.grasps
    count = grasps.count

    payload = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "length_unit": np.asarray(LENGTH_UNIT),
        "pose_convention": np.asarray(POSE_CONVENTION),
        "reference_frame": np.asarray(snapshot.reference_frame),
        "frame_id": np.asarray(snapshot.frame_id, dtype=np.int64),
        "timestamp_s": np.asarray(snapshot.timestamp_s, dtype=np.float64),
        "calibration_id": np.asarray(snapshot.calibration_id),
        "camera_serial": np.asarray(snapshot.camera_serial),
        "scene_excludes_object": np.asarray(
            snapshot.scene_excludes_object, dtype=np.bool_
        ),
        "T_reference_camera": np.asarray(
            snapshot.T_reference_camera, dtype=np.float64
        ),
        "scene_points": np.asarray(snapshot.scene_points, dtype=np.float32),
        "scene_colors": np.asarray(snapshot.scene_colors, dtype=np.float32),
        "object_points": np.asarray(snapshot.object_points, dtype=np.float32),
        "object_colors": np.asarray(snapshot.object_colors, dtype=np.float32),
        "canonical_grasp_poses": np.asarray(
            grasps.canonical_poses, dtype=np.float64
        ),
        "grasp_scores": np.asarray(grasps.scores, dtype=np.float32),
        "grasp_type_ids": np.asarray(grasps.type_ids, dtype=np.int32),
        "grasp_collision_free": np.asarray(
            grasps.collision_free, dtype=np.bool_
        ),
        "grasp_collision_checked": np.asarray(
            grasps.collision_checked, dtype=np.bool_
        ),
        "selected_grasp_index": np.asarray(
            grasps.selected_index, dtype=np.int32
        ),
        "approach_axis_local": np.asarray(
            grasps.approach_axis_local, dtype=np.float32
        ),
        "hand_poses": _optional_array(
            grasps.hand_poses, (0, 4, 4), np.float64
        ),
        "hand_angles": _optional_array(grasps.hand_angles, (0, 0), np.float32),
        "grasp_widths_m": _optional_array(grasps.widths_m, (0,), np.float32),
        "grasp_depths_m": _optional_array(grasps.depths_m, (0,), np.float32),
        "grasp_source_indices": _optional_array(
            grasps.source_indices, (0,), np.int64
        ),
        "inference_points": _optional_array(
            snapshot.inference_points, (0, 3), np.float32
        ),
        "model_name": np.asarray(snapshot.model_name),
        "checkpoint_sha256": np.asarray(snapshot.checkpoint_sha256),
        "representation_checkpoint_sha256": np.asarray(
            snapshot.representation_checkpoint_sha256
        ),
        "decision_checkpoint_sha256s": np.asarray(
            snapshot.decision_checkpoint_sha256s, dtype=np.str_
        ),
        "official_source_commit": np.asarray(snapshot.official_source_commit),
    }
    # File handles prevent NumPy from silently appending a second ``.npz``
    # suffix when callers intentionally choose another filename.
    with output.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return output


def load_snapshot_npz(path: PathLike) -> VisualizationSnapshot:
    """Load strict v1/v2 snapshots with ``allow_pickle=False``.

    Legacy v1 collision-free flags are deliberately downgraded because v1 did
    not record whether collision checking actually ran.
    """

    source = Path(path).expanduser()
    try:
        archive_context = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load snapshot {source}: {exc}") from exc

    try:
        with archive_context as archive:
            names = set(archive.files)
            if "schema_version" not in names:
                raise ValueError("snapshot is missing keys: ['schema_version']")
            version = _scalar_int(archive["schema_version"], "schema_version")
            if version == 1:
                required = _V1_REQUIRED_KEYS
            elif version == SCHEMA_VERSION:
                required = _V2_REQUIRED_KEYS
            else:
                raise ValueError(
                    f"unsupported schema_version={version}; expected 1 or {SCHEMA_VERSION}"
                )
            missing = sorted(required - names)
            unknown = sorted(names - required)
            if missing:
                raise ValueError(f"snapshot is missing keys: {missing}")
            if unknown:
                raise ValueError(
                    f"snapshot has unknown v{version} keys: {unknown}"
                )

            # Materialize every member while the archive is open.  Object arrays
            # fail here because allow_pickle=False, including in unexpected data.
            values = {name: np.asarray(archive[name]).copy() for name in names}
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("snapshot "):
            raise
        raise ValueError(f"invalid snapshot {source}: {exc}") from exc

    version = _scalar_int(values["schema_version"], "schema_version")
    if _scalar_text(values["length_unit"], "length_unit") != LENGTH_UNIT:
        raise ValueError(f"length_unit must be {LENGTH_UNIT!r}")
    if _scalar_text(values["pose_convention"], "pose_convention") != POSE_CONVENTION:
        raise ValueError(f"pose_convention must be {POSE_CONVENTION!r}")

    hand_poses = _none_if_empty(values["hand_poses"])
    hand_angles = _none_if_empty(values["hand_angles"])
    widths_m = _none_if_empty(values["grasp_widths_m"])
    depths_m = _none_if_empty(values["grasp_depths_m"])
    source_indices = _none_if_empty(values["grasp_source_indices"])
    inference_points = _none_if_empty(values["inference_points"])

    # v1 had no "collision check ran" bit and one official producer wrote
    # collision_free=True even while its metadata said not-run.  Loading must
    # remain possible, but those legacy flags cannot be elevated into evidence.
    collision_checked = (
        values["grasp_collision_checked"]
        if version >= 2
        else np.zeros_like(values["grasp_collision_free"], dtype=np.bool_)
    )
    collision_free = np.asarray(values["grasp_collision_free"], dtype=np.bool_)
    if version == 1:
        collision_free = collision_free & collision_checked

    grasps = GraspCandidates(
        canonical_poses=values["canonical_grasp_poses"],
        scores=values["grasp_scores"],
        type_ids=values["grasp_type_ids"],
        collision_free=collision_free,
        collision_checked=collision_checked,
        selected_index=_scalar_int(
            values["selected_grasp_index"], "selected_grasp_index"
        ),
        approach_axis_local=values["approach_axis_local"],
        hand_poses=hand_poses,
        hand_angles=hand_angles,
        widths_m=widths_m,
        depths_m=depths_m,
        source_indices=source_indices,
    )
    snapshot = VisualizationSnapshot(
        scene_points=values["scene_points"],
        scene_colors=values["scene_colors"],
        object_points=values["object_points"],
        object_colors=values["object_colors"],
        grasps=grasps,
        reference_frame=_scalar_text(values["reference_frame"], "reference_frame"),
        T_reference_camera=values["T_reference_camera"],
        frame_id=_scalar_int(values["frame_id"], "frame_id"),
        timestamp_s=_scalar_float(values["timestamp_s"], "timestamp_s"),
        calibration_id=_scalar_text(values["calibration_id"], "calibration_id"),
        camera_serial=_scalar_text(values["camera_serial"], "camera_serial"),
        scene_excludes_object=_scalar_bool(
            values["scene_excludes_object"], "scene_excludes_object"
        ),
        inference_points=inference_points,
        model_name=_scalar_text(values["model_name"], "model_name"),
        checkpoint_sha256=_scalar_text(
            values["checkpoint_sha256"], "checkpoint_sha256"
        ),
        representation_checkpoint_sha256=(
            _scalar_text(
                values["representation_checkpoint_sha256"],
                "representation_checkpoint_sha256",
            )
            if version >= 2
            else _scalar_text(values["checkpoint_sha256"], "checkpoint_sha256")
        ),
        decision_checkpoint_sha256s=(
            _text_vector(
                values["decision_checkpoint_sha256s"],
                "decision_checkpoint_sha256s",
            )
            if version >= 2
            else ()
        ),
        official_source_commit=(
            _scalar_text(values["official_source_commit"], "official_source_commit")
            if version >= 2
            else ""
        ),
    )
    validate_snapshot(snapshot)
    return snapshot


def _validate_grasps(grasps: GraspCandidates) -> None:
    if not isinstance(grasps, GraspCandidates):
        raise TypeError("grasps must be GraspCandidates")
    poses = np.asarray(grasps.canonical_poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            "canonical_poses must have shape [K,4,4], "
            f"got {poses.shape}"
        )
    if not np.all(np.isfinite(poses)):
        raise ValueError("canonical_poses contains non-finite values")
    count = len(poses)
    for index, pose in enumerate(poses):
        _validate_rigid_transform(pose, f"canonical_poses[{index}]")

    scores = _validate_vector(grasps.scores, count, "scores", finite=True)
    del scores
    _validate_integer_vector(grasps.type_ids, count, "type_ids")
    collision = np.asarray(grasps.collision_free)
    if collision.shape != (count,) or collision.dtype.kind != "b":
        raise ValueError(
            f"collision_free must be a boolean vector of shape ({count},)"
        )
    checked = np.asarray(grasps.collision_checked)
    if checked.shape != (count,) or checked.dtype.kind != "b":
        raise ValueError(
            f"collision_checked must be a boolean vector of shape ({count},)"
        )
    if np.any(collision & ~checked):
        raise ValueError(
            "collision_free=True requires collision_checked=True for that candidate"
        )

    if isinstance(grasps.selected_index, (bool, np.bool_)) or not isinstance(
        grasps.selected_index, (int, np.integer)
    ):
        raise ValueError("selected_index must be an integer")
    selected = int(grasps.selected_index)
    if selected < -1 or selected >= count:
        raise ValueError(
            f"selected_index={selected} is invalid for {count} candidates"
        )
    if count == 0 and selected != -1:
        raise ValueError("selected_index must be -1 when there are no candidates")

    axis = np.asarray(grasps.approach_axis_local, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("approach_axis_local must be a finite vector of shape (3,)")
    norm = float(np.linalg.norm(axis))
    if not np.isclose(norm, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(
            f"approach_axis_local must have unit norm, got {norm:.8f}"
        )

    if grasps.hand_poses is not None:
        hand_poses = np.asarray(grasps.hand_poses)
        if hand_poses.shape != (count, 4, 4):
            raise ValueError(
                f"hand_poses must have shape ({count},4,4), got {hand_poses.shape}"
            )
        for index, pose in enumerate(hand_poses):
            _validate_rigid_transform(pose, f"hand_poses[{index}]")
    if grasps.hand_angles is not None:
        angles = np.asarray(grasps.hand_angles)
        if angles.ndim != 2 or angles.shape[0] != count or angles.shape[1] == 0:
            raise ValueError(
                f"hand_angles must have shape ({count},J) with J>0, got {angles.shape}"
            )
        if not np.all(np.isfinite(angles)):
            raise ValueError("hand_angles contains non-finite values")

    _validate_optional_nonnegative_vector(grasps.widths_m, count, "widths_m")
    _validate_optional_nonnegative_vector(grasps.depths_m, count, "depths_m")
    if grasps.source_indices is not None:
        _validate_integer_vector(grasps.source_indices, count, "source_indices")


def _validate_points(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N,3], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_colors(values: np.ndarray, count: int, name: str) -> None:
    array = np.asarray(values)
    if array.shape != (count, 3):
        raise ValueError(f"{name} must have shape ({count},3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must be RGB values in [0,1]")


def _validate_rigid_transform(values: np.ndarray, name: str) -> None:
    transform = np.asarray(values, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4,4), got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(
        transform[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-6, rtol=0.0
    ):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.0):
        raise ValueError(f"{name} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(
            f"{name} rotation must have determinant +1, got {determinant:.8f}"
        )


def _validate_vector(
    values: np.ndarray, count: int, name: str, *, finite: bool
) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},), got {array.shape}")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_integer_vector(values: np.ndarray, count: int, name: str) -> None:
    array = _validate_vector(values, count, name, finite=False)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must have an integer dtype")


def _validate_optional_nonnegative_vector(
    values: Optional[np.ndarray], count: int, name: str
) -> None:
    if values is None:
        return
    array = _validate_vector(values, count, name, finite=False)
    # Upstream candidates use NaN as an explicit "unknown width/depth"
    # sentinel.  Infinity is never meaningful, and known values remain metric
    # non-negative lengths.
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} contains infinity")
    finite = np.isfinite(array)
    if np.any(array[finite] < 0.0):
        raise ValueError(f"{name} must be non-negative")


def _validate_text(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")


def _validate_sha256(value: str, name: str) -> None:
    _validate_text(value, name)
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be exactly 64 lowercase hex characters")


def _validate_optional_sha256(value: str, name: str) -> None:
    if value:
        _validate_sha256(value, name)


def has_complete_official_provenance(snapshot: VisualizationSnapshot) -> bool:
    """Return whether all released-model provenance fields are complete."""

    representation = snapshot.representation_checkpoint_sha256
    decisions = snapshot.decision_checkpoint_sha256s
    commit = snapshot.official_source_commit
    return bool(
        _SHA256_RE.fullmatch(representation)
        and len(decisions) == 8
        and all(_SHA256_RE.fullmatch(value) for value in decisions)
        and _GIT_OBJECT_ID_RE.fullmatch(commit)
    )


def _validate_nonempty_text(value: str, name: str) -> None:
    _validate_text(value, name)
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _optional_array(
    value: Optional[np.ndarray], empty_shape: tuple, dtype: np.dtype
) -> np.ndarray:
    if value is None:
        return np.empty(empty_shape, dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _none_if_empty(value: np.ndarray) -> Optional[np.ndarray]:
    return None if value.size == 0 else value


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a Unicode/string scalar")
    return str(array.item())


def _text_vector(value: np.ndarray, name: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a one-dimensional Unicode/string array")
    return tuple(str(item) for item in array.tolist())


def _scalar_int(value: np.ndarray, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer scalar")
    return int(array.item())


def _scalar_float(value: np.ndarray, name: str) -> float:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric scalar")
    return float(array.item())


def _scalar_bool(value: np.ndarray, name: str) -> bool:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean scalar")
    return bool(array.item())
