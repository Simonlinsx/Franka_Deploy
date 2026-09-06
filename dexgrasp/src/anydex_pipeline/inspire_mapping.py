"""NumPy-only mapping from GraspNet two-finger poses to Inspire Hand-R poses.

This implements the geometry in the official
``InspireHandRGraspGroup.graspgroupTR_2_TR`` without importing ``ur_toolbox``,
Open3D, robot drivers, or model code.  Input and output poses use the caller's
frame unchanged; when fed representation results that frame is expected to be
``camera_color_optical_frame``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


CAMERA_FRAME = "camera_color_optical_frame"
GRASPNET_ARRAY_LEN = 17
INSPIRE_ARRAY_LEN = 23
MIN_GRASP_WIDTH = 0.025
MAX_GRASP_WIDTH = 0.10

# Official type number -> (JSON name, allowed two-finger width range in metres).
INSPIRE_GRASP_TYPES: dict[int, tuple[str, tuple[float, float]]] = {
    1: ("Ring", (0.0, 0.11)),
    2: ("Prismatic_2_Finger", (0.0, 0.11)),
    3: ("Prismatic_3_Finger", (0.0, 0.11)),
    4: ("Large_Diameter", (0.0, 0.11)),
    5: ("Medium_Wrap", (0.0, 0.11)),
    6: ("Tripod", (0.025, 0.10)),
    7: ("Sphere_3_Finger", (0.025, 0.10)),
    8: ("Distal_Type", (0.025, 0.10)),
}


@dataclass(frozen=True)
class InspireGraspBatch:
    """Mapped Inspire palm poses and actuator values.

    ``translations`` and ``rotation_matrices`` are the official raw palm/wrist
    pose before insertion depth is applied.  The official mesh visualizer moves
    this palm along the source two-finger approach axis by ``depths``; use
    ``translations_at_depth`` or ``pose_matrices(apply_depth=True)`` to obtain
    that visualized pose.

    Angle order is ``little, ring, middle, index, thumb_bend, thumb_rotate`` and
    values are the official Inspire register-scale commands (usually 0..1000),
    not radians.
    """

    scores: np.ndarray
    depths: np.ndarray
    grasp_types: np.ndarray
    rotation_matrices: np.ndarray
    translations: np.ndarray
    angles: np.ndarray
    object_ids: np.ndarray
    widths: np.ndarray
    approach_directions: np.ndarray
    frame_id: str = CAMERA_FRAME

    def __post_init__(self) -> None:
        n = np.asarray(self.scores).reshape(-1).shape[0]
        expected = {
            "depths": (n,),
            "grasp_types": (n,),
            "rotation_matrices": (n, 3, 3),
            "translations": (n, 3),
            "angles": (n, 6),
            "object_ids": (n,),
            "widths": (n,),
            "approach_directions": (n, 3),
        }
        for name, shape in expected.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    @property
    def translations_at_depth(self) -> np.ndarray:
        """Palm translations after the official insertion-depth visualization."""

        return self.translations + self.approach_directions * self.depths[:, None]

    def pose_matrices(self, *, apply_depth: bool = False) -> np.ndarray:
        """Return homogeneous palm transforms in ``frame_id``."""

        matrices = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(self), axis=0)
        matrices[:, :3, :3] = self.rotation_matrices
        matrices[:, :3, 3] = (
            self.translations_at_depth if apply_depth else self.translations
        )
        return matrices

    def to_official_array(self) -> np.ndarray:
        """Return the official 23-value InspireHandRGraspGroup layout."""

        return np.concatenate(
            [
                self.scores[:, None],
                self.depths[:, None],
                self.grasp_types[:, None].astype(np.float64),
                self.rotation_matrices.reshape(-1, 9),
                self.translations,
                self.angles,
                self.object_ids[:, None],
                self.widths[:, None],
            ],
            axis=1,
        )


def load_inspire_mapping(path: str | Path) -> dict[str, Any]:
    """Load ``width_12Dangle_6Dangle.json`` from a file or its directory."""

    mapping_path = Path(path).expanduser()
    if mapping_path.is_dir():
        mapping_path = mapping_path / "width_12Dangle_6Dangle.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Inspire mapping JSON not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Inspire mapping JSON root must be an object")
    return data


def _as_vector(
    value: float | Sequence[float] | np.ndarray | None,
    length: int,
    *,
    default: np.ndarray,
    name: str,
) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64).copy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(length, float(array), dtype=np.float64)
    array = array.reshape(-1)
    if array.shape != (length,):
        raise ValueError(f"{name} must be scalar or have shape ({length},)")
    return array


def _normalize_grasp_types(
    grasp_types: int | Sequence[int] | np.ndarray, length: int
) -> np.ndarray:
    values = np.asarray(grasp_types)
    if values.ndim == 0:
        values = np.full(length, values.item())
    values = values.reshape(-1)
    if values.shape != (length,):
        raise ValueError(f"grasp_types must be scalar or have shape ({length},)")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("grasp_types must be numeric integers in [1, 8]")
    numeric = values.astype(np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.round(numeric)).all():
        raise ValueError("grasp_types must be numeric integers in [1, 8]")
    normalized = numeric.astype(np.int64)
    unknown = sorted(set(normalized.tolist()) - set(INSPIRE_GRASP_TYPES))
    if unknown:
        raise ValueError(f"unsupported Inspire grasp type(s): {unknown}; expected 1..8")
    return normalized


def _mapping_data(mapping: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(mapping, Mapping):
        return mapping
    return load_inspire_mapping(mapping)


def map_two_finger_grasps(
    two_finger_grasps: np.ndarray,
    grasp_types: int | Sequence[int] | np.ndarray,
    mapping: str | Path | Mapping[str, Any],
    *,
    scores: float | Sequence[float] | np.ndarray | None = None,
    depth_offsets: float | Sequence[float] | np.ndarray | None = None,
    frame_id: str = CAMERA_FRAME,
) -> InspireGraspBatch:
    """Map GraspNet 17-value candidates to official Inspire palm poses.

    Args:
        two_finger_grasps: ``(N, 17)`` standard GraspNet arrays in one frame.
        grasp_types: Inspire semantic type(s), integer 1 through 8.  Type 0 is
            invalid even though the released random-selection branch can emit it.
        mapping: Parsed mapping object, JSON file, or directory containing
            ``width_12Dangle_6Dangle.json``.
        scores: Optional decision-model scores replacing representation scores.
        depth_offsets: Optional Inspire insertion-depth predictions added to the
            two-finger depth, matching the official robot script.
        frame_id: Name propagated to the output; geometry is frame-preserving.

    Returns:
        Raw Inspire palm poses, total depths, six actuator values, and the
        clamped/quantized source widths used for JSON lookup.
    """

    grasps = np.asarray(two_finger_grasps, dtype=np.float64)
    if grasps.ndim == 1:
        grasps = grasps[None, :]
    if grasps.ndim != 2 or grasps.shape[1] != GRASPNET_ARRAY_LEN:
        raise ValueError(f"two_finger_grasps must have shape (N, {GRASPNET_ARRAY_LEN})")
    if not np.isfinite(grasps[:, :16]).all():
        raise ValueError("two_finger_grasps contains non-finite pose values")

    count = grasps.shape[0]
    types = _normalize_grasp_types(grasp_types, count)
    data = _mapping_data(mapping)
    output_scores = _as_vector(
        scores, count, default=grasps[:, 0], name="scores"
    )
    offsets = _as_vector(
        depth_offsets,
        count,
        default=np.zeros(count, dtype=np.float64),
        name="depth_offsets",
    )

    source_widths = grasps[:, 1]
    source_rotations = grasps[:, 4:13].reshape(-1, 3, 3)
    source_translations = grasps[:, 13:16]
    palm_rotations = np.empty((count, 3, 3), dtype=np.float64)
    palm_translations = np.empty((count, 3), dtype=np.float64)
    angles = np.empty((count, 6), dtype=np.float64)
    mapped_widths = np.empty(count, dtype=np.float64)

    for index, grasp_type in enumerate(types):
        type_name, (type_minimum, type_maximum) = INSPIRE_GRASP_TYPES[int(grasp_type)]
        minimum = max(type_minimum, MIN_GRASP_WIDTH)
        maximum = min(type_maximum, MAX_GRASP_WIDTH)
        width = float(np.clip(source_widths[index], minimum, maximum))
        # Exact upstream convention: metres -> centimetres, rounded to 0.1 cm.
        width_key = str(np.round(width * 100.0, 1))

        try:
            entry = data[type_name][width_key]
            offset_translation = np.asarray(entry["translation"], dtype=np.float64)
            offset_rotation = np.asarray(entry["rotation"], dtype=np.float64)
            actuator_angles = np.asarray(entry["6d"], dtype=np.float64)
        except KeyError as exc:
            raise KeyError(
                f"mapping has no entry for Inspire type {grasp_type} "
                f"({type_name}) at width {width_key} cm"
            ) from exc

        if offset_translation.shape != (3,):
            raise ValueError(f"{type_name}/{width_key} translation must have shape (3,)")
        if offset_rotation.shape != (3, 3):
            raise ValueError(f"{type_name}/{width_key} rotation must have shape (3, 3)")
        if actuator_angles.shape != (6,):
            raise ValueError(f"{type_name}/{width_key} 6d must have shape (6,)")

        two_finger_transform = np.eye(4, dtype=np.float64)
        two_finger_transform[:3, :3] = source_rotations[index]
        two_finger_transform[:3, 3] = source_translations[index]
        inspire_to_two_finger = np.eye(4, dtype=np.float64)
        inspire_to_two_finger[:3, :3] = offset_rotation
        inspire_to_two_finger[:3, 3] = offset_translation

        # Official formula:
        # T_frame_inspire = T_frame_two_finger @ inv(T_inspire_two_finger)
        palm_transform = two_finger_transform @ np.linalg.inv(inspire_to_two_finger)
        palm_rotations[index] = palm_transform[:3, :3]
        palm_translations[index] = palm_transform[:3, 3]
        angles[index] = actuator_angles
        mapped_widths[index] = width

    return InspireGraspBatch(
        scores=output_scores,
        depths=grasps[:, 3] + offsets,
        grasp_types=types,
        rotation_matrices=palm_rotations,
        translations=palm_translations,
        angles=angles,
        object_ids=grasps[:, 16].copy(),
        widths=mapped_widths,
        approach_directions=source_rotations[:, :, 0].copy(),
        frame_id=frame_id,
    )


# A readable alias for callers that use the official singular terminology even
# when operating on a candidate batch.
map_two_finger_to_inspire = map_two_finger_grasps
