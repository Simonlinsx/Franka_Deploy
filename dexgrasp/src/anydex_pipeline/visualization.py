"""Open3D geometry construction for AnyDex snapshots.

Open3D is imported lazily inside :func:`build_open3d_geometries`.  Importing
this module therefore never initializes a renderer and remains usable in a
model-only or headless process.  The builder creates CPU geometry only; it does
not create a window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .snapshot import VisualizationSnapshot, validate_snapshot


# The arguments are the validated snapshot and its non-negative selected grasp
# index.  A builder may return one legacy Open3D geometry, an iterable of them,
# or None.  Geometry must already be expressed in the snapshot reference frame.
SelectedHandLinkMeshBuilder = Callable[[VisualizationSnapshot, int], Any]


_VIRIDIS_STOPS = np.asarray(
    [
        [0.267004, 0.004874, 0.329415],
        [0.229739, 0.322361, 0.545706],
        [0.127568, 0.566949, 0.550556],
        [0.369214, 0.788888, 0.382914],
        [0.993248, 0.906157, 0.143936],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class VisualizationStyle:
    """Small, fixed-scale defaults suitable for a tabletop robot scene."""

    scene_brightness: float = 0.32
    scene_color_floor: float = 0.08
    object_color: Tuple[float, float, float] = (1.0, 0.12, 0.02)
    selected_color: Tuple[float, float, float] = (0.95, 0.98, 1.0)
    collision_rejected_color: Tuple[float, float, float] = (0.45, 0.45, 0.45)
    reference_frame_size_m: float = 0.08
    canonical_frame_size_m: float = 0.05
    hand_frame_size_m: float = 0.04
    approach_length_m: float = 0.05
    arrow_cylinder_radius_m: float = 0.0012
    arrow_cone_radius_m: float = 0.0025
    arrow_cone_fraction: float = 0.24
    origin_radius_m: float = 0.0025
    score_min: float = 0.0
    score_max: float = 1.0
    max_candidates: int = 10
    show_reference_frame: bool = True
    show_selected_hand_frame: bool = False
    show_collision_rejected: bool = False


@dataclass
class Open3DGeometryBundle:
    """Named Open3D geometry and numeric debug data for one snapshot."""

    scene_point_cloud: Any
    object_point_cloud: Any
    reference_frame: Optional[Any]
    grasp_origins: List[Any]
    approach_arrows: List[Any]
    selected_canonical_frame: Optional[Any]
    selected_hand_frame: Optional[Any]
    candidate_indices: np.ndarray
    candidate_colors: np.ndarray
    approach_segments: np.ndarray
    selected_hand_link_meshes: List[Any] = field(default_factory=list)

    def geometry_list(self) -> List[Any]:
        """Return geometries in a stable back-to-front insertion order."""

        geometries: List[Any] = [self.scene_point_cloud, self.object_point_cloud]
        if self.reference_frame is not None:
            geometries.append(self.reference_frame)
        geometries.extend(self.grasp_origins)
        geometries.extend(self.approach_arrows)
        if self.selected_canonical_frame is not None:
            geometries.append(self.selected_canonical_frame)
        if self.selected_hand_frame is not None:
            geometries.append(self.selected_hand_frame)
        geometries.extend(self.selected_hand_link_meshes)
        return geometries


def build_open3d_geometries(
    snapshot: VisualizationSnapshot,
    style: Optional[VisualizationStyle] = None,
    *,
    selected_hand_link_meshes: Optional[Any] = None,
    selected_hand_link_mesh_builder: Optional[
        SelectedHandLinkMeshBuilder
    ] = None,
) -> Open3DGeometryBundle:
    """Build scene/object/grasp geometry without creating an Open3D window.

    ``selected_hand_link_meshes`` accepts one already-built legacy Open3D
    geometry or an iterable of them.  Alternatively,
    ``selected_hand_link_mesh_builder`` is called as ``builder(snapshot,
    selected_index)``.  The two sources are mutually exclusive.  Returned or
    supplied geometry must already be expressed in ``snapshot.reference_frame``;
    this module deliberately does not implement hand FK or URDF loading.

    If the snapshot has no selected grasp, no selected-hand meshes are added
    and the callback is not invoked.  Construction remains CPU/headless: this
    function never creates a window.
    """

    validate_snapshot(snapshot)
    active_style = VisualizationStyle() if style is None else style
    _validate_style(active_style)
    _validate_selected_hand_mesh_sources(
        selected_hand_link_meshes, selected_hand_link_mesh_builder
    )

    # Deliberately delayed: snapshot I/O and pure math do not depend on Open3D.
    import open3d as o3d

    scene_points = np.asarray(snapshot.scene_points, dtype=np.float64)
    scene_rgb = np.asarray(snapshot.scene_colors, dtype=np.float64)
    scene_rgb = np.clip(
        scene_rgb * active_style.scene_brightness
        + active_style.scene_color_floor,
        0.0,
        1.0,
    )
    scene_cloud = _make_point_cloud(o3d, scene_points, scene_rgb)

    object_points = np.asarray(snapshot.object_points, dtype=np.float64)
    object_rgb = np.tile(
        np.asarray(active_style.object_color, dtype=np.float64),
        (len(object_points), 1),
    )
    object_cloud = _make_point_cloud(o3d, object_points, object_rgb)

    reference_frame = None
    if active_style.show_reference_frame:
        reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=active_style.reference_frame_size_m
        )

    grasps = snapshot.grasps
    candidate_indices = _visible_candidate_indices(
        np.asarray(grasps.scores, dtype=np.float64),
        np.asarray(grasps.collision_free, dtype=np.bool_),
        int(grasps.selected_index),
        active_style,
    )
    candidate_scores = np.asarray(grasps.scores, dtype=np.float64)[
        candidate_indices
    ]
    candidate_colors = scores_to_rgb(
        candidate_scores,
        score_min=active_style.score_min,
        score_max=active_style.score_max,
    )

    selected = int(grasps.selected_index)
    for visible_index, candidate_index in enumerate(candidate_indices):
        if not bool(np.asarray(grasps.collision_free)[candidate_index]):
            candidate_colors[visible_index] = np.asarray(
                active_style.collision_rejected_color, dtype=np.float64
            )
        if int(candidate_index) == selected:
            candidate_colors[visible_index] = np.asarray(
                active_style.selected_color, dtype=np.float64
            )

    origins: List[Any] = []
    arrows: List[Any] = []
    segments = np.empty((len(candidate_indices), 2, 3), dtype=np.float64)
    canonical_poses = np.asarray(grasps.canonical_poses, dtype=np.float64)
    local_axis = np.asarray(grasps.approach_axis_local, dtype=np.float64)

    for visible_index, candidate_index in enumerate(candidate_indices):
        pose = canonical_poses[int(candidate_index)]
        origin = pose[:3, 3]
        approach = pose[:3, :3] @ local_axis
        approach /= np.linalg.norm(approach)
        start = origin - active_style.approach_length_m * approach
        segments[visible_index, 0] = start
        segments[visible_index, 1] = origin
        color = candidate_colors[visible_index]

        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=active_style.origin_radius_m, resolution=10
        )
        sphere.translate(origin)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        origins.append(sphere)

        arrows.append(
            _make_approach_arrow(
                o3d,
                start=start,
                direction=approach,
                length=active_style.approach_length_m,
                color=color,
                style=active_style,
            )
        )

    selected_canonical_frame = None
    selected_hand_frame = None
    if selected >= 0:
        selected_canonical_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=active_style.canonical_frame_size_m
        )
        selected_canonical_frame.transform(canonical_poses[selected])
        if active_style.show_selected_hand_frame and grasps.hand_poses is not None:
            selected_hand_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=active_style.hand_frame_size_m
            )
            selected_hand_frame.transform(
                np.asarray(grasps.hand_poses, dtype=np.float64)[selected]
            )

    hand_link_meshes = _resolve_selected_hand_link_meshes(
        o3d,
        snapshot=snapshot,
        selected_index=selected,
        meshes=selected_hand_link_meshes,
        builder=selected_hand_link_mesh_builder,
    )

    return Open3DGeometryBundle(
        scene_point_cloud=scene_cloud,
        object_point_cloud=object_cloud,
        reference_frame=reference_frame,
        grasp_origins=origins,
        approach_arrows=arrows,
        selected_canonical_frame=selected_canonical_frame,
        selected_hand_frame=selected_hand_frame,
        selected_hand_link_meshes=hand_link_meshes,
        candidate_indices=candidate_indices,
        candidate_colors=candidate_colors,
        approach_segments=segments,
    )


def show_snapshot(
    snapshot: VisualizationSnapshot,
    window_name: str = "AnyDexGrasp scene/object/grasp poses",
    style: Optional[VisualizationStyle] = None,
    *,
    selected_hand_link_meshes: Optional[Any] = None,
    selected_hand_link_mesh_builder: Optional[
        SelectedHandLinkMeshBuilder
    ] = None,
) -> Open3DGeometryBundle:
    """Open an interactive Open3D window for an already-built snapshot.

    This is the only convenience entry point in this module that creates a
    window.  Callers that own a live Visualizer should instead use
    :func:`build_open3d_geometries` and manage geometry updates themselves.
    """

    if not isinstance(window_name, str) or not window_name.strip():
        raise ValueError("window_name must be a non-empty string")
    bundle = build_open3d_geometries(
        snapshot,
        style=style,
        selected_hand_link_meshes=selected_hand_link_meshes,
        selected_hand_link_mesh_builder=selected_hand_link_mesh_builder,
    )
    import open3d as o3d

    o3d.visualization.draw_geometries(
        bundle.geometry_list(), window_name=window_name
    )
    return bundle


def scores_to_rgb(
    scores: np.ndarray, *, score_min: float = 0.0, score_max: float = 1.0
) -> np.ndarray:
    """Map scores deterministically to a small, dependency-free viridis table."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"scores must have shape [K], got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores contains non-finite values")
    if not np.isfinite(score_min) or not np.isfinite(score_max):
        raise ValueError("score range must be finite")
    if score_max <= score_min:
        raise ValueError("score_max must be greater than score_min")
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.float64)

    normalized = np.clip(
        (values - float(score_min)) / float(score_max - score_min), 0.0, 1.0
    )
    scaled = normalized * (_VIRIDIS_STOPS.shape[0] - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.minimum(lower + 1, _VIRIDIS_STOPS.shape[0] - 1)
    alpha = (scaled - lower)[:, None]
    return (1.0 - alpha) * _VIRIDIS_STOPS[lower] + alpha * _VIRIDIS_STOPS[upper]


def rotation_from_z(direction: np.ndarray) -> np.ndarray:
    """Return a proper rotation that maps local ``+Z`` onto ``direction``."""

    target = np.asarray(direction, dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("direction must be a finite vector of shape (3,)")
    norm = float(np.linalg.norm(target))
    if norm <= 1e-12:
        raise ValueError("direction must be non-zero")
    target = target / norm
    source = np.asarray([0.0, 0.0, 1.0])
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine >= 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if cosine <= -1.0 + 1e-12:
        # Pi around +X maps +Z to -Z and keeps det(R)=+1.
        return np.diag([1.0, -1.0, -1.0]).astype(np.float64)

    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    axis = cross / sine
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def _visible_candidate_indices(
    scores: np.ndarray,
    collision_free: np.ndarray,
    selected_index: int,
    style: VisualizationStyle,
) -> np.ndarray:
    indices = np.arange(len(scores), dtype=np.int64)
    if not style.show_collision_rejected:
        indices = indices[collision_free]
    if len(indices):
        order = np.argsort(-scores[indices], kind="stable")
        indices = indices[order]
    indices = indices[: style.max_candidates]

    if selected_index >= 0 and selected_index not in indices:
        if len(indices) < style.max_candidates:
            indices = np.concatenate(
                [np.asarray([selected_index], dtype=np.int64), indices]
            )
        elif style.max_candidates > 0:
            indices = np.concatenate(
                [np.asarray([selected_index], dtype=np.int64), indices[:-1]]
            )
    return indices.astype(np.int64, copy=False)


def _make_point_cloud(o3d: Any, points: np.ndarray, colors: np.ndarray) -> Any:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def _validate_selected_hand_mesh_sources(
    meshes: Optional[Any],
    builder: Optional[SelectedHandLinkMeshBuilder],
) -> None:
    if meshes is not None and builder is not None:
        raise ValueError(
            "selected_hand_link_meshes and selected_hand_link_mesh_builder "
            "are mutually exclusive"
        )
    if builder is not None and not callable(builder):
        raise TypeError("selected_hand_link_mesh_builder must be callable")


def _resolve_selected_hand_link_meshes(
    o3d: Any,
    *,
    snapshot: VisualizationSnapshot,
    selected_index: int,
    meshes: Optional[Any],
    builder: Optional[SelectedHandLinkMeshBuilder],
) -> List[Any]:
    if selected_index < 0:
        return []

    source = builder(snapshot, selected_index) if builder is not None else meshes
    if source is None:
        return []

    geometry_type = o3d.geometry.Geometry
    if isinstance(source, geometry_type):
        resolved = [source]
    else:
        if isinstance(source, (str, bytes)):
            raise TypeError(
                "selected hand link meshes must be Open3D geometries, not text"
            )
        try:
            resolved = list(source)
        except TypeError as exc:
            raise TypeError(
                "selected hand link meshes must be an Open3D geometry or iterable"
            ) from exc

    for index, geometry in enumerate(resolved):
        if not isinstance(geometry, geometry_type):
            raise TypeError(
                "selected hand link mesh at index "
                f"{index} is not an open3d.geometry.Geometry"
            )
    return resolved


def _make_approach_arrow(
    o3d: Any,
    *,
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    color: np.ndarray,
    style: VisualizationStyle,
) -> Any:
    cone_height = length * style.arrow_cone_fraction
    cylinder_height = length - cone_height
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=style.arrow_cylinder_radius_m,
        cone_radius=style.arrow_cone_radius_m,
        cylinder_height=cylinder_height,
        cone_height=cone_height,
        resolution=16,
        cylinder_split=4,
        cone_split=1,
    )
    arrow.rotate(rotation_from_z(direction), center=(0.0, 0.0, 0.0))
    arrow.translate(np.asarray(start, dtype=np.float64))
    arrow.paint_uniform_color(np.asarray(color, dtype=np.float64))
    arrow.compute_vertex_normals()
    return arrow


def _validate_style(style: VisualizationStyle) -> None:
    if not isinstance(style, VisualizationStyle):
        raise TypeError("style must be VisualizationStyle")
    for name in (
        "reference_frame_size_m",
        "canonical_frame_size_m",
        "hand_frame_size_m",
        "approach_length_m",
        "arrow_cylinder_radius_m",
        "arrow_cone_radius_m",
        "origin_radius_m",
    ):
        value = float(getattr(style, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0.0 < float(style.arrow_cone_fraction) < 1.0:
        raise ValueError("arrow_cone_fraction must be in (0,1)")
    if not 0.0 <= float(style.scene_brightness) <= 1.0:
        raise ValueError("scene_brightness must be in [0,1]")
    if not 0.0 <= float(style.scene_color_floor) <= 1.0:
        raise ValueError("scene_color_floor must be in [0,1]")
    if isinstance(style.max_candidates, bool) or not isinstance(
        style.max_candidates, (int, np.integer)
    ):
        raise ValueError("max_candidates must be an integer")
    if int(style.max_candidates) < 0:
        raise ValueError("max_candidates must be non-negative")
    if not np.isfinite(style.score_min) or not np.isfinite(style.score_max):
        raise ValueError("score range must be finite")
    if float(style.score_max) <= float(style.score_min):
        raise ValueError("score_max must be greater than score_min")
    for name in ("object_color", "selected_color", "collision_rejected_color"):
        color = np.asarray(getattr(style, name), dtype=np.float64)
        if color.shape != (3,) or not np.all(np.isfinite(color)):
            raise ValueError(f"{name} must be a finite RGB triple")
        if np.any(color < 0.0) or np.any(color > 1.0):
            raise ValueError(f"{name} must be in [0,1]")
