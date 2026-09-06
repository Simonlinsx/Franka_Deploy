"""Replayable preprocessing for an installed FR3/V7/RH56 depth scene.

The filter separates two classes of points that must not be treated as static
workspace obstacles:

* returns from the robot, V7 adapter and open RH56 at the capture joint state;
* returns belonging to the bound grasp object cloud.

Installed-tool seed returns are classified by exact HPP-FCL triangle meshes and
conservative point cubes.  A second stage may remove calibration residuals only
when point-to-mesh distance, same-component colour palette, and bounded 3-D
connectivity all agree; its model-distance cap is derived from the bound object
alignment and has a non-configurable 20 mm ceiling.  Object returns use a
bounded nearest-neighbour test inside the object's expanded AABB.  All removed
indices, parameters, and hashes are returned, making the result replayable.
The result remains non-authoritative for unseen camera space and never
authorizes motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .hppfcl_installed_tool_backend import (
    HppFclInstalledToolBackend,
    InstalledReturnClassification,
    InstalledReturnNeighborhood,
)


def _array_sha256(value: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    shape = ",".join(str(item) for item in array.shape)
    digest = hashlib.sha256()
    digest.update(("dtype={};shape={};".format(dtype, shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _points(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise ValueError("{} must be a non-empty (N,3) array".format(name))
    if not np.all(np.isfinite(points)):
        raise ValueError("{} contains NaN or infinity".format(name))
    return points


def _colors(value: np.ndarray, point_count: int) -> np.ndarray:
    colors = np.asarray(value, dtype=np.float64)
    if colors.shape != (point_count, 3) or not np.all(np.isfinite(colors)):
        raise ValueError("scene_colors_base must be a finite (N,3) array")
    if np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ValueError("scene_colors_base must be normalized to [0,1]")
    return colors


def _installed_label_family(label: str) -> str:
    value = str(label)
    if value.startswith("FR3_visual_link"):
        return "FR3_visual"
    if value.startswith("RH56_Link"):
        return "RH56"
    if value == "V7_adapter":
        return "V7_adapter"
    raise ValueError("unknown installed-return reason label: {}".format(value))


def bounded_residual_self_returns(
    scene_points_base: np.ndarray,
    scene_colors_base: np.ndarray,
    exact_seed_indices: np.ndarray,
    exact_seed_reason_labels: Sequence[str],
    neighborhood: InstalledReturnNeighborhood,
    *,
    connectivity_radius_m: float = 0.010,
    palette_color_max_l2: float = 0.18,
    maximum_geodesic_m: float = 0.040,
) -> Tuple[np.ndarray, Mapping[str, Any]]:
    """Grow exact mesh-intersection seeds under three simultaneous bounds.

    A residual point is removed only when it is (1) within the HPP-FCL
    point-to-mesh cap carried by ``neighborhood``, (2) appearance-compatible
    with an original exact-intersection seed assigned to the same installed
    component family (FR3 shell, V7, or RH56), and (3) connected to a
    same-family seed through short 3-D edges with bounded accumulated length.
    Requiring a per-family appearance palette
    preserves black/white material boundaries on RH56 while preventing a
    nearby differently-coloured or disconnected obstacle from being erased.
    """

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("SciPy cKDTree is required for residual self filtering") from exc

    points = _points(scene_points_base, "scene_points_base")
    colors = _colors(scene_colors_base, len(points))
    seeds = np.asarray(exact_seed_indices, dtype=np.int64)
    if seeds.ndim != 1 or np.any(seeds < 0) or np.any(seeds >= len(points)):
        raise ValueError("exact_seed_indices must index scene_points_base")
    if not np.array_equal(seeds, np.unique(seeds)):
        raise ValueError("exact_seed_indices must be sorted and unique")
    seed_labels = tuple(str(item) for item in exact_seed_reason_labels)
    if len(seed_labels) != len(seeds) or any(not item for item in seed_labels):
        raise ValueError("exact seed indices/reason labels are inconsistent")
    candidates = np.asarray(neighborhood.candidate_indices, dtype=np.int64)
    if np.any(candidates >= len(points)):
        raise ValueError("installed neighborhood index is outside the scene")
    for value, name, lower, upper in (
        (connectivity_radius_m, "connectivity_radius_m", 0.002, 0.010),
        (palette_color_max_l2, "palette_color_max_l2", 0.01, 0.18),
        (maximum_geodesic_m, "maximum_geodesic_m", 0.002, 0.040),
    ):
        number = float(value)
        if not np.isfinite(number) or not lower <= number <= upper:
            raise ValueError("{} must be in [{},{}]".format(name, lower, upper))
    if len(seeds) == 0 or len(candidates) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, {
            "exact_seed_count": int(len(seeds)),
            "model_neighborhood_candidate_count": int(len(candidates)),
            "appearance_palette_candidate_count": 0,
            "connected_candidate_count": 0,
            "residual_removed_count": 0,
            "residual_removed_indices_sha256": _array_sha256(empty, "<i8"),
            "residual_removed_surface_distance_min_m": None,
            "residual_removed_surface_distance_max_m": None,
            "reason_counts": {},
        }

    seed_component_array = np.asarray(
        [_installed_label_family(item) for item in seed_labels], dtype=object
    )
    candidate_component_array = np.asarray(
        [_installed_label_family(item) for item in neighborhood.reason_labels],
        dtype=object,
    )
    palette_compatible = np.zeros(len(candidates), dtype=np.bool_)
    for component in sorted(set(str(item) for item in candidate_component_array)):
        seed_local = np.flatnonzero(seed_component_array == component)
        candidate_local = np.flatnonzero(candidate_component_array == component)
        if len(seed_local) == 0 or len(candidate_local) == 0:
            continue
        palette_tree = cKDTree(colors[seeds[seed_local]])
        color_distances, _ = palette_tree.query(colors[candidates[candidate_local]], k=1)
        palette_compatible[candidate_local] = (
            color_distances <= float(palette_color_max_l2)
        )

    accepted_candidates = candidates[palette_compatible]
    node_indices = np.unique(np.concatenate((seeds, accepted_candidates))).astype(
        np.int64
    )
    node_lookup = {int(original): local for local, original in enumerate(node_indices)}
    labels_by_original = {
        int(original): str(component)
        for original, component in zip(seeds, seed_component_array)
    }
    accepted_set = set(int(item) for item in accepted_candidates)
    labels_by_original.update(
        {
            int(original): str(component)
            for original, component in zip(candidates, candidate_component_array)
            if int(original) in accepted_set
        }
    )
    node_tree = cKDTree(points[node_indices])
    adjacency = [[] for _ in range(len(node_indices))]
    for left, right in node_tree.query_pairs(float(connectivity_radius_m)):
        if labels_by_original[int(node_indices[left])] == labels_by_original[int(node_indices[right])]:
            distance = float(
                np.linalg.norm(points[node_indices[left]] - points[node_indices[right]])
            )
            adjacency[left].append((right, distance))
            adjacency[right].append((left, distance))

    geodesic = np.full(len(node_indices), np.inf, dtype=np.float64)
    queue = []
    for original in seeds:
        local = node_lookup.get(int(original))
        if local is not None:
            geodesic[local] = 0.0
            heapq.heappush(queue, (0.0, local))
    while queue:
        distance, left = heapq.heappop(queue)
        if distance != geodesic[left] or distance > float(maximum_geodesic_m):
            continue
        for right, edge in adjacency[left]:
            proposed = distance + edge
            if (
                proposed <= float(maximum_geodesic_m) + 1e-12
                and proposed < geodesic[right]
            ):
                geodesic[right] = proposed
                heapq.heappush(queue, (proposed, right))

    exact_set = set(int(item) for item in seeds)
    connected = {
        int(node_indices[local])
        for local in np.flatnonzero(np.isfinite(geodesic))
        if int(node_indices[local]) not in exact_set
    }
    residual = np.asarray(
        sorted(connected.intersection(int(item) for item in accepted_candidates)),
        dtype=np.int64,
    )
    candidate_to_local = {
        int(original): local for local, original in enumerate(candidates)
    }
    residual_distances = np.asarray(
        [
            neighborhood.surface_distances_m[candidate_to_local[int(item)]]
            for item in residual
        ],
        dtype=np.float64,
    )
    residual_labels = [
        neighborhood.reason_labels[candidate_to_local[int(item)]] for item in residual
    ]
    reason_counts: Dict[str, int] = {}
    for label in residual_labels:
        reason_counts[str(label)] = reason_counts.get(str(label), 0) + 1
    evidence = {
        "exact_seed_count": int(len(seeds)),
        "model_neighborhood_candidate_count": int(len(candidates)),
        "appearance_palette_candidate_count": int(
            np.count_nonzero(palette_compatible)
        ),
        "connected_candidate_count": int(len(connected)),
        "residual_removed_count": int(len(residual)),
        "residual_removed_indices_sha256": _array_sha256(residual, "<i8"),
        "residual_removed_surface_distance_min_m": (
            float(np.min(residual_distances)) if len(residual_distances) else None
        ),
        "residual_removed_surface_distance_max_m": (
            float(np.max(residual_distances)) if len(residual_distances) else None
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    return residual, evidence


@dataclass(frozen=True)
class InstalledSceneFilterResult:
    filtered_scene_points_base: np.ndarray
    kept_original_indices: np.ndarray
    installed_return_indices: np.ndarray
    object_return_indices: np.ndarray
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        points = _points(self.filtered_scene_points_base, "filtered_scene_points_base")
        kept = np.asarray(self.kept_original_indices, dtype=np.int64)
        installed = np.asarray(self.installed_return_indices, dtype=np.int64)
        object_indices = np.asarray(self.object_return_indices, dtype=np.int64)
        for name, value in (
            ("kept_original_indices", kept),
            ("installed_return_indices", installed),
            ("object_return_indices", object_indices),
        ):
            if value.ndim != 1 or np.any(value < 0):
                raise ValueError("{} must be a non-negative vector".format(name))
            if not np.array_equal(value, np.unique(value)):
                raise ValueError("{} must be sorted and unique".format(name))
        if len(points) != len(kept):
            raise ValueError("filtered point count differs from kept index count")
        if np.intersect1d(installed, object_indices).size:
            raise ValueError("installed and object removal indices must be disjoint")
        json.dumps(dict(self.evidence), sort_keys=True, allow_nan=False)
        object.__setattr__(self, "filtered_scene_points_base", points.copy())
        object.__setattr__(self, "kept_original_indices", kept)
        object.__setattr__(self, "installed_return_indices", installed)
        object.__setattr__(self, "object_return_indices", object_indices)


def _alignment_and_object_mask(
    scene_points: np.ndarray,
    object_points: np.ndarray,
    *,
    object_return_max_distance_m: float,
    alignment_coverage_distance_m: float,
) -> Tuple[np.ndarray, Mapping[str, Any]]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("SciPy cKDTree is required for replayable object filtering") from exc

    scene = _points(scene_points, "scene_points")
    object_cloud = _points(object_points, "object_points")
    maximum = float(object_return_max_distance_m)
    coverage_distance = float(alignment_coverage_distance_m)
    for value, name in (
        (maximum, "object_return_max_distance_m"),
        (coverage_distance, "alignment_coverage_distance_m"),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("{} must be finite and positive".format(name))

    scene_tree = cKDTree(scene)
    object_to_live, _ = scene_tree.query(object_cloud, k=1, workers=1)
    object_tree = cKDTree(object_cloud)
    live_to_object, _ = object_tree.query(scene, k=1, workers=1)
    lower = np.min(object_cloud, axis=0) - maximum
    upper = np.max(object_cloud, axis=0) + maximum
    inside_expanded_aabb = np.all(scene >= lower, axis=1) & np.all(
        scene <= upper, axis=1
    )
    mask = inside_expanded_aabb & (live_to_object <= maximum)
    stats = {
        "method": "bidirectional_cKDTree_with_expanded_object_AABB",
        "object_point_count": int(len(object_cloud)),
        "live_scene_point_count": int(len(scene)),
        "object_to_live_median_m": float(np.median(object_to_live)),
        "object_to_live_p95_m": float(np.percentile(object_to_live, 95.0)),
        "object_to_live_maximum_m": float(np.max(object_to_live)),
        "alignment_coverage_distance_m": coverage_distance,
        "alignment_coverage_fraction": float(
            np.mean(object_to_live <= coverage_distance)
        ),
        "object_return_max_distance_m": maximum,
        "expanded_object_aabb_min_m": lower.tolist(),
        "expanded_object_aabb_max_m": upper.tolist(),
        "object_return_count_before_installed_precedence": int(np.count_nonzero(mask)),
    }
    return mask, stats


def filter_installed_scene(
    backend: HppFclInstalledToolBackend,
    scene_points_base: np.ndarray,
    object_points_base: np.ndarray,
    capture_q_rad: Sequence[float],
    *,
    adapter_stl_path: Path,
    T_EE_hand: np.ndarray,
    hand_link_mesh_paths: Mapping[str, Path],
    T_hand_open_link_visual: Mapping[str, np.ndarray],
    scene_colors_base: Optional[np.ndarray] = None,
    point_half_extent_m: float = 0.0025,
    installed_inflation_margin_m: float = 0.002,
    residual_model_distance_max_m: float = 0.020,
    residual_calibration_guard_m: float = 0.002,
    residual_connectivity_radius_m: float = 0.010,
    residual_palette_color_max_l2: float = 0.18,
    residual_maximum_geodesic_m: float = 0.040,
    object_return_max_distance_m: float = 0.015,
    alignment_coverage_distance_m: float = 0.015,
    alignment_median_max_m: float = 0.008,
    alignment_p95_max_m: float = 0.015,
    alignment_minimum_coverage: float = 0.80,
    require_alignment: bool = True,
) -> InstalledSceneFilterResult:
    """Filter installed-tool/object returns without claiming unknown-space safety."""

    if not isinstance(backend, HppFclInstalledToolBackend):
        raise TypeError("backend must be HppFclInstalledToolBackend")
    scene = _points(scene_points_base, "scene_points_base")
    object_cloud = _points(object_points_base, "object_points_base")
    q = np.asarray(capture_q_rad, dtype=np.float64)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError("capture_q_rad must be a finite 7-vector")

    object_mask, alignment = _alignment_and_object_mask(
        scene,
        object_cloud,
        object_return_max_distance_m=object_return_max_distance_m,
        alignment_coverage_distance_m=alignment_coverage_distance_m,
    )
    calibration_guard = float(residual_calibration_guard_m)
    hard_model_cap = float(residual_model_distance_max_m)
    if not np.isfinite(calibration_guard) or not 0.0 < calibration_guard <= 0.002:
        raise ValueError("residual_calibration_guard_m must be in (0,0.002]")
    if not np.isfinite(hard_model_cap) or not 0.005 <= hard_model_cap <= 0.020:
        raise ValueError("residual_model_distance_max_m must be in [0.005,0.020]")
    calibrated_model_cap = min(
        hard_model_cap,
        float(alignment["object_to_live_maximum_m"]) + calibration_guard,
    )

    installed: InstalledReturnClassification = backend.classify_installed_depth_returns(
        scene,
        q,
        adapter_stl_path=adapter_stl_path,
        T_EE_hand=T_EE_hand,
        hand_link_mesh_paths=hand_link_mesh_paths,
        T_hand_open_link_visual=T_hand_open_link_visual,
        point_half_extent_m=point_half_extent_m,
        inflation_margin_m=installed_inflation_margin_m,
    )
    residual_indices = np.empty(0, dtype=np.int64)
    residual_details: Dict[str, Any]
    if scene_colors_base is None:
        residual_details = {
            "enabled": False,
            "failure": "normalized scene colors are unavailable",
            "exact_seed_count": int(len(installed.removed_indices)),
            "model_neighborhood_candidate_count": 0,
            "appearance_palette_candidate_count": 0,
            "connected_candidate_count": 0,
            "residual_removed_count": 0,
            "residual_removed_indices_sha256": _array_sha256(
                residual_indices, "<i8"
            ),
            "residual_removed_surface_distance_min_m": None,
            "residual_removed_surface_distance_max_m": None,
            "reason_counts": {},
        }
    else:
        colors = _colors(scene_colors_base, len(scene))
        neighborhood = backend.installed_depth_return_neighborhood(
            scene,
            q,
            adapter_stl_path=adapter_stl_path,
            T_EE_hand=T_EE_hand,
            hand_link_mesh_paths=hand_link_mesh_paths,
            T_hand_open_link_visual=T_hand_open_link_visual,
            maximum_surface_distance_m=calibrated_model_cap,
        )
        residual_indices, grown = bounded_residual_self_returns(
            scene,
            colors,
            installed.removed_indices,
            installed.reason_labels,
            neighborhood,
            connectivity_radius_m=residual_connectivity_radius_m,
            palette_color_max_l2=residual_palette_color_max_l2,
            maximum_geodesic_m=residual_maximum_geodesic_m,
        )
        residual_details = {
            "enabled": True,
            "failure": "",
            "method": (
                "strict HPP-FCL point-to-mesh neighborhood AND same-component "
                "normalized-sRGB seed palette AND bounded same-component 3-D geodesic"
            ),
            "distance_method": neighborhood.distance_method,
            "maximum_model_surface_distance_m": float(
                neighborhood.maximum_surface_distance_m
            ),
            "hard_maximum_model_surface_distance_m": hard_model_cap,
            "calibration_alignment_maximum_m": float(
                alignment["object_to_live_maximum_m"]
            ),
            "calibration_guard_m": calibration_guard,
            "calibration_bound_formula": (
                "min(hard_maximum, object_to_live_maximum + calibration_guard)"
            ),
            "connectivity_radius_m": float(residual_connectivity_radius_m),
            "color_space": "normalized_sRGB_euclidean",
            "palette_color_max_l2": float(residual_palette_color_max_l2),
            "maximum_geodesic_m": float(residual_maximum_geodesic_m),
            "model_candidate_test_count": int(
                neighborhood.candidate_test_count
            ),
            "model_candidate_indices_sha256": _array_sha256(
                neighborhood.candidate_indices, "<i8"
            ),
            "model_candidate_surface_distances_sha256": _array_sha256(
                neighborhood.surface_distances_m, "<f8"
            ),
            **dict(grown),
        }
    alignment_failures = []
    if alignment["object_to_live_median_m"] > float(alignment_median_max_m):
        alignment_failures.append("object-to-live median exceeds threshold")
    if alignment["object_to_live_p95_m"] > float(alignment_p95_max_m):
        alignment_failures.append("object-to-live p95 exceeds threshold")
    if alignment["alignment_coverage_fraction"] < float(alignment_minimum_coverage):
        alignment_failures.append("object-to-live coverage is below threshold")
    alignment = dict(alignment)
    alignment.update(
        {
            "alignment_median_max_m": float(alignment_median_max_m),
            "alignment_p95_max_m": float(alignment_p95_max_m),
            "alignment_minimum_coverage": float(alignment_minimum_coverage),
            "passed": not alignment_failures,
            "failures": alignment_failures,
        }
    )
    if require_alignment and alignment_failures:
        raise ValueError(
            "bound object cloud does not align with live scene: {}".format(
                "; ".join(alignment_failures)
            )
        )

    installed_indices = np.unique(
        np.concatenate((installed.removed_indices, residual_indices))
    ).astype(np.int64)
    installed_mask = np.zeros(len(scene), dtype=np.bool_)
    installed_mask[installed_indices] = True
    # Installed geometry takes deterministic precedence for overlapping
    # labels; the two output index sets are deliberately disjoint.
    object_mask &= ~installed_mask
    object_indices = np.flatnonzero(object_mask).astype(np.int64)
    keep_mask = ~(installed_mask | object_mask)
    kept_indices = np.flatnonzero(keep_mask).astype(np.int64)
    filtered = scene[kept_indices]
    reason_counts: Dict[str, int] = {}
    for label in installed.reason_labels:
        reason_counts[label] = reason_counts.get(label, 0) + 1
    for label, count in residual_details.get("reason_counts", {}).items():
        key = "residual_{}".format(label)
        reason_counts[key] = reason_counts.get(key, 0) + int(count)
    evidence = {
        "schema_version": 1,
        "artifact_type": "installed_scene_return_filter_evidence",
        "input_scene_point_count": int(len(scene)),
        "filtered_scene_point_count": int(len(filtered)),
        "installed_return_count": int(len(installed_indices)),
        "object_return_count": int(len(object_indices)),
        "input_scene_points_sha256": _array_sha256(scene, "<f8"),
        "object_points_sha256": _array_sha256(object_cloud, "<f8"),
        "capture_q_rad": q.tolist(),
        "capture_q_sha256": _array_sha256(q, "<f8"),
        "kept_original_indices_sha256": _array_sha256(kept_indices, "<i8"),
        "installed_return_indices_sha256": _array_sha256(
            installed_indices, "<i8"
        ),
        "object_return_indices_sha256": _array_sha256(object_indices, "<i8"),
        "filtered_scene_points_sha256": _array_sha256(filtered, "<f8"),
        "installed_return_filter": {
            "method": (
                "official FR3 visual shell/V7/open-RH56 triangle mesh versus "
                "inflated point cube"
            ),
            "fr3_geometry_model": installed.fr3_geometry_model,
            "fr3_visual_meshes": [
                {"link": link, "path": path, "sha256": digest}
                for link, path, digest in installed.fr3_mesh_provenance
            ],
            "point_half_extent_m": float(installed.point_half_extent_m),
            "inflation_margin_m": float(installed.inflation_margin_m),
            "candidate_mesh_point_tests": int(installed.candidate_test_count),
            "reason_counts": dict(sorted(reason_counts.items())),
            "residual_self_return_filter": residual_details,
        },
        "object_alignment_and_filter": alignment,
        "unknown_space_policy_applied": False,
        "authoritative_for_unseen_camera_space": False,
        "motion_authorized": False,
        "meaning": "replayable observed-point preprocessing only; never a motion command",
    }
    return InstalledSceneFilterResult(
        filtered_scene_points_base=filtered,
        kept_original_indices=kept_indices,
        installed_return_indices=installed_indices,
        object_return_indices=object_indices,
        evidence=evidence,
    )


__all__ = [
    "InstalledSceneFilterResult",
    "bounded_residual_self_returns",
    "filter_installed_scene",
]
