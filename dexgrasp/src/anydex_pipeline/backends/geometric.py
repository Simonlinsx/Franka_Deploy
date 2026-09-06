from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..frames import pose_from_axes
from ..types import GraspCandidate, GraspResult, PointCloudObservation


@dataclass
class GeometricGraspBackend:
    """Deterministic geometry-only commissioning backend.

    This backend is intentionally named ``geometric`` and never presents its
    results as AnyDex model predictions.  It exists to validate camera
    calibration, object-cloud plumbing, pose conventions and visualization
    before the legacy CUDA model environment is available.
    """

    top_k: int = 8
    pregrasp_clearance_m: float = 0.055
    min_width_m: float = 0.025
    max_width_m: float = 0.10
    base_up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    name: str = "geometric_demo"

    def infer(self, observation: PointCloudObservation) -> GraspResult:
        started = time.perf_counter()
        points = np.asarray(observation.object_points, dtype=np.float64)
        if len(points) < 20:
            raise ValueError("geometric backend needs at least 20 object points")

        center = np.median(points, axis=0)
        centered = points - center
        covariance = centered.T @ centered / max(1, len(points) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        axes = eigenvectors[:, order]
        for column in range(3):
            anchor = int(np.argmax(np.abs(axes[:, column])))
            if axes[anchor, column] < 0:
                axes[:, column] *= -1
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1

        projections = centered @ axes
        low = np.quantile(projections, 0.02, axis=0)
        high = np.quantile(projections, 0.98, axis=0)
        extents = np.maximum(high - low, 1e-4)
        object_center = center + axes @ ((high + low) * 0.5)
        base_up = np.asarray(self.base_up_axis, dtype=np.float64)
        base_up /= np.linalg.norm(base_up)

        proposals: list[tuple[float, np.ndarray, np.ndarray, float]] = []
        # A top-down approach is useful on a tabletop, followed by both signs
        # of every PCA axis.  Each tuple is (prior, approach, closing_hint,
        # estimated width).
        smallest = int(np.argmin(extents))
        closing = axes[:, smallest]
        width = float(extents[smallest] * 1.08)
        proposals.append((1.0, -base_up, closing, width))
        for axis_index in range(3):
            approach_axis = axes[:, axis_index]
            closing_index = int(np.argmin([extents[j] if j != axis_index else np.inf for j in range(3)]))
            closing_axis = axes[:, closing_index]
            candidate_width = float(extents[closing_index] * 1.08)
            for sign in (1.0, -1.0):
                approach = sign * approach_axis
                upward_preference = 0.5 * (1.0 - float(np.dot(approach, base_up)))
                slender_preference = 1.0 - float(extents[axis_index] / (extents.max() + 1e-9))
                prior = 0.55 + 0.25 * upward_preference + 0.20 * slender_preference
                proposals.append((prior, approach, closing_axis, candidate_width))

        candidates: list[GraspCandidate] = []
        for source_index, (prior, approach, closing_hint, raw_width) in enumerate(proposals):
            width_m = float(np.clip(raw_width, self.min_width_m, self.max_width_m))
            projections_along = centered @ approach
            contact_offset = float(np.quantile(projections_along, 0.35))
            origin = center + approach * contact_offset
            pose = pose_from_axes(origin, approach, closing_hint)
            width_fit = np.exp(-3.0 * abs(width_m - raw_width) / self.max_width_m)
            score = float(np.clip(prior * width_fit, 0.0, 1.0))
            candidates.append(
                GraspCandidate(
                    T_reference_grasp=pose,
                    score=score,
                    width_m=width_m,
                    depth_m=self.pregrasp_clearance_m,
                    collision_free=True,
                    source_index=source_index,
                    metadata={
                        "raw_object_width_m": raw_width,
                        "diagnostic_only": True,
                    },
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        candidates = candidates[: max(1, int(self.top_k))]
        return GraspResult(
            candidates=candidates,
            backend_name=self.name,
            reference_frame=observation.reference_frame,
            inference_time_s=time.perf_counter() - started,
            selected_index=0,
            model_name="PCA/OBB commissioning backend (not AnyDexGrasp)",
            inference_points=observation.object_points,
        )
