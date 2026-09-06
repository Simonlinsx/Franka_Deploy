"""Robust, fail-closed helpers for static FR3/RH56 mount registration.

The fitted model frame is the AnyDex generated Inspire hand-source frame.  A
caller supplies points sampled from the rigid ``Link111`` mesh *after* the
official source-mesh offset has been applied once.  The target cloud is a
manually prompted, calibrated D435 crop expressed in ``robot_base``.

This module contains no camera, Franka, Inspire, serial, or renderer imports.
It cannot move hardware.  Live read-only acquisition is implemented by the
separate CLI so the numerical registration remains deterministic and testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

from .control_plan import inverse_rigid_transform, validate_rigid_transform


@dataclass(frozen=True)
class ICPOptions:
    """Parameters for observed-to-model trimmed point-to-point ICP."""

    correspondence_schedule_m: Tuple[float, ...] = (0.040, 0.020, 0.010, 0.006)
    iterations_per_stage: int = 25
    trim_fraction: float = 0.78
    min_correspondences: int = 80
    translation_tolerance_m: float = 2.0e-5
    rotation_tolerance_deg: float = 0.01
    lock_candidate_orientation: bool = True
    translation_seed_quantiles: Tuple[float, ...] = (0.05, 0.50, 0.95)
    translation_seed_keep: int = 3
    translation_seed_sample_points: int = 2000

    def validate(self) -> "ICPOptions":
        schedule = np.asarray(self.correspondence_schedule_m, dtype=np.float64)
        if (
            schedule.ndim != 1
            or len(schedule) < 1
            or not np.all(np.isfinite(schedule))
            or np.any(schedule <= 0.0)
        ):
            raise ValueError("correspondence schedule must contain positive finite values")
        if self.iterations_per_stage < 1:
            raise ValueError("iterations_per_stage must be >= 1")
        if not 0.25 <= float(self.trim_fraction) <= 1.0:
            raise ValueError("trim_fraction must be in [0.25, 1.0]")
        if self.min_correspondences < 6:
            raise ValueError("min_correspondences must be >= 6")
        if self.translation_tolerance_m <= 0.0 or self.rotation_tolerance_deg <= 0.0:
            raise ValueError("ICP convergence tolerances must be positive")
        quantiles = np.asarray(self.translation_seed_quantiles, dtype=np.float64)
        if (
            quantiles.ndim != 1
            or len(quantiles) < 1
            or not np.all(np.isfinite(quantiles))
            or np.any(quantiles < 0.0)
            or np.any(quantiles > 1.0)
        ):
            raise ValueError("translation seed quantiles must be in [0,1]")
        if self.translation_seed_keep < 1 or self.translation_seed_sample_points < 100:
            raise ValueError("translation seed counts are too small")
        return self


@dataclass(frozen=True)
class RegistrationThresholds:
    """Conservative geometry gates for a single-view mount measurement."""

    evaluation_inlier_distance_m: float = 0.006
    min_observed_inlier_ratio: float = 0.55
    max_observed_p50_m: float = 0.0045
    max_observed_p90_m: float = 0.010
    max_orientation_correction_deg: float = 12.0
    max_multistart_score_gap_ratio: float = 0.985
    ambiguity_min_orientation_separation_deg: float = 30.0
    ambiguity_absolute_score_gap_m: float = 0.0005
    min_observed_model_span_ratio: Tuple[float, float, float] = (0.45, 0.45, 0.10)

    def validate(self) -> "RegistrationThresholds":
        numeric = np.asarray(
            [
                self.evaluation_inlier_distance_m,
                self.min_observed_inlier_ratio,
                self.max_observed_p50_m,
                self.max_observed_p90_m,
                self.max_orientation_correction_deg,
                self.max_multistart_score_gap_ratio,
                self.ambiguity_min_orientation_separation_deg,
                self.ambiguity_absolute_score_gap_m,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(numeric)):
            raise ValueError("registration thresholds must be finite")
        if self.evaluation_inlier_distance_m <= 0.0:
            raise ValueError("evaluation inlier distance must be positive")
        if not 0.0 < self.min_observed_inlier_ratio <= 1.0:
            raise ValueError("minimum observed inlier ratio must be in (0,1]")
        if self.max_observed_p50_m <= 0.0 or self.max_observed_p90_m <= 0.0:
            raise ValueError("distance thresholds must be positive")
        if self.max_orientation_correction_deg <= 0.0:
            raise ValueError("orientation correction threshold must be positive")
        if not 0.0 <= self.max_multistart_score_gap_ratio <= 1.0:
            raise ValueError("multistart ambiguity ratio must be in [0,1]")
        if self.ambiguity_min_orientation_separation_deg <= 0.0:
            raise ValueError("ambiguity orientation separation must be positive")
        if self.ambiguity_absolute_score_gap_m < 0.0:
            raise ValueError("ambiguity absolute score gap must be nonnegative")
        spans = np.asarray(self.min_observed_model_span_ratio, dtype=np.float64)
        if (
            spans.shape != (3,)
            or not np.all(np.isfinite(spans))
            or np.any(spans < 0.0)
            or np.any(spans > 1.0)
        ):
            raise ValueError("minimum observed/model span ratios must be three values in [0,1]")
        return self


@dataclass(frozen=True)
class RegistrationMetrics:
    observed_count: int
    model_count: int
    observed_inlier_ratio: float
    observed_p50_m: float
    observed_p90_m: float
    observed_p95_m: float
    observed_trimmed_rmse_m: float
    visible_model_coverage_ratio: float
    orientation_correction_deg: float
    final_correspondences: int
    observed_model_span_ratio_xyz: Tuple[float, float, float]
    score: float


@dataclass(frozen=True)
class RegistrationResult:
    T_reference_hand: np.ndarray
    initial_T_reference_hand: np.ndarray
    orientation_candidate_label: str
    metrics: RegistrationMetrics
    geometry_gate_passed: bool
    rejection_reasons: Tuple[str, ...]
    second_best_score: float | None
    distinct_orientation_competitor_score: float | None
    ambiguous_multistart: bool

    def metrics_dict(self) -> dict:
        return asdict(self.metrics)


def _points(value: np.ndarray, name: str, *, minimum: int = 6) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N,3)")
    if len(points) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} points")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    extent = np.ptp(points, axis=0)
    if np.count_nonzero(extent > 1.0e-4) < 2:
        raise ValueError(f"{name} is geometrically degenerate")
    return points.copy()


def transform_points(T_reference_source: np.ndarray, points_source: np.ndarray) -> np.ndarray:
    transform = validate_rigid_transform(T_reference_source, "T_reference_source")
    points = _points(points_source, "points_source")
    return points @ transform[:3, :3].T + transform[:3, 3]


def rotation_angle_deg(rotation: np.ndarray) -> float:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    cosine = np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def rigid_delta(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """Return translation metres and rotation degrees from pose A to B."""

    first = validate_rigid_transform(T_a, "T_a")
    second = validate_rigid_transform(T_b, "T_b")
    relative = inverse_rigid_transform(first) @ second
    return float(np.linalg.norm(relative[:3, 3])), rotation_angle_deg(relative[:3, :3])


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Kabsch point arrays must have matching shape (N,3)")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return validate_rigid_transform(transform, "Kabsch transform")


def _initial_pose(
    model_points_hand: np.ndarray,
    observed_points_reference: np.ndarray,
    R_reference_hand: np.ndarray,
) -> np.ndarray:
    rotation = np.asarray(R_reference_hand, dtype=np.float64)
    candidate = np.eye(4, dtype=np.float64)
    candidate[:3, :3] = rotation
    validate_rigid_transform(candidate, "orientation candidate")
    model_center = np.median(model_points_hand, axis=0)
    observed_center = np.median(observed_points_reference, axis=0)
    candidate[:3, 3] = observed_center - rotation @ model_center
    return validate_rigid_transform(candidate, "initial T_reference_hand")


def _translation_seed_poses(
    model_points_hand: np.ndarray,
    observed_points_reference: np.ndarray,
    R_reference_hand: np.ndarray,
    options: ICPOptions,
) -> list[np.ndarray]:
    """Return the best robust quantile-alignment translation seeds.

    A partial view shifts the observed centroid away from the full-model
    centroid.  Per-axis low/median/high quantile alignments provide hypotheses
    for the visible silhouette boundaries before local ICP refinement.
    """

    from itertools import product
    from scipy.spatial import cKDTree

    rotation = np.asarray(R_reference_hand, dtype=np.float64)
    base = _initial_pose(model_points_hand, observed_points_reference, rotation)
    quantiles = np.asarray(options.translation_seed_quantiles, dtype=np.float64)
    # Row-vector convention: p_reference @ R = p_hand + t_reference @ R.
    observed_oriented = observed_points_reference @ rotation
    model_quantiles = np.quantile(model_points_hand, quantiles, axis=0)
    observed_quantiles = np.quantile(observed_oriented, quantiles, axis=0)
    per_axis = []
    for axis in range(3):
        values = [
            float(observed_quantiles[index, axis] - model_quantiles[index, axis])
            for index in range(len(quantiles))
        ]
        values.append(float(base[:3, 3] @ rotation[:, axis]))
        per_axis.append(tuple(dict.fromkeys(round(value, 12) for value in values)))

    rotated_model = model_points_hand @ rotation.T
    tree = cKDTree(rotated_model)
    if len(observed_points_reference) > options.translation_seed_sample_points:
        indices = np.linspace(
            0,
            len(observed_points_reference) - 1,
            options.translation_seed_sample_points,
            dtype=np.int64,
        )
        evaluation_points = observed_points_reference[indices]
    else:
        evaluation_points = observed_points_reference
    keep_count = max(
        options.min_correspondences,
        int(np.ceil(options.trim_fraction * len(evaluation_points))),
    )
    keep_count = min(keep_count, len(evaluation_points))
    scored = []
    for oriented_translation in product(*per_axis):
        oriented = np.asarray(oriented_translation, dtype=np.float64)
        translation = oriented @ rotation.T
        distances, _ = tree.query(evaluation_points - translation, k=1)
        closest = np.partition(distances, keep_count - 1)[:keep_count]
        # High quantile plus median is more discriminative than mean alone.
        score = float(np.percentile(closest, 90.0) + 0.5 * np.median(closest))
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        scored.append((score, validate_rigid_transform(pose, "translation seed")))
    scored.sort(key=lambda item: item[0])
    return [item[1] for item in scored[: options.translation_seed_keep]]


def _run_one_icp(
    model_points_hand: np.ndarray,
    observed_points_reference: np.ndarray,
    initial: np.ndarray,
    options: ICPOptions,
) -> tuple[np.ndarray, int]:
    from scipy.spatial import cKDTree

    transform = validate_rigid_transform(initial, "initial T_reference_hand")
    final_correspondences = 0
    requested_keep = max(
        options.min_correspondences,
        int(np.ceil(options.trim_fraction * len(observed_points_reference))),
    )
    if options.lock_candidate_orientation:
        rotated_model = model_points_hand @ transform[:3, :3].T
        fixed_tree = cKDTree(rotated_model)
    else:
        rotated_model = None
        fixed_tree = None
    for maximum_distance in options.correspondence_schedule_m:
        for _ in range(options.iterations_per_stage):
            if options.lock_candidate_orientation:
                assert rotated_model is not None and fixed_tree is not None
                distances, indices = fixed_tree.query(
                    observed_points_reference - transform[:3, 3], k=1
                )
                transformed_model = rotated_model + transform[:3, 3]
            else:
                transformed_model = transform_points(transform, model_points_hand)
                tree = cKDTree(transformed_model)
                distances, indices = tree.query(observed_points_reference, k=1)
            eligible = np.flatnonzero(
                np.isfinite(distances) & (distances <= float(maximum_distance))
            )
            if len(eligible) < options.min_correspondences:
                break
            keep_count = min(len(eligible), requested_keep)
            if keep_count < len(eligible):
                order = np.argpartition(distances[eligible], keep_count - 1)[:keep_count]
                keep = eligible[order]
            else:
                keep = eligible
            final_correspondences = len(keep)
            if options.lock_candidate_orientation:
                # The adapter permits translation plus assembled yaw, not
                # arbitrary roll/pitch.  Yaw is searched by supplying a dense
                # set of mechanically constructed orientation candidates; ICP
                # therefore refines translation only for each candidate.
                residual = (
                    observed_points_reference[keep]
                    - transformed_model[indices[keep]]
                )
                delta = np.eye(4, dtype=np.float64)
                delta[:3, 3] = np.median(residual, axis=0)
            else:
                delta = _kabsch(
                    transformed_model[indices[keep]],
                    observed_points_reference[keep],
                )
            transform = validate_rigid_transform(
                delta @ transform, "updated T_reference_hand"
            )
            translation_step = float(np.linalg.norm(delta[:3, 3]))
            rotation_step = rotation_angle_deg(delta[:3, :3])
            if (
                translation_step <= options.translation_tolerance_m
                and rotation_step <= options.rotation_tolerance_deg
            ):
                break
    return transform, final_correspondences


def _evaluate(
    model_points_hand: np.ndarray,
    observed_points_reference: np.ndarray,
    transform: np.ndarray,
    initial: np.ndarray,
    final_correspondences: int,
    thresholds: RegistrationThresholds,
) -> RegistrationMetrics:
    from scipy.spatial import cKDTree

    transformed_model = transform_points(transform, model_points_hand)
    model_tree = cKDTree(transformed_model)
    observed_distances, _ = model_tree.query(observed_points_reference, k=1)
    observed_distances = observed_distances[np.isfinite(observed_distances)]
    if len(observed_distances) == 0:
        raise ValueError("registration produced no finite distances")
    sorted_distances = np.sort(observed_distances)
    trimmed_count = max(1, int(np.ceil(0.80 * len(sorted_distances))))
    trimmed = sorted_distances[:trimmed_count]
    inlier_ratio = float(
        np.mean(observed_distances <= thresholds.evaluation_inlier_distance_m)
    )

    observed_tree = cKDTree(observed_points_reference)
    model_distances, _ = observed_tree.query(transformed_model, k=1)
    visible_coverage = float(
        np.mean(model_distances <= thresholds.evaluation_inlier_distance_m)
    )
    orientation_correction = rotation_angle_deg(
        initial[:3, :3].T @ transform[:3, :3]
    )
    inlier_observed = observed_points_reference[
        observed_distances <= thresholds.evaluation_inlier_distance_m
    ]
    if len(inlier_observed) >= 20:
        observed_hand = (
            inlier_observed - transform[:3, 3]
        ) @ transform[:3, :3]
        observed_span = np.quantile(observed_hand, 0.95, axis=0) - np.quantile(
            observed_hand, 0.05, axis=0
        )
        model_span = np.quantile(model_points_hand, 0.95, axis=0) - np.quantile(
            model_points_hand, 0.05, axis=0
        )
        span_ratio = np.divide(
            observed_span,
            model_span,
            out=np.zeros(3, dtype=np.float64),
            where=model_span > 1.0e-9,
        )
    else:
        span_ratio = np.zeros(3, dtype=np.float64)
    p50, p90, p95 = np.percentile(observed_distances, [50.0, 90.0, 95.0])
    trimmed_rmse = float(np.sqrt(np.mean(np.square(trimmed))))
    # Lower is better.  The inlier term prevents a small excellent patch from
    # winning over a fit that explains most of the prompted palm surface.
    score = float(
        p90
        + 0.50 * p50
        + 0.015 * (1.0 - inlier_ratio)
        + 0.00010 * orientation_correction
    )
    return RegistrationMetrics(
        observed_count=len(observed_points_reference),
        model_count=len(model_points_hand),
        observed_inlier_ratio=inlier_ratio,
        observed_p50_m=float(p50),
        observed_p90_m=float(p90),
        observed_p95_m=float(p95),
        observed_trimmed_rmse_m=trimmed_rmse,
        visible_model_coverage_ratio=visible_coverage,
        orientation_correction_deg=orientation_correction,
        final_correspondences=int(final_correspondences),
        observed_model_span_ratio_xyz=tuple(float(value) for value in span_ratio),
        score=score,
    )


def register_rigid_hand_model(
    model_points_hand: np.ndarray,
    observed_points_reference: np.ndarray,
    orientation_candidates: Iterable[tuple[str, np.ndarray]],
    *,
    options: ICPOptions = ICPOptions(),
    thresholds: RegistrationThresholds = RegistrationThresholds(),
) -> RegistrationResult:
    """Fit an AnyDex hand-source model to a prompted partial palm cloud.

    ICP correspondences are queried from every observed point to the full rigid
    model, which avoids penalising the large occluded part of ``Link111``.
    Multiple mechanically plausible orientations are evaluated independently.
    The returned gate is only a *mount measurement* gate; it is never a
    collision-model or robot-motion approval.
    """

    options.validate()
    thresholds.validate()
    model = _points(model_points_hand, "model_points_hand", minimum=100)
    observed = _points(
        observed_points_reference, "observed_points_reference", minimum=100
    )
    candidates = []
    for label, rotation in orientation_candidates:
        orientation_fits = []
        for initial in _translation_seed_poses(model, observed, rotation, options):
            fitted, correspondences = _run_one_icp(model, observed, initial, options)
            metrics = _evaluate(
                model, observed, fitted, initial, correspondences, thresholds
            )
            orientation_fits.append(
                (metrics.score, str(label), initial, fitted, metrics)
            )
        orientation_fits.sort(key=lambda item: item[0])
        candidates.append(orientation_fits[0])
    if not candidates:
        raise ValueError("at least one orientation candidate is required")
    candidates.sort(key=lambda item: item[0])
    _, label, initial, fitted, metrics = candidates[0]
    second_score = float(candidates[1][0]) if len(candidates) > 1 else None
    ambiguous = False
    distinct_competitors = [
        item
        for item in candidates[1:]
        if rotation_angle_deg(fitted[:3, :3].T @ item[3][:3, :3])
        >= thresholds.ambiguity_min_orientation_separation_deg
    ]
    distinct_score = (
        float(distinct_competitors[0][0]) if distinct_competitors else None
    )
    if distinct_score is not None and distinct_score > 0.0:
        # Scores are costs.  A best/second ratio near one means two different
        # mount-orientation branches explain the crop almost equally well.
        # Adjacent yaw-grid samples are intentionally not treated as branches.
        ratio = float(metrics.score / distinct_score)
        gap = float(distinct_score - metrics.score)
        ambiguous = (
            ratio >= thresholds.max_multistart_score_gap_ratio
            or gap <= thresholds.ambiguity_absolute_score_gap_m
        )

    reasons = []
    if metrics.observed_inlier_ratio < thresholds.min_observed_inlier_ratio:
        reasons.append("observed-to-model inlier ratio is too low")
    if metrics.observed_p50_m > thresholds.max_observed_p50_m:
        reasons.append("median observed-to-model distance is too high")
    if metrics.observed_p90_m > thresholds.max_observed_p90_m:
        reasons.append("p90 observed-to-model distance is too high")
    if metrics.orientation_correction_deg > thresholds.max_orientation_correction_deg:
        reasons.append("ICP orientation correction exceeds the mechanical prior")
    if metrics.final_correspondences < options.min_correspondences:
        reasons.append("too few final ICP correspondences")
    for axis, actual, required in zip(
        "xyz",
        metrics.observed_model_span_ratio_xyz,
        thresholds.min_observed_model_span_ratio,
    ):
        if actual < required:
            reasons.append(
                f"prompted palm coverage along hand-source {axis} is too small"
            )
    if ambiguous:
        reasons.append("multiple mount orientations have indistinguishable fit scores")
    return RegistrationResult(
        T_reference_hand=fitted,
        initial_T_reference_hand=initial,
        orientation_candidate_label=label,
        metrics=metrics,
        geometry_gate_passed=not reasons,
        rejection_reasons=tuple(reasons),
        second_best_score=second_score,
        distinct_orientation_competitor_score=distinct_score,
        ambiguous_multistart=ambiguous,
    )


def compose_T_EE_hand(
    T_reference_EE: np.ndarray, T_reference_hand: np.ndarray
) -> np.ndarray:
    reference_to_ee = validate_rigid_transform(T_reference_EE, "T_reference_EE")
    reference_to_hand = validate_rigid_transform(
        T_reference_hand, "T_reference_hand"
    )
    return validate_rigid_transform(
        inverse_rigid_transform(reference_to_ee) @ reference_to_hand,
        "T_EE_hand",
    )


def decompose_installed_mount(
    T_EE_hand: np.ndarray,
    F_T_EE: np.ndarray,
    *,
    fr3_face_to_rh56_seating_plane_m: float,
    T_mount_source_axis_basis: np.ndarray,
) -> dict:
    """Recover yaw and seating-frame source-origin offset from a fitted pose."""

    ee_to_hand = validate_rigid_transform(T_EE_hand, "T_EE_hand")
    flange_to_ee = validate_rigid_transform(F_T_EE, "F_T_EE")
    basis = validate_rigid_transform(
        T_mount_source_axis_basis, "T_mount_source_axis_basis"
    )
    distance = float(fr3_face_to_rh56_seating_plane_m)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("seating-plane distance must be finite and positive")
    T_F_hand = flange_to_ee @ ee_to_hand
    T_F_seating = np.eye(4, dtype=np.float64)
    T_F_seating[2, 3] = distance
    T_seating_hand = inverse_rigid_transform(T_F_seating) @ T_F_hand
    R_yaw = T_seating_hand[:3, :3] @ basis[:3, :3].T
    yaw = float(np.arctan2(R_yaw[1, 0], R_yaw[0, 0]))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    Rz = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    residual_rotation = Rz.T @ R_yaw
    return {
        "assembled_yaw_rad": yaw,
        "assembled_yaw_deg": float(np.rad2deg(yaw)),
        "seating_to_hand_source_origin_m": T_seating_hand[:3, 3].copy(),
        "mount_axis_residual_deg": rotation_angle_deg(residual_rotation),
        "T_F_hand": validate_rigid_transform(T_F_hand, "T_F_hand"),
        "T_seating_hand": validate_rigid_transform(
            T_seating_hand, "T_seating_hand"
        ),
    }


__all__ = [
    "ICPOptions",
    "RegistrationMetrics",
    "RegistrationResult",
    "RegistrationThresholds",
    "compose_T_EE_hand",
    "decompose_installed_mount",
    "register_rigid_hand_model",
    "rigid_delta",
    "rotation_angle_deg",
    "transform_points",
]
