"""Canonical byte-stable sampling for audited joint waypoint polylines.

All producers and consumers of a joint-path checksum must call the same
routine.  Algebraically equivalent interpolation formulas are not acceptable:
their IEEE-754 results can differ by one ULP and therefore produce different
SHA-256 values even though the geometric paths are indistinguishable.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


CANONICAL_JOINT_SAMPLING_ALGORITHM = "numpy_linspace_float64_v1"


def canonical_joint_path_samples(
    waypoints: Sequence[Sequence[float]],
    max_joint_step_rad: float,
    *,
    joint_count: int = 7,
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Return one canonical float64 path and per-segment interval counts.

    Segment samples are produced only by ``numpy.linspace`` with an explicit
    float64 dtype.  Shared waypoint rows are emitted once.  This precise
    implementation, including operation order, is the checksum contract.
    """

    maximum = float(max_joint_step_rad)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_joint_step_rad must be finite and positive")
    if isinstance(joint_count, (bool, np.bool_)) or int(joint_count) <= 0:
        raise ValueError("joint_count must be a positive integer")
    expected_shape = (int(joint_count),)
    vectors = tuple(np.asarray(item, dtype=np.float64) for item in waypoints)
    if len(vectors) < 2:
        raise ValueError("waypoints must contain at least two joint vectors")
    if any(
        item.shape != expected_shape or not np.all(np.isfinite(item))
        for item in vectors
    ):
        raise ValueError(
            "each waypoint must be a finite {}-vector".format(joint_count)
        )

    pieces = []
    intervals = []
    for index, (start, end) in enumerate(zip(vectors[:-1], vectors[1:])):
        count = max(
            1,
            int(
                math.ceil(
                    float(np.max(np.abs(end - start))) / maximum
                )
            ),
        )
        segment = np.linspace(
            start, end, count + 1, dtype=np.float64
        )
        pieces.append(segment if index == 0 else segment[1:])
        intervals.append(count)
    path = np.concatenate(pieces, axis=0)
    observed = float(np.max(np.abs(np.diff(path, axis=0))))
    if observed > maximum + 1.0e-12:
        raise RuntimeError("canonical path exceeded max_joint_step_rad")
    return path, tuple(intervals)


__all__ = [
    "CANONICAL_JOINT_SAMPLING_ALGORITHM",
    "canonical_joint_path_samples",
]
