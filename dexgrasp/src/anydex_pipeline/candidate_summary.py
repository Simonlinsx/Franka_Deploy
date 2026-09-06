"""Deterministic, hardware-free summaries of saved AnyDex candidates."""

from __future__ import annotations

import math

import numpy as np

from .snapshot import VisualizationSnapshot, validate_snapshot


def _target_text(value: float) -> str:
    rounded = round(float(value))
    if math.isclose(float(value), rounded, abs_tol=1.0e-5, rel_tol=0.0):
        return str(int(rounded))
    return f"{float(value):.3f}"


def candidate_summary_lines(
    snapshot: VisualizationSnapshot,
    *,
    selected_index: int,
    requested_top_k: int | None = None,
) -> tuple[str, ...]:
    """Return copy/paste-friendly lines while preserving immutable indices."""

    validate_snapshot(snapshot)
    grasps = snapshot.grasps
    selected = int(selected_index)
    if grasps.hand_angles is None:
        raise ValueError("candidate summary requires six-axis Inspire targets")
    requested = grasps.count if requested_top_k is None else int(requested_top_k)
    if requested < 1:
        raise ValueError("requested_top_k must be positive")
    axis_local = np.asarray(grasps.approach_axis_local, dtype=np.float64)
    lines = [
        "[candidates] returned={} requested_top_k={} selected={} frame={}".format(
            grasps.count,
            requested,
            selected,
            snapshot.reference_frame,
        )
    ]
    for index in range(grasps.count):
        pose = np.asarray(grasps.canonical_poses[index], dtype=np.float64)
        xyz = pose[:3, 3]
        approach = pose[:3, :3] @ axis_local
        approach /= np.linalg.norm(approach)
        targets = np.asarray(grasps.hand_angles[index], dtype=np.float64)
        target_text = ",".join(_target_text(value) for value in targets)
        marker = "*" if index == selected else " "
        lines.append(
            (
                "[candidate]{marker} index={index:03d} score={score:.6f} "
                "type={type_id:d} canonical_xyz_m=[{x:+.5f},{y:+.5f},{z:+.5f}] "
                "approach_{frame}=[{ax:+.5f},{ay:+.5f},{az:+.5f}] "
                "hand_targets=[{targets}] q6={q6}"
            ).format(
                marker=marker,
                index=index,
                score=float(grasps.scores[index]),
                type_id=int(grasps.type_ids[index]),
                x=float(xyz[0]),
                y=float(xyz[1]),
                z=float(xyz[2]),
                frame=snapshot.reference_frame,
                ax=float(approach[0]),
                ay=float(approach[1]),
                az=float(approach[2]),
                targets=target_text,
                q6=_target_text(targets[5]),
            )
        )
    return tuple(lines)


__all__ = ["candidate_summary_lines"]
