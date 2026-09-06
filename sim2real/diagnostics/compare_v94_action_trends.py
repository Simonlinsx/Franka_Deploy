#!/usr/bin/env python3
"""Compare read-only real V94 policy outputs with packaged simulation traces.

This is deliberately an offline diagnostic.  It opens only local NPZ/ZIP
files, has no hardware imports, and never claims that an action is safe or
semantically correct to execute.  The stationary real reset capture is
compared primarily with ``reset_idle_open``.  The moving simulation trace is
used only as a reference action family; its rows are not time-aligned with the
stationary real capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    workspace = Path(__file__).resolve().parents[2]
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from sim2real.deployment.bundle import DeployBundle
else:
    from sim2real.deployment.bundle import DeployBundle


from sim2real.workspace_paths import DEFAULT_V94_DEPLOY_BUNDLE

DEFAULT_BUNDLE = DEFAULT_V94_DEPLOY_BUNDLE
INITIAL_NPZ = "alignment/reset_idle_open/initial_observations_and_student_response.npz"
CLOSED_NPZ = "alignment/student_closed_loop/closed_loop_observation_action_pairs.npz"
MAX_REAL_NPZ_BYTES = 1024 * 1024 * 1024
SATURATION_TOLERANCE = 1.0e-6
SIGN_EPSILON = 1.0e-6

ACTION_NAMES = tuple(
    [f"franka_joint_{index}_increment" for index in range(1, 8)]
    + [
        "rh56_thumb_rotation_absolute",
        "rh56_thumb_bending_absolute",
        "rh56_index_absolute",
        "rh56_middle_absolute",
        "rh56_ring_absolute",
        "rh56_little_absolute",
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_actions(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2 or result.shape[1] != 13:
        raise ValueError(f"{name} must have shape [N,13], got {result.shape}")
    if result.shape[0] < 1 or result.dtype.kind != "f":
        raise ValueError(f"{name} must be a non-empty floating array")
    result = result.astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(np.abs(result) > 1.0 + SATURATION_TOLERANCE):
        raise ValueError(f"{name} contains values outside [-1,1]")
    return result


def _scalar_false(values: Mapping[str, np.ndarray], name: str) -> None:
    if name not in values:
        return
    value = np.asarray(values[name])
    if value.shape != () or value.dtype.kind != "b":
        raise ValueError(f"{name} must be one scalar boolean")
    if bool(value):
        raise ValueError(f"real capture reports {name}=true")


def _load_real(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"real shadow NPZ not found: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_REAL_NPZ_BYTES:
        raise ValueError(f"real shadow NPZ has unsafe size {size} bytes")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"raw_policy_action13"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"real shadow NPZ is missing {sorted(missing)}")
            selected = {
                name: archive[name].copy()
                for name in (
                    "raw_policy_action13",
                    "host_action_monotonic_s",
                    "timestamp_s",
                    "pointcloud_frame_id",
                    "proposal_mode",
                    "hardware_writes",
                    "robot_command_writes",
                    "write",
                )
                if name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid no-pickle real shadow NPZ: {exc}") from exc

    for name in ("hardware_writes", "robot_command_writes"):
        _scalar_false(selected, name)
    if "write" in selected:
        write = np.asarray(selected["write"])
        if write.dtype.kind != "b" or np.any(write):
            raise ValueError("real shadow contains a write=true row")
    return selected


def _time_axis(values: Mapping[str, np.ndarray], count: int) -> tuple[np.ndarray, str]:
    for name in ("host_action_monotonic_s", "timestamp_s"):
        if name in values:
            result = np.asarray(values[name], dtype=np.float64)
            if result.shape != (count,) or not np.all(np.isfinite(result)):
                raise ValueError(f"{name} must be finite [N]")
            if count > 1 and np.any(np.diff(result) <= 0.0):
                raise ValueError(f"{name} must be strictly increasing")
            return result, name
    return np.arange(count, dtype=np.float64), "sample_index"


def _stats(value: np.ndarray) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5.0)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def _axis_stats(actions: np.ndarray) -> list[dict[str, float]]:
    result = []
    for index in range(13):
        item = _stats(actions[:, index])
        item["saturation_fraction"] = float(
            np.mean(np.abs(actions[:, index]) >= 1.0 - SATURATION_TOLERANCE)
        )
        result.append(item)
    return result


def _sign(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.where(array > SIGN_EPSILON, 1, np.where(array < -SIGN_EPSILON, -1, 0))


def _cosine(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    first = np.asarray(query, dtype=np.float64)
    second = np.asarray(reference, dtype=np.float64)
    if second.ndim == 1:
        second = np.broadcast_to(second, first.shape)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    result = np.full(first.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 0.0
    result[valid] = np.sum(first[valid] * second[valid], axis=1) / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def _nearest(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.max(np.abs(query[:, None, :] - reference[None, :, :]), axis=2)
    index = np.argmin(distance, axis=1)
    return index, distance[np.arange(query.shape[0]), index]


def _temporal(actions: np.ndarray, time_s: np.ndarray) -> dict[str, Any]:
    duration = float(time_s[-1] - time_s[0]) if actions.shape[0] > 1 else 0.0
    if duration > 0.0:
        centered_time = time_s - np.mean(time_s)
        centered_action = actions - np.mean(actions, axis=0)
        slope = np.sum(centered_time[:, None] * centered_action, axis=0) / np.sum(
            centered_time**2
        )
    else:
        slope = np.zeros(13, dtype=np.float64)
    width = max(1, actions.shape[0] // 5)
    robust_change = np.median(actions[-width:], axis=0) - np.median(
        actions[:width], axis=0
    )
    return {
        "duration_s": duration,
        "linear_fit_slope_action_per_s": slope.tolist(),
        "linear_fit_change_over_capture": (slope * duration).tolist(),
        "last_minus_first_quintile_median": robust_change.tolist(),
        "maximum_abs_linear_fit_change": float(np.max(np.abs(slope * duration))),
        "maximum_abs_quintile_median_change": float(np.max(np.abs(robust_change))),
    }


def compare_action_trends(
    real_path: str | Path, *, bundle_path: str | Path = DEFAULT_BUNDLE
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return JSON report and compact plotting arrays for one offline comparison."""

    real_source = Path(real_path).expanduser().resolve()
    bundle_source = Path(bundle_path).expanduser().resolve()
    values = _load_real(real_source)
    real = _finite_actions(values["raw_policy_action13"], "real action")
    real_time, real_time_name = _time_axis(values, real.shape[0])

    bundle = DeployBundle(bundle_source)
    verification = bundle.verify()
    initial = bundle.load_npz(INITIAL_NPZ)
    closed = bundle.load_npz(CLOSED_NPZ)
    sim_reset = _finite_actions(initial["policy_action13"], "sim reset action")
    sim_closed = _finite_actions(closed["policy_action13"], "sim closed action")
    sim_reset_time = np.asarray(initial["time_s"], dtype=np.float64)
    sim_closed_time = np.asarray(closed["time_s"], dtype=np.float64)

    real_axis = _axis_stats(real)
    reset_axis = _axis_stats(sim_reset)
    closed_axis = _axis_stats(sim_closed)
    real_median = np.median(real, axis=0)
    reset_median = np.median(sim_reset, axis=0)
    reset_sign = _sign(reset_median)
    real_sign_match = np.mean(_sign(real) == reset_sign[None, :], axis=0)
    reset_sign_consensus = np.mean(_sign(sim_reset) == reset_sign[None, :], axis=0)

    nearest_reset_index, nearest_reset_linf = _nearest(real, sim_reset)
    nearest_closed_index, nearest_closed_linf = _nearest(real, sim_closed)
    nearest_reset_arm_cos = _cosine(
        real[:, :7], sim_reset[nearest_reset_index, :7]
    )
    arm_cos_to_reset_median = _cosine(real[:, :7], reset_median[:7])
    all_cos_to_reset_median = _cosine(real, reset_median)

    per_axis = []
    for index, name in enumerate(ACTION_NAMES):
        real_reset_offset = float(real_median[index] - reset_median[index])
        closed_reset_offset = float(sim_closed[-1, index] - reset_median[index])
        offset_direction_match: bool | None
        if abs(real_reset_offset) <= SIGN_EPSILON or abs(closed_reset_offset) <= SIGN_EPSILON:
            offset_direction_match = None
        else:
            offset_direction_match = bool(
                np.sign(real_reset_offset) == np.sign(closed_reset_offset)
            )
        item: dict[str, Any] = {
            "index": index,
            "name": name,
            "semantics": "increment" if index < 7 else "absolute_close_fraction",
            "sim_reset": reset_axis[index],
            "real_reset_shadow": real_axis[index],
            "sim_closed_loop": closed_axis[index],
            "sim_reset_dominant_sign": int(reset_sign[index]),
            "sim_reset_sign_consensus_fraction": float(reset_sign_consensus[index]),
            "real_sign_match_fraction": float(real_sign_match[index]),
            "real_minus_sim_reset_median": real_reset_offset,
            "absolute_median_shift": float(abs(real_reset_offset)),
            "sim_closed_last_minus_sim_reset_median": closed_reset_offset,
            "real_offset_direction_matches_closed_trajectory": offset_direction_match,
            "real_and_sim_reset_sample_ranges_overlap": bool(
                real_axis[index]["max"] >= reset_axis[index]["min"]
                and reset_axis[index]["max"] >= real_axis[index]["min"]
            ),
            "real_median_inside_sim_closed_sample_range": bool(
                closed_axis[index]["min"]
                <= real_median[index]
                <= closed_axis[index]["max"]
            ),
        }
        if index >= 7:
            item["sim_reset_median_close_fraction"] = float(
                (reset_median[index] + 1.0) * 0.5
            )
            item["real_median_close_fraction"] = float(
                (real_median[index] + 1.0) * 0.5
            )
            item["sim_closed_first_close_fraction"] = float(
                (sim_closed[0, index] + 1.0) * 0.5
            )
            item["sim_closed_last_close_fraction"] = float(
                (sim_closed[-1, index] + 1.0) * 0.5
            )
        per_axis.append(item)

    unique_modes: list[str] | None = None
    if "proposal_mode" in values:
        modes = np.asarray(values["proposal_mode"])
        if modes.shape != (real.shape[0],) or modes.dtype.kind not in "US":
            raise ValueError("proposal_mode must be string [N]")
        unique_modes = sorted(str(value) for value in np.unique(modes))

    frame_count: int | None = None
    if "pointcloud_frame_id" in values:
        frames = np.asarray(values["pointcloud_frame_id"])
        if frames.shape != (real.shape[0],) or frames.dtype.kind not in "iu":
            raise ValueError("pointcloud_frame_id must be integer [N]")
        frame_count = int(np.unique(frames).size)

    closed_counts = np.bincount(nearest_closed_index, minlength=sim_closed.shape[0])
    opposing_closed_trajectory = [
        {"index": item["index"], "name": item["name"]}
        for item in per_axis
        if item["real_offset_direction_matches_closed_trajectory"] is False
    ]
    largest_shifts = sorted(
        (
            {
                "index": item["index"],
                "name": item["name"],
                "absolute_median_shift": item["absolute_median_shift"],
                "signed_median_shift": item["real_minus_sim_reset_median"],
            }
            for item in per_axis
        ),
        key=lambda item: item["absolute_median_shift"],
        reverse=True,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "analysis_type": "offline_real_reset_vs_sim_action_trend_diagnostic",
        "sources": {
            "real_shadow_npz": str(real_source),
            "real_shadow_sha256": _sha256(real_source),
            "bundle_zip": str(bundle_source),
            "bundle_sha256": _sha256(bundle_source),
            "bundle_contract": verification.bundle_contract,
            "bundle_checked_files": verification.checked_files,
            "sim_reset_member": INITIAL_NPZ,
            "sim_closed_loop_member": CLOSED_NPZ,
        },
        "sample_counts": {
            "real_policy_rows": int(real.shape[0]),
            "real_unique_pointcloud_frames": frame_count,
            "sim_reset_idle_rows": int(sim_reset.shape[0]),
            "sim_closed_loop_rows": int(sim_closed.shape[0]),
        },
        "capture_contract": {
            "real_time_field": real_time_name,
            "real_proposal_modes": unique_modes,
            "real_is_stationary_reset_one_step_shadow": bool(
                unique_modes == ["one_step_from_measured_idle"]
            ),
            "hardware_writes": False,
            "robot_command_writes": False,
        },
        "comparison_semantics": {
            "primary_reference": "sim reset_idle_open",
            "closed_loop_reference_role": "action-family context only",
            "time_aligned_to_closed_loop": False,
            "reason": (
                "The real robot remained at reset and every prediction was a one-step "
                "proposal from measured idle. Simulation closed-loop rows after the first "
                "already contain policy-driven robot motion."
            ),
            "arm_axes_0_to_6": "normalized incremental joint commands",
            "hand_axes_7_to_12": (
                "normalized absolute semantic targets; close_fraction=(action+1)/2"
            ),
        },
        "reset_local_direction": {
            "sign_match_fraction_over_all_real_rows_per_axis": real_sign_match.tolist(),
            "all_13_axes_match_sim_reset_dominant_sign": bool(
                np.all(real_sign_match == 1.0)
            ),
            "arm7_cosine_to_sim_reset_median": _stats(arm_cos_to_reset_median),
            "arm7_cosine_at_nearest_sim_reset_linf_row": _stats(
                nearest_reset_arm_cos
            ),
            "all13_cosine_to_sim_reset_median": _stats(all_cos_to_reset_median),
            "nearest_sim_reset_action13_linf": _stats(nearest_reset_linf),
            "real_median_minus_sim_reset_median_linf": float(
                np.max(np.abs(real_median - reset_median))
            ),
        },
        "closed_loop_family_context": {
            "nearest_closed_loop_action13_linf": _stats(nearest_closed_linf),
            "nearest_closed_loop_row_counts": closed_counts.tolist(),
            "nearest_closed_loop_episode_steps": np.asarray(
                closed["episode_step"], dtype=np.int64
            ).tolist(),
            "fraction_closest_to_first_closed_loop_row": float(
                closed_counts[0] / real.shape[0]
            ),
            "real_median_axes_inside_closed_loop_sample_range": int(
                sum(item["real_median_inside_sim_closed_sample_range"] for item in per_axis)
            ),
            "axes_outside_closed_loop_sample_range": [
                {"index": item["index"], "name": item["name"]}
                for item in per_axis
                if not item["real_median_inside_sim_closed_sample_range"]
            ],
            "real_reset_offset_axes_opposing_closed_trajectory": (
                opposing_closed_trajectory
            ),
            "offset_direction_note": (
                "This compares the sign of the real-vs-reset magnitude offset with "
                "the sign from sim reset to the final packaged closed-loop row. It "
                "is context only, not time alignment or evidence of real closed-loop "
                "progress."
            ),
        },
        "temporal_diagnostics": {
            "real_stationary_reset": _temporal(real, real_time),
            "sim_reset_idle": _temporal(sim_reset, sim_reset_time),
            "sim_moving_closed_loop": _temporal(sim_closed, sim_closed_time),
            "sim_closed_last_minus_first": (sim_closed[-1] - sim_closed[0]).tolist(),
            "direct_trend_match_claimed": False,
        },
        "largest_reset_median_shifts": largest_shifts,
        "per_axis": per_axis,
        "interpretation": {
            "qualitative_reset_direction_consistent": bool(
                np.all(real_sign_match == 1.0)
                and np.median(arm_cos_to_reset_median) >= 0.95
            ),
            "exact_sim_reset_action_match_claimed": False,
            "closed_loop_action_trend_validated": False,
            "semantic_action_correctness_claimed": False,
            "physical_motion_authorized": False,
            "summary": (
                "The reset-local action direction is qualitatively consistent, but "
                "the magnitudes are materially shifted. The stationary capture cannot "
                "validate the moving closed-loop trend."
            ),
        },
    }

    arrays = {
        "action_names": np.asarray(ACTION_NAMES),
        "real_time_s": (real_time - real_time[0]).astype(np.float64),
        "real_action13": real.astype(np.float32),
        "sim_reset_time_s": sim_reset_time.astype(np.float64),
        "sim_reset_action13": sim_reset.astype(np.float32),
        "sim_closed_time_s": sim_closed_time.astype(np.float64),
        "sim_closed_action13": sim_closed.astype(np.float32),
        "nearest_sim_reset_index": nearest_reset_index.astype(np.int64),
        "nearest_sim_reset_linf": nearest_reset_linf.astype(np.float64),
        "nearest_sim_closed_index": nearest_closed_index.astype(np.int64),
        "nearest_sim_closed_linf": nearest_closed_linf.astype(np.float64),
        "arm7_cosine_to_sim_reset_median": arm_cos_to_reset_median.astype(np.float64),
        "all13_cosine_to_sim_reset_median": all_cos_to_reset_median.astype(np.float64),
    }
    return report, arrays


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real_shadow_npz", type=Path)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--npz-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report, arrays = compare_action_trends(
        args.real_shadow_npz, bundle_path=args.bundle
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.json_output is None:
        print(encoded, end="")
    else:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded, encoding="utf-8")
    if args.npz_output is not None:
        args.npz_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.npz_output, **arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
