#!/usr/bin/env python3
"""Compare recorded real policy inputs with successful simulation policy I/O.

This command is intentionally offline and read-only with respect to robot
hardware.  The real and simulation trajectories are not assumed to share a
clock, initial condition, length, or point ordering.  Consequently the report
contains schema checks, normalization reconstruction errors, distributional
statistics, and permutation-invariant point-cloud descriptors; it never
reports a tick-wise or point-wise trajectory error.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MAX_NPZ_BYTES = 4 * 1024 * 1024 * 1024
NORMALIZATION_RECONSTRUCTION_ATOL = 2.0e-5

POINT_RAW = "input_pointcloud_history_metric"
POINT_NORMALIZED = "input_pointcloud_history_normalized"
POINT_VALID = "input_pointcloud_valid_history"
PROPRIO_RAW = "input_proprio_history_raw"
PROPRIO_NORMALIZED = "input_proprio_history_normalized"
POINT_MEAN = "constant__normalization_pointcloud_mean"
POINT_STD = "constant__normalization_pointcloud_std"
PROPRIO_MEAN = "constant__normalization_proprio_mean"
PROPRIO_STD = "constant__normalization_proprio_std"

CORE_FIELDS = (
    POINT_RAW,
    POINT_NORMALIZED,
    POINT_VALID,
    PROPRIO_RAW,
    PROPRIO_NORMALIZED,
    POINT_MEAN,
    POINT_STD,
    PROPRIO_MEAN,
    PROPRIO_STD,
)

SEMANTIC_FIELDS: Mapping[str, tuple[str, ...]] = {
    "pointcloud_history_metric": (POINT_RAW,),
    "pointcloud_history_normalized": (POINT_NORMALIZED,),
    "proprio_history_raw": (PROPRIO_RAW,),
    "proprio_history_normalized": (PROPRIO_NORMALIZED,),
    "previous_executed_action13": (
        "input_previous_executed_action",
        "input_previous_policy_action13",
    ),
    "model_action13": ("output_model_action", "output_raw_action13"),
    "executed_action13": (
        "output_action_sent_to_env",
        "output_executed_action13",
    ),
}

CHECKPOINT_FIELDS = (
    "constant__checkpoint_sha256",
    "checkpoint_sha256",
    "policy_checkpoint_sha256",
    "checkpoint_sha",
)
METADATA_FIELDS = (
    "metadata_json",
    "constant__metadata_json",
    "policy_io_metadata_json",
)


@dataclass(frozen=True)
class PolicyIOArchive:
    path: Path
    arrays: Mapping[str, np.ndarray]
    checkpoint_sha256: str | None

    @property
    def steps(self) -> int:
        return int(self.arrays[POINT_RAW].shape[0])


def _safe_scalar_text(value: np.ndarray) -> str | None:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "SU":
        return None
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        try:
            item = item.decode("utf-8")
        except UnicodeDecodeError:
            return None
    text = str(item).strip()
    return text or None


def _valid_sha256(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        return None
    return text


def _find_checkpoint_sha(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in CHECKPOINT_FIELDS:
                result = _valid_sha256(item)
                if result is not None:
                    return result
        for item in value.values():
            result = _find_checkpoint_sha(item)
            if result is not None:
                return result
    elif isinstance(value, list):
        for item in value:
            result = _find_checkpoint_sha(item)
            if result is not None:
                return result
    return None


def _checkpoint_from_arrays(arrays: Mapping[str, np.ndarray]) -> str | None:
    for field in CHECKPOINT_FIELDS:
        if field not in arrays:
            continue
        result = _valid_sha256(_safe_scalar_text(arrays[field]))
        if result is not None:
            return result
    for field in METADATA_FIELDS:
        if field not in arrays:
            continue
        text = _safe_scalar_text(arrays[field])
        if text is None:
            continue
        try:
            result = _find_checkpoint_sha(json.loads(text))
        except (TypeError, ValueError):
            continue
        if result is not None:
            return result
    return None


def _checkpoint_from_sidecars(path: Path, search_root: Path) -> str | None:
    """Find a checksum in nearby JSON metadata without scanning arbitrary files."""

    root = search_root.resolve()
    current = path.parent.resolve()
    while True:
        for name in ("manifest.json", "summary.json", "verification_summary.json"):
            candidate = current / name
            if not candidate.is_file() or candidate.stat().st_size > 16 * 1024 * 1024:
                continue
            try:
                result = _find_checkpoint_sha(
                    json.loads(candidate.read_text(encoding="utf-8"))
                )
            except (OSError, UnicodeError, ValueError):
                continue
            if result is not None:
                return result
        if current == root or root not in current.parents:
            break
        current = current.parent
    return None


def _finite_float(array: np.ndarray, field: str, path: Path) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype.kind != "f":
        raise ValueError(f"{path}: {field} must be floating, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{path}: {field} contains non-finite values")
    return value


def _finite_mask(array: np.ndarray, field: str, path: Path) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype.kind not in "bifu":
        raise ValueError(f"{path}: {field} must be a numeric/bool mask, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{path}: {field} contains non-finite values")
    return value


def _validate_core(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    missing = sorted(set(CORE_FIELDS) - set(arrays))
    if missing:
        raise ValueError(f"{path}: missing core policy-I/O fields {missing}")

    points = _finite_float(arrays[POINT_RAW], POINT_RAW, path)
    points_normalized = _finite_float(
        arrays[POINT_NORMALIZED], POINT_NORMALIZED, path
    )
    valid = _finite_mask(arrays[POINT_VALID], POINT_VALID, path)
    proprio = _finite_float(arrays[PROPRIO_RAW], PROPRIO_RAW, path)
    proprio_normalized = _finite_float(
        arrays[PROPRIO_NORMALIZED], PROPRIO_NORMALIZED, path
    )
    if points.ndim != 4 or points.shape[0] < 1 or points.shape[-1] < 3:
        raise ValueError(
            f"{path}: {POINT_RAW} must be non-empty [T,H,N,D>=3], got {points.shape}"
        )
    if points_normalized.shape != points.shape:
        raise ValueError(
            f"{path}: normalized point shape {points_normalized.shape} differs from raw {points.shape}"
        )
    if valid.shape != points.shape[:3]:
        raise ValueError(
            f"{path}: {POINT_VALID} must have shape {points.shape[:3]}, got {valid.shape}"
        )
    if np.any(valid < -1.0e-6) or np.any(valid > 1.0 + 1.0e-6):
        raise ValueError(f"{path}: {POINT_VALID} contains values outside [0,1]")
    if proprio.ndim != 3 or proprio.shape[0] != points.shape[0]:
        raise ValueError(
            f"{path}: {PROPRIO_RAW} must be [T,H,P] with T={points.shape[0]}, got {proprio.shape}"
        )
    if proprio.shape[1] != points.shape[1]:
        raise ValueError(
            f"{path}: point/proprio history lengths differ: {points.shape[1]} vs {proprio.shape[1]}"
        )
    if proprio_normalized.shape != proprio.shape:
        raise ValueError(
            f"{path}: normalized proprio shape {proprio_normalized.shape} differs from raw {proprio.shape}"
        )

    for field, raw in ((POINT_MEAN, points), (POINT_STD, points)):
        constant = _finite_float(arrays[field], field, path)
        try:
            np.broadcast_to(constant, raw.shape)
        except ValueError as exc:
            raise ValueError(
                f"{path}: {field} shape {constant.shape} does not broadcast to {raw.shape}"
            ) from exc
    for field, raw in ((PROPRIO_MEAN, proprio), (PROPRIO_STD, proprio)):
        constant = _finite_float(arrays[field], field, path)
        try:
            np.broadcast_to(constant, raw.shape)
        except ValueError as exc:
            raise ValueError(
                f"{path}: {field} shape {constant.shape} does not broadcast to {raw.shape}"
            ) from exc
    if np.any(arrays[POINT_STD] <= 0.0):
        raise ValueError(f"{path}: {POINT_STD} must be strictly positive")
    if np.any(arrays[PROPRIO_STD] <= 0.0):
        raise ValueError(f"{path}: {PROPRIO_STD} must be strictly positive")


def _load_npz(path: Path, search_root: Path) -> PolicyIOArchive:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"policy-I/O NPZ not found: {source}")
    size = source.stat().st_size
    if size <= 0 or size > MAX_NPZ_BYTES:
        raise ValueError(f"{source}: unsafe NPZ size {size} bytes")
    try:
        with np.load(source, allow_pickle=False) as archive:
            names = set(archive.files)
            wanted = set(CORE_FIELDS)
            wanted.update(CHECKPOINT_FIELDS)
            wanted.update(METADATA_FIELDS)
            for aliases in SEMANTIC_FIELDS.values():
                wanted.update(aliases)
            arrays = {name: archive[name].copy() for name in names & wanted}
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid no-pickle policy-I/O NPZ {source}: {exc}") from exc
    _validate_core(source, arrays)
    checkpoint = _checkpoint_from_arrays(arrays)
    if checkpoint is None:
        checkpoint = _checkpoint_from_sidecars(source, search_root)
    return PolicyIOArchive(source, arrays, checkpoint)


def _resolve_sim_sources(path: Path) -> tuple[list[Path], Path]:
    source = path.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != ".npz":
            raise ValueError(f"--sim file must be an NPZ, got {source}")
        return [source], source.parent
    if not source.is_dir():
        raise FileNotFoundError(f"simulation NPZ/directory not found: {source}")
    files = sorted(candidate.resolve() for candidate in source.rglob("policy_io.npz"))
    if not files:
        raise FileNotFoundError(f"no policy_io.npz found recursively under {source}")
    return files, source


def _stats(value: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5.0)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def _reconstruction_report(archive: PolicyIOArchive) -> dict[str, Any]:
    arrays = archive.arrays
    reconstructed_points = (
        arrays[POINT_RAW].astype(np.float64)
        - arrays[POINT_MEAN].astype(np.float64)
    ) / arrays[POINT_STD].astype(np.float64)
    reconstructed_proprio = (
        arrays[PROPRIO_RAW].astype(np.float64)
        - arrays[PROPRIO_MEAN].astype(np.float64)
    ) / arrays[PROPRIO_STD].astype(np.float64)
    point_error = np.abs(
        reconstructed_points - arrays[POINT_NORMALIZED].astype(np.float64)
    )
    proprio_error = np.abs(
        reconstructed_proprio - arrays[PROPRIO_NORMALIZED].astype(np.float64)
    )

    def item(error: np.ndarray) -> dict[str, Any]:
        maximum = float(np.max(error))
        return {
            "max_abs_error": maximum,
            "p95_abs_error": float(np.percentile(error, 95.0)),
            "within_float32_tolerance": maximum <= NORMALIZATION_RECONSTRUCTION_ATOL,
            "tolerance": NORMALIZATION_RECONSTRUCTION_ATOL,
        }

    return {"pointcloud": item(point_error), "proprio": item(proprio_error)}


def _constant_comparison(real: np.ndarray, sim: np.ndarray) -> dict[str, Any]:
    first = np.asarray(real)
    second = np.asarray(sim)
    result: dict[str, Any] = {
        "real_shape": list(first.shape),
        "sim_shape": list(second.shape),
    }
    if first.shape != second.shape:
        result.update({"shape_compatible": False, "match": False})
        return result
    error = np.abs(first.astype(np.float64) - second.astype(np.float64))
    maximum = float(np.max(error)) if error.size else 0.0
    result.update(
        {
            "shape_compatible": True,
            "match": bool(maximum == 0.0),
            "max_abs_difference": maximum,
        }
    )
    return result


def _semantic_array(
    archive: PolicyIOArchive, aliases: Iterable[str]
) -> tuple[str | None, np.ndarray | None]:
    for name in aliases:
        if name in archive.arrays:
            return name, np.asarray(archive.arrays[name])
    return None, None


def _axis_distribution(real: np.ndarray, sim: np.ndarray) -> dict[str, Any] | None:
    first = np.asarray(real, dtype=np.float64)
    second = np.asarray(sim, dtype=np.float64)
    # A one-dimensional array represents scalar samples.  Its last dimension
    # is the sample count, not a feature axis, and expanding it produced huge,
    # misleading reports.  Only arrays with an explicit, reasonably-sized
    # final feature axis receive per-axis statistics.
    if (
        first.ndim < 2
        or second.ndim < 2
        or first.shape[-1] != second.shape[-1]
        or not 1 <= first.shape[-1] <= 128
    ):
        return None
    dimension = first.shape[-1]
    first = first.reshape(-1, dimension)
    second = second.reshape(-1, dimension)
    first_mean = np.mean(first, axis=0)
    second_mean = np.mean(second, axis=0)
    first_std = np.std(first, axis=0)
    second_std = np.std(second, axis=0)
    standardized = np.abs(first_mean - second_mean) / np.maximum(second_std, 1.0e-12)
    order = np.argsort(standardized)[::-1][: min(12, dimension)]
    top = [
        {
            "axis": int(index),
            "real_mean": float(first_mean[index]),
            "sim_mean": float(second_mean[index]),
            "real_std": float(first_std[index]),
            "sim_std": float(second_std[index]),
            "real_minus_sim_mean": float(first_mean[index] - second_mean[index]),
            "abs_mean_shift_in_sim_std": float(standardized[index]),
            "std_ratio_real_over_sim": float(
                first_std[index] / max(second_std[index], 1.0e-12)
            ),
        }
        for index in order
    ]
    return {"last_axis_dimension": int(dimension), "top_mean_shift_axes": top}


def _distribution_report(real: np.ndarray, sim: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "real": _stats(real),
        "sim": _stats(sim),
        "real_shape": list(np.asarray(real).shape),
        "sim_aggregate_shape": list(np.asarray(sim).shape),
    }
    axis = _axis_distribution(real, sim)
    if axis is not None:
        result["last_axis_distribution"] = axis
    return result


def _pointcloud_descriptors(
    points: np.ndarray, valid: np.ndarray
) -> dict[str, np.ndarray]:
    xyz = np.asarray(points, dtype=np.float64)[..., :3]
    mask = np.asarray(valid) > 0.5
    flat_xyz = xyz.reshape(-1, xyz.shape[-2], 3)
    flat_mask = mask.reshape(-1, mask.shape[-1])
    count = np.sum(flat_mask, axis=1).astype(np.float64)
    fraction = count / float(flat_mask.shape[1])
    centroid = np.full((flat_xyz.shape[0], 3), np.nan, dtype=np.float64)
    extent = np.full_like(centroid, np.nan)
    radius_median = np.full(flat_xyz.shape[0], np.nan, dtype=np.float64)
    radius_p95 = np.full_like(radius_median, np.nan)
    for index, (cloud, selected) in enumerate(zip(flat_xyz, flat_mask)):
        selected_cloud = cloud[selected]
        if selected_cloud.shape[0] == 0:
            continue
        center = np.mean(selected_cloud, axis=0)
        radii = np.linalg.norm(selected_cloud - center, axis=1)
        centroid[index] = center
        extent[index] = np.max(selected_cloud, axis=0) - np.min(selected_cloud, axis=0)
        radius_median[index] = np.median(radii)
        radius_p95[index] = np.percentile(radii, 95.0)
    return {
        "valid_point_count": count,
        "valid_fraction": fraction,
        "centroid_xyz_m": centroid,
        "extent_xyz_m": extent,
        "radius_median_m": radius_median,
        "radius_p95_m": radius_p95,
    }


def _schema_report(real: PolicyIOArchive, simulations: Sequence[PolicyIOArchive]) -> dict[str, Any]:
    sim0 = simulations[0]
    fields: dict[str, Any] = {}
    for name in CORE_FIELDS:
        first = real.arrays[name]
        second = sim0.arrays[name]
        if name in (POINT_MEAN, POINT_STD, PROPRIO_MEAN, PROPRIO_STD):
            compatible = first.shape == second.shape
        else:
            compatible = first.shape[1:] == second.shape[1:]
        fields[name] = {
            "real_shape": list(first.shape),
            "real_dtype": str(first.dtype),
            "sim_shape_example": list(second.shape),
            "sim_dtype_example": str(second.dtype),
            "compatible_ignoring_step_count": bool(compatible),
        }
    return {
        "core_fields_present": True,
        "all_core_shapes_compatible_ignoring_step_count": all(
            item["compatible_ignoring_step_count"] for item in fields.values()
        ),
        "fields": fields,
    }


def _assert_sim_consistency(simulations: Sequence[PolicyIOArchive]) -> None:
    reference = simulations[0]
    for archive in simulations[1:]:
        for name in CORE_FIELDS:
            first = reference.arrays[name]
            second = archive.arrays[name]
            if name in (POINT_MEAN, POINT_STD, PROPRIO_MEAN, PROPRIO_STD):
                if first.shape != second.shape or not np.array_equal(first, second):
                    raise ValueError(
                        f"simulation normalization constant {name} differs between "
                        f"{reference.path} and {archive.path}"
                    )
            elif first.shape[1:] != second.shape[1:]:
                raise ValueError(
                    f"simulation field {name} trailing shape differs between "
                    f"{reference.path} ({first.shape}) and {archive.path} ({second.shape})"
                )


def _checkpoint_report(
    real: PolicyIOArchive, simulations: Sequence[PolicyIOArchive]
) -> dict[str, Any]:
    sim_values = sorted(
        {archive.checkpoint_sha256 for archive in simulations if archive.checkpoint_sha256}
    )
    real_value = real.checkpoint_sha256
    if len(sim_values) > 1:
        status = "inconsistent_simulation_checkpoints"
        match: bool | None = False
    elif real_value is None:
        status = "unknown_real_checkpoint"
        match = None
    elif not sim_values:
        status = "unknown_simulation_checkpoint"
        match = None
    elif real_value == sim_values[0]:
        status = "match"
        match = True
    else:
        status = "mismatch"
        match = False
    return {
        "real_sha256": real_value,
        "simulation_sha256_values": sim_values,
        "match": match,
        "status": status,
    }


def compare_policy_io(real_path: str | Path, sim_path: str | Path) -> dict[str, Any]:
    """Build a JSON-serializable, distributional policy-I/O comparison."""

    real_source = Path(real_path).expanduser().resolve()
    sim_sources, sim_root = _resolve_sim_sources(Path(sim_path))
    real = _load_npz(real_source, real_source.parent)
    simulations = [_load_npz(source, sim_root) for source in sim_sources]
    _assert_sim_consistency(simulations)

    sim_arrays = {
        name: np.concatenate([archive.arrays[name] for archive in simulations], axis=0)
        for name in (
            POINT_RAW,
            POINT_NORMALIZED,
            POINT_VALID,
            PROPRIO_RAW,
            PROPRIO_NORMALIZED,
        )
    }

    normalization = {
        "real_reconstruction": _reconstruction_report(real),
        "simulation_reconstruction": [
            {
                "path": str(archive.path),
                **_reconstruction_report(archive),
            }
            for archive in simulations
        ],
        "constants_real_vs_sim": {
            name: _constant_comparison(real.arrays[name], simulations[0].arrays[name])
            for name in (POINT_MEAN, POINT_STD, PROPRIO_MEAN, PROPRIO_STD)
        },
    }

    distributions: dict[str, Any] = {}
    for semantic_name, aliases in SEMANTIC_FIELDS.items():
        real_field, real_value = _semantic_array(real, aliases)
        sim_parts: list[np.ndarray] = []
        sim_fields: list[str] = []
        for archive in simulations:
            field, value = _semantic_array(archive, aliases)
            if field is not None and value is not None:
                sim_fields.append(field)
                sim_parts.append(value)
        if real_value is None or len(sim_parts) != len(simulations):
            distributions[semantic_name] = {
                "available": False,
                "real_field": real_field,
                "simulation_fields": sorted(set(sim_fields)),
            }
            continue
        trailing = {part.shape[1:] for part in sim_parts}
        if len(trailing) != 1:
            raise ValueError(f"simulation semantic field {semantic_name} has mixed shapes")
        sim_value = np.concatenate(sim_parts, axis=0)
        distributions[semantic_name] = {
            "available": True,
            "real_field": real_field,
            "simulation_fields": sorted(set(sim_fields)),
            **_distribution_report(real_value, sim_value),
        }

    real_descriptors = _pointcloud_descriptors(
        real.arrays[POINT_RAW], real.arrays[POINT_VALID]
    )
    sim_descriptors = _pointcloud_descriptors(
        sim_arrays[POINT_RAW], sim_arrays[POINT_VALID]
    )
    descriptor_report = {
        "method": (
            "valid-point centroid, axis-aligned extent, and radial quantiles over every "
            "history cloud; invariant to point permutation"
        ),
        "point_order_assumed_aligned": False,
        "pointwise_rmse_computed": False,
        "descriptors": {
            name: _distribution_report(real_descriptors[name], sim_descriptors[name])
            for name in real_descriptors
        },
    }

    return {
        "format": "sim2real_policy_io_distributional_comparison_v1",
        "comparison_semantics": {
            "distributional_only": True,
            "exact_tick_alignment": False,
            "exact_time_alignment": False,
            "trajectory_initial_conditions_assumed_equal": False,
            "point_order_assumed_equal": False,
            "physical_control_performed": False,
            "physical_motion_authorized": False,
            "note": (
                "Different real/simulation trajectories are not paired samples; field "
                "statistics and point-cloud descriptors must not be interpreted as a "
                "per-tick control error."
            ),
        },
        "sources": {
            "real": str(real.path),
            "real_steps": real.steps,
            "simulation_input": str(Path(sim_path).expanduser().resolve()),
            "simulation_archives": [str(archive.path) for archive in simulations],
            "simulation_archive_count": len(simulations),
            "simulation_total_steps": int(sum(archive.steps for archive in simulations)),
        },
        "checkpoint": _checkpoint_report(real, simulations),
        "schema": _schema_report(real, simulations),
        "normalization": normalization,
        "distributional_fields": distributions,
        "pointcloud_permutation_invariant": descriptor_report,
    }


def _atomic_write_json(path: Path, report: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, type=Path, help="real recorded policy_io NPZ")
    parser.add_argument(
        "--sim",
        required=True,
        type=Path,
        help="one simulation NPZ or directory recursively containing policy_io.npz",
    )
    parser.add_argument("--output", type=Path, help="optional atomically-written JSON report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = compare_policy_io(args.real, args.sim)
        if args.output is not None:
            _atomic_write_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    except (OSError, ValueError) as exc:
        print(f"policy I/O comparison failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
