#!/usr/bin/python3
"""Offline, replayable filtering of FR3/V7/RH56 and object depth returns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from anydex_pipeline.hppfcl_installed_tool_backend import (  # noqa: E402
    HppFclInstalledToolBackend,
)
from anydex_pipeline.control_plan import is_official_snapshot  # noqa: E402
from anydex_pipeline.inspire_hand_model import InspireHandModel  # noqa: E402
from anydex_pipeline.inspire_open_configuration import (  # noqa: E402
    OFFICIAL_OPEN_JOINT_POSITIONS_RAD,
    official_open_configuration_provenance,
)
from anydex_pipeline.installed_scene_filter import filter_installed_scene  # noqa: E402
from anydex_pipeline.installed_tool_audit import V7_T_EE_HAND  # noqa: E402
from anydex_pipeline.snapshot import load_snapshot_npz  # noqa: E402


DEFAULT_ADAPTER = ROOT / "assets/adapter/V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.stl"
DEFAULT_ANYDEX_ROOT = ROOT / "third_party/AnyDexGrasp"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_indices(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(
        ("dtype=<i8;shape={};".format(",".join(map(str, array.shape)))).encode(
            "ascii"
        )
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _scalar(archive, key: str):
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError("live scene {} must be a scalar".format(key))
    return value.item()


def _load_live_scene(path: Path):
    source = path.expanduser().resolve()
    with np.load(str(source), allow_pickle=False) as archive:
        required = {
            "artifact_type",
            "schema_name",
            "schema_version",
            "scene_points",
            "scene_colors",
            "reference_frame",
            "T_reference_camera",
            "calibration_id",
            "calibration_source",
            "calibration_sha256",
            "camera_serial",
            "camera_name",
            "capture_q_rad",
            "capture_q_source",
            "frame_ids",
            "frame_timestamps_s",
            "frame_count",
            "capture_started_at_unix_s",
            "capture_completed_at_unix_s",
            "captured_at_unix_s",
            "points_per_frame",
            "config_path",
            "config_sha256",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError("live scene is missing {}".format(missing))
        if (
            str(_scalar(archive, "artifact_type")) != "dexgrasp_live_scene"
            or str(_scalar(archive, "schema_name")) != "dexgrasp_live_scene"
            or int(_scalar(archive, "schema_version")) != 1
        ):
            raise ValueError("live scene artifact type/schema is invalid")
        if str(_scalar(archive, "reference_frame")) != "robot_base":
            raise ValueError("live scene reference_frame must be robot_base")
        if str(_scalar(archive, "capture_q_source")) != "cli_asserted":
            raise ValueError(
                "live capture_q_source must be cli_asserted; runtime live-q replay "
                "remains mandatory"
            )
        points = np.asarray(archive["scene_points"], dtype=np.float64)
        colors = np.asarray(archive["scene_colors"], dtype=np.float64)
        q = np.asarray(archive["capture_q_rad"], dtype=np.float64)
        transform = np.asarray(archive["T_reference_camera"], dtype=np.float64)
        frame_ids = np.asarray(archive["frame_ids"], dtype=np.int64)
        frame_timestamps = np.asarray(
            archive["frame_timestamps_s"], dtype=np.float64
        )
        points_per_frame = np.asarray(archive["points_per_frame"], dtype=np.int64)
        frame_count = int(_scalar(archive, "frame_count"))
        capture_started = float(_scalar(archive, "capture_started_at_unix_s"))
        capture_completed = float(
            _scalar(archive, "capture_completed_at_unix_s")
        )
        captured_at = float(_scalar(archive, "captured_at_unix_s"))
        config_path = Path(str(_scalar(archive, "config_path"))).expanduser().resolve()
        calibration_path = Path(
            str(_scalar(archive, "calibration_source"))
        ).expanduser().resolve()
        config_sha256 = str(_scalar(archive, "config_sha256"))
        calibration_sha256 = str(_scalar(archive, "calibration_sha256"))
        metadata = {
            "captured_at_unix_s": captured_at,
            "calibration_id": str(_scalar(archive, "calibration_id")),
            "camera_serial": str(_scalar(archive, "camera_serial")),
            "camera_name": str(_scalar(archive, "camera_name")),
            "capture_q_source": "cli_asserted",
            "capture_provenance": {
                "artifact_type": "dexgrasp_live_scene",
                "schema_version": 1,
                "reference_frame": "robot_base",
                "capture_q_source": "cli_asserted",
                "frame_count": frame_count,
                "capture_started_at_unix_s": capture_started,
                "capture_completed_at_unix_s": capture_completed,
                "config_path": str(config_path),
                "config_sha256": config_sha256,
                "calibration_source": str(calibration_path),
                "calibration_sha256": calibration_sha256,
            },
        }
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise ValueError("live scene_points must be non-empty (N,3)")
    if colors.shape != points.shape:
        raise ValueError("live scene_colors must match scene_points")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(colors)):
        raise ValueError("live scene points/colors contain NaN or infinity")
    if np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ValueError("live scene colors must lie in [0,1]")
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError("live capture_q_rad must be a finite 7-vector")
    if (
        transform.shape != (4, 4)
        or not np.all(np.isfinite(transform))
        or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
        or not np.allclose(
            transform[:3, :3].T @ transform[:3, :3],
            np.eye(3),
            atol=1e-6,
            rtol=0.0,
        )
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-6)
    ):
        raise ValueError("live T_reference_camera must be a rigid transform")
    if (
        frame_count < 1
        or frame_ids.shape != (frame_count,)
        or frame_timestamps.shape != (frame_count,)
        or points_per_frame.shape != (frame_count,)
        or len(np.unique(frame_ids)) != frame_count
        or not np.all(np.isfinite(frame_timestamps))
        or np.any(points_per_frame <= 0)
        or int(np.sum(points_per_frame)) != len(points)
    ):
        raise ValueError("live frame provenance/counts are inconsistent")
    if (
        not np.isfinite(capture_started)
        or not np.isfinite(capture_completed)
        or not np.isfinite(captured_at)
        or capture_started <= 0.0
        or capture_completed < capture_started
        or not np.isclose(captured_at, capture_completed, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("live capture wall-clock provenance is inconsistent")
    for provenance_path, expected_hash, name in (
        (config_path, config_sha256, "camera config"),
        (calibration_path, calibration_sha256, "eye-to-hand calibration"),
    ):
        if not provenance_path.is_file():
            raise ValueError("live {} provenance file is missing".format(name))
        if _sha256_file(provenance_path) != expected_hash:
            raise ValueError("live {} provenance hash differs".format(name))
    if not metadata["calibration_id"] or not metadata["camera_serial"]:
        raise ValueError("live calibration_id/camera_serial must be non-empty")
    return source, points, colors, q, metadata


def _voxel_representatives(points: np.ndarray, voxel_m: float) -> np.ndarray:
    keys = np.floor(np.asarray(points, dtype=np.float64) / float(voxel_m)).astype(
        np.int64
    )
    _, indices = np.unique(keys, axis=0, return_index=True)
    return np.sort(indices.astype(np.int64))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove only replayably classified installed-tool and bound-object "
            "returns. This is offline preprocessing and never authorizes motion."
        )
    )
    parser.add_argument("--live-scene", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=None)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--anydex-root", type=Path, default=DEFAULT_ANYDEX_ROOT)
    parser.add_argument("--scene-voxel-m", type=float, default=0.005)
    parser.add_argument("--installed-inflation-margin-m", type=float, default=0.002)
    parser.add_argument("--object-return-max-distance-m", type=float, default=0.015)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--open-hand-q12",
        type=float,
        nargs=12,
        metavar=tuple("Q{}".format(index) for index in range(1, 13)),
        help=(
            "optional commissioned override; default is the official actuator-to-"
            "URDF result for all-six [1000]"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.002 <= float(args.scene_voxel_m) <= 0.02:
        raise SystemExit("--scene-voxel-m must be in [0.002,0.02]")
    if not 0.0 <= float(args.installed_inflation_margin_m) <= 0.02:
        raise SystemExit("--installed-inflation-margin-m must be in [0,0.02]")
    if not 0.002 <= float(args.object_return_max_distance_m) <= 0.04:
        raise SystemExit("--object-return-max-distance-m must be in [0.002,0.04]")
    output = args.output.expanduser().resolve()
    evidence_path = (
        args.evidence.expanduser().resolve()
        if args.evidence is not None
        else output.with_suffix(output.suffix + ".evidence.json")
    )
    for path in (output, evidence_path):
        if path.exists() and not args.overwrite:
            raise SystemExit("output exists; pass --overwrite: {}".format(path))

    live_input = args.live_scene.expanduser().resolve()
    snapshot_input = args.snapshot.expanduser().resolve()
    adapter_input = args.adapter.expanduser().resolve()
    before_hashes = {
        "live_scene": _sha256_file(live_input),
        "snapshot": _sha256_file(snapshot_input),
        "adapter": _sha256_file(adapter_input),
    }
    live_path, raw_points, raw_colors, capture_q, live_metadata = _load_live_scene(
        live_input
    )
    snapshot_path = snapshot_input
    snapshot = load_snapshot_npz(snapshot_path)
    if not is_official_snapshot(snapshot):
        raise SystemExit("snapshot is not provenance-complete official AnyDexGrasp output")
    if snapshot.reference_frame != "robot_base":
        raise SystemExit("snapshot reference_frame must be robot_base")
    if snapshot.calibration_id != str(live_metadata.get("calibration_id", "")):
        raise SystemExit("snapshot/live-scene calibration_id mismatch")
    if snapshot.camera_serial != str(live_metadata.get("camera_serial", "")):
        raise SystemExit("snapshot/live-scene camera_serial mismatch")

    representative_indices = _voxel_representatives(
        raw_points, float(args.scene_voxel_m)
    )
    scene_points = raw_points[representative_indices]
    scene_colors = raw_colors[representative_indices]
    hand = InspireHandModel.from_anydex_root(
        args.anydex_root, mesh_resolution="full"
    )
    open_q12 = (
        np.asarray(args.open_hand_q12, dtype=np.float64)
        if args.open_hand_q12 is not None
        else OFFICIAL_OPEN_JOINT_POSITIONS_RAD.copy()
    )
    open_configuration_source = (
        {
            "method": "explicit CLI commissioned override",
            "joint_positions_rad": open_q12.tolist(),
        }
        if args.open_hand_q12 is not None
        else dict(official_open_configuration_provenance(args.anydex_root))
    )
    open_transforms = hand.link_mesh_transforms(np.eye(4), open_q12)
    mesh_directory = hand.urdf_path.parent.parent / "meshes"
    mesh_paths = {
        link.name: (mesh_directory / link.mesh_filename).resolve()
        for link in hand.links
    }
    backend = HppFclInstalledToolBackend()
    result = filter_installed_scene(
        backend,
        scene_points,
        snapshot.object_points,
        capture_q,
        adapter_stl_path=adapter_input,
        T_EE_hand=V7_T_EE_HAND,
        hand_link_mesh_paths=mesh_paths,
        T_hand_open_link_visual=open_transforms,
        scene_colors_base=scene_colors,
        point_half_extent_m=0.5 * float(args.scene_voxel_m),
        installed_inflation_margin_m=float(args.installed_inflation_margin_m),
        object_return_max_distance_m=float(args.object_return_max_distance_m),
    )
    after_hashes = {
        "live_scene": _sha256_file(live_path),
        "snapshot": _sha256_file(snapshot_path),
        "adapter": _sha256_file(adapter_input),
    }
    if after_hashes != before_hashes:
        raise SystemExit("a live-scene/filter input changed during processing")
    kept_raw_indices = representative_indices[result.kept_original_indices]
    installed_raw_indices = representative_indices[result.installed_return_indices]
    object_raw_indices = representative_indices[result.object_return_indices]
    filtered_colors = scene_colors[result.kept_original_indices]
    filter_created_at_unix_s = time.time()
    source_capture_time = float(live_metadata.get("captured_at_unix_s", 0.0))
    source_scene_age_s = (
        max(0.0, filter_created_at_unix_s - source_capture_time)
        if source_capture_time > 0.0
        else None
    )

    evidence = dict(result.evidence)
    evidence.update(
        {
            "live_scene_path": str(live_path),
            "live_scene_sha256": before_hashes["live_scene"],
            "snapshot_path": str(snapshot_path),
            "snapshot_sha256": before_hashes["snapshot"],
            "adapter_path": str(adapter_input),
            "adapter_sha256": before_hashes["adapter"],
            "raw_scene_point_count": int(len(raw_points)),
            "voxel_representative_count": int(len(representative_indices)),
            "voxel_representative_raw_indices_sha256": _sha256_indices(
                representative_indices
            ),
            "kept_raw_indices_sha256": _sha256_indices(kept_raw_indices),
            "installed_return_raw_indices_sha256": _sha256_indices(
                installed_raw_indices
            ),
            "object_return_raw_indices_sha256": _sha256_indices(object_raw_indices),
            "open_hand_joint_positions_rad": open_q12.tolist(),
            "open_hand_fk_commissioned": True,
            "open_hand_configuration_provenance": open_configuration_source,
            "filter_created_at_unix_s": filter_created_at_unix_s,
            "source_scene_age_s": source_scene_age_s,
            "fresh_for_motion_audit": False,
            "freshness_note": (
                "saved/offline filtering cannot establish execution-time scene freshness"
            ),
            "runtime_operator_clearance_confirmation_required": True,
            "live_scene_capture_provenance": live_metadata[
                "capture_provenance"
            ],
        }
    )
    evidence_bytes = (
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    evidence_sha256 = _sha256_bytes(evidence_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output),
        artifact_type=np.asarray("installed_filtered_live_scene"),
        schema_version=np.asarray(1, dtype=np.int64),
        scene_points=np.asarray(result.filtered_scene_points_base, dtype=np.float32),
        filtered_scene_points=np.asarray(
            result.filtered_scene_points_base, dtype=np.float32
        ),
        scene_colors=np.asarray(filtered_colors, dtype=np.float32),
        reference_frame=np.asarray("robot_base"),
        scene_excludes_object=np.asarray(True, dtype=np.bool_),
        capture_q_rad=np.asarray(capture_q, dtype=np.float64),
        capture_q_source=np.asarray(str(live_metadata["capture_q_source"])),
        captured_at_unix_s=np.asarray(
            float(live_metadata.get("captured_at_unix_s", 0.0)), dtype=np.float64
        ),
        calibration_id=np.asarray(str(live_metadata.get("calibration_id", ""))),
        camera_serial=np.asarray(str(live_metadata.get("camera_serial", ""))),
        source_live_scene_path=np.asarray(str(live_path)),
        source_live_scene_sha256=np.asarray(before_hashes["live_scene"]),
        filter_evidence_path=np.asarray(str(evidence_path)),
        filter_evidence_sha256=np.asarray(evidence_sha256),
        installed_inflation_margin_m=np.asarray(
            float(args.installed_inflation_margin_m), dtype=np.float64
        ),
        kept_raw_indices=np.asarray(kept_raw_indices, dtype=np.int64),
        installed_return_raw_indices=np.asarray(installed_raw_indices, dtype=np.int64),
        object_return_raw_indices=np.asarray(object_raw_indices, dtype=np.int64),
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = evidence_path.with_name(
        ".{}.{}.tmp".format(evidence_path.name, os.getpid())
    )
    temporary.write_bytes(evidence_bytes)
    os.replace(str(temporary), str(evidence_path))
    print(
        "[filter] raw={} voxel={} installed={} object={} kept={}".format(
            len(raw_points),
            len(representative_indices),
            len(result.installed_return_indices),
            len(result.object_return_indices),
            len(result.kept_original_indices),
        )
    )
    print("[filter] output={}".format(output))
    print("[filter] evidence={}".format(evidence_path))
    print("[filter] motion_authorized=false; unknown camera space remains unverified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
