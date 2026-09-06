"""Hardware-free regression benchmark for recorded object-mask sequences.

The recorded videos do not have dense human labels.  A case-specific HSV
component therefore provides conservative dense temporal proxy metrics.  An
optional, content-addressed set of sparse visually reviewed visible-object masks
adds true IoU/recall/precision/contamination on selected key frames.  The two
evidence sources remain separate instead of presenting the HSV proxy as ground
truth.  The same evaluator accepts masks exported by guarded_v2, SAM2, Cutie or
any future implementation.

No camera, Franka or RH56 module is imported here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import cv2
import numpy as np

from dynamic_pcd.segmentation.sam2_video_backend import (
    PRODUCTION_VOS_COMPONENT_COMPILE_MODES,
    PRODUCTION_VOS_COMPONENT_DYNAMIC,
    PRODUCTION_VOS_COMPILE_MODE,
    SAM2_MEMORY_ATTENTION_STRIDE,
    VOS_COMPILE_PREWARM_CONTRACT,
)


SCHEMA_VERSION = 1
SPARSE_GROUND_TRUTH_SCHEMA = "reviewed_visible_object_masks_v1"
SPARSE_GROUND_TRUTH_MANIFEST_SCHEMA = "reviewed_sparse_mask_manifest_v1"
REVIEWED_MOTION_TRACK_SCHEMA = "reviewed_visible_bbox_track_v1"
PRODUCTION_REPLAY_SUMMARY_SCHEMA = "guarded_v2_real_rgb_replay_summary_v2"
PRODUCTION_CASE_NAMES = (
    "fast_green_ball_entry",
    "rolling_green_ball",
    "rolling_red_cylinder",
    "static_green_ball_rh56",
    "rh56_heavy_occlusion",
)
SPARSE_GROUND_TRUTH_REVIEW_STATUSES = frozenset(
    {"human_confirmed", "independent_visual_review"}
)
REVIEWED_MOTION_TRACK_THRESHOLDS = {
    "reviewed_centroid_error_p95_max": 0.35,
    "reviewed_centroid_error_max": 0.75,
    "reviewed_motion_residual_p95_max": 0.40,
}
PRODUCTION_SUITE_COVERAGE_SCHEMA = "guarded_v2_suite_coverage_v1"
PRODUCTION_COMPLETE_OCCLUSION_EVENTS = (
    ("static_green_ball_rh56", (245, 245), 246),
    ("static_green_ball_rh56", (290, 292), 293),
    ("static_green_ball_rh56", (307, 310), 311),
)
PRODUCTION_HEAVY_PARTIAL_OCCLUSION_SOURCES = (91, 121, 211, 271)
PRODUCTION_POINTCLOUD_ACCEPTANCE_SCOPE = (
    "separate_saved_rgbd_guarded_v2_current_provider_exact_128_point_gate"
)


class OfflineBenchmarkError(RuntimeError):
    """Raised when a benchmark artifact is incomplete or inconsistent."""


def _resolve_path(value: Any, base: Path, *, description: str) -> Path:
    candidates = value if isinstance(value, list) else [value]
    attempted: list[str] = []
    for raw in candidates:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        attempted.append(str(path))
        if path.exists():
            return path
    raise OfflineBenchmarkError(
        f"no existing {description}; attempted: {', '.join(attempted)}"
    )


def _sample_indices(
    frame_count: int,
    *,
    source_fps: float,
    start_frame: int,
    evaluation_fps: float | None,
) -> list[int]:
    if frame_count <= 0:
        return []
    if start_frame < 0 or start_frame >= frame_count:
        raise OfflineBenchmarkError(
            f"source_start_frame={start_frame} is outside 0..{frame_count - 1}"
        )
    if evaluation_fps is None:
        return list(range(start_frame, frame_count))
    if not (0.0 < evaluation_fps <= source_fps + 1e-9):
        raise OfflineBenchmarkError(
            f"evaluation_fps_hz must be in (0, source_fps={source_fps}]"
        )
    step = source_fps / evaluation_fps
    indices: list[int] = []
    sample = 0
    while True:
        # Matches ffmpeg's 30 -> 20 Hz selection used to create the historical
        # comparison clips: 0, 2, 3, 5, 6, 8, ... .
        index = start_frame + int(math.floor(sample * step + 0.5))
        if index >= frame_count:
            break
        if not indices or index != indices[-1]:
            indices.append(index)
        sample += 1
    return indices


def _iter_sampled_video(
    path: Path,
    *,
    start_frame: int = 0,
    evaluation_fps: float | None = None,
) -> tuple[float, list[int], Iterator[np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OfflineBenchmarkError(f"cannot open video: {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(source_fps) or source_fps <= 0.0 or frame_count <= 0:
        capture.release()
        raise OfflineBenchmarkError(f"video has invalid FPS/frame count: {path}")
    indices = _sample_indices(
        frame_count,
        source_fps=source_fps,
        start_frame=start_frame,
        evaluation_fps=evaluation_fps,
    )

    def generate() -> Iterator[np.ndarray]:
        next_position = 0
        try:
            for source_index in range(frame_count):
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise OfflineBenchmarkError(
                        f"video ended at frame {source_index}/{frame_count}: {path}"
                    )
                if next_position >= len(indices):
                    break
                if source_index == indices[next_position]:
                    next_position += 1
                    yield frame
        finally:
            capture.release()

    return source_fps, indices, generate()


def _mask_files(path: Path) -> list[Path]:
    files = sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not files:
        raise OfflineBenchmarkError(f"mask directory is empty: {path}")
    return files


def _iter_masks(spec: dict[str, Any], base: Path) -> tuple[int, Iterator[np.ndarray]]:
    kind = str(spec.get("kind", "directory"))
    path = _resolve_path(spec.get("path"), base, description="candidate mask path")
    threshold = int(spec.get("threshold", 127))
    if kind == "directory":
        files = _mask_files(path)

        def generate_directory() -> Iterator[np.ndarray]:
            for item in files:
                mask = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise OfflineBenchmarkError(f"cannot read mask: {item}")
                yield mask > threshold

        return len(files), generate_directory()
    if kind == "video":
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise OfflineBenchmarkError(f"cannot open mask video: {path}")
        source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        mask_start = int(spec.get("start_frame", 0))
        mask_evaluation_fps = spec.get("evaluation_fps_hz")
        if mask_evaluation_fps is not None:
            mask_evaluation_fps = float(mask_evaluation_fps)
        indices = _sample_indices(
            source_count,
            source_fps=source_fps,
            start_frame=mask_start,
            evaluation_fps=mask_evaluation_fps,
        )

        def generate_video() -> Iterator[np.ndarray]:
            position = 0
            try:
                for source_index in range(source_count):
                    ok, frame = capture.read()
                    if not ok:
                        raise OfflineBenchmarkError(
                            f"mask video ended at {source_index}/{source_count}: {path}"
                        )
                    if position >= len(indices):
                        break
                    if source_index != indices[position]:
                        continue
                    position += 1
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    yield gray > threshold
            finally:
                capture.release()

        return len(indices), generate_video()
    raise OfflineBenchmarkError(f"unsupported mask kind={kind!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise OfflineBenchmarkError(f"missing {description}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfflineBenchmarkError(f"invalid {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OfflineBenchmarkError(f"{description} must be a JSON object: {path}")
    return payload


def _video_sampling_contract(case: dict[str, Any], base: Path) -> dict[str, Any]:
    name = str(case.get("name", ""))
    video = _resolve_path(case.get("video"), base, description=f"{name} video")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OfflineBenchmarkError(f"cannot open video: {video}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        size_wh = [
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ]
    finally:
        capture.release()
    evaluation_fps = float(case.get("evaluation_fps_hz", fps))
    source_start = int(case.get("source_start_frame", 0))
    indices = _sample_indices(
        frame_count,
        source_fps=fps,
        start_frame=source_start,
        evaluation_fps=evaluation_fps,
    )
    return {
        "video": video,
        "sha256": _sha256_file(video),
        "fps_hz": fps,
        "frame_count": frame_count,
        "size_wh": size_wh,
        "evaluation_fps_hz": evaluation_fps,
        "sampled_source_indices": indices,
    }


def _verify_candidate_artifacts(
    *,
    case_name: str,
    case_root: Path,
    case_summary: dict[str, Any],
    expected_source_indices: list[int],
) -> str:
    artifact = case_summary.get("candidate_artifacts")
    if not isinstance(artifact, dict):
        raise OfflineBenchmarkError(
            f"case {case_name}: replay summary lacks candidate_artifacts"
        )
    raw_masks = artifact.get("masks")
    if not isinstance(raw_masks, list) or len(raw_masks) != len(
        expected_source_indices
    ):
        raise OfflineBenchmarkError(
            f"case {case_name}: artifact mask manifest is incomplete"
        )
    masks_dir = case_root / "masks"
    if not masks_dir.is_dir():
        raise OfflineBenchmarkError(f"case {case_name}: missing masks dir")
    expected_names = {f"{index:06d}.png" for index in range(len(raw_masks))}
    actual_names = {
        item.name
        for item in masks_dir.iterdir()
        if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    if actual_names != expected_names:
        raise OfflineBenchmarkError(
            f"case {case_name}: mask filenames must be exact zero-based PNGs; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    normalized_masks: list[dict[str, Any]] = []
    for frame_index, source_frame_index in enumerate(expected_source_indices):
        raw = raw_masks[frame_index]
        expected_relative = f"masks/{frame_index:06d}.png"
        if not isinstance(raw, dict):
            raise OfflineBenchmarkError(
                f"case {case_name}: mask artifact {frame_index} is not an object"
            )
        expected_record = {
            "frame_index": frame_index,
            "source_frame_index": int(source_frame_index),
            "path": expected_relative,
            "sha256": _sha256_file(case_root / expected_relative),
        }
        normalized = {
            "frame_index": int(raw.get("frame_index", -1)),
            "source_frame_index": int(raw.get("source_frame_index", -1)),
            "path": str(raw.get("path", "")),
            "sha256": str(raw.get("sha256", "")).lower(),
        }
        if normalized != expected_record:
            raise OfflineBenchmarkError(
                f"case {case_name}: mask artifact {frame_index} digest/index/path "
                "does not match produced bytes"
            )
        normalized_masks.append(normalized)
    states_path = case_root / "states.jsonl"
    states = artifact.get("states")
    expected_states = {
        "path": "states.jsonl",
        "sha256": _sha256_file(states_path) if states_path.is_file() else "",
    }
    if not isinstance(states, dict) or {
        "path": str(states.get("path", "")),
        "sha256": str(states.get("sha256", "")).lower(),
    } != expected_states:
        raise OfflineBenchmarkError(
            f"case {case_name}: states.jsonl digest/path does not match summary"
        )
    content = {"masks": normalized_masks, "states": expected_states}
    aggregate = _canonical_json_sha256(content)
    if str(artifact.get("aggregate_sha256", "")).lower() != aggregate:
        raise OfflineBenchmarkError(
            f"case {case_name}: candidate artifact aggregate digest mismatch"
        )
    return aggregate


def _verify_production_sam2_readiness_provenance(
    *,
    case_name: str,
    case_summary: dict[str, Any],
    config_provenance: dict[str, Any],
    source_size_wh: list[int],
    seed_source_frame: int,
) -> None:
    """Require both service compile readiness and bbox-session readiness."""

    bbox = case_summary.get("online_sam2_bbox_prewarm")
    if not isinstance(bbox, dict):
        raise OfflineBenchmarkError(
            f"case {case_name}: missing online-SAM2 bbox prewarm provenance"
        )
    try:
        bbox_timings = [
            float(bbox[name])
            for name in (
                "max_rpc_ms",
                "exact_reseed_rpc_ms",
                "total_ms",
            )
        ]
        required_tracks = int(bbox["required_stable_tracks"])
        achieved_tracks = int(bbox["achieved_stable_tracks"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise OfflineBenchmarkError(
            f"case {case_name}: malformed bbox prewarm provenance"
        ) from exc
    if (
        bbox.get("enabled") is not True
        or bbox.get("attempted") is not True
        or bbox.get("passed") is not True
        or int(bbox.get("seed_frame_id", -1)) != int(seed_source_frame)
        or required_tracks < 1
        or achieved_tracks < required_tracks
        or bbox.get("status")
        != "discarded_tracks_stable_then_exact_seed_restored"
        or any(not np.isfinite(value) or value < 0.0 for value in bbox_timings)
    ):
        raise OfflineBenchmarkError(
            f"case {case_name}: online-SAM2 bbox prewarm did not prove a hot, "
            "exactly re-seeded session"
        )

    health = case_summary.get("online_sam2_service_health")
    if not isinstance(health, dict):
        raise OfflineBenchmarkError(
            f"case {case_name}: missing online-SAM2 service-health provenance"
        )
    expected_image_size = int(case_summary.get("sam2_image_size", -1))
    rope_side = expected_image_size // SAM2_MEMORY_ATTENTION_STRIDE
    expected_rope_grid = [rope_side, rope_side]
    expected_rope_tokens = rope_side * rope_side
    token_counts = health.get("vos_memory_attention_rope_cache_token_counts")
    try:
        prewarm_timings = [
            float(health[name])
            for name in (
                "compile_prewarm_initialize_box_ms",
                "compile_prewarm_track_ms",
                "compile_prewarm_initialize_mask_ms",
                "compile_prewarm_mask_track_ms",
            )
        ]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise OfflineBenchmarkError(
            f"case {case_name}: malformed service compile-prewarm timings"
        ) from exc
    expected_shape_hw = [int(source_size_wh[1]), int(source_size_wh[0])]
    if (
        health.get("backend") != "official-sam2-video-predictor"
        or health.get("loaded") is not True
        or health.get("initialized") is not False
        or str(health.get("checkpoint_sha256", "")).lower()
        != str(config_provenance.get("checkpoint_sha256", "")).lower()
        or str(health.get("model_config_sha256", "")).lower()
        != str(config_provenance.get("model_config_sha256", "")).lower()
        or int(health.get("image_size", -1)) != expected_image_size
        or health.get("vos_optimized") is not True
        or health.get("vos_compile_mode") != PRODUCTION_VOS_COMPILE_MODE
        or health.get("vos_compile_cuda_graphs") is not False
        or health.get("vos_component_compile_modes")
        != dict(PRODUCTION_VOS_COMPONENT_COMPILE_MODES)
        or health.get("vos_component_compile_dynamic")
        != dict(PRODUCTION_VOS_COMPONENT_DYNAMIC)
        or health.get("vos_memory_attention_rope_grid_hw") != expected_rope_grid
        or health.get("vos_memory_attention_rope_expected_tokens")
        != expected_rope_tokens
        or health.get("vos_memory_attention_rope_caches_verified") is not True
        or not isinstance(token_counts, list)
        or not token_counts
        or any(int(value) != expected_rope_tokens for value in token_counts)
        or int(health.get("vos_memory_attention_rope_cache_count", -1))
        != len(token_counts)
        or health.get("compile_prewarm_required") is not True
        or health.get("compile_prewarm_completed") is not True
        or health.get("compile_prewarm_contract") != VOS_COMPILE_PREWARM_CONTRACT
        or health.get("compile_prewarm_shape_hw") != expected_shape_hw
        or any(not np.isfinite(value) or value < 0.0 for value in prewarm_timings)
    ):
        raise OfflineBenchmarkError(
            f"case {case_name}: online-SAM2 service health differs from the "
            "compiled production identity/readiness contract"
        )


def _verify_replay_candidate_root(
    *,
    root: Path,
    cases: list[dict[str, Any]],
    manifest: dict[str, Any],
    manifest_base: Path,
    require_production: bool,
) -> dict[str, dict[str, Any]]:
    root_summary_path = root / "summary.json"
    root_summary = _load_json_object(
        root_summary_path, description="candidate-root replay summary"
    )
    if root_summary.get("schema") != PRODUCTION_REPLAY_SUMMARY_SCHEMA:
        raise OfflineBenchmarkError(
            f"candidate root schema must be {PRODUCTION_REPLAY_SUMMARY_SCHEMA!r}"
        )
    expected_names = tuple(str(case.get("name", "")) for case in cases)
    root_cases = root_summary.get("cases")
    if not isinstance(root_cases, list):
        raise OfflineBenchmarkError("candidate root cases must be a list")
    root_names = tuple(
        str(item.get("case", "")) if isinstance(item, dict) else ""
        for item in root_cases
    )
    if root_names != expected_names or int(root_summary.get("case_count", -1)) != len(
        cases
    ):
        raise OfflineBenchmarkError(
            f"candidate root case set/order {root_names} != benchmark {expected_names}"
        )
    actual_case_dirs = {
        item.name
        for item in root.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    }
    if actual_case_dirs != set(expected_names):
        raise OfflineBenchmarkError(
            "candidate root must contain exactly the declared case directories; "
            f"missing={sorted(set(expected_names) - actual_case_dirs)}, "
            f"extra={sorted(actual_case_dirs - set(expected_names))}"
        )
    if require_production:
        if expected_names != PRODUCTION_CASE_NAMES:
            raise OfflineBenchmarkError(
                "production candidate evaluation requires exactly the fixed five cases"
            )
        expected_replay_manifest_sha = str(
            manifest.get("expected_replay_manifest_sha256", "")
        ).lower()
        if not expected_replay_manifest_sha:
            raise OfflineBenchmarkError(
                "production benchmark must pin expected_replay_manifest_sha256"
            )
        if str(root_summary.get("replay_manifest_sha256", "")).lower() != (
            expected_replay_manifest_sha
        ):
            raise OfflineBenchmarkError("candidate root replay-manifest digest mismatch")
        for key in (
            "production_acceptance_eligible",
            "production_defaults_used",
            "realtime_schedule_preserved",
        ):
            if root_summary.get(key) is not True:
                raise OfflineBenchmarkError(
                    f"candidate root production field {key} must be true"
                )
        if root_summary.get("diagnostic_only") is not False or root_summary.get(
            "diagnostic_only_not_production_acceptance"
        ) is not False:
            raise OfflineBenchmarkError("diagnostic replay cannot receive production PASS")
        if root_summary.get("production_ineligibility_reasons") != []:
            raise OfflineBenchmarkError(
                "production replay has non-empty ineligibility reasons"
            )
        contract = root_summary.get("production_acceptance_contract")
        config_provenance = root_summary.get("effective_config_provenance")
        if not isinstance(contract, dict) or not isinstance(config_provenance, dict):
            raise OfflineBenchmarkError(
                "candidate root lacks production/config provenance contract"
            )
        if tuple(contract.get("required_case_names") or ()) != PRODUCTION_CASE_NAMES:
            raise OfflineBenchmarkError("production contract case names are not fixed")
        expected_contract = {
            "default_config_sha256": config_provenance.get(
                "source_config_sha256"
            ),
            "checkpoint_sha256": config_provenance.get("checkpoint_sha256"),
            "model_config_sha256": config_provenance.get("model_config_sha256"),
            "default_neutral_depth_m": 1.0,
            "default_sam2_image_size": 512,
        }
        for key, expected in expected_contract.items():
            if contract.get(key) != expected:
                raise OfflineBenchmarkError(
                    f"production contract {key} does not match effective provenance"
                )
        source_contract_fields = {
            "implementation_source_manifest",
            "implementation_source_manifest_sha256",
            "implementation_source_set_sha256",
        }
        supplied_source_fields = source_contract_fields.intersection(contract)
        if supplied_source_fields:
            if supplied_source_fields != source_contract_fields:
                raise OfflineBenchmarkError(
                    "production implementation-source contract is incomplete"
                )
            source_provenance = root_summary.get(
                "implementation_source_provenance"
            )
            if not isinstance(source_provenance, dict) or source_provenance.get(
                "validated"
            ) is not True:
                raise OfflineBenchmarkError(
                    "candidate root lacks validated implementation-source provenance"
                )
            if source_provenance.get("manifest_sha256") != contract.get(
                "implementation_source_manifest_sha256"
            ):
                raise OfflineBenchmarkError(
                    "candidate implementation source-manifest digest mismatch"
                )
            if source_provenance.get("source_set_sha256") != contract.get(
                "implementation_source_set_sha256"
            ):
                raise OfflineBenchmarkError(
                    "candidate implementation source-set digest mismatch"
                )
            expected_source_manifest_sha = str(
                manifest.get("expected_implementation_source_manifest_sha256", "")
            ).lower()
            expected_source_set_sha = str(
                manifest.get("expected_implementation_source_set_sha256", "")
            ).lower()
            expected_source_pins = (
                expected_source_manifest_sha,
                expected_source_set_sha,
            )
            if any(expected_source_pins) and not all(expected_source_pins):
                raise OfflineBenchmarkError(
                    "benchmark implementation-source pins are incomplete"
                )
            if expected_source_manifest_sha and (
                source_provenance.get("manifest_sha256")
                != expected_source_manifest_sha
            ):
                raise OfflineBenchmarkError(
                    "candidate source-manifest digest differs from benchmark pin"
                )
            if expected_source_set_sha and (
                source_provenance.get("source_set_sha256")
                != expected_source_set_sha
            ):
                raise OfflineBenchmarkError(
                    "candidate source-set digest differs from benchmark pin"
                )

    case_by_name: dict[str, dict[str, Any]] = {}
    chain = root_summary.get("case_summary_chain")
    if not isinstance(chain, list) or len(chain) != len(cases):
        raise OfflineBenchmarkError("candidate root case_summary_chain is incomplete")
    normalized_chain: list[dict[str, Any]] = []
    root_config = root_summary.get("effective_config_provenance")
    for position, (case, root_case, raw_chain) in enumerate(
        zip(cases, root_cases, chain)
    ):
        name = str(case["name"])
        case_root = root / name
        case_summary_path = case_root / "summary.json"
        case_summary = _load_json_object(
            case_summary_path, description=f"{name} replay summary"
        )
        if case_summary != root_case:
            raise OfflineBenchmarkError(
                f"case {name}: root and per-case replay summaries differ"
            )
        sampling = _video_sampling_contract(case, manifest_base)
        expected_indices = sampling["sampled_source_indices"]
        mandatory_equal = {
            "schema": PRODUCTION_REPLAY_SUMMARY_SCHEMA,
            "case": name,
            "source_video_sha256": sampling["sha256"],
            "sampled_frames": len(expected_indices),
            "sampled_source_indices": expected_indices,
            "evaluation_fps_hz": 20.0,
        }
        for key, expected in mandatory_equal.items():
            if case_summary.get(key) != expected:
                raise OfflineBenchmarkError(
                    f"case {name}: replay summary {key}={case_summary.get(key)!r} "
                    f"!= expected {expected!r}"
                )
        source_metadata = case_summary.get("source_video_metadata")
        if not isinstance(source_metadata, dict) or source_metadata != {
            "fps_hz": sampling["fps_hz"],
            "frame_count": sampling["frame_count"],
            "size_wh": sampling["size_wh"],
            "sha256": sampling["sha256"],
        }:
            raise OfflineBenchmarkError(
                f"case {name}: source-video metadata provenance mismatch"
            )
        if require_production and case_summary.get("source_video_pin_complete") is not True:
            raise OfflineBenchmarkError(f"case {name}: source-video pin is incomplete")
        seed_source = int(case_summary.get("seed_source_frame", -1))
        seed_position = int(case_summary.get("seed_position", -1))
        if (
            seed_position < 0
            or seed_position >= len(expected_indices)
            or expected_indices[seed_position] != seed_source
        ):
            raise OfflineBenchmarkError(f"case {name}: invalid seed position/source")
        if "seed_source_frame" in case and seed_source != int(
            case["seed_source_frame"]
        ):
            raise OfflineBenchmarkError(
                f"case {name}: replay seed source differs from pinned manifest"
            )
        expected_seed_sha = case.get("seed_video_sha256")
        if expected_seed_sha is not None:
            seed_video = case_summary.get("seed_video_provenance")
            if not isinstance(seed_video, dict):
                raise OfflineBenchmarkError(
                    f"case {name}: seed-video provenance is missing"
                )
            for key, expected in (
                ("sha256", str(expected_seed_sha).lower()),
                ("fps_hz", case.get("seed_video_fps_hz")),
                ("frame_count", case.get("seed_video_frame_count")),
                ("size_wh", case.get("seed_video_size_wh")),
            ):
                if seed_video.get(key) != expected:
                    raise OfflineBenchmarkError(
                        f"case {name}: seed-video {key} differs from pin"
                    )
            if seed_video.get("frame_index") != seed_source:
                raise OfflineBenchmarkError(
                    f"case {name}: seed-video provenance frame mismatch"
                )
            if require_production and case_summary.get(
                "seed_video_pin_complete"
            ) is not True:
                raise OfflineBenchmarkError(
                    f"case {name}: seed-video pin is incomplete"
                )
        sparse_spec = dict(case.get("sparse_ground_truth") or {})
        pinned_source_sha = str(sparse_spec.get("source_video_sha256", "")).lower()
        if pinned_source_sha and pinned_source_sha != sampling["sha256"]:
            raise OfflineBenchmarkError(
                f"case {name}: source video is not pinned by sparse ground truth"
            )
        if require_production and not pinned_source_sha:
            raise OfflineBenchmarkError(
                f"case {name}: production source video lacks sparse-GT SHA pin"
            )
        if case_summary.get("effective_config_provenance") != root_config:
            raise OfflineBenchmarkError(
                f"case {name}: effective config provenance differs from root"
            )
        config_provenance = case_summary.get("effective_config_provenance")
        if not isinstance(config_provenance, dict):
            raise OfflineBenchmarkError(
                f"case {name}: effective config provenance is missing"
            )
        for digest_key in (
            "source_config_sha256",
            "canonical_effective_config_sha256",
            "checkpoint_sha256",
            "model_config_sha256",
        ):
            digest = str(config_provenance.get(digest_key, ""))
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise OfflineBenchmarkError(
                    f"case {name}: invalid {digest_key} provenance"
                )
        if require_production:
            required_profile = {
                "requested_object_mask_mode": "guarded_v2",
                "effective_mask_publication_mode": "guarded_sam2_primary",
                "effective_recovery_publication_mode": "unified_three_evidence",
                "diagnostic_only": False,
                "diagnostic_only_not_production_acceptance": False,
                "production_acceptance_eligible": True,
                "production_defaults_used": True,
                "realtime_schedule_preserved": True,
                "sam2_image_size": 512,
                "neutral_synthetic_depth_m": 1.0,
                "hardware_interfaces_opened": False,
            }
            for key, expected in required_profile.items():
                if case_summary.get(key) != expected:
                    raise OfflineBenchmarkError(
                        f"case {name}: production profile {key} must be {expected!r}"
                    )
            if case_summary.get("production_ineligibility_reasons") != []:
                raise OfflineBenchmarkError(
                    f"case {name}: production ineligibility reasons are non-empty"
                )
            _verify_production_sam2_readiness_provenance(
                case_name=name,
                case_summary=case_summary,
                config_provenance=config_provenance,
                source_size_wh=sampling["size_wh"],
                seed_source_frame=seed_source,
            )
        bootstrap_final = case_summary.get("guarded_v2_bootstrap_final")
        if require_production and not isinstance(bootstrap_final, dict):
            raise OfflineBenchmarkError(
                f"case {name}: production replay lacks final bootstrap provenance"
            )
        if bootstrap_final is not None:
            if not isinstance(bootstrap_final, dict):
                raise OfflineBenchmarkError(
                    f"case {name}: guarded_v2_bootstrap_final must be an object"
                )
            commissioned_frame_id = bootstrap_final.get("commissioned_frame_id")
            if (
                bootstrap_final.get("phase") != "commissioned"
                or bootstrap_final.get("commission_count") != 1
                or isinstance(commissioned_frame_id, bool)
                or not isinstance(commissioned_frame_id, int)
                or int(commissioned_frame_id) not in expected_indices
                or int(commissioned_frame_id) <= seed_source
            ):
                raise OfflineBenchmarkError(
                    f"case {name}: final bootstrap must be commissioned exactly "
                    "once on a sampled source frame after the seed"
                )
        aggregate = _verify_candidate_artifacts(
            case_name=name,
            case_root=case_root,
            case_summary=case_summary,
            expected_source_indices=expected_indices,
        )
        expected_chain = {
            "case": name,
            "summary_path": f"{name}/summary.json",
            "summary_sha256": _sha256_file(case_summary_path),
            "candidate_artifacts_sha256": aggregate,
        }
        if not isinstance(raw_chain, dict) or raw_chain != expected_chain:
            raise OfflineBenchmarkError(
                f"case {name}: root case summary/artifact digest chain mismatch"
            )
        normalized_chain.append(expected_chain)
        case_by_name[name] = {
            "summary": case_summary,
            "sampled_source_indices": expected_indices,
            "seed_position": seed_position,
            "guarded_v2_bootstrap_final": bootstrap_final,
        }
    if _canonical_json_sha256(normalized_chain) != str(
        root_summary.get("candidate_root_artifacts_sha256", "")
    ).lower():
        raise OfflineBenchmarkError("candidate-root aggregate artifact digest mismatch")
    return case_by_name


def _load_sparse_ground_truth(
    spec: dict[str, Any],
    base: Path,
    *,
    sampled_source_frames: set[int],
    expected_source_video: Path,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    """Load immutable, human-reviewed visible-object masks.

    Dense HSV/motion proxies remain useful for temporal coverage.  These sparse
    labels are deliberately stricter: each PNG must be binary, non-empty,
    explicitly human-confirmed and content-addressed in the manifest.
    """

    if not spec:
        return {}, None
    schema = str(spec.get("schema", ""))
    if schema != SPARSE_GROUND_TRUTH_SCHEMA:
        raise OfflineBenchmarkError(
            f"unsupported sparse_ground_truth schema={schema!r}; expected "
            f"{SPARSE_GROUND_TRUTH_SCHEMA!r}"
        )
    records = spec.get("records")
    if not isinstance(records, list) or not records:
        raise OfflineBenchmarkError(
            "sparse_ground_truth.records must be a non-empty list"
        )
    required = int(spec.get("min_reviewed_frames", 5))
    if required < 1:
        raise OfflineBenchmarkError(
            "sparse_ground_truth.min_reviewed_frames must be positive"
        )
    if len(records) < required:
        raise OfflineBenchmarkError(
            f"sparse_ground_truth has {len(records)} records; at least "
            f"{required} are required"
        )
    source_video = _resolve_path(
        spec.get("source_video"), base, description="sparse ground-truth source video"
    )
    declared_source_video_sha256 = str(
        spec.get("source_video_sha256", "")
    ).lower()
    actual_source_video_sha256 = _sha256_file(source_video)
    if declared_source_video_sha256 != actual_source_video_sha256:
        raise OfflineBenchmarkError(
            f"sparse ground-truth source-video SHA-256 mismatch for {source_video}: "
            f"manifest={declared_source_video_sha256!r}, "
            f"actual={actual_source_video_sha256}"
        )
    benchmark_source_sha256 = _sha256_file(expected_source_video.resolve())
    if benchmark_source_sha256 != actual_source_video_sha256:
        raise OfflineBenchmarkError(
            "sparse ground-truth source bytes are not the benchmark case video: "
            f"labels_sha256={actual_source_video_sha256}, "
            f"benchmark_sha256={benchmark_source_sha256}"
        )

    loaded: dict[int, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise OfflineBenchmarkError(
                f"sparse_ground_truth.records[{index}] must be an object"
            )
        source_frame = int(raw.get("source_frame", -1))
        if source_frame in loaded:
            raise OfflineBenchmarkError(
                f"duplicate sparse ground-truth source_frame={source_frame}"
            )
        if source_frame not in sampled_source_frames:
            raise OfflineBenchmarkError(
                f"sparse ground-truth source_frame={source_frame} is not in the "
                "case's sampled 20 Hz frame sequence"
            )
        review_status = str(raw.get("review_status", ""))
        label_method = str(raw.get("label_method", "")).strip()
        review_notes = str(raw.get("review_notes", "")).strip()
        if review_status not in SPARSE_GROUND_TRUTH_REVIEW_STATUSES:
            raise OfflineBenchmarkError(
                f"sparse ground-truth source_frame={source_frame} must have "
                "an explicitly reviewed status in "
                f"{sorted(SPARSE_GROUND_TRUTH_REVIEW_STATUSES)}"
            )
        if not label_method or not review_notes:
            raise OfflineBenchmarkError(
                f"sparse ground-truth source_frame={source_frame} requires "
                "label_method and review_notes"
            )
        path = _resolve_path(
            raw.get("path"), base, description="sparse ground-truth mask"
        )
        declared_sha256 = str(raw.get("sha256", "")).lower()
        actual_sha256 = _sha256_file(path)
        if declared_sha256 != actual_sha256:
            raise OfflineBenchmarkError(
                f"sparse ground-truth SHA-256 mismatch for {path}: manifest="
                f"{declared_sha256!r}, actual={actual_sha256}"
            )
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OfflineBenchmarkError(
                f"cannot read sparse ground-truth mask: {path}"
            )
        values = set(int(value) for value in np.unique(image))
        if not values.issubset({0, 255}):
            raise OfflineBenchmarkError(
                f"sparse ground-truth mask must contain only 0/255: {path}; "
                f"values={sorted(values)}"
            )
        mask = image == 255
        if not np.any(mask):
            raise OfflineBenchmarkError(
                f"reviewed visible-object mask is empty: {path}; do not invent "
                "a fully-occluded target label"
            )
        metadata = {
            "source_video": str(source_video),
            "source_video_sha256": actual_source_video_sha256,
            "source_frame": source_frame,
            "path": str(path),
            "sha256": actual_sha256,
            "review_status": review_status,
            "label_method": label_method,
            "review_notes": review_notes,
            "area_px": int(mask.sum()),
        }
        loaded[source_frame] = {"mask": mask, "metadata": metadata}
        provenance.append(metadata)
    return loaded, {
        "schema": schema,
        "source_video": str(source_video),
        "source_video_sha256": actual_source_video_sha256,
        "min_reviewed_frames": required,
        "records": provenance,
    }


def _load_reviewed_motion_track(
    spec: dict[str, Any],
    base: Path,
    *,
    case_name: str,
    expected_source_video: Path,
    expected_source_fps: float,
    expected_source_indices: list[int],
    expected_source_start_frame: int,
    expected_evaluation_fps: float,
    expected_reviewed_visible_intervals: list[tuple[int, int]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    """Load a dense, human-reviewed visible-object bbox/centroid track.

    The track is intentionally a separate, content-addressed artifact rather
    than a value inferred from the HSV proxy under test.  It must cover every
    sampled source frame inside the reviewed-visible intervals.  A missing
    track keeps the historical proxy motion gate unchanged; a configured but
    malformed or incompletely reviewed track fails closed.
    """

    if not spec:
        return {}, None
    path = _resolve_path(
        spec.get("path"), base, description=f"{case_name} reviewed motion track"
    )
    declared_sha256 = str(spec.get("sha256", "")).lower()
    actual_sha256 = _sha256_file(path)
    if declared_sha256 != actual_sha256:
        raise OfflineBenchmarkError(
            f"case {case_name}: reviewed motion track SHA-256 mismatch for "
            f"{path}: manifest={declared_sha256!r}, actual={actual_sha256}"
        )
    payload = _load_json_object(path, description="reviewed motion track")
    if payload.get("schema") != REVIEWED_MOTION_TRACK_SCHEMA:
        raise OfflineBenchmarkError(
            f"case {case_name}: reviewed motion track schema must be "
            f"{REVIEWED_MOTION_TRACK_SCHEMA!r}"
        )
    if str(payload.get("case", "")) != case_name:
        raise OfflineBenchmarkError(
            f"case {case_name}: reviewed motion track case identity mismatch"
        )

    source_video = expected_source_video.resolve()
    source_sha256 = _sha256_file(source_video)
    if str(payload.get("source_video_sha256", "")).lower() != source_sha256:
        raise OfflineBenchmarkError(
            f"case {case_name}: reviewed motion track source-video hash mismatch"
        )
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise OfflineBenchmarkError(f"cannot open video: {source_video}")
    try:
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_size_wh = [
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ]
    finally:
        capture.release()
    expected_metadata = {
        "source_video_fps_hz": float(expected_source_fps),
        "source_video_frame_count": source_frame_count,
        "source_video_size_wh": source_size_wh,
        "source_start_frame": int(expected_source_start_frame),
        "evaluation_fps_hz": float(expected_evaluation_fps),
        "sampled_source_indices": [int(value) for value in expected_source_indices],
        "reviewed_visible_source_intervals": [
            [int(start), int(end)]
            for start, end in expected_reviewed_visible_intervals
        ],
    }
    for key, expected in expected_metadata.items():
        if payload.get(key) != expected:
            raise OfflineBenchmarkError(
                f"case {case_name}: reviewed motion track {key} does not match "
                "the exact benchmark sampling contract"
            )

    expected_reviewed_sources = [
        int(source)
        for source in expected_source_indices
        if _inside_source_intervals(source, expected_reviewed_visible_intervals)
    ]
    if payload.get("reviewed_visible_sampled_source_indices") != (
        expected_reviewed_sources
    ):
        raise OfflineBenchmarkError(
            f"case {case_name}: reviewed motion track visible sampled-source "
            "indices are incomplete or misordered"
        )
    raw_thresholds = payload.get("thresholds")
    if not isinstance(raw_thresholds, dict) or set(raw_thresholds) != set(
        REVIEWED_MOTION_TRACK_THRESHOLDS
    ):
        raise OfflineBenchmarkError(
            f"case {case_name}: reviewed motion track thresholds must be exactly "
            f"{sorted(REVIEWED_MOTION_TRACK_THRESHOLDS)}"
        )
    thresholds: dict[str, float] = {}
    for key, commissioned in REVIEWED_MOTION_TRACK_THRESHOLDS.items():
        try:
            value = float(raw_thresholds[key])
        except (TypeError, ValueError) as exc:
            raise OfflineBenchmarkError(
                f"case {case_name}: reviewed motion threshold {key} is invalid"
            ) from exc
        if not np.isfinite(value) or value != commissioned:
            raise OfflineBenchmarkError(
                f"case {case_name}: reviewed motion threshold {key}={value!r} "
                f"must remain commissioned at {commissioned}"
            )
        thresholds[key] = value

    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise OfflineBenchmarkError(
            f"case {case_name}: reviewed motion track records must be a list"
        )
    raw_sources: list[int] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise OfflineBenchmarkError(
                f"case {case_name}: reviewed motion record {index} is not an object"
            )
        raw_sources.append(int(raw.get("source_frame", -1)))
    if len(raw_sources) != len(set(raw_sources)):
        raise OfflineBenchmarkError(
            f"case {case_name}: reviewed motion track contains duplicate source frames"
        )
    if raw_sources != expected_reviewed_sources:
        raise OfflineBenchmarkError(
            f"case {case_name}: reviewed motion track record order/coverage "
            f"{raw_sources} != exact reviewed samples {expected_reviewed_sources}"
        )

    frame_width, frame_height = source_size_wh
    loaded: dict[int, dict[str, Any]] = {}
    provenance_records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_records):
        source_frame = raw_sources[index]
        review_status = str(raw.get("review_status", ""))
        label_method = str(raw.get("label_method", "")).strip()
        review_notes = str(raw.get("review_notes", "")).strip()
        if review_status not in SPARSE_GROUND_TRUTH_REVIEW_STATUSES:
            raise OfflineBenchmarkError(
                f"case {case_name}: reviewed motion source_frame={source_frame} "
                "is not human-confirmed; leave the draft unreferenced until every "
                "record has been visually reviewed"
            )
        if not label_method or not review_notes:
            raise OfflineBenchmarkError(
                f"case {case_name}: reviewed motion source_frame={source_frame} "
                "requires label_method and review_notes"
            )
        raw_bbox = raw.get("bbox_xyxy")
        raw_centroid = raw.get("centroid_xy")
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise OfflineBenchmarkError(
                f"case {case_name}: reviewed motion source_frame={source_frame} "
                "bbox_xyxy must contain four values"
            )
        if not isinstance(raw_centroid, list) or len(raw_centroid) != 2:
            raise OfflineBenchmarkError(
                f"case {case_name}: reviewed motion source_frame={source_frame} "
                "centroid_xy must contain two values"
            )
        try:
            bbox = tuple(float(value) for value in raw_bbox)
            centroid = tuple(float(value) for value in raw_centroid)
        except (TypeError, ValueError) as exc:
            raise OfflineBenchmarkError(
                f"case {case_name}: reviewed motion source_frame={source_frame} "
                "geometry must be numeric"
            ) from exc
        if not all(np.isfinite(value) for value in (*bbox, *centroid)):
            raise OfflineBenchmarkError(
                f"case {case_name}: reviewed motion source_frame={source_frame} "
                "geometry must be finite"
            )
        x1, y1, x2, y2 = bbox
        cx, cy = centroid
        if not (0.0 <= x1 < x2 <= frame_width and 0.0 <= y1 < y2 <= frame_height):
            raise OfflineBenchmarkError(
                f"case {case_name}: reviewed motion source_frame={source_frame} "
                f"bbox={bbox} lies outside source image {source_size_wh}"
            )
        if not (x1 <= cx < x2 and y1 <= cy < y2):
            raise OfflineBenchmarkError(
                f"case {case_name}: reviewed motion source_frame={source_frame} "
                "centroid lies outside bbox"
            )
        record = {
            "source_frame": source_frame,
            "bbox_xyxy": bbox,
            "centroid_xy": centroid,
            "bbox_diagonal_px": float(math.hypot(x2 - x1, y2 - y1)),
            "review_status": review_status,
            "label_method": label_method,
            "review_notes": review_notes,
        }
        loaded[source_frame] = record
        provenance_records.append(
            {
                **record,
                "bbox_xyxy": list(bbox),
                "centroid_xy": list(centroid),
            }
        )
    return loaded, {
        "schema": REVIEWED_MOTION_TRACK_SCHEMA,
        "path": str(path),
        "sha256": actual_sha256,
        "source_video": str(source_video),
        "source_video_sha256": source_sha256,
        "sampled_source_indices": expected_metadata["sampled_source_indices"],
        "reviewed_visible_sampled_source_indices": expected_reviewed_sources,
        "reviewed_record_count": len(loaded),
        "thresholds": thresholds,
        "records": provenance_records,
    }


def _load_sparse_ground_truth_manifest(
    raw_path: Any, base: Path
) -> tuple[dict[str, dict[str, Any]], Path] | tuple[dict[str, dict[str, Any]], None]:
    if raw_path is None:
        return {}, None
    path = _resolve_path(
        raw_path, base, description="sparse ground-truth manifest"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("schema", "")) != SPARSE_GROUND_TRUTH_MANIFEST_SCHEMA:
        raise OfflineBenchmarkError(
            f"unsupported sparse ground-truth manifest schema in {path}: "
            f"{payload.get('schema')!r}"
        )
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise OfflineBenchmarkError(
            f"sparse ground-truth manifest cases must be non-empty: {path}"
        )
    result: dict[str, dict[str, Any]] = {}
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise OfflineBenchmarkError(
                f"sparse ground-truth manifest case {index} must be an object"
            )
        name = str(raw_case.get("name", "")).strip()
        spec = raw_case.get("sparse_ground_truth")
        if not name or not isinstance(spec, dict):
            raise OfflineBenchmarkError(
                f"sparse ground-truth manifest case {index} requires name and "
                "sparse_ground_truth"
            )
        if name in result:
            raise OfflineBenchmarkError(
                f"duplicate sparse ground-truth case name={name!r}"
            )
        # Record paths are relative to the sparse manifest, not whichever
        # benchmark manifest references it.
        normalized = dict(spec)
        source_video = Path(str(normalized.get("source_video", ""))).expanduser()
        if not source_video.is_absolute():
            normalized["source_video"] = str(
                (path.parent / source_video).resolve()
            )
        normalized_records = []
        for record in normalized.get("records") or []:
            copied = dict(record)
            record_path = Path(str(copied.get("path", ""))).expanduser()
            if not record_path.is_absolute():
                copied["path"] = str((path.parent / record_path).resolve())
            normalized_records.append(copied)
        normalized["records"] = normalized_records
        result[name] = normalized
    return result, path


def _verify_production_suite_coverage_contract(
    *,
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    sparse_by_case: dict[str, dict[str, Any]],
) -> None:
    """Bind production PASS to the five reviewed scenario roles.

    The heavy-occlusion video contains difficult *partial* visibility, while
    the static RH56 case contains the reviewed complete-occlusion and
    reappearance events.  Keeping those roles explicit prevents a report from
    silently claiming that the heavy case contains a fully hidden target.
    """

    raw = manifest.get("production_suite_coverage_contract")
    if not isinstance(raw, dict):
        raise OfflineBenchmarkError(
            "production benchmark lacks production_suite_coverage_contract"
        )
    expected_keys = {
        "schema",
        "exact_20hz_case_names",
        "visible_part_accuracy_case_names",
        "fast_entry_case_name",
        "rolling_case_names",
        "complete_occlusion_events",
        "heavy_partial_occlusion_case_name",
        "heavy_partial_occlusion_reviewed_source_frames",
        "pointcloud_acceptance_scope",
    }
    if set(raw) != expected_keys:
        raise OfflineBenchmarkError(
            "production suite coverage fields must be exactly "
            f"{sorted(expected_keys)}"
        )
    if raw.get("schema") != PRODUCTION_SUITE_COVERAGE_SCHEMA:
        raise OfflineBenchmarkError(
            "production suite coverage schema must be "
            f"{PRODUCTION_SUITE_COVERAGE_SCHEMA!r}"
        )
    if tuple(raw.get("exact_20hz_case_names") or ()) != PRODUCTION_CASE_NAMES:
        raise OfflineBenchmarkError(
            "production suite exact-20Hz roles must cover the fixed five cases"
        )
    if tuple(raw.get("visible_part_accuracy_case_names") or ()) != (
        PRODUCTION_CASE_NAMES
    ):
        raise OfflineBenchmarkError(
            "production suite visible-part accuracy must cover all five cases"
        )
    if raw.get("fast_entry_case_name") != "fast_green_ball_entry":
        raise OfflineBenchmarkError("production fast-entry role is not pinned")
    if tuple(raw.get("rolling_case_names") or ()) != (
        "rolling_green_ball",
        "rolling_red_cylinder",
    ):
        raise OfflineBenchmarkError("production rolling-object roles are not pinned")
    if raw.get("heavy_partial_occlusion_case_name") != "rh56_heavy_occlusion":
        raise OfflineBenchmarkError(
            "production heavy-partial-occlusion role is not pinned"
        )
    if tuple(raw.get("heavy_partial_occlusion_reviewed_source_frames") or ()) != (
        PRODUCTION_HEAVY_PARTIAL_OCCLUSION_SOURCES
    ):
        raise OfflineBenchmarkError(
            "production heavy-partial-occlusion reviewed frames are not pinned"
        )
    if raw.get("pointcloud_acceptance_scope") != (
        PRODUCTION_POINTCLOUD_ACCEPTANCE_SCOPE
    ):
        raise OfflineBenchmarkError(
            "production suite must keep final 128-point acceptance in the "
            "separate saved-RGBD current-provider gate"
        )

    case_by_name = {str(case.get("name", "")): case for case in cases}
    if tuple(case_by_name) != PRODUCTION_CASE_NAMES:
        raise OfflineBenchmarkError(
            "production suite coverage requires the fixed five case order"
        )
    for name in PRODUCTION_CASE_NAMES:
        case = case_by_name[name]
        if float(case.get("evaluation_fps_hz", 0.0)) != 20.0:
            raise OfflineBenchmarkError(
                f"case {name}: production suite must remain exact 20 Hz"
            )
        sparse = sparse_by_case.get(name)
        if not isinstance(sparse, dict) or len(sparse.get("records") or ()) < int(
            sparse.get("min_reviewed_frames", 5)
        ):
            raise OfflineBenchmarkError(
                f"case {name}: visible-part accuracy lacks sufficient reviewed masks"
            )

    normalized_events: list[tuple[str, tuple[int, int], int]] = []
    for index, event in enumerate(raw.get("complete_occlusion_events") or ()):
        if not isinstance(event, dict) or set(event) != {
            "case",
            "source_interval",
            "reappearance_source_frame",
        }:
            raise OfflineBenchmarkError(
                f"complete-occlusion event {index} is malformed"
            )
        interval = event.get("source_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            raise OfflineBenchmarkError(
                f"complete-occlusion event {index} interval must be [start,end]"
            )
        normalized_events.append(
            (
                str(event.get("case", "")),
                (int(interval[0]), int(interval[1])),
                int(event.get("reappearance_source_frame", -1)),
            )
        )
    if tuple(normalized_events) != PRODUCTION_COMPLETE_OCCLUSION_EVENTS:
        raise OfflineBenchmarkError(
            "production complete-occlusion/reappearance events are not pinned"
        )
    static_case = case_by_name["static_green_ball_rh56"]
    static_absent = {
        tuple(int(value) for value in interval)
        for interval in static_case.get("reviewed_absent_source_intervals") or ()
    }
    static_reappear = {
        int(value)
        for value in static_case.get("reviewed_reappearance_source_frames") or ()
    }
    for _name, interval, reappearance in PRODUCTION_COMPLETE_OCCLUSION_EVENTS:
        if interval not in static_absent or reappearance not in static_reappear:
            raise OfflineBenchmarkError(
                "complete-occlusion suite role is not backed by reviewed static "
                "RH56 absence/reappearance intervals"
            )
    static_thresholds = dict(static_case.get("thresholds") or {})
    if "reviewed_absent_false_positive_max" not in static_thresholds or (
        "reviewed_reappearance_ticks_max" not in static_thresholds
    ):
        raise OfflineBenchmarkError(
            "complete-occlusion suite role lacks fail-closed/recovery gates"
        )

    rolling_green_thresholds = dict(
        case_by_name["rolling_green_ball"].get("thresholds") or {}
    )
    if "reviewed_absent_false_positive_max" not in rolling_green_thresholds:
        raise OfflineBenchmarkError(
            "rolling-green pre-entry reviewed absence is not a gating check"
        )
    heavy_sparse_records = {
        int(record.get("source_frame", -1)): record
        for record in sparse_by_case["rh56_heavy_occlusion"].get("records") or ()
    }
    for source in PRODUCTION_HEAVY_PARTIAL_OCCLUSION_SOURCES:
        record = heavy_sparse_records.get(source)
        if (
            record is None
            or str(record.get("review_status", ""))
            not in SPARSE_GROUND_TRUTH_REVIEW_STATUSES
            or not str(record.get("review_notes", "")).strip()
        ):
            raise OfflineBenchmarkError(
                "heavy partial-occlusion role lacks reviewed visible-part evidence "
                f"at source frame {source}"
            )


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _component_proxy(
    frame: np.ndarray,
    previous_frame: np.ndarray | None,
    config: dict[str, Any],
    previous_center: tuple[float, float] | None,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    raw = np.zeros(frame.shape[:2], dtype=np.uint8)
    ranges = config.get("hsv_ranges")
    if not isinstance(ranges, list) or not ranges:
        raise OfflineBenchmarkError("target_proxy.hsv_ranges must be non-empty")
    for bounds in ranges:
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise OfflineBenchmarkError("each HSV range must be [[lo], [hi]]")
        lower = np.asarray(bounds[0], dtype=np.uint8)
        upper = np.asarray(bounds[1], dtype=np.uint8)
        raw = cv2.bitwise_or(raw, cv2.inRange(hsv, lower, upper))

    roi = config.get("workspace_roi_normalized")
    if roi is not None:
        if not isinstance(roi, list) or len(roi) != 4:
            raise OfflineBenchmarkError("workspace_roi_normalized must have 4 values")
        height, width = raw.shape
        x1 = int(np.clip(round(float(roi[0]) * width), 0, width))
        y1 = int(np.clip(round(float(roi[1]) * height), 0, height))
        x2 = int(np.clip(round(float(roi[2]) * width), 0, width))
        y2 = int(np.clip(round(float(roi[3]) * height), 0, height))
        outside = np.ones(raw.shape, dtype=bool)
        outside[y1:y2, x1:x2] = False
        raw[outside] = 0

    motion_threshold = config.get("motion_threshold_bgr")
    if motion_threshold is not None:
        if previous_frame is None:
            # A motion proxy has no evidence on its first frame.  Returning an
            # empty proxy prevents an unrelated static object of the same
            # colour from becoming the temporal seed.
            raw[:] = 0
        else:
            difference = cv2.absdiff(frame, previous_frame)
            moving = np.max(difference, axis=2) >= int(motion_threshold)
            raw[~moving] = 0

    open_kernel = int(config.get("open_kernel", 3))
    close_kernel = int(config.get("close_kernel", 7))
    if open_kernel > 1:
        raw = cv2.morphologyEx(
            raw,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)),
        )
    if close_kernel > 1:
        raw = cv2.morphologyEx(
            raw,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)),
        )

    count, labels, stats, centers = cv2.connectedComponentsWithStats(raw, 8)
    min_area = int(config.get("min_area_px", 60))
    max_area = int(config.get("max_area_px", frame.shape[0] * frame.shape[1]))
    min_aspect = float(config.get("min_aspect", 0.2))
    max_aspect = float(config.get("max_aspect", 5.0))
    min_fill = float(config.get("min_fill", 0.15))
    candidates: list[tuple[float, int]] = []
    selection_center = previous_center
    if selection_center is None and config.get("initial_seed_xy") is not None:
        seed = config["initial_seed_xy"]
        if not isinstance(seed, list) or len(seed) != 2:
            raise OfflineBenchmarkError("initial_seed_xy must have two values")
        selection_center = (float(seed[0]), float(seed[1]))
    if selection_center is None and config.get("initial_seed_normalized") is not None:
        seed = config["initial_seed_normalized"]
        if not isinstance(seed, list) or len(seed) != 2:
            raise OfflineBenchmarkError(
                "initial_seed_normalized must have two values"
            )
        selection_center = (
            float(seed[0]) * frame.shape[1],
            float(seed[1]) * frame.shape[0],
        )
    for label in range(1, count):
        _x, _y, width, height, area = (int(v) for v in stats[label])
        aspect = width / max(1.0, float(height))
        fill = area / max(1.0, float(width * height))
        if not (
            min_area <= area <= max_area
            and min_aspect <= aspect <= max_aspect
            and fill >= min_fill
        ):
            continue
        center = (float(centers[label, 0]), float(centers[label, 1]))
        selection_mode = str(config.get("component_selection", "nearest"))
        if selection_center is None or selection_mode == "largest_within_jump":
            score = -float(area)
        else:
            score = float(np.linalg.norm(np.subtract(center, selection_center)))
        if selection_center is not None:
            distance = float(np.linalg.norm(np.subtract(center, selection_center)))
            max_jump = config.get("max_center_jump_px")
            if max_jump is not None and distance > float(max_jump):
                continue
        candidates.append((score, label))
    if not candidates:
        return np.zeros(raw.shape, dtype=bool), None
    candidates.sort()
    selected = int(candidates[0][1])
    center = (float(centers[selected, 0]), float(centers[selected, 1]))
    return labels == selected, center


def _load_timings(spec: dict[str, Any], base: Path) -> dict[int, float]:
    states = spec.get("states")
    if states is None:
        return {}
    path = _resolve_path(states, base, description="candidate states JSONL")
    frame_field = str(spec.get("states_frame_field", "frame_index"))
    latency_field = str(spec.get("states_latency_field", "processing_ms"))
    evaluable_field_raw = spec.get(
        "states_latency_evaluable_field", "latency_evaluable"
    )
    evaluable_field = (
        None if evaluable_field_raw is None else str(evaluable_field_raw)
    )
    strict = bool(spec.get("strict_states_contract", False))
    expected_source_raw = spec.get("expected_source_indices")
    expected_source_indices = (
        None
        if expected_source_raw is None
        else [int(value) for value in expected_source_raw]
    )
    seed_position = int(spec.get("seed_position", -1))
    timings: dict[int, float] = {}
    observed_frame_ids: list[int] = []
    strict_state_records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("record must be an object")
                frame_id = int(record[frame_field])
                if frame_id in observed_frame_ids:
                    raise ValueError(f"duplicate {frame_field}={frame_id}")
                observed_frame_ids.append(frame_id)
                if strict and evaluable_field is None:
                    raise TypeError("strict states require a latency_evaluable field")
                if evaluable_field is not None and (
                    strict or evaluable_field in record
                ):
                    evaluable = record[evaluable_field]
                    if not isinstance(evaluable, bool):
                        raise TypeError(
                            f"{evaluable_field} must be boolean"
                        )
                else:
                    evaluable = True
                if strict:
                    if expected_source_indices is None:
                        raise TypeError("strict states require expected_source_indices")
                    if not 0 <= frame_id < len(expected_source_indices):
                        raise ValueError(
                            f"{frame_field}={frame_id} is outside candidate range"
                        )
                    source_frame = int(record["source_frame_index"])
                    if source_frame != expected_source_indices[frame_id]:
                        raise ValueError(
                            f"source_frame_index={source_frame} != expected "
                            f"{expected_source_indices[frame_id]} for frame {frame_id}"
                        )
                    bootstrap_present = "guarded_v2_bootstrap" in record
                    bootstrap = record.get("guarded_v2_bootstrap")
                    if bootstrap_present and not isinstance(bootstrap, dict):
                        raise TypeError("guarded_v2_bootstrap must be an object")
                    strict_state_records.append(
                        {
                            "frame_id": frame_id,
                            "source_frame_index": source_frame,
                            "latency_evaluable": evaluable,
                            "valid": record.get("valid"),
                            "mask_source": record.get("mask_source"),
                            "bootstrap_present": bootstrap_present,
                            "bootstrap": bootstrap,
                        }
                    )
                if latency_field in record:
                    latency = float(record[latency_field])
                    if not np.isfinite(latency) or latency < 0.0:
                        raise ValueError(
                            f"{latency_field} must be finite and nonnegative"
                        )
                elif evaluable:
                    raise KeyError(latency_field)
                else:
                    latency = 0.0
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise OfflineBenchmarkError(
                    f"invalid timing record {path}:{line_number}: {exc}"
                ) from exc
            if evaluable:
                timings[frame_id] = latency
    if strict:
        if expected_source_indices is None:
            raise OfflineBenchmarkError("strict states lack expected source indices")
        expected_frame_ids = list(range(len(expected_source_indices)))
        if observed_frame_ids != expected_frame_ids:
            raise OfflineBenchmarkError(
                f"states frame IDs/order {observed_frame_ids} != exact expected "
                f"{expected_frame_ids}"
            )
        bootstrap_records = [
            record
            for record in strict_state_records
            if int(record["frame_id"]) >= seed_position
        ]
        has_bootstrap_contract = any(
            bool(record["bootstrap_present"]) for record in bootstrap_records
        )
        if has_bootstrap_contract:
            expected_bootstrap_final = spec.get(
                "expected_guarded_v2_bootstrap_final"
            )
            if expected_bootstrap_final is not None and not isinstance(
                expected_bootstrap_final, dict
            ):
                raise OfflineBenchmarkError(
                    "expected_guarded_v2_bootstrap_final must be an object"
                )
            missing = [
                int(record["frame_id"])
                for record in bootstrap_records
                if not bool(record["bootstrap_present"])
            ]
            if missing:
                raise OfflineBenchmarkError(
                    "strict guarded-v2 states omit guarded_v2_bootstrap at "
                    f"frames {missing}"
                )

            commission_position: Optional[int] = None
            commissioned_source_frame: Optional[int] = None
            boundary_expiry_seen = False
            boundary_rearm_seen = False
            previous_boundary_rearm_count = 0
            for record in bootstrap_records:
                frame_id = int(record["frame_id"])
                source_frame = int(record["source_frame_index"])
                state = record["bootstrap"]
                phase = str(state.get("phase", ""))
                count = state.get("commission_count")
                commissioned_frame_id = state.get("commissioned_frame_id")
                boundary_rearm_count = state.get("boundary_rearm_count", 0)
                if isinstance(count, bool) or not isinstance(count, int):
                    raise OfflineBenchmarkError(
                        "guarded_v2_bootstrap.commission_count must be an "
                        f"integer at frame {frame_id}"
                    )
                if (
                    isinstance(boundary_rearm_count, bool)
                    or not isinstance(boundary_rearm_count, int)
                    or boundary_rearm_count not in (0, 1)
                ):
                    raise OfflineBenchmarkError(
                        "guarded_v2_bootstrap.boundary_rearm_count must be "
                        f"integer 0 or 1 at frame {frame_id}"
                    )
                if boundary_rearm_count < previous_boundary_rearm_count:
                    raise OfflineBenchmarkError(
                        "guarded-v2 boundary rearm count regressed at frame "
                        f"{frame_id}"
                    )
                rearm_transition = (
                    boundary_rearm_count > previous_boundary_rearm_count
                )
                if rearm_transition:
                    seed_frame_id = state.get("seed_frame_id")
                    status = str(state.get("status", ""))
                    raw_recovery_rearm = bool(
                        record.get("mask_source") == "recovery_committed"
                        and status
                        == (
                            "rearmed once from confirmed interior recovery "
                            "after boundary-clipped seed"
                        )
                    )
                    fine_core_rearm = bool(
                        record.get("mask_source")
                        == "lost_reappearance_visible_core"
                        and status
                        == (
                            "rearmed once from sealed boundary-seed fine "
                            "visible core"
                        )
                    )
                    if (
                        previous_boundary_rearm_count != 0
                        or boundary_rearm_count != 1
                        or boundary_rearm_seen
                        or not boundary_expiry_seen
                        or phase != "provisional"
                        or count != 0
                        or commissioned_frame_id is not None
                        or record.get("valid") is not True
                        or isinstance(seed_frame_id, bool)
                        or not isinstance(seed_frame_id, int)
                        or int(seed_frame_id) != source_frame
                        or not (raw_recovery_rearm or fine_core_rearm)
                    ):
                        raise OfflineBenchmarkError(
                            "guarded-v2 boundary bootstrap rearm lacks the "
                            "one-shot confirmed-interior recovery contract at "
                            f"frame {frame_id}"
                        )
                    boundary_rearm_seen = True
                if phase == "commissioned":
                    if count != 1 or (
                        isinstance(commissioned_frame_id, bool)
                        or not isinstance(commissioned_frame_id, int)
                    ):
                        raise OfflineBenchmarkError(
                            "commissioned bootstrap state must carry exactly "
                            f"one integer commissioned_frame_id at frame {frame_id}"
                        )
                    if commission_position is None:
                        if int(commissioned_frame_id) != source_frame:
                            raise OfflineBenchmarkError(
                                "bootstrap commissioned_frame_id="
                                f"{commissioned_frame_id} does not match the "
                                f"transition source frame {source_frame}"
                            )
                        commission_position = frame_id
                        commissioned_source_frame = int(commissioned_frame_id)
                    elif int(commissioned_frame_id) != commissioned_source_frame:
                        raise OfflineBenchmarkError(
                            "bootstrap commissioned_frame_id changed after "
                            f"commission at frame {frame_id}"
                        )
                else:
                    if commission_position is not None:
                        raise OfflineBenchmarkError(
                            "guarded-v2 bootstrap regressed from commissioned "
                            f"to {phase!r} at frame {frame_id}"
                        )
                    if count != 0 or commissioned_frame_id is not None:
                        raise OfflineBenchmarkError(
                            "pre-commission guarded-v2 bootstrap must keep "
                            "commission_count=0 and no "
                            f"commissioned_frame_id at frame {frame_id}; "
                            f"phase={phase!r}"
                        )
                    if phase == "expired":
                        if (
                            boundary_rearm_seen
                            or boundary_rearm_count != 0
                            or state.get("status")
                            != (
                                "expired: raw candidate touches the effective "
                                "image boundary"
                            )
                        ):
                            raise OfflineBenchmarkError(
                                "pre-commission expired bootstrap is allowed "
                                "only for the exact boundary-clipped seed "
                                f"contract at frame {frame_id}"
                            )
                        boundary_expiry_seen = True
                    elif phase == "provisional":
                        if boundary_expiry_seen and not boundary_rearm_seen:
                            raise OfflineBenchmarkError(
                                "guarded-v2 bootstrap returned from boundary "
                                "expiry without the one-shot confirmed-interior "
                                f"rearm contract at frame {frame_id}"
                            )
                    else:
                        raise OfflineBenchmarkError(
                            "pre-commission guarded-v2 bootstrap phase must be "
                            "provisional or exact boundary-expired at frame "
                            f"{frame_id}; phase={phase!r}"
                        )
                if boundary_expiry_seen and phase == "commissioned" and (
                    not boundary_rearm_seen or boundary_rearm_count != 1
                ):
                    raise OfflineBenchmarkError(
                        "guarded-v2 boundary-expired bootstrap commissioned "
                        "without one-shot confirmed-interior rearm at frame "
                        f"{frame_id}"
                    )
                previous_boundary_rearm_count = boundary_rearm_count
            if commission_position is None:
                raise OfflineBenchmarkError(
                    "strict guarded-v2 states never reached one bootstrap commission"
                )
            if expected_bootstrap_final is not None and (
                expected_bootstrap_final.get("phase") != "commissioned"
                or expected_bootstrap_final.get("commission_count") != 1
                or expected_bootstrap_final.get("commissioned_frame_id")
                != commissioned_source_frame
                or (
                    "boundary_rearm_count" in expected_bootstrap_final
                    and expected_bootstrap_final.get("boundary_rearm_count")
                    != previous_boundary_rearm_count
                )
            ):
                raise OfflineBenchmarkError(
                    "states bootstrap commission does not match the replay "
                    "summary's final commissioned-once provenance"
                )
            for record in strict_state_records:
                frame_id = int(record["frame_id"])
                source_frame = int(record["source_frame_index"])
                expected_evaluable = source_frame > int(
                    commissioned_source_frame
                )
                if bool(record["latency_evaluable"]) is not expected_evaluable:
                    raise OfflineBenchmarkError(
                        "latency_evaluable="
                        f"{record['latency_evaluable']} for frame {frame_id}; "
                        "guarded-v2 timing must remain false through bootstrap "
                        "commission source frame "
                        f"{commissioned_source_frame} and true thereafter"
                    )
            expected_latency_ids = {
                int(record["frame_id"])
                for record in strict_state_records
                if int(record["source_frame_index"])
                > int(commissioned_source_frame)
            }
        else:
            # Backwards compatibility for content-addressed replay artifacts
            # produced before bootstrap state was embedded in every record.
            for record in strict_state_records:
                frame_id = int(record["frame_id"])
                expected_evaluable = frame_id > seed_position
                if bool(record["latency_evaluable"]) is not expected_evaluable:
                    raise OfflineBenchmarkError(
                        f"latency_evaluable={record['latency_evaluable']} for "
                        f"frame {frame_id}; expected {expected_evaluable} from "
                        f"seed_position={seed_position}"
                    )
            expected_latency_ids = set(
                range(seed_position + 1, len(expected_source_indices))
            )
        if set(timings) != expected_latency_ids:
            raise OfflineBenchmarkError(
                f"states evaluable latency IDs {sorted(timings)} != expected "
                f"formal ticks {sorted(expected_latency_ids)}"
            )
    return timings


def _load_latency_summary(
    spec: dict[str, Any], base: Path
) -> dict[str, float] | None:
    raw_path = spec.get("latency_summary")
    if raw_path is None:
        return None
    path = _resolve_path(raw_path, base, description="latency summary JSON")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = spec.get("latency_summary_fields")
    if not isinstance(fields, dict) or not fields:
        raise OfflineBenchmarkError(
            "candidate.latency_summary_fields must map metric names to JSON fields"
        )
    result: dict[str, float] = {}
    for output_name, source_name in fields.items():
        value: Any = payload
        for component in str(source_name).split("."):
            if not isinstance(value, dict) or component not in value:
                raise OfflineBenchmarkError(
                    f"latency summary field {source_name!r} is absent from {path}"
                )
            value = value[component]
        number = float(value)
        if not np.isfinite(number) or number < 0.0:
            raise OfflineBenchmarkError(
                f"latency summary field {source_name!r} is not finite/nonnegative"
            )
        result[str(output_name)] = number
    return result


def _stats(values: Iterable[float]) -> dict[str, float] | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return {
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _reviewed_source_intervals(
    case: dict[str, Any],
    key: str,
    *,
    sampled_source_frames: list[int],
) -> list[tuple[int, int]]:
    """Validate optional inclusive human-reviewed source-frame intervals.

    These intervals deliberately remain separate from the colour/motion proxy.
    In particular, a one-frame proxy dropout must not silently redefine a
    reviewed visible interval as target-absent.  Manifests that predate these
    fields remain valid and simply return no reviewed temporal metric.
    """

    raw = case.get(key)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise OfflineBenchmarkError(f"case {case.get('name')}: {key} must be a list")
    if not sampled_source_frames:
        raise OfflineBenchmarkError(
            f"case {case.get('name')}: cannot validate {key} without sampled frames"
        )
    sampled = set(int(value) for value in sampled_source_frames)
    first_source = int(sampled_source_frames[0])
    last_source = int(sampled_source_frames[-1])
    intervals: list[tuple[int, int]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, list) or len(item) != 2:
            raise OfflineBenchmarkError(
                f"case {case.get('name')}: {key}[{index}] must be [start, end]"
            )
        start, end = (int(value) for value in item)
        if start > end:
            raise OfflineBenchmarkError(
                f"case {case.get('name')}: {key}[{index}] start={start} exceeds "
                f"end={end}"
            )
        if start < first_source or end > last_source:
            raise OfflineBenchmarkError(
                f"case {case.get('name')}: {key}[{index}]={start}..{end} is "
                f"outside sampled source range {first_source}..{last_source}"
            )
        if not any(start <= source <= end for source in sampled):
            raise OfflineBenchmarkError(
                f"case {case.get('name')}: {key}[{index}]={start}..{end} "
                "contains no sampled source frame"
            )
        intervals.append((start, end))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] <= previous[1]:
            raise OfflineBenchmarkError(
                f"case {case.get('name')}: {key} intervals overlap: "
                f"{previous} and {current}"
            )
    return intervals


def _inside_source_intervals(
    source_frame: int, intervals: list[tuple[int, int]]
) -> bool:
    return any(start <= int(source_frame) <= end for start, end in intervals)


def _longest_true_run(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _draw_diagnostic(
    frame: np.ndarray,
    candidate: np.ndarray,
    proxy: np.ndarray,
    contamination: np.ndarray,
    *,
    title: str,
) -> np.ndarray:
    image = frame.copy()
    image[candidate] = (
        0.55 * image[candidate]
        + 0.45 * np.asarray([255, 0, 255], dtype=np.float32)
    ).astype(np.uint8)
    image[contamination] = (
        0.25 * image[contamination]
        + 0.75 * np.asarray([0, 0, 255], dtype=np.float32)
    ).astype(np.uint8)
    proxy_contours, _ = cv2.findContours(
        proxy.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, proxy_contours, -1, (0, 255, 0), 2)
    cv2.putText(
        image,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _draw_sparse_ground_truth_diagnostic(
    frame: np.ndarray,
    candidate: np.ndarray,
    ground_truth: np.ndarray,
    *,
    source_frame: int,
    iou: float,
    recall: float,
    precision: float,
) -> np.ndarray:
    image = frame.copy()
    true_positive = np.logical_and(candidate, ground_truth)
    false_positive = np.logical_and(candidate, ~ground_truth)
    false_negative = np.logical_and(~candidate, ground_truth)
    image[true_positive] = (
        0.45 * image[true_positive]
        + 0.55 * np.asarray([0, 220, 0], dtype=np.float32)
    ).astype(np.uint8)
    image[false_positive] = (
        0.25 * image[false_positive]
        + 0.75 * np.asarray([0, 0, 255], dtype=np.float32)
    ).astype(np.uint8)
    image[false_negative] = (
        0.25 * image[false_negative]
        + 0.75 * np.asarray([255, 0, 0], dtype=np.float32)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        ground_truth.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, (0, 255, 255), 2)
    cv2.putText(
        image,
        (
            f"sparse GT source={source_frame} IoU={iou:.3f} "
            f"R={recall:.3f} P={precision:.3f}"
        ),
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _update_worst(
    entries: list[tuple[float, int, np.ndarray]],
    score: float,
    frame_id: int,
    image: np.ndarray,
    *,
    limit: int = 3,
) -> None:
    if not np.isfinite(score):
        return
    entries.append((float(score), int(frame_id), image.copy()))
    entries.sort(key=lambda item: item[0], reverse=True)
    del entries[limit:]


def _threshold_results(
    summary: dict[str, Any],
    thresholds: dict[str, Any],
    *,
    non_gating_thresholds: set[str] | None = None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    diagnostic_only = set() if non_gating_thresholds is None else non_gating_thresholds
    paths = {
        "target_coverage_min": ("target_coverage", "value", "min"),
        "proxy_recall_p05_min": ("proxy_recall", "p05", "min"),
        "contamination_p95_max": ("contamination_proxy", "p95", "max"),
        "bbox_growth_p95_max": ("bbox_area_growth", "p95", "max"),
        "excess_jump_p95_max": ("excess_centroid_jump_normalized", "p95", "max"),
        "reviewed_centroid_error_p95_max": (
            "reviewed_centroid_error_normalized",
            "p95",
            "max",
        ),
        "reviewed_centroid_error_max": (
            "reviewed_centroid_error_normalized",
            "max",
            "max",
        ),
        "reviewed_motion_residual_p95_max": (
            "reviewed_motion_residual_normalized",
            "p95",
            "max",
        ),
        "latency_p95_ms_max": ("latency_ms", "p95", "max"),
        "latency_max_ms_max": ("latency_ms", "max", "max"),
        "absent_false_positive_max": ("absent_false_positive", "value", "max"),
        "reviewed_visible_longest_invalid_run_max": (
            "reviewed_visible_longest_invalid_run",
            "value",
            "max",
        ),
        "reviewed_visible_nonempty_coverage_min": (
            "reviewed_visible_nonempty_coverage",
            "value",
            "min",
        ),
        "reviewed_reappearance_ticks_max": (
            "reviewed_reappearance_ticks",
            "max",
            "max",
        ),
        "latency_over_50ms_fraction_max": (
            "latency_over_50ms_fraction",
            "value",
            "max",
        ),
        "latency_over_50ms_longest_run_max": (
            "latency_over_50ms_longest_run",
            "value",
            "max",
        ),
        "reviewed_absent_false_positive_max": (
            "reviewed_absent_false_positive",
            "value",
            "max",
        ),
        "sparse_gt_reviewed_frames_min": (
            "sparse_gt_reviewed_frames",
            "value",
            "min",
        ),
        "sparse_gt_iou_p05_min": ("sparse_gt_iou", "p05", "min"),
        "sparse_gt_recall_p05_min": ("sparse_gt_recall", "p05", "min"),
        "sparse_gt_precision_p05_min": (
            "sparse_gt_precision",
            "p05",
            "min",
        ),
        "sparse_gt_contamination_p95_max": (
            "sparse_gt_contamination",
            "p95",
            "max",
        ),
    }
    if not thresholds:
        raise OfflineBenchmarkError("acceptance thresholds must be non-empty")
    unknown_thresholds = set(thresholds).difference(paths)
    if unknown_thresholds:
        raise OfflineBenchmarkError(
            f"unknown acceptance threshold keys: {sorted(unknown_thresholds)}"
        )
    for key, (metric_name, field, relation) in paths.items():
        if key not in thresholds:
            continue
        gating = key not in diagnostic_only
        metric = summary.get(metric_name)
        actual = None if metric is None else metric.get(field)
        expected = float(thresholds[key])
        if not np.isfinite(expected):
            raise OfflineBenchmarkError(
                f"acceptance threshold {key} must be finite"
            )
        if actual is None:
            passed = False
            detail = "metric unavailable"
        elif relation == "min":
            passed = float(actual) >= expected
            detail = f"{float(actual):.6g} >= {expected:.6g}"
        else:
            passed = float(actual) <= expected
            detail = f"{float(actual):.6g} <= {expected:.6g}"
        checks.append(
            {
                "threshold": key,
                "metric": metric_name,
                "gating": gating,
                "passed": bool(passed),
                "detail": (
                    detail if gating else f"{detail}; diagnostic only (non-gating)"
                ),
            }
        )
    if not checks:
        raise OfflineBenchmarkError("acceptance produced no recognized checks")
    return checks


def evaluate_case(
    case: dict[str, Any],
    *,
    manifest_base: Path,
    output_root: Path,
) -> dict[str, Any]:
    name = str(case.get("name", "")).strip()
    if not name:
        raise OfflineBenchmarkError("every case requires a non-empty name")
    video_path = _resolve_path(case.get("video"), manifest_base, description="video")
    candidate_spec = dict(case.get("candidate") or {})
    candidate_name = str(candidate_spec.get("name", "candidate"))
    mask_count, masks = _iter_masks(candidate_spec, manifest_base)
    timings = _load_timings(candidate_spec, manifest_base)
    provided_latency_summary = _load_latency_summary(
        candidate_spec, manifest_base
    )
    start_frame = int(case.get("source_start_frame", 0))
    evaluation_fps = case.get("evaluation_fps_hz")
    if evaluation_fps is not None:
        evaluation_fps = float(evaluation_fps)
    source_fps, source_indices, frames = _iter_sampled_video(
        video_path,
        start_frame=start_frame,
        evaluation_fps=evaluation_fps,
    )
    if mask_count != len(source_indices):
        raise OfflineBenchmarkError(
            f"case {name}: candidate masks={mask_count}, sampled video frames="
            f"{len(source_indices)}; fix start/rate alignment instead of truncating"
        )
    sparse_gt, sparse_gt_provenance = _load_sparse_ground_truth(
        dict(case.get("sparse_ground_truth") or {}),
        manifest_base,
        sampled_source_frames=set(source_indices),
        expected_source_video=video_path,
    )
    proxy_cfg = dict(case.get("target_proxy") or {})
    first_visible_source_frame = case.get("target_first_visible_source_frame")
    if first_visible_source_frame is not None:
        first_visible_source_frame = int(first_visible_source_frame)
        if first_visible_source_frame < start_frame:
            raise OfflineBenchmarkError(
                f"case {name}: target_first_visible_source_frame="
                f"{first_visible_source_frame} precedes source_start_frame="
                f"{start_frame}"
            )
    reviewed_visible_intervals = _reviewed_source_intervals(
        case,
        "reviewed_visible_source_intervals",
        sampled_source_frames=source_indices,
    )
    reviewed_absent_intervals = _reviewed_source_intervals(
        case,
        "reviewed_absent_source_intervals",
        sampled_source_frames=source_indices,
    )
    reviewed_visible_sources = {
        source
        for source in source_indices
        if _inside_source_intervals(source, reviewed_visible_intervals)
    }
    reviewed_absent_sources = {
        source
        for source in source_indices
        if _inside_source_intervals(source, reviewed_absent_intervals)
    }
    reviewed_overlap = reviewed_visible_sources.intersection(reviewed_absent_sources)
    if reviewed_overlap:
        raise OfflineBenchmarkError(
            f"case {name}: reviewed visible/absent intervals overlap on sampled "
            f"source frames {sorted(reviewed_overlap)}"
        )
    reviewed_motion_track, reviewed_motion_provenance = (
        _load_reviewed_motion_track(
            dict(case.get("reviewed_motion_track") or {}),
            manifest_base,
            case_name=name,
            expected_source_video=video_path,
            expected_source_fps=source_fps,
            expected_source_indices=source_indices,
            expected_source_start_frame=start_frame,
            expected_evaluation_fps=(
                source_fps if evaluation_fps is None else evaluation_fps
            ),
            expected_reviewed_visible_intervals=reviewed_visible_intervals,
        )
    )
    reviewed_motion_seed_source_frame: int | None = None
    if reviewed_motion_provenance is not None and case.get("seed_source_frame") is not None:
        reviewed_motion_seed_source_frame = int(case["seed_source_frame"])
        if reviewed_motion_seed_source_frame not in source_indices:
            raise OfflineBenchmarkError(
                f"case {name}: seed_source_frame={reviewed_motion_seed_source_frame} "
                "must be an exact sampled source frame when reviewed motion is used"
            )
    raw_reappearance_sources = case.get("reviewed_reappearance_source_frames")
    if raw_reappearance_sources is None:
        reviewed_reappearance_sources: list[int] = []
    else:
        if not isinstance(raw_reappearance_sources, list):
            raise OfflineBenchmarkError(
                f"case {name}: reviewed_reappearance_source_frames must be a list"
            )
        reviewed_reappearance_sources = [
            int(value) for value in raw_reappearance_sources
        ]
        if len(reviewed_reappearance_sources) != len(
            set(reviewed_reappearance_sources)
        ):
            raise OfflineBenchmarkError(
                f"case {name}: reviewed_reappearance_source_frames has duplicates"
            )
        if reviewed_reappearance_sources != sorted(reviewed_reappearance_sources):
            raise OfflineBenchmarkError(
                f"case {name}: reviewed_reappearance_source_frames must be sorted"
            )
        unknown_reappearance = set(reviewed_reappearance_sources).difference(
            reviewed_visible_sources
        )
        if unknown_reappearance:
            raise OfflineBenchmarkError(
                f"case {name}: reviewed reappearance frames must be sampled and "
                f"inside reviewed-visible intervals: {sorted(unknown_reappearance)}"
            )
        for source_frame in reviewed_reappearance_sources:
            event_position = source_indices.index(source_frame)
            if event_position == 0:
                raise OfflineBenchmarkError(
                    f"case {name}: reappearance {source_frame} has no prior sample"
                )
            previous_source = source_indices[event_position - 1]
            if previous_source not in reviewed_absent_sources:
                raise OfflineBenchmarkError(
                    f"case {name}: reappearance {source_frame} must immediately "
                    f"follow an explicitly reviewed-absent sampled frame; previous "
                    f"source frame is {previous_source}"
                )
    min_visible_proxy = int(proxy_cfg.get("min_visible_area_px", 60))
    coverage_recall = float(case.get("coverage_recall_threshold", 0.20))
    dilation_px = int(case.get("contamination_proxy_dilation_px", 7))
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
    )

    records: list[dict[str, Any]] = []
    previous_frame: np.ndarray | None = None
    previous_proxy_center: tuple[float, float] | None = None
    previous_candidate_center: tuple[float, float] | None = None
    previous_visible_proxy_center: tuple[float, float] | None = None
    previous_pair_centers: tuple[tuple[float, float], tuple[float, float]] | None = None
    previous_reviewed_motion_pair: tuple[
        int,
        tuple[float, float] | None,
        tuple[float, float],
        float,
    ] | None = None
    worst_contamination: list[tuple[float, int, np.ndarray]] = []
    worst_jump: list[tuple[float, int, np.ndarray]] = []
    sparse_gt_diagnostics: list[tuple[int, np.ndarray]] = []
    candidate_areas: list[float] = []
    candidate_boxes: list[tuple[int, int, int, int] | None] = []
    candidate_centers: list[tuple[float, float] | None] = []
    proxy_centers: list[tuple[float, float] | None] = []

    for position, (frame, candidate) in enumerate(zip(frames, masks)):
        if candidate.shape != frame.shape[:2]:
            raise OfflineBenchmarkError(
                f"case {name} frame {position}: mask shape={candidate.shape}, "
                f"video shape={frame.shape[:2]}"
            )
        source_frame = source_indices[position]
        before_known_entry = (
            first_visible_source_frame is not None
            and source_frame < first_visible_source_frame
        )
        if before_known_entry:
            # Some recordings contain unrelated same-colour motion before the
            # labelled target enters the image.  Treat this interval as known
            # target-absent instead of allowing a colour proxy to invent an
            # earlier target.  This boundary is a property of the fixed real
            # test case, not a hint available to the candidate tracker.
            proxy = np.zeros(frame.shape[:2], dtype=bool)
            proxy_center = None
        else:
            proxy, proxy_center = _component_proxy(
                frame, previous_frame, proxy_cfg, previous_proxy_center
            )
        if proxy_center is not None:
            previous_proxy_center = proxy_center
        visible_proxy_area = int(proxy.sum())
        expected_visible = visible_proxy_area >= min_visible_proxy
        candidate_area = int(candidate.sum())
        intersection = int(np.logical_and(candidate, proxy).sum())
        recall = (
            intersection / float(visible_proxy_area) if expected_visible else math.nan
        )
        precision = (
            intersection / float(candidate_area)
            if expected_visible and candidate_area > 0
            else math.nan
        )
        dilated_proxy = cv2.dilate(proxy.astype(np.uint8), dilation_kernel) > 0
        contamination_mask = np.logical_and(candidate, ~dilated_proxy)
        contamination = (
            int(contamination_mask.sum()) / float(candidate_area)
            if expected_visible and candidate_area > 0
            else math.nan
        )
        sparse_label = sparse_gt.get(source_frame)
        sparse_gt_area = None
        sparse_gt_iou = math.nan
        sparse_gt_recall = math.nan
        sparse_gt_precision = math.nan
        sparse_gt_contamination = math.nan
        if sparse_label is not None:
            ground_truth = np.asarray(sparse_label["mask"], dtype=bool)
            if ground_truth.shape != frame.shape[:2]:
                raise OfflineBenchmarkError(
                    f"case {name} source frame {source_frame}: sparse ground-truth "
                    f"shape={ground_truth.shape}, video shape={frame.shape[:2]}"
                )
            sparse_gt_area = int(ground_truth.sum())
            sparse_intersection = int(
                np.logical_and(candidate, ground_truth).sum()
            )
            sparse_union = int(np.logical_or(candidate, ground_truth).sum())
            sparse_gt_iou = sparse_intersection / float(sparse_union)
            sparse_gt_recall = sparse_intersection / float(sparse_gt_area)
            sparse_gt_precision = (
                sparse_intersection / float(candidate_area)
                if candidate_area > 0
                else 0.0
            )
            sparse_gt_contamination = 1.0 - sparse_gt_precision
            sparse_gt_diagnostics.append(
                (
                    source_frame,
                    _draw_sparse_ground_truth_diagnostic(
                        frame,
                        candidate,
                        ground_truth,
                        source_frame=source_frame,
                        iou=sparse_gt_iou,
                        recall=sparse_gt_recall,
                        precision=sparse_gt_precision,
                    ),
                )
            )
        candidate_box = _bbox(candidate)
        candidate_center = _centroid(candidate)
        reviewed_motion_label = reviewed_motion_track.get(source_frame)
        reviewed_centroid_error = math.nan
        reviewed_motion_residual = math.nan
        # Production replay deliberately emits an empty, fail-closed mask before
        # its pinned seed frame.  If the target becomes visible one sampled tick
        # before that seed, temporal visibility coverage must still count the
        # miss, but the dense geometry metric must not charge the same known
        # non-tracking frame a full image-diagonal error a second time.  This
        # exemption is deliberately narrow: a non-empty pre-seed output remains
        # scored, and every empty frame at/after the seed keeps the full missing
        # penalty so a mid-track disappearance cannot game the motion gates.
        reviewed_motion_preseed_fail_closed = bool(
            reviewed_motion_label is not None
            and reviewed_motion_seed_source_frame is not None
            and source_frame < reviewed_motion_seed_source_frame
            and candidate_center is None
        )
        reviewed_motion_score_eligible = bool(
            reviewed_motion_label is not None
            and not reviewed_motion_preseed_fail_closed
        )
        if reviewed_motion_score_eligible:
            reviewed_center = tuple(reviewed_motion_label["centroid_xy"])
            reviewed_diagonal = float(
                reviewed_motion_label["bbox_diagonal_px"]
            )
            missing_penalty_px = float(math.hypot(*frame.shape[:2][::-1]))
            reviewed_centroid_error = (
                missing_penalty_px
                if candidate_center is None
                else float(
                    np.linalg.norm(
                        np.subtract(candidate_center, reviewed_center)
                    )
                )
            ) / max(1.0, reviewed_diagonal)
            if previous_reviewed_motion_pair is not None:
                (
                    previous_position,
                    previous_candidate,
                    previous_reviewed,
                    previous_diagonal,
                ) = previous_reviewed_motion_pair
                if position == previous_position + 1:
                    motion_scale = max(
                        1.0, 0.5 * (previous_diagonal + reviewed_diagonal)
                    )
                    if candidate_center is None or previous_candidate is None:
                        reviewed_motion_residual = missing_penalty_px / motion_scale
                    else:
                        candidate_delta = np.subtract(
                            candidate_center, previous_candidate
                        )
                        reviewed_delta = np.subtract(
                            reviewed_center, previous_reviewed
                        )
                        reviewed_motion_residual = float(
                            np.linalg.norm(candidate_delta - reviewed_delta)
                        ) / motion_scale
            previous_reviewed_motion_pair = (
                position,
                candidate_center,
                reviewed_center,
                reviewed_diagonal,
            )
        elif reviewed_motion_label is not None:
            # The seed starts a new candidate-motion chain.  Never create a
            # missing->seed residual from an intentional pre-seed empty mask.
            previous_reviewed_motion_pair = None
        elif reviewed_motion_track:
            # Multiple reviewed-visible intervals may be separated by an
            # explicitly absent/fully-occluded interval.  Never score a motion
            # delta across that gap.
            previous_reviewed_motion_pair = None
        target_hit = bool(expected_visible and recall >= coverage_recall)
        absent_false_positive = bool(not expected_visible and candidate_area > 0)

        centroid_jump = math.nan
        if candidate_center is not None and previous_candidate_center is not None:
            centroid_jump = float(
                np.linalg.norm(np.subtract(candidate_center, previous_candidate_center))
            )
        if candidate_center is not None:
            previous_candidate_center = candidate_center

        excess_jump = math.nan
        if expected_visible and candidate_center is not None and proxy_center is not None:
            if previous_pair_centers is not None:
                previous_candidate, previous_proxy = previous_pair_centers
                candidate_delta = np.subtract(candidate_center, previous_candidate)
                proxy_delta = np.subtract(proxy_center, previous_proxy)
                excess_jump = float(np.linalg.norm(candidate_delta - proxy_delta))
            previous_pair_centers = (candidate_center, proxy_center)
            previous_visible_proxy_center = proxy_center
        elif expected_visible:
            previous_pair_centers = None
            previous_visible_proxy_center = proxy_center

        record = {
            "frame": position,
            "source_frame": source_frame,
            "expected_visible": expected_visible,
            "reviewed_visible": source_frame in reviewed_visible_sources,
            "reviewed_absent": source_frame in reviewed_absent_sources,
            "reviewed_reappearance": source_frame
            in reviewed_reappearance_sources,
            "candidate_area_px": candidate_area,
            "proxy_area_px": visible_proxy_area,
            "target_hit": target_hit,
            "absent_false_positive": absent_false_positive,
            "proxy_recall": recall,
            "proxy_precision": precision,
            "contamination_proxy": contamination,
            "sparse_gt_available": sparse_label is not None,
            "sparse_gt_area_px": sparse_gt_area,
            "sparse_gt_iou": sparse_gt_iou,
            "sparse_gt_recall": sparse_gt_recall,
            "sparse_gt_precision": sparse_gt_precision,
            "sparse_gt_contamination": sparse_gt_contamination,
            "reviewed_motion_track_available": reviewed_motion_label is not None,
            "reviewed_motion_score_eligible": reviewed_motion_score_eligible,
            "reviewed_motion_score_exclusion": (
                "pre_seed_fail_closed"
                if reviewed_motion_preseed_fail_closed
                else None
            ),
            "reviewed_centroid_x": (
                None
                if reviewed_motion_label is None
                else float(reviewed_motion_label["centroid_xy"][0])
            ),
            "reviewed_centroid_y": (
                None
                if reviewed_motion_label is None
                else float(reviewed_motion_label["centroid_xy"][1])
            ),
            "reviewed_centroid_error_normalized": reviewed_centroid_error,
            "reviewed_motion_residual_normalized": reviewed_motion_residual,
            "bbox_x1": None if candidate_box is None else candidate_box[0],
            "bbox_y1": None if candidate_box is None else candidate_box[1],
            "bbox_x2": None if candidate_box is None else candidate_box[2],
            "bbox_y2": None if candidate_box is None else candidate_box[3],
            "centroid_jump_px": centroid_jump,
            "excess_centroid_jump_px": excess_jump,
            "latency_ms": timings.get(position),
        }
        records.append(record)
        candidate_areas.append(float(candidate_area))
        candidate_boxes.append(candidate_box)
        candidate_centers.append(candidate_center)
        proxy_centers.append(proxy_center)
        title = (
            f"{name} frame={position} recall={recall:.3f} "
            f"contam={contamination:.3f}"
        )
        diagnostic = _draw_diagnostic(
            frame, candidate, proxy, contamination_mask, title=title
        )
        _update_worst(
            worst_contamination,
            contamination,
            position,
            diagnostic,
        )
        _update_worst(worst_jump, excess_jump, position, diagnostic)
        previous_frame = frame

    reference_limit = int(case.get("reference_window_frames", 15))
    reference_boxes = [
        box
        for record, box in zip(records, candidate_boxes)
        if box is not None
        and record["expected_visible"]
        and record["proxy_recall"] >= max(0.5, coverage_recall)
    ][:reference_limit]
    if not reference_boxes:
        reference_boxes = [box for box in candidate_boxes if box is not None][
            :reference_limit
        ]
    if reference_boxes:
        ref_width = float(
            np.median([box[2] - box[0] for box in reference_boxes])
        )
        ref_height = float(
            np.median([box[3] - box[1] for box in reference_boxes])
        )
        ref_area = max(1.0, ref_width * ref_height)
        ref_diagonal = max(1.0, math.hypot(ref_width, ref_height))
    else:
        ref_width = ref_height = ref_area = ref_diagonal = math.nan

    for record, box in zip(records, candidate_boxes):
        if box is None or not np.isfinite(ref_area):
            record["bbox_area_growth"] = math.nan
        else:
            width = box[2] - box[0]
            height = box[3] - box[1]
            record["bbox_area_growth"] = width * height / ref_area
        jump = record["excess_centroid_jump_px"]
        record["excess_centroid_jump_normalized"] = (
            math.nan
            if jump is None or not np.isfinite(jump) or not np.isfinite(ref_diagonal)
            else float(jump) / ref_diagonal
        )

    visible_records = [record for record in records if record["expected_visible"]]
    absent_records = [record for record in records if not record["expected_visible"]]
    reviewed_visible_records = [
        record for record in records if record["reviewed_visible"]
    ]
    reviewed_absent_records = [
        record for record in records if record["reviewed_absent"]
    ]
    reviewed_interval_diagnostics: list[dict[str, Any]] = []
    for interval_start, interval_end in reviewed_visible_intervals:
        interval_records = [
            record
            for record in records
            if interval_start <= int(record["source_frame"]) <= interval_end
        ]
        reviewed_interval_diagnostics.append(
            {
                "source_start": interval_start,
                "source_end": interval_end,
                "sampled_frames": len(interval_records),
                "invalid_frames": sum(
                    int(record["candidate_area_px"]) == 0
                    for record in interval_records
                ),
                "longest_consecutive_invalid_frames": _longest_true_run(
                    int(record["candidate_area_px"]) == 0
                    for record in interval_records
                ),
            }
        )
    reviewed_visible_longest_invalid_run = (
        None
        if not reviewed_visible_intervals
        else {
            "value": max(
                int(item["longest_consecutive_invalid_frames"])
                for item in reviewed_interval_diagnostics
            ),
            "reviewed_visible_frames": len(reviewed_visible_records),
            "intervals": reviewed_interval_diagnostics,
        }
    )
    reviewed_visible_nonempty_coverage = (
        None
        if not reviewed_visible_intervals
        else {
            "value": sum(
                int(record["candidate_area_px"]) > 0
                for record in reviewed_visible_records
            )
            / float(len(reviewed_visible_records)),
            "reviewed_visible_frames": len(reviewed_visible_records),
            "nonempty_frames": sum(
                int(record["candidate_area_px"]) > 0
                for record in reviewed_visible_records
            ),
        }
    )

    reappearance_events: list[dict[str, Any]] = []
    for source_frame in reviewed_reappearance_sources:
        event_position = next(
            int(record["frame"])
            for record in records
            if int(record["source_frame"]) == source_frame
        )
        interval_end = next(
            end
            for start, end in reviewed_visible_intervals
            if start <= source_frame <= end
        )
        previous_record = records[event_position - 1]
        prior_reviewed_absent_empty = bool(
            previous_record["reviewed_absent"]
            and int(previous_record["candidate_area_px"]) == 0
        )
        first_valid = (
            next(
                (
                    record
                    for record in records[event_position:]
                    if int(record["source_frame"]) <= interval_end
                    and int(record["candidate_area_px"]) > 0
                ),
                None,
            )
            if prior_reviewed_absent_empty
            else None
        )
        reappearance_events.append(
            {
                "source_frame": source_frame,
                "preceding_reviewed_absent_source_frame": int(
                    previous_record["source_frame"]
                ),
                "preceding_reviewed_absent_was_empty": prior_reviewed_absent_empty,
                "resolved": first_valid is not None,
                "first_nonempty_source_frame": (
                    None if first_valid is None else int(first_valid["source_frame"])
                ),
                "ticks_to_first_nonempty": (
                    None
                    if first_valid is None
                    else int(first_valid["frame"]) - event_position
                ),
            }
        )
    reappearance_ticks_values = [
        int(event["ticks_to_first_nonempty"])
        for event in reappearance_events
        if event["ticks_to_first_nonempty"] is not None
    ]
    reviewed_reappearance_ticks = (
        None
        if not reviewed_reappearance_sources
        else {
            # A missing recovery is not encoded as an arbitrary large number.
            # Keeping max unavailable makes a configured threshold fail closed.
            "max": (
                max(reappearance_ticks_values)
                if len(reappearance_ticks_values) == len(reappearance_events)
                else None
            ),
            "event_count": len(reappearance_events),
            "resolved_event_count": len(reappearance_ticks_values),
            "events": reappearance_events,
        }
    )
    reviewed_absent_false_positive = (
        None
        if not reviewed_absent_intervals
        else {
            "value": (
                sum(
                    int(record["candidate_area_px"]) > 0
                    for record in reviewed_absent_records
                )
                / float(len(reviewed_absent_records))
            ),
            "absent_frames": len(reviewed_absent_records),
            "nonempty_frames": sum(
                int(record["candidate_area_px"]) > 0
                for record in reviewed_absent_records
            ),
            "intervals": [list(interval) for interval in reviewed_absent_intervals],
        }
    )
    measured_latency = _stats(
        record["latency_ms"]
        for record in records
        if record["latency_ms"] is not None
    )
    latency_records = [
        record for record in records if record["latency_ms"] is not None
    ]
    latency_over_50ms_fraction = (
        None
        if not latency_records
        else {
            "value": sum(
                float(record["latency_ms"]) > 50.0 for record in latency_records
            )
            / float(len(latency_records)),
            "deadline_ms": 50.0,
            "samples": len(latency_records),
            "misses": sum(
                float(record["latency_ms"]) > 50.0 for record in latency_records
            ),
        }
    )
    latency_over_50ms_longest_run = (
        None
        if not latency_records
        else {
            "value": _longest_true_run(
                float(record["latency_ms"]) > 50.0
                for record in latency_records
            ),
            "deadline_ms": 50.0,
            "samples": len(latency_records),
        }
    )
    latency_summary = measured_latency or provided_latency_summary
    sparse_gt_records = [record for record in records if record["sparse_gt_available"]]
    sparse_gt_iou_stats = _stats(
        record["sparse_gt_iou"] for record in sparse_gt_records
    )
    sparse_gt_recall_stats = _stats(
        record["sparse_gt_recall"] for record in sparse_gt_records
    )
    sparse_gt_precision_stats = _stats(
        record["sparse_gt_precision"] for record in sparse_gt_records
    )
    sparse_gt_contamination_stats = _stats(
        record["sparse_gt_contamination"] for record in sparse_gt_records
    )
    reviewed_motion_records = [
        record for record in records if record["reviewed_motion_score_eligible"]
    ]
    reviewed_motion_preseed_fail_closed_sources = [
        int(record["source_frame"])
        for record in records
        if record["reviewed_motion_score_exclusion"] == "pre_seed_fail_closed"
    ]
    reviewed_centroid_error_stats = _stats(
        record["reviewed_centroid_error_normalized"]
        for record in reviewed_motion_records
    )
    reviewed_motion_residual_stats = _stats(
        record["reviewed_motion_residual_normalized"]
        for record in reviewed_motion_records
    )
    summary: dict[str, Any] = {
        "name": name,
        "candidate": candidate_name,
        "video": str(video_path),
        "source_fps_hz": source_fps,
        "evaluation_fps_hz": evaluation_fps or source_fps,
        "source_start_frame": start_frame,
        "target_first_visible_source_frame": first_visible_source_frame,
        "evaluated_frames": len(records),
        "visible_proxy_frames": len(visible_records),
        "proxy_notice": (
            "HSV target pixels are a conservative diagnostic proxy, not labelled "
            "ground-truth masks or true IoU"
        ),
        "target_coverage": {
            "value": (
                sum(bool(record["target_hit"]) for record in visible_records)
                / float(len(visible_records))
                if visible_records
                else None
            ),
            "threshold_recall": coverage_recall,
        },
        "nonempty_coverage": {
            "value": (
                sum(record["candidate_area_px"] > 0 for record in visible_records)
                / float(len(visible_records))
                if visible_records
                else None
            )
        },
        "absent_false_positive": {
            "value": (
                sum(bool(record["absent_false_positive"]) for record in absent_records)
                / float(len(absent_records))
                if absent_records
                else None
            ),
            "absent_frames": len(absent_records),
        },
        "reviewed_visible_source_intervals": [
            list(interval) for interval in reviewed_visible_intervals
        ],
        "reviewed_absent_source_intervals": [
            list(interval) for interval in reviewed_absent_intervals
        ],
        "reviewed_visible_longest_invalid_run": (
            reviewed_visible_longest_invalid_run
        ),
        "reviewed_visible_nonempty_coverage": reviewed_visible_nonempty_coverage,
        "reviewed_reappearance_ticks": reviewed_reappearance_ticks,
        "reviewed_absent_false_positive": reviewed_absent_false_positive,
        "proxy_recall": _stats(record["proxy_recall"] for record in visible_records),
        "proxy_precision": _stats(
            record["proxy_precision"] for record in visible_records
        ),
        "contamination_proxy": _stats(
            record["contamination_proxy"] for record in visible_records
        ),
        "sparse_gt_reviewed_frames": {"value": len(sparse_gt_records)},
        "sparse_gt_iou": sparse_gt_iou_stats,
        "sparse_gt_recall": sparse_gt_recall_stats,
        "sparse_gt_precision": sparse_gt_precision_stats,
        "sparse_gt_contamination": sparse_gt_contamination_stats,
        "reviewed_motion_track": (
            None
            if reviewed_motion_provenance is None
            else {
                **reviewed_motion_provenance,
                "notice": (
                    "Dense human-reviewed visible-object bbox/centroid evidence; "
                    "scored candidate errors are normalized by the reviewed visible "
                    "bbox diagonal. Motion residual compares consecutive candidate "
                    "and reviewed centroid displacements and never spans an absent "
                    "interval. Empty fail-closed samples before the pinned seed are "
                    "excluded only from geometry scoring; reviewed visibility "
                    "coverage still counts them as misses. Non-empty pre-seed and "
                    "all post-seed samples remain scored."
                ),
                "geometry_scoring_seed_source_frame": (
                    reviewed_motion_seed_source_frame
                ),
                "geometry_scored_frame_count": len(reviewed_motion_records),
                "geometry_excluded_pre_seed_fail_closed_source_frames": (
                    reviewed_motion_preseed_fail_closed_sources
                ),
            }
        ),
        "reviewed_centroid_error_normalized": reviewed_centroid_error_stats,
        "reviewed_motion_residual_normalized": reviewed_motion_residual_stats,
        "sparse_ground_truth": (
            None
            if sparse_gt_provenance is None
            else {
                **sparse_gt_provenance,
                "notice": (
                    "Human-reviewed sparse visible-object masks; no hidden pixels "
                    "are filled through RH56 occlusion. Dense proxy metrics remain "
                    "separate temporal diagnostics."
                ),
                "iou": sparse_gt_iou_stats,
                "recall": sparse_gt_recall_stats,
                "precision": sparse_gt_precision_stats,
                "contamination": sparse_gt_contamination_stats,
            }
        ),
        "bbox_area_growth": _stats(
            record["bbox_area_growth"] for record in visible_records
        ),
        "centroid_jump_px": _stats(
            record["centroid_jump_px"] for record in records
        ),
        "excess_centroid_jump_normalized": _stats(
            record["excess_centroid_jump_normalized"] for record in visible_records
        ),
        "latency_ms": latency_summary,
        "latency_source": (
            "states_jsonl"
            if measured_latency is not None
            else ("summary_json" if provided_latency_summary is not None else None)
        ),
        "latency_samples": sum(record["latency_ms"] is not None for record in records),
        # Keep the historical scalar while exposing a structured metric that
        # can participate in threshold checks and reports.
        "latency_deadline_miss_fraction_50ms": (
            None
            if latency_over_50ms_fraction is None
            else latency_over_50ms_fraction["value"]
        ),
        "latency_over_50ms_fraction": latency_over_50ms_fraction,
        "latency_over_50ms_longest_run": latency_over_50ms_longest_run,
        "reference_bbox_wh_px": (
            [ref_width, ref_height]
            if np.isfinite(ref_width) and np.isfinite(ref_height)
            else None
        ),
    }
    thresholds = dict(case.get("thresholds") or {})
    if sparse_gt_provenance is not None:
        sparse_thresholds = dict(
            (case.get("sparse_ground_truth") or {}).get("thresholds") or {}
        )
        duplicate_thresholds = set(thresholds).intersection(sparse_thresholds)
        if duplicate_thresholds:
            raise OfflineBenchmarkError(
                f"case {name}: duplicate sparse/main thresholds: "
                f"{sorted(duplicate_thresholds)}"
            )
        thresholds.update(sparse_thresholds)
    if reviewed_motion_provenance is not None:
        motion_thresholds = dict(reviewed_motion_provenance["thresholds"])
        duplicate_thresholds = set(thresholds).intersection(motion_thresholds)
        if duplicate_thresholds:
            raise OfflineBenchmarkError(
                f"case {name}: duplicate reviewed-motion/main thresholds: "
                f"{sorted(duplicate_thresholds)}"
            )
        thresholds.update(motion_thresholds)
    sufficient_sparse_ground_truth = bool(
        sparse_gt_provenance is not None
        and len(sparse_gt_records)
        >= int(sparse_gt_provenance.get("min_reviewed_frames", 1))
    )
    non_gating_thresholds: set[str] = set()
    if sufficient_sparse_ground_truth:
        # HSV/motion masks deliberately contain only conservative colour
        # support.  Once adequate human-reviewed masks exist, proxy recall and
        # contamination remain useful drift diagnostics but cannot overrule
        # true visible-object IoU/recall/precision.
        non_gating_thresholds.update(
            {"proxy_recall_p05_min", "contamination_p95_max"}
        )
    if reviewed_motion_provenance is not None:
        # The HSV component is not object identity ground truth and can select
        # a same-colour distractor during fast motion.  Only a complete,
        # content-addressed, every-frame human track can demote its historical
        # jump metric; all older manifests retain the exact legacy gate.
        non_gating_thresholds.add("excess_jump_p95_max")
    if (
        reviewed_visible_intervals
        and "reviewed_visible_nonempty_coverage_min" in thresholds
    ):
        non_gating_thresholds.add("target_coverage_min")
    if (
        reviewed_absent_intervals
        and "reviewed_absent_false_positive_max" in thresholds
    ):
        non_gating_thresholds.add("absent_false_positive_max")
    if case.get("proxy_absence_gating") is False:
        non_gating_thresholds.add("absent_false_positive_max")
    summary["proxy_absence_role"] = (
        "gating"
        if "absent_false_positive_max" not in non_gating_thresholds
        else "diagnostic_non_gating"
    )
    summary["dense_proxy_quality_role"] = (
        "diagnostic_non_gating_when_human_sparse_gt_is_sufficient"
        if sufficient_sparse_ground_truth
        else "gating_without_sufficient_human_sparse_gt"
    )
    summary["proxy_motion_role"] = (
        "diagnostic_non_gating_with_complete_reviewed_motion_track"
        if reviewed_motion_provenance is not None
        else "legacy_gating_without_complete_reviewed_motion_track"
    )
    summary["checks"] = _threshold_results(
        summary,
        thresholds,
        non_gating_thresholds=non_gating_thresholds,
    )
    summary["passed"] = all(
        (not bool(check["gating"])) or bool(check["passed"])
        for check in summary["checks"]
    )

    case_output = output_root / name / candidate_name
    case_output.mkdir(parents=True, exist_ok=True)
    frame_path = case_output / "frames.csv"
    with frame_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (case_output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for label, entries in (
        ("contamination", worst_contamination),
        ("jump", worst_jump),
    ):
        for rank, (score, frame_id, image) in enumerate(entries, 1):
            path = case_output / f"worst_{label}_{rank}_frame_{frame_id:04d}.png"
            if not cv2.imwrite(str(path), image):
                raise OfflineBenchmarkError(f"failed to save diagnostic: {path}")
    for source_frame, image in sparse_gt_diagnostics:
        path = case_output / f"sparse_gt_source_{source_frame:06d}.png"
        if not cv2.imwrite(str(path), image):
            raise OfflineBenchmarkError(f"failed to save diagnostic: {path}")
    return summary


def _markdown_report(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# guarded_v2 offline mask benchmark",
        "",
        (
            "These scores use case-specific colour components as conservative "
            "visible-target proxies. They are regression diagnostics, not labelled IoU."
        ),
        (
            "When sufficient human sparse ground truth is present, proxy recall and "
            "contamination are explicitly non-gating; reviewed visible intervals and "
            "sparse masks provide the formal coverage/quality gates."
        ),
        "",
        "| Case | Candidate | Proxy coverage | Proxy recall p05 | Proxy contam p95 | BBox growth p95 | Jump p95 | Proxy-absent frames | Proxy-absent FP | Latency p95 ms | PASS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    def value(summary: dict[str, Any], metric: str, field: str) -> str:
        item = summary.get(metric)
        raw = None if item is None else item.get(field)
        return "n/a" if raw is None else f"{float(raw):.3f}"

    for summary in summaries:
        lines.append(
            "| {name} | {candidate} | {coverage} | {recall} | {contam} | "
            "{growth} | {jump} | {absent_frames} | {absent_fp} | {latency} | "
            "{passed} |".format(
                name=summary["name"],
                candidate=summary["candidate"],
                coverage=value(summary, "target_coverage", "value"),
                recall=value(summary, "proxy_recall", "p05"),
                contam=value(summary, "contamination_proxy", "p95"),
                growth=value(summary, "bbox_area_growth", "p95"),
                jump=value(summary, "excess_centroid_jump_normalized", "p95"),
                absent_frames=int(
                    summary["absent_false_positive"]["absent_frames"]
                ),
                absent_fp=value(summary, "absent_false_positive", "value"),
                latency=value(summary, "latency_ms", "p95"),
                passed="yes" if summary["passed"] else "NO",
            )
        )
    lines.append("")
    if any(
        summary.get("reviewed_visible_source_intervals")
        or summary.get("reviewed_absent_source_intervals")
        for summary in summaries
    ):
        lines.extend(
            [
                "## Human-reviewed temporal gates",
                "",
                (
                    "These intervals are explicit source-frame evidence and do not "
                    "inherit one-frame visibility dropouts from the HSV proxy."
                ),
                "",
                "| Case | Reviewed visible frames | Nonempty coverage | Max invalid run | Reappearance events | Max recovery ticks | Reviewed absent frames | Absent FP | >50 ms fraction |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for summary in summaries:
            visible = summary.get("reviewed_visible_longest_invalid_run")
            reviewed_coverage = summary.get(
                "reviewed_visible_nonempty_coverage"
            )
            reappearance = summary.get("reviewed_reappearance_ticks")
            absent = summary.get("reviewed_absent_false_positive")
            latency_miss = summary.get("latency_over_50ms_fraction")
            lines.append(
                "| {name} | {visible_frames} | {coverage} | {invalid} | {events} | "
                "{recovery} | {absent_frames} | {absent_fp} | {miss} |".format(
                    name=summary["name"],
                    visible_frames=(
                        "n/a"
                        if visible is None
                        else int(visible["reviewed_visible_frames"])
                    ),
                    coverage=(
                        "n/a"
                        if reviewed_coverage is None
                        else f"{float(reviewed_coverage['value']):.3f}"
                    ),
                    invalid=(
                        "n/a" if visible is None else int(visible["value"])
                    ),
                    events=(
                        "n/a"
                        if reappearance is None
                        else int(reappearance["event_count"])
                    ),
                    recovery=(
                        "n/a"
                        if reappearance is None or reappearance["max"] is None
                        else int(reappearance["max"])
                    ),
                    absent_frames=(
                        "n/a" if absent is None else int(absent["absent_frames"])
                    ),
                    absent_fp=(
                        "n/a" if absent is None else f"{float(absent['value']):.3f}"
                    ),
                    miss=(
                        "n/a"
                        if latency_miss is None
                        else f"{float(latency_miss['value']):.3f}"
                    ),
                )
            )
        lines.append("")
    if any(summary.get("sparse_ground_truth") is not None for summary in summaries):
        lines.extend(
            [
                "## Human-reviewed sparse ground truth",
                "",
                (
                    "Only labelled source frames are scored here. Red means candidate "
                    "contamination, blue means missed visible target, and hidden target "
                    "pixels behind RH56 are never invented."
                ),
                "",
                "| Case | Reviewed frames | IoU p05 | Recall p05 | Precision p05 | Contam p95 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for summary in summaries:
            if summary.get("sparse_ground_truth") is None:
                continue
            lines.append(
                "| {name} | {count} | {iou} | {recall} | {precision} | "
                "{contamination} |".format(
                    name=summary["name"],
                    count=int(summary["sparse_gt_reviewed_frames"]["value"]),
                    iou=value(summary, "sparse_gt_iou", "p05"),
                    recall=value(summary, "sparse_gt_recall", "p05"),
                    precision=value(summary, "sparse_gt_precision", "p05"),
                    contamination=value(
                        summary, "sparse_gt_contamination", "p95"
                    ),
                )
            )
        lines.append("")
    if any(summary.get("reviewed_motion_track") is not None for summary in summaries):
        lines.extend(
            [
                "## Human-reviewed dense bbox motion track",
                "",
                (
                    "Centroid error is normalized by the reviewed visible bbox "
                    "diagonal. Motion residual compares consecutive candidate and "
                    "reviewed displacements. With a complete content-addressed "
                    "track, the HSV excess-jump score is diagnostic only."
                ),
                "",
                "| Case | Geometry-scored frames | Centroid p95 | Centroid max | Motion residual p95 | HSV jump role |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for summary in summaries:
            track = summary.get("reviewed_motion_track")
            if track is None:
                continue
            lines.append(
                "| {name} | {count} | {centroid_p95} | {centroid_max} | "
                "{motion_p95} | {role} |".format(
                    name=summary["name"],
                    count=int(track["geometry_scored_frame_count"]),
                    centroid_p95=value(
                        summary, "reviewed_centroid_error_normalized", "p95"
                    ),
                    centroid_max=value(
                        summary, "reviewed_centroid_error_normalized", "max"
                    ),
                    motion_p95=value(
                        summary, "reviewed_motion_residual_normalized", "p95"
                    ),
                    role=summary["proxy_motion_role"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def evaluate_manifest(
    manifest_path: str | Path,
    output: str | Path,
    *,
    candidate_root: str | Path | None = None,
    candidate_name: str = "guarded_v2",
    candidate_root_mode: str = "production",
    selected_cases: Iterable[str] | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise OfflineBenchmarkError(
            f"unsupported schema_version={manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise OfflineBenchmarkError("manifest cases must be a non-empty list")
    if not all(isinstance(case, dict) for case in cases):
        raise OfflineBenchmarkError("every manifest case must be an object")
    full_case_names = tuple(str(case.get("name", "")).strip() for case in cases)
    if any(not name for name in full_case_names) or len(full_case_names) != len(
        set(full_case_names)
    ):
        raise OfflineBenchmarkError("benchmark case names must be non-empty and unique")
    sparse_by_case, sparse_manifest_path = _load_sparse_ground_truth_manifest(
        manifest.get("sparse_ground_truth_manifest"), manifest_file.parent
    )
    expected_sparse_sha = str(
        manifest.get("expected_sparse_ground_truth_manifest_sha256", "")
    ).lower()
    if sparse_manifest_path is not None and expected_sparse_sha:
        actual_sparse_sha = _sha256_file(sparse_manifest_path)
        if actual_sparse_sha != expected_sparse_sha:
            raise OfflineBenchmarkError(
                "sparse ground-truth manifest digest differs from benchmark pin"
            )
    if candidate_root is not None and candidate_root_mode == "production":
        if sparse_manifest_path is None or not expected_sparse_sha:
            raise OfflineBenchmarkError(
                "production evaluation requires a pinned sparse GT manifest"
            )
        if full_case_names != PRODUCTION_CASE_NAMES:
            raise OfflineBenchmarkError(
                "production benchmark requires exactly the fixed five cases"
            )
        if set(sparse_by_case) != set(PRODUCTION_CASE_NAMES):
            raise OfflineBenchmarkError(
                "production sparse GT manifest must contain exactly all five cases"
            )
        _verify_production_suite_coverage_contract(
            manifest=manifest,
            cases=[dict(case) for case in cases],
            sparse_by_case=sparse_by_case,
        )
    if sparse_by_case:
        benchmark_case_names = {
            str(case.get("name", "")).strip()
            for case in cases
            if isinstance(case, dict)
        }
        unknown = set(sparse_by_case).difference(benchmark_case_names)
        if unknown:
            raise OfflineBenchmarkError(
                "sparse ground-truth manifest contains unknown benchmark cases: "
                f"{sorted(unknown)}"
            )
        cases = [
            {
                **dict(raw_case),
                **(
                    {"sparse_ground_truth": sparse_by_case[str(raw_case["name"])]}
                    if str(raw_case.get("name", "")) in sparse_by_case
                    else {}
                ),
            }
            for raw_case in cases
        ]
    requested = (
        None if selected_cases is None else {str(name) for name in selected_cases}
    )
    if requested is not None:
        unknown_requested = requested.difference(full_case_names)
        if unknown_requested:
            raise OfflineBenchmarkError(
                f"unknown selected cases: {sorted(unknown_requested)}"
            )
        if candidate_root_mode == "production":
            raise OfflineBenchmarkError(
                "selected_cases is diagnostic/partial only and cannot receive "
                "production PASS"
            )
        cases = [case for case in cases if str(case["name"]) in requested]
        if not cases:
            raise OfflineBenchmarkError("selected_cases produced an empty benchmark")
    if candidate_root is not None:
        root = Path(candidate_root).expanduser().resolve()
        allowed_modes = {"production", "diagnostic_partial", "diagnostic_legacy"}
        if candidate_root_mode not in allowed_modes:
            raise OfflineBenchmarkError(
                f"candidate_root_mode must be one of {sorted(allowed_modes)}"
            )
        replay_contract: dict[str, dict[str, Any]] = {}
        if candidate_root_mode != "diagnostic_legacy":
            replay_contract = _verify_replay_candidate_root(
                root=root,
                cases=[dict(case) for case in cases],
                manifest=manifest,
                manifest_base=manifest_file.parent,
                require_production=candidate_root_mode == "production",
            )
        overridden_cases: list[dict[str, Any]] = []
        for raw_case in cases:
            case = dict(raw_case)
            name = str(case.get("name", "")).strip()
            case_root = root / name
            states = case_root / "states.jsonl"
            candidate: dict[str, Any] = {
                "name": str(candidate_name),
                "kind": "directory",
                "path": str(case_root / "masks"),
            }
            if states.is_file():
                candidate["states"] = str(states)
                candidate["states_latency_field"] = "compute_processing_ms"
                candidate["states_latency_evaluable_field"] = (
                    "latency_evaluable"
                )
            if candidate_root_mode != "diagnostic_legacy":
                case_contract = replay_contract[name]
                candidate["strict_states_contract"] = True
                candidate["expected_source_indices"] = case_contract[
                    "sampled_source_indices"
                ]
                candidate["seed_position"] = int(case_contract["seed_position"])
                if case_contract.get("guarded_v2_bootstrap_final") is not None:
                    candidate["expected_guarded_v2_bootstrap_final"] = (
                        case_contract["guarded_v2_bootstrap_final"]
                    )
            case["candidate"] = candidate
            overridden_cases.append(case)
        cases = overridden_cases
    elif selected_cases is not None:
        raise OfflineBenchmarkError("selected_cases requires candidate_root")
    output_root = Path(output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        evaluate_case(
            dict(case), manifest_base=manifest_file.parent, output_root=output_root
        )
        for case in cases
    ]
    diagnostic_checks_passed = all(summary["passed"] for summary in summaries)
    production_eligible = bool(
        candidate_root is not None
        and candidate_root_mode == "production"
        and tuple(summary["name"] for summary in summaries) == PRODUCTION_CASE_NAMES
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest_file),
        "sparse_ground_truth_manifest": (
            None if sparse_manifest_path is None else str(sparse_manifest_path)
        ),
        "sparse_ground_truth_manifest_sha256": (
            None
            if sparse_manifest_path is None
            else _sha256_file(sparse_manifest_path)
        ),
        "candidate_root_override": (
            None
            if candidate_root is None
            else str(Path(candidate_root).expanduser().resolve())
        ),
        "candidate_root_mode": (
            None if candidate_root is None else candidate_root_mode
        ),
        "case_count": len(summaries),
        "diagnostic_checks_passed": diagnostic_checks_passed,
        "production_acceptance_eligible": production_eligible,
        "diagnostic_only": bool(candidate_root is not None and not production_eligible),
        "passed": bool(diagnostic_checks_passed and production_eligible)
        if candidate_root is not None
        else diagnostic_checks_passed,
        "cases": summaries,
    }
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "REPORT.md").write_text(
        _markdown_report(summaries), encoding="utf-8"
    )
    return result
