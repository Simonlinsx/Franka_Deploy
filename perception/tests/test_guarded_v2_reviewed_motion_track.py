from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dynamic_pcd.evaluation.guarded_v2_offline import (
    OfflineBenchmarkError,
    REVIEWED_MOTION_TRACK_THRESHOLDS,
    evaluate_manifest,
)


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height)
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reviewed_motion_fixture(
    tmp_path: Path,
    *,
    candidate_jump: bool = False,
    candidate_empty_frames: tuple[int, ...] = (),
    include_track: bool = True,
    seed_source_frame: int | None = None,
) -> tuple[Path, Path, dict]:
    video = tmp_path / "source.mp4"
    masks = tmp_path / "masks"
    masks.mkdir()
    frames: list[np.ndarray] = []
    records: list[dict] = []
    for frame_index in range(6):
        target_center = (20 + 8 * frame_index, 42)
        proxy_center = (82 - 10 * frame_index, 15)
        frame = np.full((64, 96, 3), 20, dtype=np.uint8)
        cv2.circle(frame, target_center, 5, (20, 220, 20), -1)
        cv2.circle(frame, proxy_center, 5, (240, 20, 20), -1)
        frames.append(frame)

        candidate_center = target_center
        if candidate_jump and frame_index == 3:
            candidate_center = proxy_center
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if frame_index not in candidate_empty_frames:
            cv2.circle(mask, candidate_center, 5, 255, -1)
        assert cv2.imwrite(str(masks / f"{frame_index:06d}.png"), mask)
        records.append(
            {
                "source_frame": frame_index,
                "bbox_xyxy": [
                    target_center[0] - 5,
                    target_center[1] - 5,
                    target_center[0] + 6,
                    target_center[1] + 6,
                ],
                "centroid_xy": [float(target_center[0]), float(target_center[1])],
                "review_status": "human_confirmed",
                "label_method": "synthetic exact visible target fixture",
                "review_notes": "Every source frame is deterministically reviewed.",
            }
        )
    _write_video(video, frames)
    track_payload = {
        "schema": "reviewed_visible_bbox_track_v1",
        "case": "reviewed_motion",
        "source_video_sha256": _sha256(video),
        "source_video_fps_hz": 20.0,
        "source_video_frame_count": 6,
        "source_video_size_wh": [96, 64],
        "source_start_frame": 0,
        "evaluation_fps_hz": 20.0,
        "sampled_source_indices": list(range(6)),
        "reviewed_visible_source_intervals": [[0, 5]],
        "reviewed_visible_sampled_source_indices": list(range(6)),
        "thresholds": dict(REVIEWED_MOTION_TRACK_THRESHOLDS),
        "records": records,
    }
    track = tmp_path / "reviewed_motion.json"
    track.write_text(
        json.dumps(track_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    case = {
        "name": "reviewed_motion",
        "video": str(video),
        "candidate": {"kind": "directory", "path": str(masks)},
        "evaluation_fps_hz": 20.0,
        "reviewed_visible_source_intervals": [[0, 5]],
        "target_proxy": {
            # Deliberately tracks a moving blue distractor instead of the green
            # target.  This makes the legacy excess-jump proxy objectively bad.
            "hsv_ranges": [[[105, 80, 40], [135, 255, 255]]],
            "initial_seed_xy": [82, 15],
            "min_area_px": 20,
            "min_visible_area_px": 20,
            "max_center_jump_px": 30,
            "open_kernel": 1,
            "close_kernel": 1,
        },
        "thresholds": {"excess_jump_p95_max": 0.05},
    }
    if include_track:
        case["reviewed_motion_track"] = {
            "path": str(track),
            "sha256": _sha256(track),
        }
    if seed_source_frame is not None:
        case["seed_source_frame"] = int(seed_source_frame)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "cases": [case]}), encoding="utf-8"
    )
    return manifest, track, track_payload


def test_complete_reviewed_track_replaces_broken_hsv_motion_gate(tmp_path):
    manifest, _track, _payload = _reviewed_motion_fixture(tmp_path)
    result = evaluate_manifest(manifest, tmp_path / "output")
    case = result["cases"][0]
    checks = {check["threshold"]: check for check in case["checks"]}

    assert result["passed"]
    assert checks["reviewed_centroid_error_p95_max"]["passed"]
    assert checks["reviewed_centroid_error_max"]["passed"]
    assert checks["reviewed_motion_residual_p95_max"]["passed"]
    assert not checks["excess_jump_p95_max"]["passed"]
    assert not checks["excess_jump_p95_max"]["gating"]
    assert case["proxy_motion_role"] == (
        "diagnostic_non_gating_with_complete_reviewed_motion_track"
    )


def test_reviewed_track_rejects_candidate_distractor_jump(tmp_path):
    manifest, _track, _payload = _reviewed_motion_fixture(
        tmp_path, candidate_jump=True
    )
    result = evaluate_manifest(manifest, tmp_path / "output")
    case = result["cases"][0]
    checks = {check["threshold"]: check for check in case["checks"]}

    assert not result["passed"]
    assert not checks["reviewed_centroid_error_max"]["passed"]
    assert not checks["reviewed_motion_residual_p95_max"]["passed"]
    assert checks["reviewed_centroid_error_max"]["gating"]


def test_reviewed_track_does_not_double_penalize_preseed_fail_closed(tmp_path):
    manifest, _track, _payload = _reviewed_motion_fixture(
        tmp_path,
        candidate_empty_frames=(0,),
        seed_source_frame=1,
    )

    result = evaluate_manifest(manifest, tmp_path / "output")
    case = result["cases"][0]
    checks = {check["threshold"]: check for check in case["checks"]}
    track = case["reviewed_motion_track"]

    assert result["passed"]
    assert case["reviewed_visible_nonempty_coverage"]["value"] == pytest.approx(
        5.0 / 6.0
    )
    assert track["geometry_scoring_seed_source_frame"] == 1
    assert track["geometry_scored_frame_count"] == 5
    assert track["geometry_excluded_pre_seed_fail_closed_source_frames"] == [0]
    assert checks["reviewed_centroid_error_max"]["passed"]
    assert checks["reviewed_motion_residual_p95_max"]["passed"]


def test_reviewed_track_still_penalizes_midtrack_disappearance(tmp_path):
    manifest, _track, _payload = _reviewed_motion_fixture(
        tmp_path,
        candidate_empty_frames=(3,),
        seed_source_frame=1,
    )

    result = evaluate_manifest(manifest, tmp_path / "output")
    case = result["cases"][0]
    checks = {check["threshold"]: check for check in case["checks"]}
    track = case["reviewed_motion_track"]

    assert not result["passed"]
    assert track["geometry_excluded_pre_seed_fail_closed_source_frames"] == []
    assert not checks["reviewed_centroid_error_max"]["passed"]
    assert not checks["reviewed_motion_residual_p95_max"]["passed"]


def test_reviewed_track_still_scores_nonempty_preseed_distractor(tmp_path):
    manifest, _track, _payload = _reviewed_motion_fixture(
        tmp_path,
        candidate_jump=True,
        seed_source_frame=4,
    )

    result = evaluate_manifest(manifest, tmp_path / "output")
    case = result["cases"][0]
    checks = {check["threshold"]: check for check in case["checks"]}

    assert not result["passed"]
    assert case["reviewed_motion_track"][
        "geometry_excluded_pre_seed_fail_closed_source_frames"
    ] == []
    assert not checks["reviewed_centroid_error_max"]["passed"]
    assert not checks["reviewed_motion_residual_p95_max"]["passed"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "misordered"])
def test_reviewed_track_rejects_incomplete_or_duplicate_records(tmp_path, mutation):
    manifest, track, payload = _reviewed_motion_fixture(tmp_path)
    if mutation == "missing":
        payload["records"].pop(2)
    elif mutation == "duplicate":
        payload["records"][2]["source_frame"] = 1
    else:
        payload["records"][1], payload["records"][2] = (
            payload["records"][2],
            payload["records"][1],
        )
    track.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["cases"][0]["reviewed_motion_track"]["sha256"] = _sha256(
        track
    )
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    with pytest.raises(
        OfflineBenchmarkError, match="record order/coverage|duplicate source frames"
    ):
        evaluate_manifest(manifest, tmp_path / "output")


def test_reviewed_track_rejects_unreviewed_candidate_proposal(tmp_path):
    manifest, track, payload = _reviewed_motion_fixture(tmp_path)
    payload["records"][2]["review_status"] = "needs_human_review"
    track.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["cases"][0]["reviewed_motion_track"]["sha256"] = _sha256(
        track
    )
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    with pytest.raises(OfflineBenchmarkError, match="is not human-confirmed"):
        evaluate_manifest(manifest, tmp_path / "output")


def test_reviewed_track_rejects_wrong_exact20hz_sampling_contract(tmp_path):
    manifest, track, payload = _reviewed_motion_fixture(tmp_path)
    payload["sampled_source_indices"][2] = 99
    track.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["cases"][0]["reviewed_motion_track"]["sha256"] = _sha256(
        track
    )
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    with pytest.raises(OfflineBenchmarkError, match="exact benchmark sampling"):
        evaluate_manifest(manifest, tmp_path / "output")


def test_reviewed_track_rejects_hash_mismatch(tmp_path):
    manifest, _track, _payload = _reviewed_motion_fixture(tmp_path)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["cases"][0]["reviewed_motion_track"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    with pytest.raises(OfflineBenchmarkError, match="SHA-256 mismatch"):
        evaluate_manifest(manifest, tmp_path / "output")


def test_legacy_hsv_jump_gate_is_unchanged_without_reviewed_track(tmp_path):
    manifest, _track, _payload = _reviewed_motion_fixture(
        tmp_path, include_track=False
    )
    result = evaluate_manifest(manifest, tmp_path / "output")
    case = result["cases"][0]
    checks = {check["threshold"]: check for check in case["checks"]}

    assert not result["passed"]
    assert not checks["excess_jump_p95_max"]["passed"]
    assert checks["excess_jump_p95_max"]["gating"]
    assert case["reviewed_motion_track"] is None
    assert case["proxy_motion_role"] == (
        "legacy_gating_without_complete_reviewed_motion_track"
    )
