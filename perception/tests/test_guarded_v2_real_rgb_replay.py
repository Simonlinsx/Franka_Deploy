import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dynamic_pcd.evaluation import guarded_v2_real_rgb_replay as replay_module
from dynamic_pcd.evaluation.guarded_v2_real_rgb_replay import (
    NeutralDepthVideoCamera,
    RealRGBReplayError,
    _seed_mask,
    replay_manifest,
)
from dynamic_pcd.types import MaskResult


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 30.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_neutral_depth_camera_preserves_source_ids_and_source_timestamps():
    frames = [np.zeros((12, 20, 3), dtype=np.uint8) for _ in range(3)]
    camera = NeutralDepthVideoCamera(
        frames,
        [10, 12, 13],
        next_position=1,
        rate_hz=20.0,
        source_rate_hz=30.0,
        depth_m=1.0,
        device_serial="offline",
        realtime=False,
    )
    camera.start()
    first = camera.get_frame()
    second = camera.get_frame()

    assert first.frame_id == 12
    assert second.frame_id == 13
    assert abs(second.timestamp - first.timestamp - (1.0 / 30.0)) < 1e-9
    assert np.all(first.depth_raw == 1000)
    assert np.allclose(first.depth_m, 1.0)
    assert first.timestamp_domain == "offline_source_video"


def test_single_frame_proxy_seed_does_not_inherit_an_earlier_wrong_component(tmp_path):
    sampled = []
    for center in ((80, 10), (80, 10), (18, 34)):
        frame = np.full((60, 100, 3), 15, dtype=np.uint8)
        cv2.circle(frame, center, 8, (20, 220, 20), -1)
        sampled.append(frame)
    # Make the old top-right green distractor remain present when the target
    # enters at the final frame.
    cv2.circle(sampled[2], (80, 10), 8, (20, 220, 20), -1)
    case = {
        "name": "fast_entry",
        "seed_source_frame": 3,
        "seed": {"kind": "target_proxy_single_frame"},
        "target_proxy": {
            "hsv_ranges": [[[35, 80, 40], [95, 255, 255]]],
            "motion_threshold_bgr": 15,
            "initial_seed_xy": [18, 34],
            "max_center_jump_px": 30,
            "min_area_px": 40,
            "open_kernel": 1,
            "close_kernel": 1,
        },
    }
    position, mask, source = _seed_mask(
        case,
        base=tmp_path,
        source_frames=[sampled[0], sampled[0], sampled[1], sampled[2]],
        sampled_frames=sampled,
        sampled_source_indices=[0, 2, 3],
    )

    ys, xs = np.nonzero(mask)
    assert position == 2
    assert float(xs.mean()) < 30.0
    assert float(ys.mean()) > 25.0
    assert source.startswith("single_frame_conservative")


def test_mask_video_seed_forbids_future_or_past_frame_lookahead(tmp_path):
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    cv2.circle(frame, (12, 12), 5, (255, 255, 255), -1)
    seed_video = tmp_path / "seed.mp4"
    _write_video(seed_video, [frame, frame, frame])
    case = {
        "name": "lookahead",
        "seed_source_frame": 0,
        "seed": {
            "kind": "mask_video_frame",
            "path": str(seed_video),
            "frame_index": 1,
        },
    }
    with pytest.raises(RealRGBReplayError, match="lookahead is forbidden"):
        _seed_mask(
            case,
            base=tmp_path,
            source_frames=[frame, frame, frame],
            sampled_frames=[frame, frame, frame],
            sampled_source_indices=[0, 1, 2],
        )


def test_replay_manifest_exports_exact_candidate_contract_without_hardware(
    tmp_path, monkeypatch
):
    frames = []
    binary_frames = []
    for index in range(7):
        frame = np.full((48, 64, 3), 20, dtype=np.uint8)
        cv2.circle(frame, (20 + index, 28), 7, (20, 220, 20), -1)
        frames.append(frame)
        binary = np.zeros_like(frame)
        cv2.circle(binary, (20 + index, 28), 7, (255, 255, 255), -1)
        binary_frames.append(binary)
    video = tmp_path / "source.mp4"
    seed_video = tmp_path / "seed.mp4"
    _write_video(video, frames)
    _write_video(seed_video, binary_frames)
    manifest = {
        "schema": "guarded_v2_real_rgb_replay_v1",
        "cases": [
            {
                "name": "synthetic",
                "video": str(video),
                "evaluation_fps_hz": 20.0,
                "seed_source_frame": 0,
                "seed": {
                    "kind": "mask_video_frame",
                    "path": str(seed_video),
                    "frame_index": 0,
                    "threshold": 96,
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class FakeProvider:
        hardware_opened = False
        constructed_configs = []

        def __init__(self, cfg, *, camera):
            self.constructed_configs.append(cfg)
            self.camera = camera
            self.last_mask_result = None
            self.last_timings_ms = {}
            self.published_mask_source = "fake_tracked"
            self.online_sam2_status = "tracked_exact"

        def start(self):
            self.camera.start()

        def stop(self):
            self.camera.stop()

        def initialize_from_bbox(self, frame, bbox):
            x1, y1, x2, y2 = (int(value) for value in bbox)
            mask = np.zeros(frame.depth_raw.shape, dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            self.last_mask_result = MaskResult(
                mask=mask,
                bbox_xyxy=np.asarray(bbox, dtype=np.int32),
                score=1.0,
                valid=True,
                message="fake initialization",
            )
            return True

        def step_mask_only(self):
            frame = self.camera.get_frame()
            green = cv2.inRange(
                cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2HSV),
                (35, 50, 30),
                (95, 255, 255),
            )
            ys, xs = np.nonzero(green)
            bbox = np.asarray(
                [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                dtype=np.int32,
            )
            self.last_timings_ms = {"total": 2.0}
            return frame, MaskResult(
                mask=(green > 0).astype(np.uint8),
                bbox_xyxy=bbox,
                score=1.0,
                valid=True,
                message="fake tracked",
            )

    monkeypatch.setattr(replay_module, "ObjectPCDProvider", FakeProvider)
    output = tmp_path / "candidate"
    result = replay_manifest(
        manifest_path,
        output,
        config_path=(
            Path(__file__).resolve().parents[1] / "configs" / "d435_default.yaml"
        ),
        realtime=False,
    )

    # 30 -> 20 Hz over seven source frames selects 0,2,3,5,6.
    mask_files = sorted((output / "synthetic" / "masks").glob("*.png"))
    records = [
        json.loads(line)
        for line in (output / "synthetic" / "states.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(mask_files) == 5
    assert [record["source_frame_index"] for record in records] == [0, 2, 3, 5, 6]
    assert records[0]["latency_evaluable"] is False
    assert all(record["latency_evaluable"] is True for record in records[1:])
    assert all(
        record["appearance_probability_cache"]
        == {"build_count": 0, "query_count": 0, "hit_count": 0}
        for record in records
    )
    assert all(
        record["appearance_gate_cache"]
        == {
            "stats_build_count": 0,
            "stats_hit_count": 0,
            "stats_miss_count": 0,
            "stats_bypass_count": 0,
            "supported_build_count": 0,
            "supported_hit_count": 0,
            "supported_miss_count": 0,
            "supported_bypass_count": 0,
        }
        for record in records
    )
    assert all(record["mask_gate_breakdown_ms"] == {} for record in records)
    assert result["hardware_interfaces_opened"] is False
    assert result["effective_config_provenance"]["vos_optimized"] is True
    assert (
        FakeProvider.constructed_configs[0]["online_sam2"]["mask_publication_mode"]
        == "guarded_sam2_primary"
    )
    assert (
        result["cases"][0]["effective_mask_publication_mode"] == "guarded_sam2_primary"
    )
    assert result["cases"][0]["evidence_scope"] == "2d_mask_identity_state_timing_only"
    assert result["cases"][0]["appearance_probability_cache_final"] == {
        "build_count": 0,
        "query_count": 0,
        "hit_count": 0,
    }
    assert result["cases"][0]["appearance_gate_cache_final"] == {
        "stats_build_count": 0,
        "stats_hit_count": 0,
        "stats_miss_count": 0,
        "stats_bypass_count": 0,
        "supported_build_count": 0,
        "supported_hit_count": 0,
        "supported_miss_count": 0,
        "supported_bypass_count": 0,
    }
    assert result["cases"][0]["validity_continuity_after_seed"] == {
        "longest_consecutive_valid_frames": 5,
        "longest_consecutive_invalid_frames": 0,
        "validity_transitions": 0,
    }


def test_latency_evaluable_starts_after_bootstrap_commission_frame():
    class Provider:
        guarded_v2_bootstrap_state = {"commissioned_frame_id": None}

    provider = Provider()
    assert (
        replay_module._latency_evaluable_after_bootstrap(provider, frame_id=40)
        is False
    )
    provider.guarded_v2_bootstrap_state = {"commissioned_frame_id": 45}
    assert (
        replay_module._latency_evaluable_after_bootstrap(provider, frame_id=45)
        is False
    )
    assert (
        replay_module._latency_evaluable_after_bootstrap(provider, frame_id=47)
        is True
    )


def test_latency_evaluable_keeps_legacy_provider_behavior():
    class LegacyProvider:
        pass

    assert (
        replay_module._latency_evaluable_after_bootstrap(
            LegacyProvider(), frame_id=1
        )
        is True
    )


def test_diagnostic_replay_is_explicit_semantic_sam2_not_production(
    tmp_path, monkeypatch
):
    frame = np.full((32, 48, 3), 20, dtype=np.uint8)
    cv2.circle(frame, (20, 18), 6, (20, 220, 20), -1)
    video = tmp_path / "source.mp4"
    seed_video = tmp_path / "seed.mp4"
    _write_video(video, [frame, frame, frame])
    binary = np.zeros_like(frame)
    cv2.circle(binary, (20, 18), 6, (255, 255, 255), -1)
    _write_video(seed_video, [binary, binary, binary])
    manifest = {
        "schema": "guarded_v2_real_rgb_replay_v1",
        "cases": [
            {
                "name": "diagnostic",
                "video": str(video),
                "evaluation_fps_hz": 20.0,
                "seed_source_frame": 0,
                "seed": {
                    "kind": "mask_video_frame",
                    "path": str(seed_video),
                    "frame_index": 0,
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class FakeProvider:
        constructed_configs = []

        def __init__(self, cfg, *, camera):
            self.constructed_configs.append(cfg)
            self.camera = camera
            self.last_mask_result = None
            self.last_timings_ms = {}
            self.published_mask_source = "semantic_sam2"
            self.online_sam2_status = "tracked_exact"

        def start(self):
            self.camera.start()

        def stop(self):
            self.camera.stop()

        def initialize_from_bbox(self, frame, bbox):
            mask = np.zeros(frame.depth_raw.shape, dtype=np.uint8)
            x1, y1, x2, y2 = (int(value) for value in bbox)
            mask[y1:y2, x1:x2] = 1
            self.last_mask_result = MaskResult(
                mask=mask,
                bbox_xyxy=np.asarray(bbox, dtype=np.int32),
                score=1.0,
                valid=True,
                message="initialized",
            )
            return True

        def step_mask_only(self):
            frame = self.camera.get_frame()
            mask = np.zeros(frame.depth_raw.shape, dtype=np.uint8)
            mask[12:24, 14:26] = 1
            return frame, MaskResult(
                mask=mask,
                bbox_xyxy=np.asarray([14, 12, 26, 24], dtype=np.int32),
                score=1.0,
                valid=True,
                message="tracked",
            )

    monkeypatch.setattr(replay_module, "ObjectPCDProvider", FakeProvider)
    result = replay_manifest(
        manifest_path,
        tmp_path / "candidate",
        config_path=(
            Path(__file__).resolve().parents[1] / "configs" / "d435_default.yaml"
        ),
        realtime=False,
        diagnostic_semantic_sam2_direct=True,
    )

    assert (
        FakeProvider.constructed_configs[0]["online_sam2"]["mask_publication_mode"]
        == "semantic_sam2"
    )
    assert result["diagnostic_only_not_production_acceptance"] is True
    assert result["cases"][0]["effective_mask_publication_mode"] == "semantic_sam2"
    assert (
        result["cases"][0]["effective_recovery_publication_mode"]
        == "semantic_sam2_direct"
    )
