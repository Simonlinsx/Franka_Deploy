import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dynamic_pcd.evaluation.guarded_v2_offline import (
    OfflineBenchmarkError,
    _verify_production_suite_coverage_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "configs" / "guarded_v2_sparse_ground_truth.json"
REPLAY_MANIFEST = PROJECT_ROOT / "configs" / "guarded_v2_real_rgb_replay_cases.json"
BENCHMARK_MANIFEST = (
    PROJECT_ROOT / "configs" / "guarded_v2_benchmark_long_vos_baseline.json"
)
PRODUCTION_CONFIG = PROJECT_ROOT / "configs" / "d435_default.yaml"
ASSET_ROOT = PROJECT_ROOT / "testdata" / "guarded_v2_sparse_ground_truth"
REVIEW_STATUSES = {"human_confirmed", "independent_visual_review"}


def test_reviewed_sparse_ground_truth_assets_are_complete_and_content_addressed():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema"] == "reviewed_sparse_mask_manifest_v1"
    assert len(payload["cases"]) == 5

    referenced: set[Path] = set()
    total_records = 0
    for case in payload["cases"]:
        spec = case["sparse_ground_truth"]
        assert spec["schema"] == "reviewed_visible_object_masks_v1"
        assert len(spec["records"]) >= 5
        assert len(spec["records"]) >= spec["min_reviewed_frames"]
        assert len(spec["source_video_sha256"]) == 64
        assert spec["source_video"].endswith(".mp4")
        source_frames = [int(record["source_frame"]) for record in spec["records"]]
        assert source_frames == sorted(set(source_frames))
        for record in spec["records"]:
            path = (MANIFEST.parent / record["path"]).resolve()
            referenced.add(path)
            total_records += 1
            assert record["review_status"] in REVIEW_STATUSES
            assert record["label_method"].strip()
            assert record["review_notes"].strip()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            assert mask is not None
            assert mask.shape == (480, 848)
            assert set(int(value) for value in np.unique(mask)).issubset({0, 255})
            assert int(np.count_nonzero(mask)) > 0

    on_disk = set(ASSET_ROOT.glob("*/*.png"))
    assert total_records == 28
    assert referenced == {path.resolve() for path in on_disk}


def test_replay_source_and_seed_videos_match_pinned_bytes_and_metadata():
    replay = json.loads(REPLAY_MANIFEST.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK_MANIFEST.read_text(encoding="utf-8"))
    assert hashlib.sha256(REPLAY_MANIFEST.read_bytes()).hexdigest() == benchmark[
        "expected_replay_manifest_sha256"
    ]
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == benchmark[
        "expected_sparse_ground_truth_manifest_sha256"
    ]
    assert hashlib.sha256(PRODUCTION_CONFIG.read_bytes()).hexdigest() == replay[
        "production_acceptance_contract"
    ]["default_config_sha256"]
    assert [case["name"] for case in replay["cases"]] == [
        "fast_green_ball_entry",
        "rolling_green_ball",
        "rolling_red_cylinder",
        "static_green_ball_rh56",
        "rh56_heavy_occlusion",
    ]
    missing_assets = []
    for case in replay["cases"]:
        source = Path(case["video"])
        if not source.is_file():
            missing_assets.append(source)
        seed = case["seed"]
        if seed["kind"] == "mask_video_frame":
            seed_path = (REPLAY_MANIFEST.parent / seed["path"]).resolve()
            if not seed_path.is_file():
                missing_assets.append(seed_path)
    if missing_assets:
        pytest.skip(
            "optional external replay assets are not installed: "
            + ", ".join(str(path) for path in missing_assets)
        )
    for case in replay["cases"]:
        source = Path(case["video"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == case[
            "source_video_sha256"
        ]
        capture = cv2.VideoCapture(str(source))
        assert capture.isOpened()
        assert abs(capture.get(cv2.CAP_PROP_FPS) - case["source_video_fps_hz"]) < 1e-6
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == case[
            "source_video_frame_count"
        ]
        assert [
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ] == case["source_video_size_wh"]
        capture.release()
        seed = case["seed"]
        if seed["kind"] != "mask_video_frame":
            continue
        seed_path = (REPLAY_MANIFEST.parent / seed["path"]).resolve()
        assert seed["frame_index"] == case["seed_source_frame"]
        assert hashlib.sha256(seed_path.read_bytes()).hexdigest() == seed["sha256"]
        seed_capture = cv2.VideoCapture(str(seed_path))
        assert seed_capture.isOpened()
        assert abs(seed_capture.get(cv2.CAP_PROP_FPS) - seed["fps_hz"]) < 1e-6
        assert int(seed_capture.get(cv2.CAP_PROP_FRAME_COUNT)) == seed["frame_count"]
        assert [
            int(seed_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(seed_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ] == seed["size_wh"]
        seed_capture.release()


def test_fixed_temporal_review_contract_covers_preentry_and_reappearance():
    benchmark = json.loads(BENCHMARK_MANIFEST.read_text(encoding="utf-8"))
    cases = {case["name"]: case for case in benchmark["cases"]}
    assert cases["fast_green_ball_entry"]["reviewed_visible_source_intervals"] == [
        [38, 123],
    ]
    assert cases["fast_green_ball_entry"]["reviewed_absent_source_intervals"] == [
        [0, 37],
    ]
    assert cases["rolling_green_ball"]["proxy_absence_gating"] is False
    assert cases["rolling_green_ball"]["reviewed_visible_source_intervals"] == [
        [68, 183],
    ]
    assert cases["rolling_green_ball"]["reviewed_absent_source_intervals"] == [
        [60, 66],
    ]
    assert cases["rolling_red_cylinder"]["target_first_visible_source_frame"] == 90
    assert cases["rolling_red_cylinder"]["reviewed_visible_source_intervals"] == [
        [90, 95],
        [101, 171],
    ]
    assert cases["rolling_red_cylinder"]["reviewed_absent_source_intervals"] == [
        [96, 99],
        [173, 204],
    ]
    assert cases["static_green_ball_rh56"][
        "reviewed_visible_source_intervals"
    ] == [[75, 244], [246, 289], [293, 306], [311, 352]]
    assert cases["static_green_ball_rh56"][
        "reviewed_absent_source_intervals"
    ] == [[245, 245], [290, 292], [307, 310], [353, 419]]
    assert cases["static_green_ball_rh56"][
        "reviewed_reappearance_source_frames"
    ] == [246, 293, 311]
    suite = benchmark["production_suite_coverage_contract"]
    assert suite["complete_occlusion_events"] == [
        {
            "case": "static_green_ball_rh56",
            "source_interval": [245, 245],
            "reappearance_source_frame": 246,
        },
        {
            "case": "static_green_ball_rh56",
            "source_interval": [290, 292],
            "reappearance_source_frame": 293,
        },
        {
            "case": "static_green_ball_rh56",
            "source_interval": [307, 310],
            "reappearance_source_frame": 311,
        },
    ]
    assert suite["heavy_partial_occlusion_case_name"] == "rh56_heavy_occlusion"
    assert suite["heavy_partial_occlusion_reviewed_source_frames"] == [
        91,
        121,
        211,
        271,
    ]
    for case in benchmark["cases"]:
        thresholds = case["thresholds"]
        assert case["evaluation_fps_hz"] == 20.0
        assert thresholds["latency_max_ms_max"] == 100.0
        assert thresholds["latency_over_50ms_longest_run_max"] == 1
    assert cases["rolling_green_ball"]["thresholds"][
        "reviewed_absent_false_positive_max"
    ] == 0.05


def test_production_suite_coverage_contract_is_grounded_in_reviewed_evidence():
    benchmark = json.loads(BENCHMARK_MANIFEST.read_text(encoding="utf-8"))
    sparse = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sparse_by_case = {
        case["name"]: case["sparse_ground_truth"] for case in sparse["cases"]
    }

    _verify_production_suite_coverage_contract(
        manifest=benchmark,
        cases=benchmark["cases"],
        sparse_by_case=sparse_by_case,
    )

    mutated = json.loads(json.dumps(benchmark))
    mutated["production_suite_coverage_contract"]["complete_occlusion_events"][0][
        "case"
    ] = "rh56_heavy_occlusion"
    with pytest.raises(OfflineBenchmarkError, match="complete-occlusion"):
        _verify_production_suite_coverage_contract(
            manifest=mutated,
            cases=mutated["cases"],
            sparse_by_case=sparse_by_case,
        )

    mutated = json.loads(json.dumps(benchmark))
    del {
        case["name"]: case for case in mutated["cases"]
    }["rolling_green_ball"]["thresholds"]["reviewed_absent_false_positive_max"]
    with pytest.raises(OfflineBenchmarkError, match="pre-entry reviewed absence"):
        _verify_production_suite_coverage_contract(
            manifest=mutated,
            cases=mutated["cases"],
            sparse_by_case=sparse_by_case,
        )
