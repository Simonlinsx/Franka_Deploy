import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from dynamic_pcd.apps.record_rgbd_case import (
    _frame_record,
    _write_integrity_manifest,
)
from dynamic_pcd.apps.replay_rgbd_case import replay
from dynamic_pcd.apps.replay_rgbd_case import _configure_object_mask_mode
from dynamic_pcd.config import load_config
from dynamic_pcd.types import CameraIntrinsics, RGBDFrame


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_frame(index: int, *, initialization: bool = False) -> RGBDFrame:
    height, width = 120, 160
    color = np.full((height, width, 3), 35, dtype=np.uint8)
    depth = np.full((height, width), 1050, dtype=np.uint16)
    center = (80 + (0 if initialization else index), 60)
    cv2.circle(color, center, 18, (20, 30, 230), -1, cv2.LINE_8)
    cv2.circle(depth, center, 18, 900, -1, cv2.LINE_8)
    sequence = 1 if initialization else index + 2
    return RGBDFrame(
        color_bgr=color,
        depth_raw=depth,
        depth_scale=0.001,
        intrinsics=CameraIntrinsics(
            width=width,
            height=height,
            fx=150.0,
            fy=150.0,
            ppx=80.0,
            ppy=60.0,
            model="none",
            distortion=(0.0,) * 5,
        ),
        timestamp=1000.0 + sequence / 30.0,
        frame_id=sequence,
        retrieved_at_s=1000.01 + sequence / 30.0,
        timestamp_domain="timestamp_domain.global_time",
        depth_timestamp_s=1000.001 + sequence / 30.0,
        color_depth_timestamp_skew_s=0.001,
        color_depth_epoch_timestamp_skew_s=0.001,
        sensor_frame_number=100 + sequence,
        depth_sensor_frame_number=200 + sequence,
        retrieved_monotonic_s=500.0 + sequence / 30.0,
        host_clock_pair_span_s=1.0e-6,
        capture_diagnostic={"outcome": "accepted"},
    )


def _make_case(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "config.yaml"
    config = {
        "camera": {
            "serial": None,
            "width": 160,
            "height": 120,
            "fps": 30,
            "z_min": 0.25,
            "z_max": 1.20,
            "runtime_frame_timeout_ms": 100,
        },
        "tracker": {
            "mode": "adaptive_color_depth",
            "init_method": "grabcut",
            "interactive_roi_padding_px": 16,
            "min_area": 40,
        },
        "sam2": {"enabled": False},
        "online_sam2": {"enabled": False},
        "extrinsics": {
            "calibration_file": None,
            "require_calibration": False,
            "strict_camera_serial": False,
            "calibrated": False,
            "T_base_camera": np.eye(4).tolist(),
        },
    }
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    case = tmp_path / "case"
    (case / "initialization").mkdir(parents=True)
    (case / "color").mkdir()
    (case / "depth").mkdir()
    initialization = _synthetic_frame(-1, initialization=True)
    assert cv2.imwrite(
        str(case / "initialization" / "color.png"),
        initialization.color_bgr,
    )
    assert cv2.imwrite(
        str(case / "initialization" / "depth.png"),
        initialization.depth_raw,
    )
    initialization_record = _frame_record(-1, initialization)
    initialization_record.update(
        {
            "role": "tracker_initialization",
            "color_path": "initialization/color.png",
            "depth_path": "initialization/depth.png",
        }
    )
    records = []
    for index in range(4):
        frame = _synthetic_frame(index)
        assert cv2.imwrite(
            str(case / "color" / f"{index:06d}.png"), frame.color_bgr
        )
        assert cv2.imwrite(
            str(case / "depth" / f"{index:06d}.png"), frame.depth_raw
        )
        records.append(_frame_record(index, frame))
    (case / "frames.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (case / "capture_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    manifest = {
        "schema": "dynamic_object_pcd_rgbd_case_v1",
        "complete": True,
        "case": "static_ball",
        "frame_count": len(records),
        "initialization_frame": initialization_record,
        "integrity_manifest": "MANIFEST.sha256",
        "image_width": 160,
        "image_height": 120,
        "nominal_fps": 30,
        "depth_scale_m_per_unit": 0.001,
        "camera_serial": "",
        "camera_intrinsics": initialization.intrinsics.to_dict(),
        "camera_K": initialization.intrinsics.as_matrix().tolist(),
        "T_base_camera": np.eye(4).tolist(),
        "provider_initialization_roi_xyxy": [46, 26, 114, 94],
        "source_config_sha256": _sha256(config_path),
    }
    (case / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_integrity_manifest(case)
    return case, config_path


def test_offline_replay_evaluates_every_recorded_video_frame(
    tmp_path: Path,
) -> None:
    case, config = _make_case(tmp_path)
    output = tmp_path / "replay"
    replay(
        argparse.Namespace(
            case_dir=case,
            config=config,
            output=output,
            disable_online_sam2=True,
            as_fast_as_possible=True,
            rate=1.0,
            max_frames=None,
            vis=False,
        )
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["frames"] == 4
    assert summary["evaluated_all_recorded_video_frames"] is True
    assert summary["config_matches_capture"] is True
    assert summary["object_mask_mode"] == "guarded"
    assert summary["requested_object_mask_mode"] == "guarded"
    assert summary["effective_object_mask_mode"] == "adaptive_only"
    assert summary["provider_mask_publication_mode"] == "adaptive_fusion"
    assert summary["recovery_publication_mode"] == "unified_three_evidence"
    with np.load(output / "results.npz", allow_pickle=False) as result:
        assert result["frame_id"].tolist() == [2, 3, 4, 5]
        assert result["policy128_xyzrgb_base"].shape == (4, 128, 6)
        assert result["policy128_valid"].shape == (4, 128)
        assert np.all(result["mask_valid"])
        assert np.all(result["provider_published_mask_valid"])
        assert set(result["provider_published_mask_source"].tolist()) == {
            "adaptive"
        }
        assert set(result["provider_online_sam2_status"].tolist()) == {
            "no_exact_result"
        }
        assert set(
            result["projector_effective_policy_mask_provenance"].tolist()
        ) == {"current_frame_projector_input_from_provider_mask"}
        np.testing.assert_array_equal(
            result["projector_effective_policy_mask_source_frame_id"],
            result["frame_id"],
        )
        assert result["object_mask_mode"].item() == "guarded"
        assert result["requested_object_mask_mode"].item() == "guarded"
        assert result["effective_object_mask_mode"].item() == "adaptive_only"
        assert (
            result["recovery_publication_mode"].item()
            == "unified_three_evidence"
        )


def test_programmatic_replay_rejects_unknown_object_mask_mode(
    tmp_path: Path,
) -> None:
    case, config = _make_case(tmp_path)
    with pytest.raises(ValueError, match="object_mask_mode must be one of"):
        replay(
            argparse.Namespace(
                case_dir=case,
                config=config,
                output=tmp_path / "invalid-mode",
                disable_online_sam2=False,
                object_mask_mode="guardde",
                as_fast_as_possible=True,
                rate=1.0,
                max_frames=None,
                vis=False,
            )
        )


def test_guarded_v2_mode_maps_to_production_semantic_primary() -> None:
    cfg = load_config()
    effective_mode, recovery_mode = _configure_object_mask_mode(
        cfg,
        requested_mode="guarded_v2",
        disable_online_sam2=False,
    )

    assert effective_mode == "guarded_v2"
    assert recovery_mode == "unified_three_evidence"
    assert cfg["online_sam2"]["enabled"] is True
    assert cfg["online_sam2"]["require_for_bbox_init"] is True
    assert cfg["online_sam2"]["mask_publication_mode"] == (
        "guarded_sam2_primary"
    )
    assert cfg["tracker"]["recovery_publication_mode"] == (
        "unified_three_evidence"
    )


def test_guarded_v2_no_sam2_ab_is_explicitly_adaptive_only() -> None:
    cfg = load_config()
    effective_mode, recovery_mode = _configure_object_mask_mode(
        cfg,
        requested_mode="guarded",
        disable_online_sam2=True,
    )

    assert effective_mode == "adaptive_only"
    assert recovery_mode == "unified_three_evidence"
    assert cfg["online_sam2"]["enabled"] is False
    assert cfg["online_sam2"]["mask_publication_mode"] == "adaptive_fusion"
