from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import inspect

import numpy as np

import sim2real.observation.roi_selector as selector
from sim2real.deployment.bundle import DeployBundle
from sim2real.contracts.v94 import V94Contract


BUNDLE = Path(__file__).resolve().parents[2] / "data/test_fixtures/sim2real/deploy.zip"


def test_camera_only_selector_returns_numeric_roi_and_stops_provider(
    monkeypatch, tmp_path
):
    import sim2real.observation.live_preview as preview

    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    roi = (410, 215, 120, 110)
    stopped = []
    provider = SimpleNamespace(
        last_bbox_initialization_evidence=SimpleNamespace(
            source="grabcut",
            prompt_bbox_xyxy=np.asarray(
                [roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]],
                dtype=np.int32,
            ),
            mask_bbox_xyxy=np.asarray([430, 235, 500, 300], dtype=np.int32),
            mask_area_px=1200,
        ),
        extrinsics=SimpleNamespace(
            camera_serial=contract.camera_serial,
            calibration_id=contract.calibration_id,
            T_base_camera=contract.T_base_camera_optical.copy(),
        ),
        stop=lambda: stopped.append(True),
    )

    def initialize(_config, requested_roi, **kwargs):
        assert requested_roi is None
        assert kwargs == {"disable_online_sam2": False}
        return provider, object()

    monkeypatch.setattr(preview, "_initialize_provider", initialize)
    result = selector._select(
        bundle_path=BUNDLE,
        pcd_config_path=tmp_path / "unused.yaml",
    )

    assert result["roi_xywh"] == list(roi)
    assert result["camera_serial"] == contract.camera_serial
    assert result["calibration_id"] == contract.calibration_id
    assert result["mask_source"] == "grabcut"
    assert stopped == [True]


def test_selector_reports_python_failure_over_private_pipe(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        selector,
        "_select",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic GUI initialization failure")
        ),
    )
    read_fd, write_fd = os.pipe()
    try:
        code = selector.main(
            [
                "--bundle",
                str(tmp_path / "deploy.zip"),
                "--pcd-config",
                str(tmp_path / "pcd.yaml"),
                "--nonce",
                "a" * 32,
                "--result-fd",
                str(write_fd),
            ]
        )
        write_fd = -1
        payload = os.read(read_fd, selector.MAX_RESULT_BYTES)
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)

    record = json.loads(payload)
    assert code == 1
    assert record == {
        "protocol": selector.ROI_SELECTOR_PROTOCOL,
        "nonce": "a" * 32,
        "status": "error",
        "error": "RuntimeError: synthetic GUI initialization failure",
    }


def test_grounding_depth_admission_rejects_far_background_candidate():
    depth_raw = np.full((240, 424), 900, dtype=np.uint16)
    depth_raw[194:202, 117:122] = 6634
    frame = SimpleNamespace(depth_raw=depth_raw, depth_scale=0.001)

    accepted, reason = selector._grounding_candidate_depth_admission(
        frame,
        np.asarray([117, 194, 122, 202], dtype=np.int32),
        z_min_m=0.25,
        z_max_m=1.65,
    )

    assert accepted is False
    assert "measured_p50=6.634m" in reason
    assert "range [0.25,1.65]m" in reason


def test_grounding_depth_admission_accepts_current_target_depth_with_holes():
    depth_raw = np.zeros((240, 424), dtype=np.uint16)
    depth_raw[80:90, 100:112] = 920
    depth_raw[80:82, 100:112] = 0
    frame = SimpleNamespace(depth_raw=depth_raw, depth_scale=0.001)

    accepted, reason = selector._grounding_candidate_depth_admission(
        frame,
        np.asarray([100, 80, 112, 90], dtype=np.int32),
        z_min_m=0.25,
        z_max_m=1.65,
    )

    assert accepted is True
    assert "depth PASS" in reason
    assert "p50=0.920m" in reason


def test_grounding_depth_admission_rejects_clipped_bottom_candidate():
    depth_raw = np.full((240, 424), 1317, dtype=np.uint16)
    frame = SimpleNamespace(depth_raw=depth_raw, depth_scale=0.001)

    accepted, reason = selector._grounding_candidate_depth_admission(
        frame,
        np.asarray([91, 227, 112, 240], dtype=np.int32),
        z_min_m=0.25,
        z_max_m=1.65,
    )

    assert accepted is False
    assert "touches image boundary" in reason


def test_grounding_depth_admission_rejects_sparse_in_range_depth():
    depth_raw = np.full((240, 424), 5000, dtype=np.uint16)
    depth_raw[100:105, 100:110] = 1000
    frame = SimpleNamespace(depth_raw=depth_raw, depth_scale=0.001)

    accepted, reason = selector._grounding_candidate_depth_admission(
        frame,
        np.asarray([100, 100, 110, 110], dtype=np.int32),
        z_min_m=0.25,
        z_max_m=1.65,
        minimum_valid_depth_ratio=0.60,
    )

    assert accepted is False
    assert "in_range=50/100 required=60" in reason


def test_tabletop_grounding_workspace_rejects_static_lab_distractor():
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    depth_raw = np.zeros((480, 848), dtype=np.uint16)
    bbox = np.asarray([184, 119, 206, 142], dtype=np.int32)
    depth_raw[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 1470
    frame = SimpleNamespace(
        depth_raw=depth_raw,
        depth_scale=0.001,
        intrinsics=SimpleNamespace(
            fx=contract.camera_K[0, 0],
            fy=contract.camera_K[1, 1],
            ppx=contract.camera_K[0, 2],
            ppy=contract.camera_K[1, 2],
        ),
    )

    accepted, reason = selector._grounding_candidate_depth_admission(
        frame,
        bbox,
        z_min_m=0.25,
        z_max_m=2.0,
        T_base_camera=contract.T_base_camera_optical,
        workspace_center_base_m=[0.5896658057, -0.1356170101, 0.0522189354],
        workspace_max_horizontal_distance_m=0.40,
    )

    assert accepted is False
    assert "outside the task acquisition corridor" in reason
    assert "horizontal_distance=" in reason


def test_tabletop_grounding_workspace_accepts_real_held_cylinder_region():
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    depth_raw = np.zeros((480, 848), dtype=np.uint16)
    bbox = np.asarray([189, 178, 255, 236], dtype=np.int32)
    depth_raw[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 1139
    frame = SimpleNamespace(
        depth_raw=depth_raw,
        depth_scale=0.001,
        intrinsics=SimpleNamespace(
            fx=contract.camera_K[0, 0],
            fy=contract.camera_K[1, 1],
            ppx=contract.camera_K[0, 2],
            ppy=contract.camera_K[1, 2],
        ),
    )

    accepted, reason = selector._grounding_candidate_depth_admission(
        frame,
        bbox,
        z_min_m=0.25,
        z_max_m=2.0,
        T_base_camera=contract.T_base_camera_optical,
        workspace_center_base_m=[0.5896658057, -0.1356170101, 0.0522189354],
        workspace_max_horizontal_distance_m=0.40,
    )

    assert accepted is True
    assert "workspace_distance=" in reason


def test_grounding_preview_isolated_process_does_not_load_opencv_qt():
    owner_source = inspect.getsource(selector._GroundingSearchPreview)
    child_source = inspect.getsource(selector._grounding_preview_process)

    assert "cv2" not in owner_source
    assert "cv2" not in child_source
    assert 'mp.get_context("spawn")' in owner_source
    assert "tkinter" in child_source
