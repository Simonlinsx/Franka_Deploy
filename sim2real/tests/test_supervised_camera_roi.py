"""Headless fixed-ROI contract for the supervised V94 camera path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sim2real.deployment.bundle import DeployBundle
from sim2real.runtime.supervised_v94_runtime import (
    VERIFIED_OBJECT_ROI_EVIDENCE,
    VERIFIED_OBJECT_ROI_EVIDENCE_SHA256,
    VERIFIED_OBJECT_ROI_XYWH,
    SupervisedV94RuntimeError,
    _load_verified_object_roi,
    _resolve_object_roi,
)
from sim2real.contracts.v94 import V94Contract
from sim2real.runtime.v94_live_observation_owner import LiveD435ProviderFactory


BUNDLE = Path(__file__).resolve().parents[2] / "data/test_fixtures/sim2real/deploy.zip"
PRIMARY_CHECKPOINT_SHA256 = (
    DeployBundle(BUNDLE).verify().primary_checkpoint_sha256
)


def test_pinned_fixed_roi_matches_same_camera_provider_soak() -> None:
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))

    if not VERIFIED_OBJECT_ROI_EVIDENCE.is_file():
        with pytest.raises(
            SupervisedV94RuntimeError,
            match="verified fixed-ROI evidence is missing",
        ):
            _load_verified_object_roi(
                bundle_path=BUNDLE,
                contract=contract,
                selected_checkpoint_sha256=PRIMARY_CHECKPOINT_SHA256,
            )
        return

    evidence = _load_verified_object_roi(
        bundle_path=BUNDLE,
        contract=contract,
        selected_checkpoint_sha256=PRIMARY_CHECKPOINT_SHA256,
    )

    assert evidence.xywh == VERIFIED_OBJECT_ROI_XYWH == (460, 235, 120, 100)
    assert evidence.evidence_sha256 == VERIFIED_OBJECT_ROI_EVIDENCE_SHA256
    assert evidence.camera_serial == contract.camera_serial
    assert evidence.calibration_id == contract.calibration_id
    assert evidence.valid_publications == 1801
    assert evidence.invalid_publications == 0
    assert evidence.acquisition_elapsed_s >= 60.0
    assert evidence.position_source == "pinned_fixed_roi"
    assert evidence.preflight_valid_frames == 0


def test_camera_preflighted_current_roi_replaces_only_run_position(
    monkeypatch,
) -> None:
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    monkeypatch.setattr(
        "sim2real.runtime.supervised_v94_runtime._load_verified_object_roi",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("current ROI must not load historical fixed-ROI evidence")
        ),
    )
    request = SimpleNamespace(
        object_roi_xywh=(400, 210, 130, 120),
        object_roi_source="operator_interactive_camera_preflight",
        object_roi_preflight_sha256="a" * 64,
        object_roi_preflight_valid_frames=3,
        object_roi_preflight_invalid_frames=0,
        object_roi_preflight_min_policy_points=800,
        object_roi_preflight_depth_p50_m=0.91,
        object_roi_preflight_mask_bbox_xyxy=(430, 235, 500, 305),
        object_roi_preflight_mask_area_px=1200,
        object_roi_preflight_mask_source="grabcut",
        object_roi_preflight_bundle_sha256="b" * 64,
        object_roi_preflight_pcd_config_sha256="c" * 64,
        object_roi_preflight_checkpoint_sha256="d" * 64,
    )

    resolved = _resolve_object_roi(
        request=request,
        bundle_path=BUNDLE,
        contract=contract,
        checkpoint_sha256="d" * 64,
        bundle_sha256="b" * 64,
        pcd_config_sha256="c" * 64,
    )

    assert resolved.xywh == (400, 210, 130, 120)
    assert resolved.position_source == "operator_interactive_camera_preflight"
    assert resolved.preflight_sha256 == "a" * 64
    assert resolved.preflight_valid_frames == 3
    assert resolved.preflight_min_policy_points == 800
    assert resolved.preflight_depth_p50_m == pytest.approx(0.91)
    assert resolved.preflight_mask_bbox_xyxy == (430, 235, 500, 305)
    assert resolved.preflight_mask_area_px == 1200
    assert resolved.preflight_mask_source == "grabcut"
    assert resolved.evidence_path is None
    assert resolved.evidence_sha256 == ""
    assert resolved.camera_serial == contract.camera_serial
    assert resolved.calibration_id == contract.calibration_id
    assert resolved.valid_publications == 3
    assert resolved.invalid_publications == 0


def test_camera_preflighted_current_roi_accepts_sam2_box_mask() -> None:
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    request = SimpleNamespace(
        object_roi_xywh=(353, 247, 82, 82),
        object_roi_source="operator_interactive_camera_preflight",
        object_roi_preflight_sha256="a" * 64,
        object_roi_preflight_valid_frames=3,
        object_roi_preflight_invalid_frames=7,
        object_roi_preflight_min_policy_points=1496,
        object_roi_preflight_depth_p50_m=0.929,
        object_roi_preflight_mask_bbox_xyxy=(367, 256, 424, 329),
        object_roi_preflight_mask_area_px=2824,
        object_roi_preflight_mask_source="online_sam2_box",
        object_roi_preflight_bundle_sha256="b" * 64,
        object_roi_preflight_pcd_config_sha256="c" * 64,
        object_roi_preflight_checkpoint_sha256="d" * 64,
    )

    resolved = _resolve_object_roi(
        request=request,
        bundle_path=BUNDLE,
        contract=contract,
        checkpoint_sha256="d" * 64,
        bundle_sha256="b" * 64,
        pcd_config_sha256="c" * 64,
    )

    assert resolved.xywh == (353, 247, 82, 82)
    assert resolved.preflight_invalid_frames == 7
    assert resolved.preflight_min_policy_points == 1496
    assert resolved.preflight_mask_source == "online_sam2_box"


def test_unvalidated_current_roi_is_rejected(monkeypatch) -> None:
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    monkeypatch.setattr(
        "sim2real.runtime.supervised_v94_runtime._load_verified_object_roi",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("current ROI must not load historical fixed-ROI evidence")
        ),
    )
    request = SimpleNamespace(
        object_roi_xywh=(400, 210, 130, 120),
        object_roi_source="operator_interactive_camera_preflight",
        object_roi_preflight_sha256="",
        object_roi_preflight_valid_frames=0,
        object_roi_preflight_invalid_frames=0,
        object_roi_preflight_min_policy_points=0,
        object_roi_preflight_depth_p50_m=None,
        object_roi_preflight_mask_bbox_xyxy=(),
        object_roi_preflight_mask_area_px=0,
        object_roi_preflight_mask_source="",
        object_roi_preflight_bundle_sha256="",
        object_roi_preflight_pcd_config_sha256="",
    )

    with pytest.raises(
        SupervisedV94RuntimeError,
        match="preflight",
    ):
        _resolve_object_roi(
            request=request,
            bundle_path=BUNDLE,
            contract=contract,
            checkpoint_sha256="d" * 64,
            bundle_sha256="b" * 64,
            pcd_config_sha256="c" * 64,
        )


def test_external_checkpoint_cannot_reuse_primary_fixed_roi_evidence() -> None:
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    if not VERIFIED_OBJECT_ROI_EVIDENCE.is_file():
        pytest.skip("fixed ROI evidence is not installed")

    with pytest.raises(
        SupervisedV94RuntimeError,
        match="external --checkpoint requires",
    ):
        _load_verified_object_roi(
            bundle_path=BUNDLE,
            contract=contract,
            selected_checkpoint_sha256="0" * 64,
        )


def test_live_factory_rejects_interactive_roi_without_touching_provider(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    import sim2real.runtime.v94_live_observation_owner as module

    monkeypatch.setattr(
        module,
        "_initialize_provider",
        lambda *_args, **_kwargs: calls.append("provider") or (object(), object()),
    )

    with pytest.raises(ValueError, match="interactive selectROI is forbidden"):
        LiveD435ProviderFactory(
            config_path=tmp_path / "unused.yaml",
            roi_xywh=None,
            disable_online_sam2=True,
        )

    assert calls == []


def test_missing_or_changed_roi_evidence_fails_before_hardware_preflight(
    monkeypatch, tmp_path: Path
) -> None:
    import sim2real.runtime.supervised_v94_runtime as module

    hardware_calls = []
    monkeypatch.setattr(
        module,
        "_validate_request",
        lambda _request: (
            tmp_path / "unused-audit.json",
            {},
            SimpleNamespace(),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        module,
        "_load_verified_object_roi",
        lambda **_kwargs: (_ for _ in ()).throw(
            module.SupervisedV94RuntimeError("ROI evidence changed")
        ),
    )
    monkeypatch.setattr(
        module,
        "_fresh_franka_preflight",
        lambda **_kwargs: hardware_calls.append("franka"),
    )
    monkeypatch.setattr(
        module,
        "_fresh_rh56_preflight",
        lambda **_kwargs: hardware_calls.append("rh56"),
    )
    bundle = tmp_path / "unused.zip"
    profile = tmp_path / "unused-profile.json"
    pcd_config = tmp_path / "unused-pcd.yaml"
    bundle.write_bytes(b"unused")
    profile.write_text("{}", encoding="utf-8")
    pcd_config.write_text("{}\n", encoding="utf-8")
    request = SimpleNamespace(
        bundle=bundle,
        profile=profile,
        pcd_config=pcd_config,
    )
    monkeypatch.setattr(
        module,
        "prepare_supervised_v94_artifacts",
        lambda _request: SimpleNamespace(
            output=tmp_path / "unused-audit.json",
            profile={},
            contract=SimpleNamespace(),
            envelope=SimpleNamespace(),
            native_build=SimpleNamespace(),
            bundle_sha256="a" * 64,
            profile_file_sha256="b" * 64,
            pcd_config_sha256="c" * 64,
            checkpoint_source="bundle_primary",
            checkpoint_path=None,
            checkpoint_sha256="d" * 64,
            pinned_checkpoint_bytes=b"checkpoint",
        ),
    )

    with pytest.raises(module.SupervisedV94RuntimeError, match="ROI evidence changed"):
        module.run_supervised_v94(request)

    assert hardware_calls == []


def test_live_factory_is_inert_then_passes_only_numeric_roi_to_initializer(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    provider = object()
    import sim2real.runtime.v94_live_observation_owner as module

    def initialize(config_path, roi, *, disable_online_sam2, object_mask_mode):
        calls.append((config_path, roi, disable_online_sam2, object_mask_mode))
        return provider, object()

    monkeypatch.setattr(module, "_initialize_provider", initialize)
    config = tmp_path / "unused.yaml"
    factory = LiveD435ProviderFactory(
        config_path=config,
        roi_xywh=VERIFIED_OBJECT_ROI_XYWH,
        disable_online_sam2=True,
    )

    assert calls == []
    assert factory() is provider
    assert calls == [(config, VERIFIED_OBJECT_ROI_XYWH, True, "guarded")]


def test_live_factory_binds_formal_runtime_frame_timeout(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    provider = object()
    import sim2real.runtime.v94_live_observation_owner as module

    def initialize(
        config_path,
        roi,
        *,
        disable_online_sam2,
        object_mask_mode,
        required_runtime_frame_timeout_ms,
    ):
        calls.append(
            (
                config_path,
                roi,
                disable_online_sam2,
                object_mask_mode,
                required_runtime_frame_timeout_ms,
            )
        )
        return provider, object()

    monkeypatch.setattr(module, "_initialize_provider", initialize)
    config = tmp_path / "unused.yaml"
    factory = LiveD435ProviderFactory(
        config_path=config,
        roi_xywh=VERIFIED_OBJECT_ROI_XYWH,
        disable_online_sam2=True,
        required_runtime_frame_timeout_ms=100,
    )

    assert calls == []
    assert factory() is provider
    assert calls == [
        (config, VERIFIED_OBJECT_ROI_XYWH, True, "guarded", 100)
    ]


def test_live_factory_preserves_all_versioned_mask_ab_modes(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    import sim2real.runtime.v94_live_observation_owner as module

    def initialize(config_path, roi, **options):
        calls.append((config_path, roi, options))
        return object(), object()

    monkeypatch.setattr(module, "_initialize_provider", initialize)
    modes = ("guarded", "guarded_v2", "guarded_v1", "legacy")
    for mode in modes:
        factory = LiveD435ProviderFactory(
            config_path=tmp_path / "provider.yaml",
            roi_xywh=VERIFIED_OBJECT_ROI_XYWH,
            object_mask_mode=mode,
        )
        factory()
    assert [call[2]["object_mask_mode"] for call in calls] == list(modes)
    with pytest.raises(ValueError, match="guarded.*legacy"):
        LiveD435ProviderFactory(
            config_path=tmp_path / "provider.yaml",
            roi_xywh=VERIFIED_OBJECT_ROI_XYWH,
            object_mask_mode="invalid",
        )


@pytest.mark.parametrize(
    "roi",
    ((1, 2, 0, 4), (-1, 2, 3, 4), (1, 2, 3.5, 4), (True, 2, 3, 4)),
)
def test_live_factory_rejects_invalid_numeric_roi_inertly(
    roi, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="live D435 ROI"):
        LiveD435ProviderFactory(
            config_path=tmp_path / "unused.yaml",
            roi_xywh=roi,
            disable_online_sam2=True,
        )
