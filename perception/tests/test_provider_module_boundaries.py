"""Checks for the provider's hardware-inert supporting modules."""

from dynamic_pcd.provider.config import validated_runtime_frame_timeout_ms
from dynamic_pcd.provider.mask_geometry import (
    _binary_mask_centroid,
    _integer_translation_overlap_counts,
    _mask_overlap_registration_shift,
    _scale_mask_about_bbox_center,
)
from dynamic_pcd.provider.object_pcd_provider import (
    ObjectPCDProvider,
    _validated_runtime_frame_timeout_ms,
)


def test_provider_config_compatibility_import_is_identical() -> None:
    assert _validated_runtime_frame_timeout_ms is validated_runtime_frame_timeout_ms


def test_provider_geometry_methods_use_the_split_pure_functions() -> None:
    assert ObjectPCDProvider._binary_mask_centroid is _binary_mask_centroid
    assert (
        ObjectPCDProvider._integer_translation_overlap_counts
        is _integer_translation_overlap_counts
    )
    assert (
        ObjectPCDProvider._mask_overlap_registration_shift
        is _mask_overlap_registration_shift
    )
    assert (
        ObjectPCDProvider._scale_mask_about_bbox_center
        is _scale_mask_about_bbox_center
    )
