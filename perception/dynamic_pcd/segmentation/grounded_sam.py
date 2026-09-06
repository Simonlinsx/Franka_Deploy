"""Generic Grounded-SAM backend.

The implementation lives in ``grounded_sam2`` for compatibility with an early
prototype.  New code should import from this module; both native GroundingDINO
+ SAM1 and optional Transformers + SAM2 combinations are supported.
"""

from dynamic_pcd.segmentation.grounded_sam2 import (  # noqa: F401
    GroundedSAM2Backend,
    GroundedSAMBackend,
    NativeGroundingDINODetector,
    PromptModelDependencyError,
    SAM1BoxMaskPredictor,
    SAM2BoxMaskPredictor,
    TextDetection,
    TransformersGroundingDINODetector,
    bbox_iou,
    non_maximum_suppression,
    rank_detections,
)

__all__ = [
    "GroundedSAMBackend",
    "GroundedSAM2Backend",
    "NativeGroundingDINODetector",
    "TransformersGroundingDINODetector",
    "SAM1BoxMaskPredictor",
    "SAM2BoxMaskPredictor",
    "TextDetection",
    "PromptModelDependencyError",
    "bbox_iou",
    "non_maximum_suppression",
    "rank_detections",
]
