"""Offline, hardware-free evaluation helpers for object-mask pipelines."""

from .guarded_v2_offline import evaluate_manifest

__all__ = ["evaluate_manifest"]
