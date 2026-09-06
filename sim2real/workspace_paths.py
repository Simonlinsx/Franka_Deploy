"""Canonical workspace paths shared by sim-to-real tools.

Only repository layout belongs here. Runtime policy, device access, and file
creation remain with their owning modules. Keeping these constants in one
place prevents command modules and tests from reconstructing asset roots with
different ``Path.parents`` expressions.
"""

from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = WORKSPACE_ROOT / "data"
CHECKPOINT_ROOT = DATA_ROOT / "checkpoints"
PERCEPTION_CORPUS_ROOT = DATA_ROOT / "perception_corpus"
RUN_OUTPUT_ROOT = DATA_ROOT / "runs"
TEST_FIXTURE_ROOT = DATA_ROOT / "test_fixtures"
DEFAULT_V94_DEPLOY_BUNDLE = TEST_FIXTURE_ROOT / "sim2real" / "deploy.zip"


__all__ = [
    "CHECKPOINT_ROOT",
    "DATA_ROOT",
    "DEFAULT_V94_DEPLOY_BUNDLE",
    "PERCEPTION_CORPUS_ROOT",
    "RUN_OUTPUT_ROOT",
    "TEST_FIXTURE_ROOT",
    "WORKSPACE_ROOT",
]
