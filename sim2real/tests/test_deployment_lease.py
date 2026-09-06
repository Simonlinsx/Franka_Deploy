from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim2real.deployment.lease import (
    DeploymentLeaseError,
    acquire_v94_deployment_lease,
)


FINGERPRINT = "a" * 64


def _acquire(tmp_path: Path, run_id: str):
    return acquire_v94_deployment_lease(
        run_id=run_id,
        runs_dir=tmp_path,
        audit_path=tmp_path / f"v94_supervised_real_{run_id}.json",
        request_fingerprint=FINGERPRINT,
    )


def test_claim_is_permanent_and_same_run_id_cannot_be_reused(tmp_path: Path):
    with _acquire(tmp_path, "run-001") as lease:
        lease.mark_phase("runtime")
        lease.finalize(result="PASS")

    claim = json.loads(lease.info.claim_path.read_text(encoding="utf-8"))
    assert claim["state"] == "PASS"
    with pytest.raises(DeploymentLeaseError, match="already been claimed"):
        _acquire(tmp_path, "run-001")


def test_live_hardware_lock_refuses_a_second_process_slot_without_claim(
    tmp_path: Path,
):
    with _acquire(tmp_path, "run-001") as first:
        with pytest.raises(DeploymentLeaseError, match="currently owns"):
            _acquire(tmp_path, "run-002")
        assert not (
            tmp_path / ".v94_supervised_claims" / "run-002.json"
        ).exists()
        first.finalize(result="PASS")


def test_exception_finalizes_failure_and_releases_lock_for_new_run(
    tmp_path: Path,
):
    with pytest.raises(RuntimeError, match="synthetic"):
        with _acquire(tmp_path, "run-001") as lease:
            lease.mark_phase("reset")
            raise RuntimeError("synthetic")
    claim = json.loads(lease.info.claim_path.read_text(encoding="utf-8"))
    assert claim["state"] == "FAILED"

    with _acquire(tmp_path, "run-002") as second:
        second.finalize(result="PASS")


def test_keyboard_interrupt_is_recorded_and_not_suppressed(tmp_path: Path):
    with pytest.raises(KeyboardInterrupt):
        with _acquire(tmp_path, "run-001") as lease:
            raise KeyboardInterrupt
    claim = json.loads(lease.info.claim_path.read_text(encoding="utf-8"))
    assert claim["state"] == "INTERRUPTED"
