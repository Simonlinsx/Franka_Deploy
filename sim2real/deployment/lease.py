"""Cross-process lease for the V94 real-robot deployment path.

The persistent lock file is never removed: the kernel ``flock`` is the sole
authority for hardware ownership.  Per-run claim files are permanent
tombstones so a run id cannot be silently reused after a reset failure or
process crash.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import stat
import threading
import time
from typing import Any, Mapping, Optional
import uuid


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,80}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LOCK_NAME = ".v94_supervised_hardware.lock"
_CLAIMS_DIR_NAME = ".v94_supervised_claims"
_INHERITED_FDS: set[int] = set()
_INHERITED_FDS_LOCK = threading.Lock()


class DeploymentLeaseError(RuntimeError):
    """The process could not obtain an exclusive real-hardware lease."""


def _close_inherited_fds_after_fork() -> None:
    # A forked visualization child must not keep the parent's flock alive.
    # Do not acquire a Python lock in the child: another vanished parent
    # thread could have owned it at fork time.
    inherited = tuple(_INHERITED_FDS)
    _INHERITED_FDS.clear()
    for descriptor in inherited:
        try:
            os.close(descriptor)
        except OSError:
            pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_close_inherited_fds_after_fork)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while recording deployment lease")
        view = view[written:]


def _validate_owned_regular_file(descriptor: int, label: str) -> None:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
    ):
        raise DeploymentLeaseError(
            f"{label} must be a single-link regular file owned by this user"
        )


def _validate_runs_dir(path: Path) -> Path:
    source = path.expanduser().resolve()
    try:
        metadata = source.stat()
    except OSError as exc:
        raise DeploymentLeaseError(f"deployment runs directory is unavailable: {exc}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or not os.access(source, os.W_OK | os.X_OK)
    ):
        raise DeploymentLeaseError(
            "deployment runs directory must be owner-controlled and writable"
        )
    return source


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class DeploymentLeaseInfo:
    run_id: str
    lock_path: Path
    claim_path: Path
    audit_path: Path
    claim_nonce: str


class DeploymentLease:
    """Held global flock plus a permanent, atomically-created run claim."""

    def __init__(
        self,
        *,
        descriptor: int,
        info: DeploymentLeaseInfo,
        claim: Mapping[str, Any],
    ) -> None:
        self._descriptor = int(descriptor)
        self.info = info
        self._claim = dict(claim)
        self._closed = False
        self._finalized = False

    @property
    def run_id(self) -> str:
        return self.info.run_id

    def assert_active(self, run_id: str) -> None:
        if (
            self._closed
            or self._descriptor < 0
            or str(run_id) != self.info.run_id
        ):
            raise DeploymentLeaseError("deployment hardware lease is not active")
        try:
            _validate_owned_regular_file(
                self._descriptor, "deployment hardware lock"
            )
        except OSError as exc:
            raise DeploymentLeaseError(
                "deployment hardware lease descriptor is no longer valid"
            ) from exc

    def _replace_claim(self, updates: Mapping[str, Any]) -> None:
        self.assert_active(self.info.run_id)
        claim = dict(self._claim)
        claim.update(dict(updates))
        claim["updated_realtime_s"] = time.time()
        temporary = self.info.claim_path.with_name(
            f".{self.info.claim_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary, flags, 0o600)
        try:
            _validate_owned_regular_file(descriptor, "deployment claim temporary")
            payload = (
                json.dumps(claim, sort_keys=True, allow_nan=False) + "\n"
            ).encode("utf-8")
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.replace(temporary, self.info.claim_path)
            _fsync_directory(self.info.claim_path.parent)
        except BaseException:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise
        self._claim = claim

    def mark_phase(self, phase: str) -> None:
        phase_text = str(phase).strip()
        if not phase_text:
            raise DeploymentLeaseError("deployment phase cannot be empty")
        self._replace_claim({"state": "ACTIVE", "phase": phase_text})

    def finalize(
        self,
        *,
        result: str,
        error: Optional[str] = None,
    ) -> None:
        result_text = str(result).upper()
        if result_text not in {"PASS", "FAILED", "INTERRUPTED"}:
            raise DeploymentLeaseError("invalid deployment lease terminal result")
        self._replace_claim(
            {
                "state": result_text,
                "phase": "terminal",
                "error": None if error is None else str(error),
            }
        )
        self._finalized = True

    def _release(self) -> None:
        if self._closed:
            return
        descriptor = self._descriptor
        self._descriptor = -1
        self._closed = True
        with _INHERITED_FDS_LOCK:
            _INHERITED_FDS.discard(descriptor)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> "DeploymentLease":
        self.assert_active(self.info.run_id)
        return self

    def __exit__(self, exc_type, exc, _traceback) -> bool:
        finalize_error: Optional[BaseException] = None
        if not self._finalized:
            if exc_type is None:
                result = "FAILED"
                detail = "lease exited without an explicit PASS result"
            elif issubclass(exc_type, KeyboardInterrupt):
                result = "INTERRUPTED"
                detail = f"{exc_type.__name__}: {exc}"
            else:
                result = "FAILED"
                detail = f"{exc_type.__name__}: {exc}"
            try:
                self.finalize(result=result, error=detail)
            except BaseException as failure:
                finalize_error = failure
        self._release()
        if exc_type is None and finalize_error is not None:
            raise DeploymentLeaseError(
                f"could not finalize deployment claim: {finalize_error}"
            ) from finalize_error
        return False


def acquire_v94_deployment_lease(
    *,
    run_id: str,
    runs_dir: Path,
    audit_path: Path,
    request_fingerprint: str,
) -> DeploymentLease:
    """Acquire the global hardware lock and permanently claim ``run_id``."""

    run_id = str(run_id)
    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise DeploymentLeaseError("run-id is not safe for a deployment claim")
    fingerprint = str(request_fingerprint).lower()
    if _SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise DeploymentLeaseError("request fingerprint must be a SHA-256")
    directory = _validate_runs_dir(Path(runs_dir))
    audit = Path(audit_path).expanduser().resolve()
    if audit.parent != directory:
        raise DeploymentLeaseError("audit path is outside the deployment runs directory")

    lock_path = directory / _LOCK_NAME
    lock_flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(lock_path, lock_flags, 0o600)
    try:
        _validate_owned_regular_file(descriptor, "deployment hardware lock")
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DeploymentLeaseError(
                "another V94 deployment process currently owns Franka/RH56"
            ) from exc
        with _INHERITED_FDS_LOCK:
            _INHERITED_FDS.add(descriptor)

        lock_payload = {
            "schema_version": 1,
            "run_id": run_id,
            "pid": os.getpid(),
            "uid": os.geteuid(),
            "host": socket.gethostname(),
            "acquired_realtime_s": time.time(),
        }
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        _write_all(
            descriptor,
            (json.dumps(lock_payload, sort_keys=True) + "\n").encode("utf-8"),
        )
        os.fsync(descriptor)

        claims_dir = directory / _CLAIMS_DIR_NAME
        try:
            claims_dir.mkdir(mode=0o700)
            _fsync_directory(directory)
        except FileExistsError:
            pass
        claims_metadata = claims_dir.lstat()
        if (
            not stat.S_ISDIR(claims_metadata.st_mode)
            or claims_metadata.st_uid != os.geteuid()
        ):
            raise DeploymentLeaseError(
                "deployment claims path must be an owner-controlled directory"
            )
        os.chmod(claims_dir, 0o700)
        claim_path = claims_dir / f"{run_id}.json"
        nonce = uuid.uuid4().hex
        claim_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            claim_descriptor = os.open(claim_path, claim_flags, 0o600)
        except FileExistsError as exc:
            raise DeploymentLeaseError(
                f"run-id has already been claimed: {run_id}"
            ) from exc
        claim = {
            "schema_version": 1,
            "kind": "v94_supervised_run_claim",
            "state": "CLAIMED",
            "phase": "admission",
            "run_id": run_id,
            "claim_nonce": nonce,
            "request_fingerprint": fingerprint,
            "audit_path": str(audit),
            "pid": os.getpid(),
            "uid": os.geteuid(),
            "host": socket.gethostname(),
            "claimed_realtime_s": time.time(),
        }
        try:
            _validate_owned_regular_file(
                claim_descriptor, "deployment run claim"
            )
            _write_all(
                claim_descriptor,
                (json.dumps(claim, sort_keys=True) + "\n").encode("utf-8"),
            )
            os.fsync(claim_descriptor)
        finally:
            os.close(claim_descriptor)
        _fsync_directory(claims_dir)
        return DeploymentLease(
            descriptor=descriptor,
            info=DeploymentLeaseInfo(
                run_id=run_id,
                lock_path=lock_path,
                claim_path=claim_path,
                audit_path=audit,
                claim_nonce=nonce,
            ),
            claim=claim,
        )
    except BaseException:
        with _INHERITED_FDS_LOCK:
            _INHERITED_FDS.discard(descriptor)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)
        raise


__all__ = [
    "DeploymentLease",
    "DeploymentLeaseError",
    "DeploymentLeaseInfo",
    "acquire_v94_deployment_lease",
]
