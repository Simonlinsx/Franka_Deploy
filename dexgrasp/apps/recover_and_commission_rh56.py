#!/usr/bin/env python3
"""Recover one interrupted installed RH56 Stage-2 run and recommission it.

This is deliberately a thin, fail-closed process orchestrator.  It never
imports a hardware transport and never reconstructs hand targets from command
line values.  The exact failed Stage-2 evidence is the only authority for the
base profile, official snapshot, candidate index, six actuator targets,
historical Stage-1 lineage, and waypoint step size.  After recovery, this
workflow creates and verifies a fresh current-source q6=900 Stage-1.

Every motion-capable child runs in its own process session.  Ctrl+C is
forwarded as SIGINT and the orchestrator waits for the child's reviewed cleanup
path; it never escalates to SIGTERM or SIGKILL.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
import uuid


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for _path in (ROOT / "src", WORKSPACE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from anydex_pipeline.control_config import (  # noqa: E402
    load_control_config,
    verify_adapter_assets,
)
from anydex_pipeline.rh56_commissioning import (  # noqa: E402
    build_stage1_prerequisite_binding,
    json_sha256,
    load_evidence,
    seal_evidence,
    sha256_file,
    verify_applied_config,
    verify_evidence,
)
from anydex_pipeline.rh56_interrupted_recovery import (  # noqa: E402
    InterruptedRecoveryPlan,
    RECOVERY_EVIDENCE_KIND,
    RECOVERY_MODE_COUPLED_CLOSE_PREFIX,
    RECOVERY_MODE_Q6_RETURN,
    build_interrupted_recovery_plan,
)
from anydex_pipeline.rh56_reset_open import (  # noqa: E402
    RESET_Q6_ARRIVAL_TOLERANCE_UNITS,
    RESET_Q6_DEADBAND_ESCAPE_UNITS,
    RESET_Q6_FIRST_STEP_UNITS,
    RESET_Q6_STEP_UNITS,
)
from anydex_pipeline.viewer_ready import normalize_unfollowed_leaf  # noqa: E402


INSTALLED_TOKEN = "RH56_INSTALLED_ON_FR3"
POWER_TOKEN = "RH56_24V_CUTOFF_READY"
STOP_TOKEN = "FR3_STOP_READY"
CLEAR_TOKEN = "INSTALLED_AIR_WORKSPACE_CLEAR"
NO_CONTACT_TOKEN = "PLA_LOW_SPEED_NO_CONTACT"
RECOVERY_TOKEN = "RH56_INTERRUPTED_OPEN_RECOVERY"
WIDE_Q6_TOKEN = "RH56_Q6_WIDE_RANGE_COMMISSIONING"
COUPLED_TOKEN = "RH56_COUPLED_AIR_CLOSURE"
EXACT_AIR_TARGET_TOKEN = "RH56_EXACT_CANDIDATE_AIR_TARGET"

RECEIPT_SCHEMA_VERSION = 2
RECEIPT_KIND = "recover_and_commission_rh56_workflow_receipt_v2"
RECOVERY_SOURCE_NAMES = frozenset(
    (
        "recovery_cli",
        "recovery_module",
        "rh56_reset_open",
        "rh56_sequence_driver",
        "rh56_hand_path",
        "rh56_register_api",
        "franka_sequence_driver",
        "adapter_mesh",
        "adapter_provenance",
    )
)
IDLE_STATUSES = frozenset((2,))
BEND_OPEN_MIN_ANGLE = 980
MAX_DISABLED_CURRENT_MA = 100
RECOVERY_ROUTE_Q6_RETURN = "sealed_stage2_q6_return_v1"
RECOVERY_ROUTE_NEAR_OPEN_RESET = "profile_near_open_reset_v1"
RECOVERY_ROUTE_COUPLED_PREFIX = "sealed_coupled_close_prefix_v1"
DEFAULT_RESET_SPEEDS = (1000,) * 6
DEFAULT_RESET_FORCES = (500,) * 6


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class FileBinding:
    name: str
    path: Path
    sha256: str
    device: int
    inode: int
    size: int
    mode: int
    mtime_ns: int
    ctime_ns: int

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "sha256": self.sha256,
            "identity": {
                "device": self.device,
                "inode": self.inode,
                "size": self.size,
                "mode": self.mode,
                "mtime_ns": self.mtime_ns,
                "ctime_ns": self.ctime_ns,
            },
        }


@dataclass(frozen=True)
class WorkflowContext:
    workflow_id: str
    failed_stage2: Path
    output_dir: Path
    recovery_output: Path
    fresh_stage1_output: Path
    stage2_output: Path
    receipt_output: Path
    base_config_path: Path
    base_config: Mapping[str, Any]
    snapshot_path: Path
    candidate_index: int
    candidate_score: float
    hand_targets: Tuple[int, int, int, int, int, int]
    historical_stage1_evidence_path: Path
    q6_step: int
    derived_profile_path: Path
    recovery_plan: InterruptedRecoveryPlan
    immutable_inputs: Tuple[FileBinding, ...]
    historical_commissioning_sources: Tuple[Mapping[str, str], ...]
    recovery_runtime_sources: Tuple[FileBinding, ...]


@dataclass(frozen=True)
class ChildResult:
    returncode: int
    interrupted: bool = False


class StageFailure(RuntimeError):
    def __init__(self, stage: str, returncode: int, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.returncode = int(returncode)


class WorkflowInterrupted(StageFailure):
    pass


ChildRunner = Callable[[str, Sequence[str]], ChildResult]
ContextLoader = Callable[[Path, Path], WorkflowContext]
RecoveryValidator = Callable[[Path, WorkflowContext], Mapping[str, Any]]
Stage2Validator = Callable[[Path, WorkflowContext], Mapping[str, Any]]
Stage1Validator = Callable[[Path, WorkflowContext], Mapping[str, Any]]
AppliedValidator = Callable[[Path, Path, WorkflowContext], Mapping[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evidence-bound RH56 recovery -> fresh current-source q6=900 "
            "Stage-1 -> exact Stage-2 recommission -> verified derived profile. "
            "No target/candidate override exists."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the exact evidence-bound workflow")
    run.add_argument("--failed-stage2", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--confirm-installed", metavar=INSTALLED_TOKEN)
    run.add_argument("--confirm-24v-cutoff", metavar=POWER_TOKEN)
    run.add_argument("--confirm-franka-stop", metavar=STOP_TOKEN)
    run.add_argument("--confirm-workspace-clear", metavar=CLEAR_TOKEN)
    run.add_argument("--confirm-no-contact", metavar=NO_CONTACT_TOKEN)
    run.add_argument("--confirm-recovery", metavar=RECOVERY_TOKEN)
    run.add_argument("--confirm-wide-q6", metavar=WIDE_Q6_TOKEN)
    run.add_argument("--confirm-coupled-closure", metavar=COUPLED_TOKEN)
    run.add_argument("--confirm-exact-air-target", metavar=EXACT_AIR_TARGET_TOKEN)
    return parser


def _require_tokens(args: argparse.Namespace) -> None:
    required = (
        ("--confirm-installed", args.confirm_installed, INSTALLED_TOKEN),
        ("--confirm-24v-cutoff", args.confirm_24v_cutoff, POWER_TOKEN),
        ("--confirm-franka-stop", args.confirm_franka_stop, STOP_TOKEN),
        ("--confirm-workspace-clear", args.confirm_workspace_clear, CLEAR_TOKEN),
        ("--confirm-no-contact", args.confirm_no_contact, NO_CONTACT_TOKEN),
        ("--confirm-recovery", args.confirm_recovery, RECOVERY_TOKEN),
        ("--confirm-wide-q6", args.confirm_wide_q6, WIDE_Q6_TOKEN),
        (
            "--confirm-coupled-closure",
            args.confirm_coupled_closure,
            COUPLED_TOKEN,
        ),
        (
            "--confirm-exact-air-target",
            args.confirm_exact_air_target,
            EXACT_AIR_TARGET_TOKEN,
        ),
    )
    missing = [
        "{} {}".format(flag, expected)
        for flag, actual, expected in required
        if actual != expected
    ]
    if missing:
        raise ValueError(
            "exact confirmations required before any child process: "
            + "; ".join(missing)
        )


def _canonical_bound_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("{} must be a non-empty canonical absolute path".format(name))
    supplied = Path(value).expanduser()
    if not supplied.is_absolute():
        raise ValueError("{} must be absolute".format(name))
    resolved = normalize_unfollowed_leaf(supplied)
    if str(resolved) != value:
        raise ValueError("{} must use its resolved canonical spelling".format(name))
    try:
        metadata = resolved.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError("{} is missing: {}".format(name, resolved)) from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("{} must not be a final symlink alias".format(name))
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("{} must be a regular file".format(name))
    return resolved


def _strict_six(value: Any, name: str, *, allow_disabled: bool = False) -> Tuple[int, ...]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError("{} must be a six-integer JSON array".format(name))
    minimum = -1 if allow_disabled else 0
    output = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("{}[{}] must be an integer".format(name, index))
        if not minimum <= item <= 1000:
            raise ValueError("{}[{}] is outside {}..1000".format(name, index, minimum))
        output.append(int(item))
    return tuple(output)


def _open_regular_nofollow(path: Path, name: str) -> tuple[Path, int, os.stat_result]:
    source = normalize_unfollowed_leaf(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(source), flags)
    except OSError as exc:
        raise ValueError(
            "{} cannot be opened as a non-symlink file {}: {}".format(
                name, source, exc
            )
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("{} is not a regular file: {}".format(name, source))
        return source, descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise


def _digest_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _file_binding(name: str, path: Path) -> FileBinding:
    source, descriptor, metadata = _open_regular_nofollow(path, name)
    try:
        digest = _digest_descriptor(descriptor)
    finally:
        os.close(descriptor)
    return FileBinding(
        name=name,
        path=source,
        sha256=digest,
        device=int(metadata.st_dev),
        inode=int(metadata.st_ino),
        size=int(metadata.st_size),
        mode=stat.S_IMODE(metadata.st_mode),
        mtime_ns=int(metadata.st_mtime_ns),
        ctime_ns=int(metadata.st_ctime_ns),
    )


def _assert_file_binding(binding: FileBinding) -> None:
    current = _file_binding(binding.name, binding.path)
    expected = (
        binding.sha256,
        binding.device,
        binding.inode,
        binding.size,
        binding.mode,
        binding.mtime_ns,
        binding.ctime_ns,
    )
    actual = (
        current.sha256,
        current.device,
        current.inode,
        current.size,
        current.mode,
        current.mtime_ns,
        current.ctime_ns,
    )
    if actual != expected:
        raise RuntimeError("immutable input changed: {}".format(binding.path))


def _reject_existing_leaf(path: Path, name: str) -> Path:
    target = normalize_unfollowed_leaf(path)
    try:
        target.lstat()
    except FileNotFoundError:
        return target
    raise FileExistsError(
        "{} path already exists (including a symlink): {}".format(name, target)
    )


def _require_readonly_regular_output(path: Path, name: str) -> FileBinding:
    binding = _file_binding(name, path)
    if binding.mode != 0o444:
        raise ValueError("{} mode must be exactly 0444".format(name))
    return binding


def _directory_identity(path: Path, name: str) -> tuple[Path, int, int, int]:
    target = normalize_unfollowed_leaf(path)
    try:
        metadata = target.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError("{} disappeared: {}".format(name, target)) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("{} is not a real directory: {}".format(name, target))
    return (
        target,
        int(metadata.st_dev),
        int(metadata.st_ino),
        stat.S_IMODE(metadata.st_mode),
    )


def _assert_directory_identity(
    expected: tuple[Path, int, int, int], name: str
) -> None:
    if _directory_identity(expected[0], name) != expected:
        raise RuntimeError("{} identity changed: {}".format(name, expected[0]))


def _workflow_source_paths() -> Mapping[str, Path]:
    return {
        "workflow_cli": Path(__file__).resolve(),
        "workflow_wrapper": (
            ROOT / "scripts/recover_and_commission_rh56.sh"
        ).resolve(),
        "recovery_wrapper": (ROOT / "scripts/recover_installed_rh56.sh").resolve(),
        "commission_wrapper": (
            ROOT / "scripts/commission_installed_rh56.sh"
        ).resolve(),
    }


def _recovery_runtime_source_paths(assets: Any) -> Mapping[str, Path]:
    """Return every current file that can influence the recovery motion."""

    return {
        "recovery_cli": (ROOT / "apps/recover_installed_rh56.py").resolve(),
        "recovery_module": (
            ROOT / "src/anydex_pipeline/rh56_interrupted_recovery.py"
        ).resolve(),
        "rh56_reset_open": (
            ROOT / "src/anydex_pipeline/rh56_reset_open.py"
        ).resolve(),
        "rh56_sequence_driver": (
            ROOT / "src/anydex_pipeline/inspire_sequence_driver.py"
        ).resolve(),
        "rh56_hand_path": (
            ROOT / "src/anydex_pipeline/rh56_hand_path.py"
        ).resolve(),
        "rh56_register_api": (WORKSPACE / "examples/inspire_rh56_test.py").resolve(),
        "franka_sequence_driver": (
            ROOT / "src/anydex_pipeline/franka_sequence_driver.py"
        ).resolve(),
        "adapter_mesh": Path(assets.mesh_path).resolve(),
        "adapter_provenance": Path(assets.provenance_path).resolve(),
    }


def _historical_source_provenance(value: Any) -> Tuple[Mapping[str, str], ...]:
    """Copy sealed historical source bindings without treating them as live code.

    The exact failed artifact and all its semantics are accepted only by
    ``build_interrupted_recovery_plan``.  These entries therefore remain
    provenance: current execution authority is recorded separately below.
    """

    if not isinstance(value, list) or not value:
        raise ValueError("failed Stage-2 source_bindings are missing")
    result = []
    names = set()
    for position, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"name", "path", "sha256"}:
            raise ValueError(
                "failed Stage-2 source_bindings[{}] is malformed".format(position)
            )
        name = item.get("name")
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("failed Stage-2 source binding names must be unique")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("failed Stage-2 source path must be absolute")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("failed Stage-2 source binding SHA-256 is malformed")
        names.add(name)
        result.append({"name": name, "path": path, "sha256": digest})
    return tuple(result)


def _derive_context(failed_stage2: Path, output_dir: Path) -> WorkflowContext:
    failed_path = normalize_unfollowed_leaf(failed_stage2)
    failed_input = _file_binding("failed_stage2", failed_path)
    failed_before = failed_input.sha256
    evidence, resolved_failed = load_evidence(failed_path)
    if resolved_failed != failed_path:
        raise ValueError("failed Stage-2 path resolution changed")

    profile = evidence.get("control_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("failed Stage-2 evidence has no control_profile binding")
    base_path = _canonical_bound_path(profile.get("path"), "control_profile.path")
    base_input = _file_binding("base_config", base_path)
    base_config, canonical_base_path = load_control_config(base_path)
    if canonical_base_path != base_path:
        raise ValueError("failed Stage-2 base profile path is not canonical")
    _assert_file_binding(base_input)
    assets = verify_adapter_assets(base_config, base_path)

    plan = build_interrupted_recovery_plan(
        failed_path,
        expected_config=base_config,
        expected_config_path=base_path,
    )
    _assert_file_binding(failed_input)
    failed_after = _file_binding("failed_stage2", failed_path).sha256
    if failed_before != failed_after or failed_after != plan.failed_file_sha256:
        raise ValueError("failed Stage-2 evidence changed during workflow derivation")

    # Reload the exact bytes validated by the recovery-plan builder.  The plan
    # builder is the single authority for historical-profile lineage, including
    # its one reviewed old-FR3-limits -> current-FR3-limits migration.  From
    # this point onward the live, already-validated base bytes are immutable;
    # do not compare them a second time to the historical Stage-2 snapshot.
    evidence, _ = load_evidence(failed_path)
    _assert_file_binding(failed_input)
    if failed_input.sha256 != plan.failed_file_sha256:
        raise ValueError("failed Stage-2 evidence changed after recovery-plan validation")
    profile = evidence.get("control_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("failed Stage-2 evidence lost its control_profile binding")
    _assert_file_binding(base_input)
    reloaded_base_config, reloaded_base_path = load_control_config(base_path)
    _assert_file_binding(base_input)
    if reloaded_base_path != base_path or reloaded_base_config != base_config:
        raise ValueError("base profile changed after recovery-plan validation")
    snapshot_binding = evidence.get("snapshot_candidate")
    if not isinstance(snapshot_binding, Mapping):
        raise ValueError("failed Stage-2 evidence has no official snapshot binding")
    snapshot_path = _canonical_bound_path(
        snapshot_binding.get("path"), "snapshot_candidate.path"
    )
    snapshot_input = _file_binding("official_snapshot", snapshot_path)
    candidate = snapshot_binding.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("failed Stage-2 evidence has no candidate binding")
    index = candidate.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("failed Stage-2 candidate index is invalid")
    score = candidate.get("official_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise ValueError("failed Stage-2 candidate score is invalid")
    targets = _strict_six(candidate.get("hand_targets"), "candidate.hand_targets")
    if targets != tuple(plan.coupled_targets):
        raise ValueError("candidate hand targets differ from the recovery plan")
    if targets[5] != int(plan.target_q6):
        raise ValueError("candidate q6 target differs from the recovery plan")

    stage1 = evidence.get("stage1_prerequisite")
    if not isinstance(stage1, Mapping):
        raise ValueError("failed Stage-2 evidence has no Stage-1 prerequisite")
    stage1_path = _canonical_bound_path(
        stage1.get("path"), "stage1_prerequisite.path"
    )
    historical_stage1_input = _file_binding(
        "historical_stage1_evidence", stage1_path
    )
    request = evidence.get("request")
    if not isinstance(request, Mapping):
        raise ValueError("failed Stage-2 evidence has no request binding")
    if request.get("target_source") != "official_snapshot_candidate":
        raise ValueError("failed Stage-2 target was not snapshot-derived")
    if request.get("step_units") != plan.step_units:
        raise ValueError("failed Stage-2 step size differs from the recovery plan")
    if request.get("coupled_targets") != list(targets):
        raise ValueError("failed Stage-2 request targets differ from its candidate")

    destination = _reject_existing_leaf(output_dir, "workflow output directory")
    workflow_id = str(uuid.uuid4())
    suffix = base_path.suffix or ".json"
    derived = base_path.with_name(
        "{}_candidate{}_commissioned_{}{}".format(
            base_path.stem, int(index), workflow_id.replace("-", "")[:12], suffix
        )
    )
    derived = _reject_existing_leaf(derived, "derived profile")

    authority_inputs = (
        failed_input,
        base_input,
        snapshot_input,
        historical_stage1_input,
    )
    expected_hashes = {
        "failed_stage2": plan.failed_file_sha256,
        # The recovery plan has already validated the precise historical to
        # current profile migration.  Runtime authority is the immutable live
        # file, not the superseded hash stored in the historical evidence.
        "base_config": base_input.sha256,
        "official_snapshot": snapshot_binding.get("file_sha256"),
        "historical_stage1_evidence": stage1.get("file_sha256"),
    }
    for binding in authority_inputs:
        if expected_hashes[binding.name] != binding.sha256:
            raise ValueError(
                "{} changed after strict evidence validation".format(binding.name)
            )
    historical_sources = _historical_source_provenance(
        evidence.get("source_bindings")
    )
    workflow_sources = tuple(
        _file_binding(name, path) for name, path in _workflow_source_paths().items()
    )
    recovery_runtime_sources = tuple(
        _file_binding(name, path)
        for name, path in _recovery_runtime_source_paths(assets).items()
    )
    immutable = authority_inputs + workflow_sources + recovery_runtime_sources
    return WorkflowContext(
        workflow_id=workflow_id,
        failed_stage2=failed_path,
        output_dir=destination,
        recovery_output=destination / "recovery.json",
        fresh_stage1_output=destination / "fresh_stage1.json",
        stage2_output=destination / "stage2.json",
        receipt_output=destination / "workflow_receipt.json",
        base_config_path=base_path,
        base_config=base_config,
        snapshot_path=snapshot_path,
        candidate_index=int(index),
        candidate_score=float(score),
        hand_targets=targets,  # type: ignore[arg-type]
        historical_stage1_evidence_path=stage1_path,
        q6_step=int(plan.step_units),
        derived_profile_path=derived,
        recovery_plan=plan,
        immutable_inputs=immutable,
        historical_commissioning_sources=historical_sources,
        recovery_runtime_sources=recovery_runtime_sources,
    )


def _assert_immutable_inputs(context: WorkflowContext) -> None:
    for binding in context.immutable_inputs:
        _assert_file_binding(binding)


def _validate_context_layout(context: WorkflowContext) -> None:
    output_dir = normalize_unfollowed_leaf(context.output_dir)
    if output_dir != context.output_dir:
        raise ValueError("workflow output directory is not canonical")
    expected_children = {
        "recovery_output": output_dir / "recovery.json",
        "fresh_stage1_output": output_dir / "fresh_stage1.json",
        "stage2_output": output_dir / "stage2.json",
        "receipt_output": output_dir / "workflow_receipt.json",
    }
    for field, expected in expected_children.items():
        actual = Path(getattr(context, field))
        if actual != expected:
            raise ValueError("{} is an aliased workflow output path".format(field))
    derived = normalize_unfollowed_leaf(context.derived_profile_path)
    if derived != context.derived_profile_path:
        raise ValueError("derived profile path is not canonical")
    if derived.parent != context.base_config_path.parent:
        raise ValueError("derived profile must be a unique sibling of the base profile")
    occupied = {binding.path for binding in context.immutable_inputs}
    if derived in occupied or derived == context.base_config_path:
        raise ValueError("derived profile aliases an immutable input")
    if len(set(expected_children.values()) | {derived}) != 5:
        raise ValueError("workflow output paths must be unique")


def _assert_runtime_state(
    context: WorkflowContext,
    output_identity: tuple[Path, int, int, int],
) -> None:
    _assert_directory_identity(output_identity, "workflow output directory")
    _assert_immutable_inputs(context)


def _validate_source_bindings(
    value: Any,
    expected_sources: Sequence[FileBinding],
) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("recovery source_bindings must be a non-empty array")
    expected = {binding.name: binding for binding in expected_sources}
    if set(expected) != RECOVERY_SOURCE_NAMES:
        raise ValueError("workflow recovery runtime source authority is incomplete")
    names = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"name", "path", "sha256"}:
            raise ValueError("recovery source_bindings[{}] is malformed".format(index))
        name = item.get("name")
        digest = item.get("sha256")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("recovery source binding names must be unique")
        names.add(name)
        path = _canonical_bound_path(
            item.get("path"), "source_bindings[{}].path".format(index)
        )
        binding = _file_binding("recovery_source." + name, path)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or binding.sha256 != digest
        ):
            raise ValueError("recovery source binding hash mismatch: {}".format(path))
        authority = expected.get(name)
        if (
            authority is None
            or path != authority.path
            or digest != authority.sha256
        ):
            raise ValueError(
                "recovery source binding differs from workflow runtime authority: "
                + str(name)
            )
    if names != RECOVERY_SOURCE_NAMES:
        raise ValueError(
            "recovery source binding names differ: expected={} actual={}".format(
                sorted(RECOVERY_SOURCE_NAMES), sorted(names)
            )
        )


def _strict_profile_open_feedback_policy(
    config: Mapping[str, Any],
) -> tuple[Tuple[int, ...], Tuple[int, int]]:
    """Derive the reviewed installed-hand open feedback thresholds.

    The five bend axes use the fixed canonical 980 feedback floor.  The q6
    actuator has a measured endpoint offset, so its floor is the profile's
    open command minus the same arrival tolerance used by the reset driver.
    """

    inspire = config.get("inspire")
    if not isinstance(inspire, Mapping):
        raise ValueError("recovery profile inspire policy is missing")
    targets = inspire.get("open_targets")
    validated_range = inspire.get("thumb_rotate_validated_realtime_range")
    tolerance = inspire.get("arrival_tolerance_units")
    if (
        not isinstance(targets, list)
        or len(targets) != 6
        or any(type(value) is not int or not 0 <= value <= 1000 for value in targets)
    ):
        raise ValueError("recovery profile open_targets are malformed")
    if (
        not isinstance(validated_range, list)
        or len(validated_range) != 2
        or any(
            type(value) is not int or not 0 <= value <= 1000
            for value in validated_range
        )
        or validated_range[0] > validated_range[1]
    ):
        raise ValueError("recovery profile q6 validated range is malformed")
    if type(tolerance) is not int or not 0 <= tolerance <= 30:
        raise ValueError("recovery profile arrival tolerance is malformed")
    if targets != [1000] * 6:
        raise ValueError("recovery profile canonical open targets changed")
    if targets[5] != validated_range[1]:
        raise ValueError("recovery profile q6 open target is not the range endpoint")
    q6_minimum = targets[5] - tolerance
    if not validated_range[0] <= q6_minimum <= validated_range[1]:
        raise ValueError("recovery profile q6 open feedback floor is outside its range")
    return (
        (BEND_OPEN_MIN_ANGLE,) * 5 + (q6_minimum,),
        (validated_range[0], validated_range[1]),
    )


def _expected_recovery_routes(plan: InterruptedRecoveryPlan) -> Tuple[str, ...]:
    if plan.recovery_mode == RECOVERY_MODE_Q6_RETURN:
        return (RECOVERY_ROUTE_Q6_RETURN, RECOVERY_ROUTE_NEAR_OPEN_RESET)
    if plan.recovery_mode == RECOVERY_MODE_COUPLED_CLOSE_PREFIX:
        return (RECOVERY_ROUTE_COUPLED_PREFIX,)
    raise ValueError("recovery plan mode has no reviewed execution route")


def _canonical_near_open_reset_commands(
    initial_q6: int,
    q6_range: Tuple[int, int],
    q6_open_minimum: int,
) -> Tuple[int, ...]:
    lower, upper = q6_range
    if not lower <= initial_q6 <= upper:
        raise ValueError("near-open reset preflight q6 is outside the profile range")
    if initial_q6 >= q6_open_minimum:
        return ()
    if upper - initial_q6 < RESET_Q6_DEADBAND_ESCAPE_UNITS:
        backoff = max(
            lower,
            min(
                initial_q6 - RESET_Q6_DEADBAND_ESCAPE_UNITS,
                upper
                - RESET_Q6_DEADBAND_ESCAPE_UNITS
                - RESET_Q6_ARRIVAL_TOLERANCE_UNITS,
            ),
        )
        if initial_q6 - backoff < RESET_Q6_DEADBAND_ESCAPE_UNITS:
            raise ValueError("near-open reset cannot form its deadband escape")
        return (backoff, upper)
    commands = []
    previous = initial_q6
    first = True
    while previous < upper:
        increment = (
            RESET_Q6_FIRST_STEP_UNITS if first else RESET_Q6_STEP_UNITS
        )
        target = min(upper, previous + increment)
        commands.append(target)
        previous = target
        first = False
    return tuple(commands)


def _strict_recovery_route(
    evidence: Mapping[str, Any],
    context: WorkflowContext,
) -> tuple[str, bool, Tuple[int, ...]]:
    """Validate the explicit route selected by the current recovery runtime."""

    request = evidence.get("request")
    result = evidence.get("result")
    if not isinstance(request, Mapping) or not isinstance(result, Mapping):
        raise ValueError("recovery route binding is missing")
    expected_routes = _expected_recovery_routes(context.recovery_plan)
    if request.get("allowed_recovery_routes") != list(expected_routes):
        raise ValueError("recovery allowed route list differs from the reviewed plan")
    route = request.get("selected_recovery_route")
    if (
        not isinstance(route, str)
        or route not in expected_routes
        or result.get("recovery_route") != route
    ):
        raise ValueError("recovery selected route is missing, unsupported, or inconsistent")
    open_minimums, q6_range = _strict_profile_open_feedback_policy(
        context.base_config
    )
    plan_binding = context.recovery_plan.as_binding()
    if plan_binding.get("allowed_recovery_routes") != list(expected_routes):
        raise ValueError("recovery plan allowed route list is malformed")
    if plan_binding.get("profile_near_open_q6_range") != list(q6_range):
        raise ValueError("recovery plan near-open q6 range differs from the profile")
    if plan_binding.get("q6_open_min_angle") != open_minimums[5]:
        raise ValueError("recovery plan q6 open threshold differs from the profile")
    if request.get("profile_near_open_q6_range") != list(q6_range):
        raise ValueError("recovery near-open q6 range differs from the profile")
    if request.get("q6_open_min_angle") != open_minimums[5]:
        raise ValueError("recovery q6 open threshold differs from the profile")
    expected_historical_path = route != RECOVERY_ROUTE_NEAR_OPEN_RESET
    if result.get("evidence_bound_path_executed") is not expected_historical_path:
        raise ValueError("recovery route misstates whether the sealed path executed")
    route_initial_q6 = result.get("recovery_route_initial_q6")
    if (
        type(route_initial_q6) is not int
        or request.get("route_dispatch_initial_q6") != route_initial_q6
    ):
        raise ValueError("recovery route dispatch q6 binding is missing or inconsistent")
    route_dispatch = evidence.get("route_dispatch")
    if not isinstance(route_dispatch, Mapping):
        raise ValueError("recovery route dispatch boundary sample is missing")
    if route_dispatch.get("phase") != "boundary_state_snapshot":
        raise ValueError("recovery route dispatch boundary phase is invalid")
    dispatch_targets = _strict_six(
        route_dispatch.get("angle_targets"),
        "route_dispatch.angle_targets",
        allow_disabled=True,
    )
    dispatch_angles = _strict_six(
        route_dispatch.get("angles"), "route_dispatch.angles"
    )
    dispatch_currents = _strict_six_signed(
        route_dispatch.get("currents"), "route_dispatch.currents"
    )
    dispatch_errors = _strict_six(
        route_dispatch.get("errors"), "route_dispatch.errors"
    )
    dispatch_statuses = _strict_six(
        route_dispatch.get("statuses"), "route_dispatch.statuses"
    )
    dispatch_temperatures = _strict_six(
        route_dispatch.get("temperatures"), "route_dispatch.temperatures"
    )
    if (
        dispatch_targets != (-1,) * 6
        or dispatch_angles[5] != route_initial_q6
        or any(abs(value) > MAX_DISABLED_CURRENT_MA for value in dispatch_currents)
        or dispatch_errors != (0,) * 6
        or dispatch_statuses != (2,) * 6
        or any(value >= 50 for value in dispatch_temperatures)
    ):
        raise ValueError("recovery route dispatch boundary is not disabled and safe")
    telemetry = evidence.get("telemetry")
    if (
        not isinstance(telemetry, list)
        or [
            sample
            for sample in telemetry
            if isinstance(sample, Mapping)
            and sample.get("phase") == "boundary_state_snapshot"
        ]
        != [route_dispatch]
    ):
        raise ValueError("recovery route dispatch is not uniquely bound in telemetry")
    executed_q6 = evidence.get("executed_q6_return_waypoints")
    executed_reset = evidence.get("executed_reset_q6_waypoints")
    if (
        not isinstance(executed_q6, list)
        or not isinstance(executed_reset, list)
        or any(type(value) is not int or not 0 <= value <= 1000 for value in executed_q6)
        or any(
            type(value) is not int or not 0 <= value <= 1000
            for value in executed_reset
        )
    ):
        raise ValueError("recovery executed waypoint records are malformed")
    if route == RECOVERY_ROUTE_NEAR_OPEN_RESET:
        if executed_q6:
            raise ValueError("near-open reset cannot claim the sealed q6 return path")
        reset_preflights = [
            sample
            for sample in telemetry
            if isinstance(sample, Mapping)
            and sample.get("phase") == "reset_open_preflight"
        ]
        if len(reset_preflights) != 1:
            raise ValueError("near-open recovery lacks one reset-open preflight")
        reset_preflight = reset_preflights[0]
        reset_targets = _strict_six(
            reset_preflight.get("angle_targets"),
            "reset_open_preflight.angle_targets",
            allow_disabled=True,
        )
        reset_angles = _strict_six(
            reset_preflight.get("angles"), "reset_open_preflight.angles"
        )
        reset_errors = _strict_six(
            reset_preflight.get("errors"), "reset_open_preflight.errors"
        )
        reset_statuses = _strict_six(
            reset_preflight.get("statuses"), "reset_open_preflight.statuses"
        )
        if (
            reset_targets != (-1,) * 6
            or reset_errors != (0,) * 6
            or any(value not in (2, 0xFF) for value in reset_statuses)
            or any(value < BEND_OPEN_MIN_ANGLE for value in reset_angles[:5])
        ):
            raise ValueError("near-open reset preflight is not disabled/open/idle")
        expected_reset = _canonical_near_open_reset_commands(
            reset_angles[5],
            q6_range,
            open_minimums[5],
        )
        if (
            not q6_range[0] <= route_initial_q6 <= q6_range[1]
            or any(
                not q6_range[0] <= value <= q6_range[1]
                for value in executed_reset
            )
            or any(value < BEND_OPEN_MIN_ANGLE for value in dispatch_angles[:5])
            or tuple(executed_reset) != expected_reset
        ):
            raise ValueError(
                "near-open dispatch or reset path differs from the canonical profile path"
            )
    else:
        if executed_reset:
            raise ValueError("sealed recovery route cannot claim reset-driver waypoints")
        if any(
            isinstance(sample, Mapping)
            and sample.get("phase") == "reset_open_preflight"
            for sample in telemetry
        ):
            raise ValueError("sealed recovery route contains reset-open telemetry")
    if route == RECOVERY_ROUTE_Q6_RETURN and not (
        context.recovery_plan.permitted_live_q6_range[0]
        <= route_initial_q6
        <= context.recovery_plan.permitted_live_q6_range[1]
    ):
        raise ValueError("sealed q6 recovery dispatch is outside its evidence range")
    if route == RECOVERY_ROUTE_Q6_RETURN and (
        not executed_q6 or executed_q6[-1] != 1000
    ):
        raise ValueError("sealed q6 recovery route did not record a complete return")
    return route, expected_historical_path, open_minimums


def _strict_recovery_evidence(
    evidence_path: Path, context: WorkflowContext
) -> Mapping[str, Any]:
    source = normalize_unfollowed_leaf(evidence_path)
    runtime_binding = _require_readonly_regular_output(
        source, "recovery evidence"
    )
    evidence, resolved = load_evidence(source)
    _assert_file_binding(runtime_binding)
    if resolved != source:
        raise ValueError("recovery evidence changed while being loaded")
    if evidence.get("schema_version") != 1:
        raise ValueError("recovery evidence schema_version is not 1")
    if evidence.get("kind") != RECOVERY_EVIDENCE_KIND:
        raise ValueError("recovery evidence kind is invalid")
    if evidence.get("motion_authorized") is not False:
        raise ValueError("recovery evidence motion_authorized must remain false")
    if evidence.get("commissioning_unlock_claimed") is not False:
        raise ValueError("recovery evidence cannot claim commissioning unlock")
    integrity = evidence.get("integrity")
    if not isinstance(integrity, Mapping) or set(integrity) != {
        "algorithm",
        "payload_sha256",
    }:
        raise ValueError("recovery integrity object is malformed")
    if integrity.get("algorithm") != "sha256-canonical-json-without-integrity":
        raise ValueError("recovery integrity algorithm is invalid")
    unsigned = {key: value for key, value in evidence.items() if key != "integrity"}
    if integrity.get("payload_sha256") != json_sha256(unsigned):
        raise ValueError("recovery canonical payload SHA-256 mismatch")

    profile = evidence.get("control_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("recovery control_profile binding is missing")
    if Path(str(profile.get("path", ""))).expanduser().resolve() != context.base_config_path:
        raise ValueError("recovery belongs to a different base profile")
    if profile.get("file_sha256") != sha256_file(context.base_config_path):
        raise ValueError("recovery base-profile file hash mismatch")
    if profile.get("parsed_sha256") != json_sha256(context.base_config):
        raise ValueError("recovery parsed profile hash mismatch")
    if profile.get("snapshot") != context.base_config:
        raise ValueError("recovery profile snapshot differs from the base profile")
    if evidence.get("failed_commissioning_binding") != context.recovery_plan.as_binding():
        raise ValueError("recovery failed-Stage2 binding differs from the strict plan")
    _validate_source_bindings(
        evidence.get("source_bindings"),
        context.recovery_runtime_sources,
    )
    confirmations = evidence.get("operator_confirmations")
    if not isinstance(confirmations, Mapping) or any(
        confirmations.get(name) is not True
        for name in (
            "installed_on_fr3",
            "24v_cutoff_ready",
            "franka_stop_ready",
            "workspace_clear",
            "no_contact_PLA_scope",
            "interrupted_open_recovery_confirmed",
        )
    ):
        raise ValueError("recovery evidence lost an operator confirmation")
    request = evidence.get("request")
    if not isinstance(request, Mapping):
        raise ValueError("recovery request binding is missing")
    if request.get("recovery_mode") != context.recovery_plan.recovery_mode:
        raise ValueError("recovery mode differs from the strict recovery plan")
    if request.get("evidence_bound_q6_return_waypoints") != list(
        context.recovery_plan.q6_return_waypoints
    ):
        raise ValueError("recovery q6 return path differs from the strict plan")
    if request.get("permitted_live_q6_range") != list(
        context.recovery_plan.permitted_live_q6_range
    ):
        raise ValueError("recovery live-q6 range differs from the strict plan")

    result = evidence.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("recovery result is missing")
    required_pass = (
        "adopted_disabled_verified",
        "recovered_open_verified",
        "disabled_verified",
    )
    if result.get("status") != "pass" or any(
        result.get(name) is not True for name in required_pass
    ):
        raise ValueError("recovery did not prove adopt/open/disable PASS")
    if (
        result.get("operation_error") is not None
        or result.get("stop_error") is not None
        or result.get("cleanup_error") is not None
    ):
        raise ValueError("recovery PASS contains an operation or stop error")
    route, _historical_path_executed, open_minimums = _strict_recovery_route(
        evidence, context
    )
    franka = evidence.get("franka_read_only")
    if not isinstance(franka, Mapping) or franka.get("verified") is not True:
        raise ValueError("recovery did not prove the Franka read-only gate")

    final = evidence.get("final")
    if not isinstance(final, Mapping):
        raise ValueError("recovery final feedback is missing")
    if _strict_six(final.get("angle_targets"), "final.angle_targets", allow_disabled=True) != (-1,) * 6:
        raise ValueError("recovery final ANGLE_SET is not all-six disabled")
    angles = _strict_six(final.get("angles"), "final.angles")
    currents = _strict_six_signed(final.get("currents"), "final.currents")
    errors = _strict_six(final.get("errors"), "final.errors")
    statuses = _strict_six(final.get("statuses"), "final.statuses")
    if any(value < minimum for value, minimum in zip(angles, open_minimums)):
        raise ValueError("recovery final angles are not canonical open")
    if any(abs(value) > MAX_DISABLED_CURRENT_MA for value in currents):
        raise ValueError("recovery final disabled current is too high")
    if errors != (0,) * 6 or any(value not in IDLE_STATUSES for value in statuses):
        raise ValueError("recovery final feedback is faulted or non-idle")
    expected_speeds = (
        DEFAULT_RESET_SPEEDS
        if route == RECOVERY_ROUTE_NEAR_OPEN_RESET
        else tuple(context.recovery_plan.original_speeds)
    )
    expected_forces = (
        DEFAULT_RESET_FORCES
        if route == RECOVERY_ROUTE_NEAR_OPEN_RESET
        else tuple(context.recovery_plan.original_forces)
    )
    expected_restore_target = (
        "reset_defaults"
        if route == RECOVERY_ROUTE_NEAR_OPEN_RESET
        else "failed_run_snapshot"
    )
    if (
        final.get("original_settings_restored") is not True
        or final.get("settings_restore_target") != expected_restore_target
    ):
        raise ValueError("recovery settings restore proof differs from its route")
    snapshot = final.get("snapshot_after_disable")
    if not isinstance(snapshot, Mapping):
        raise ValueError("recovery has no post-disable snapshot")
    if _strict_six(snapshot.get("angle_targets"), "snapshot.angle_targets", allow_disabled=True) != (-1,) * 6:
        raise ValueError("post-disable snapshot is not all-six disabled")
    if any(
        value < minimum
        for value, minimum in zip(
            _strict_six(snapshot.get("angles"), "snapshot.angles"),
            open_minimums,
        )
    ):
        raise ValueError("post-disable snapshot is not all-six open")
    if _strict_six(snapshot.get("errors"), "snapshot.errors") != (0,) * 6:
        raise ValueError("post-disable snapshot contains a device error")
    if any(
        abs(value) > MAX_DISABLED_CURRENT_MA
        for value in _strict_six_signed(snapshot.get("currents"), "snapshot.currents")
    ):
        raise ValueError("post-disable snapshot current is too high")
    if any(
        value not in IDLE_STATUSES
        for value in _strict_six(snapshot.get("statuses"), "snapshot.statuses")
    ):
        raise ValueError("post-disable snapshot is not idle")
    if tuple(snapshot.get("speeds", ())) != expected_speeds:
        raise ValueError("post-disable speeds differ from the route restore policy")
    if tuple(snapshot.get("force_limits", ())) != expected_forces:
        raise ValueError("post-disable forces differ from the route restore policy")
    _assert_file_binding(runtime_binding)
    return evidence


def _strict_six_signed(value: Any, name: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError("{} must be a six-integer JSON array".format(name))
    output = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or not -5000 <= item <= 5000:
            raise ValueError("{}[{}] is not a signed register".format(name, index))
        output.append(int(item))
    return tuple(output)


def _strict_fresh_stage1_evidence(
    evidence_path: Path, context: WorkflowContext
) -> Mapping[str, Any]:
    """Prove the newly generated q6=900 prerequisite against current code."""

    source = normalize_unfollowed_leaf(evidence_path)
    runtime_binding = _require_readonly_regular_output(
        source, "fresh Stage-1 evidence"
    )
    prerequisite = build_stage1_prerequisite_binding(
        source,
        expected_config=context.base_config,
        expected_config_path=context.base_config_path,
    )
    _assert_file_binding(runtime_binding)
    if prerequisite.get("path") != str(source):
        raise ValueError("fresh Stage-1 binding points to another file")
    if prerequisite.get("file_sha256") != runtime_binding.sha256:
        raise ValueError("fresh Stage-1 binding file hash differs")
    if prerequisite.get("target_q6") != 900:
        raise ValueError("fresh Stage-1 target is not q6=900")
    _assert_file_binding(runtime_binding)
    return prerequisite


def _strict_stage2_evidence(
    evidence_path: Path, context: WorkflowContext
) -> Mapping[str, Any]:
    source = normalize_unfollowed_leaf(evidence_path)
    runtime_binding = _require_readonly_regular_output(source, "Stage-2 evidence")
    verification = verify_evidence(
        source,
        config_path=context.base_config_path,
        require_coupled=True,
    )
    _assert_file_binding(runtime_binding)
    if not verification.passed:
        raise ValueError(
            "Stage-2 evidence verification failed: " + "; ".join(verification.blockers)
        )
    evidence, resolved = load_evidence(source)
    _assert_file_binding(runtime_binding)
    if resolved != source:
        raise ValueError("Stage-2 evidence path changed while being loaded")
    binding = evidence.get("snapshot_candidate")
    candidate = binding.get("candidate") if isinstance(binding, Mapping) else None
    if not isinstance(binding, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("Stage-2 output lost its snapshot candidate binding")
    if Path(str(binding.get("path", ""))).expanduser().resolve() != context.snapshot_path:
        raise ValueError("Stage-2 output binds a different snapshot")
    snapshot_input = next(
        item for item in context.immutable_inputs if item.name == "official_snapshot"
    )
    if binding.get("file_sha256") != snapshot_input.sha256:
        raise ValueError("Stage-2 output binds a different snapshot hash")
    if candidate.get("index") != context.candidate_index:
        raise ValueError("Stage-2 output binds a different candidate")
    score = candidate.get("official_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or float(score) != context.candidate_score
    ):
        raise ValueError("Stage-2 output binds a different candidate score")
    if candidate.get("hand_targets") != list(context.hand_targets):
        raise ValueError("Stage-2 output binds different hand targets")
    stage1 = evidence.get("stage1_prerequisite")
    if not isinstance(stage1, Mapping):
        raise ValueError("Stage-2 output lost its Stage-1 prerequisite")
    if (
        Path(str(stage1.get("path", ""))).expanduser().resolve()
        != context.fresh_stage1_output
    ):
        raise ValueError("Stage-2 output binds a different Stage-1 prerequisite")
    stage1_input = _require_readonly_regular_output(
        context.fresh_stage1_output, "fresh Stage-1 evidence"
    )
    if stage1.get("file_sha256") != stage1_input.sha256:
        raise ValueError("Stage-2 output binds a different Stage-1 hash")
    _assert_file_binding(stage1_input)
    request = evidence.get("request")
    if not isinstance(request, Mapping):
        raise ValueError("Stage-2 output lost its request binding")
    if request.get("target_source") != "official_snapshot_candidate":
        raise ValueError("Stage-2 output target source is not the official candidate")
    if request.get("step_units") != context.q6_step:
        raise ValueError("Stage-2 output uses a different q6 step")
    if request.get("coupled_targets") != list(context.hand_targets):
        raise ValueError("Stage-2 output request binds different coupled targets")
    if evidence.get("result", {}).get("status") != "pass":
        raise ValueError("Stage-2 output is not a PASS")
    _assert_file_binding(stage1_input)
    _assert_file_binding(runtime_binding)
    return evidence


def _strict_applied_profile(
    evidence_path: Path, profile_path: Path, context: WorkflowContext
) -> Mapping[str, Any]:
    evidence_binding = _require_readonly_regular_output(
        evidence_path, "Stage-2 evidence"
    )
    profile_binding = _require_readonly_regular_output(
        profile_path, "derived profile"
    )
    result = verify_applied_config(
        evidence_path,
        profile_path,
        require_coupled=True,
    )
    _assert_file_binding(evidence_binding)
    _assert_file_binding(profile_binding)
    if not result.passed:
        raise ValueError(
            "derived profile verification failed: " + "; ".join(result.blockers)
        )
    resolved = normalize_unfollowed_leaf(profile_path)
    if resolved.parent != context.base_config_path.parent:
        raise ValueError("derived profile is not beside the base profile")
    return {"proposal": copy.deepcopy(result.proposal)}


def _run_child(stage: str, command: Sequence[str]) -> ChildResult:
    argv = [str(value) for value in command]
    print("[workflow:{}] {}".format(stage, " ".join(argv)), flush=True)
    process = subprocess.Popen(
        argv,
        cwd=str(ROOT),
        start_new_session=True,
    )
    interrupted = False
    while True:
        try:
            returncode = int(process.wait())
            return ChildResult(returncode=returncode, interrupted=interrupted)
        except KeyboardInterrupt:
            interrupted = True
            print(
                "[workflow:{}] Ctrl+C received; forwarding SIGINT and waiting "
                "for child cleanup (no kill escalation)".format(stage),
                file=sys.stderr,
                flush=True,
            )
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass


def _stage_record(
    name: str,
    command: Sequence[str],
    *,
    started_at: str,
    completed_at: str,
    returncode: Optional[int],
    interrupted: bool,
    error: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "argv": [str(value) for value in command],
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "returncode": returncode,
        "interrupted": bool(interrupted),
        "error": error,
    }


def _execute_stage(
    name: str,
    command: Sequence[str],
    stages: list[dict[str, Any]],
    runner: ChildRunner,
) -> None:
    started = _utc_now()
    try:
        result = runner(name, tuple(str(value) for value in command))
    except KeyboardInterrupt as exc:
        stages.append(
            _stage_record(
                name,
                command,
                started_at=started,
                completed_at=_utc_now(),
                returncode=130,
                interrupted=True,
                error="KeyboardInterrupt",
            )
        )
        raise WorkflowInterrupted(name, 130, "workflow interrupted") from exc
    except BaseException as exc:
        stages.append(
            _stage_record(
                name,
                command,
                started_at=started,
                completed_at=_utc_now(),
                returncode=None,
                interrupted=False,
                error="{}: {}".format(type(exc).__name__, exc),
            )
        )
        raise
    if not isinstance(result, ChildResult):
        raise TypeError("child runner must return ChildResult")
    stages.append(
        _stage_record(
            name,
            command,
            started_at=started,
            completed_at=_utc_now(),
            returncode=int(result.returncode),
            interrupted=bool(result.interrupted),
        )
    )
    if result.interrupted:
        raise WorkflowInterrupted(
            name,
            130,
            "{} was interrupted after child cleanup".format(name),
        )
    if int(result.returncode) != 0:
        raise StageFailure(
            name,
            int(result.returncode),
            "{} failed with exit {}".format(name, result.returncode),
        )


def _optional_output_binding(name: str, path: Path) -> dict[str, Any]:
    target = normalize_unfollowed_leaf(path)
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return {
            "name": name,
            "path": str(target),
            "exists": False,
            "safe_regular_file": False,
            "sha256": None,
            "read_only": False,
        }
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        return {
            "name": name,
            "path": str(target),
            "exists": True,
            "safe_regular_file": False,
            "sha256": None,
            "read_only": not bool(metadata.st_mode & 0o222),
        }
    binding = _file_binding(name, target)
    return {
        "name": name,
        "path": str(target),
        "exists": True,
        "safe_regular_file": True,
        "sha256": binding.sha256,
        "read_only": binding.mode == 0o444,
        "identity": binding.as_json()["identity"],
    }


def _workflow_source_bindings(context: WorkflowContext) -> list[dict[str, Any]]:
    names = set(_workflow_source_paths())
    return [
        binding.as_json()
        for binding in context.immutable_inputs
        if binding.name in names
    ]


def _recovery_runtime_source_bindings(
    context: WorkflowContext,
) -> list[dict[str, Any]]:
    return [binding.as_json() for binding in context.recovery_runtime_sources]


def _recovery_execution_receipt_payload(
    evidence: Optional[Mapping[str, Any]],
    *,
    require_pass: bool,
) -> dict[str, Any]:
    request = evidence.get("request") if isinstance(evidence, Mapping) else None
    result = evidence.get("result") if isinstance(evidence, Mapping) else None
    request = request if isinstance(request, Mapping) else {}
    result = result if isinstance(result, Mapping) else {}
    telemetry = evidence.get("telemetry") if isinstance(evidence, Mapping) else None
    reset_preflights = (
        [
            sample
            for sample in telemetry
            if isinstance(sample, Mapping)
            and sample.get("phase") == "reset_open_preflight"
        ]
        if isinstance(telemetry, list)
        else []
    )
    reset_preflight_q6 = (
        reset_preflights[0].get("angles", [None] * 6)[5]
        if len(reset_preflights) == 1
        and isinstance(reset_preflights[0].get("angles"), list)
        and len(reset_preflights[0]["angles"]) == 6
        else None
    )
    payload = {
        "route": result.get("recovery_route"),
        "evidence_bound_path_executed": result.get(
            "evidence_bound_path_executed"
        ),
        "allowed_recovery_routes": request.get("allowed_recovery_routes"),
        "route_dispatch_initial_q6": request.get(
            "route_dispatch_initial_q6"
        ),
        "reset_preflight_initial_q6": reset_preflight_q6,
        "executed_q6_return_waypoints": (
            evidence.get("executed_q6_return_waypoints")
            if isinstance(evidence, Mapping)
            else None
        ),
        "executed_reset_q6_waypoints": (
            evidence.get("executed_reset_q6_waypoints")
            if isinstance(evidence, Mapping)
            else None
        ),
    }
    if require_pass:
        route = payload["route"]
        allowed = payload["allowed_recovery_routes"]
        if (
            not isinstance(route, str)
            or not isinstance(allowed, list)
            or route not in allowed
            or type(payload["evidence_bound_path_executed"]) is not bool
            or type(payload["route_dispatch_initial_q6"]) is not int
            or (
                route == RECOVERY_ROUTE_NEAR_OPEN_RESET
                and type(payload["reset_preflight_initial_q6"]) is not int
            )
            or (
                route != RECOVERY_ROUTE_NEAR_OPEN_RESET
                and payload["reset_preflight_initial_q6"] is not None
            )
            or not isinstance(payload["executed_q6_return_waypoints"], list)
            or not isinstance(payload["executed_reset_q6_waypoints"], list)
        ):
            raise ValueError("validated recovery cannot form a PASS receipt route")
    return payload


def _write_receipt(
    context: WorkflowContext,
    args: argparse.Namespace,
    *,
    output_identity: tuple[Path, int, int, int],
    started_at: str,
    stages: Sequence[Mapping[str, Any]],
    status: str,
    failed_stage: Optional[str],
    error: Optional[str],
    recovery_evidence: Optional[Mapping[str, Any]] = None,
) -> Path:
    payload = seal_evidence(
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "kind": RECEIPT_KIND,
            "workflow_id": context.workflow_id,
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "motion_authorized": False,
            "receipt_is_motion_authority": False,
            "inputs": [binding.as_json() for binding in context.immutable_inputs],
            "historical_commissioning_sources": [
                dict(item) for item in context.historical_commissioning_sources
            ],
            "recovery_runtime_sources": _recovery_runtime_source_bindings(context),
            "selection": {
                "source": "failed_stage2_snapshot_candidate_binding",
                "candidate_index": context.candidate_index,
                "official_score": context.candidate_score,
                "hand_targets": list(context.hand_targets),
                "q6_step": context.q6_step,
                "manual_target_or_candidate_override_allowed": False,
            },
            "recovery_plan_binding": context.recovery_plan.as_binding(),
            "recovery_execution": _recovery_execution_receipt_payload(
                recovery_evidence,
                require_pass=status == "pass",
            ),
            "operator_confirmations": {
                "installed_on_fr3": args.confirm_installed == INSTALLED_TOKEN,
                "24v_cutoff_ready": args.confirm_24v_cutoff == POWER_TOKEN,
                "franka_stop_ready": args.confirm_franka_stop == STOP_TOKEN,
                "workspace_clear": args.confirm_workspace_clear == CLEAR_TOKEN,
                "no_contact_PLA_scope": args.confirm_no_contact == NO_CONTACT_TOKEN,
                "interrupted_recovery": args.confirm_recovery == RECOVERY_TOKEN,
                "wide_q6": args.confirm_wide_q6 == WIDE_Q6_TOKEN,
                "coupled_closure": args.confirm_coupled_closure == COUPLED_TOKEN,
                "exact_air_target": (
                    args.confirm_exact_air_target == EXACT_AIR_TARGET_TOKEN
                ),
            },
            "stages": [dict(item) for item in stages],
            "outputs": [
                _optional_output_binding("recovery_evidence", context.recovery_output),
                _optional_output_binding(
                    "stage1_evidence", context.fresh_stage1_output
                ),
                _optional_output_binding("stage2_evidence", context.stage2_output),
                _optional_output_binding(
                    "materialized_staging_profile",
                    context.output_dir / "materialized_profile.json",
                ),
                _optional_output_binding("derived_profile", context.derived_profile_path),
            ],
            "workflow_sources": _workflow_source_bindings(context),
            "result": {
                "status": status,
                "failed_stage": failed_stage,
                "error": error,
                "final_profile_verified": status == "pass",
                "profile_path": (
                    str(context.derived_profile_path) if status == "pass" else None
                ),
            },
        }
    )
    _write_exclusive_readonly_json(
        context.receipt_output,
        payload,
        parent_identity=output_identity,
    )
    _assert_directory_identity(output_identity, "workflow output directory")
    if context.receipt_output.stat().st_mode & 0o222:
        raise RuntimeError("workflow receipt is not read-only")
    return context.receipt_output


def _write_exclusive_readonly_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    parent_identity: Optional[tuple[Path, int, int, int]] = None,
) -> None:
    """Create one final receipt with O_EXCL and fixed 0444 permissions."""

    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_exclusive_readonly_bytes(
        path, encoded, parent_identity=parent_identity
    )


def _write_exclusive_readonly_bytes(
    path: Path,
    encoded: bytes,
    *,
    parent_identity: Optional[tuple[Path, int, int, int]] = None,
) -> FileBinding:
    if parent_identity is None:
        output = normalize_unfollowed_leaf(path)
    else:
        raw = Path(path).expanduser()
        if not raw.is_absolute():
            raw = Path.cwd() / raw
        if raw.name in ("", ".", "..") or raw.parent != parent_identity[0]:
            raise ValueError("exclusive output is not inside its bound directory")
        output = raw
    output.parent.mkdir(parents=True, exist_ok=True)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(str(output.parent), directory_flags)
    directory_metadata = os.fstat(directory_fd)
    if parent_identity is not None and (
        int(directory_metadata.st_dev),
        int(directory_metadata.st_ino),
        stat.S_IMODE(directory_metadata.st_mode),
    ) != parent_identity[1:]:
        os.close(directory_fd)
        raise RuntimeError("exclusive output parent directory identity changed")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    created = False
    created_identity: Optional[tuple[int, int]] = None
    try:
        descriptor = os.open(output.name, flags, 0o444, dir_fd=directory_fd)
        created = True
        metadata = os.fstat(descriptor)
        created_identity = (int(metadata.st_dev), int(metadata.st_ino))
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating workflow receipt")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    except BaseException:
        try:
            if descriptor is not None:
                os.close(descriptor)
                descriptor = None
            if created and created_identity is not None:
                try:
                    current = os.stat(
                        output.name, dir_fd=directory_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    pass
                else:
                    if (
                        not stat.S_ISLNK(current.st_mode)
                        and (int(current.st_dev), int(current.st_ino))
                        == created_identity
                    ):
                        os.unlink(output.name, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    binding = _require_readonly_regular_output(output, "exclusive output")
    if binding.sha256 != hashlib.sha256(encoded).hexdigest():
        raise RuntimeError("exclusive output content changed after creation")
    if created_identity != (binding.device, binding.inode):
        raise RuntimeError("exclusive output identity changed after creation")
    return binding


def _publish_readonly_copy(source: Path, destination: Path) -> FileBinding:
    source_binding = _require_readonly_regular_output(source, "materialized staging profile")
    opened, descriptor, metadata = _open_regular_nofollow(
        source, "materialized staging profile"
    )
    del opened
    try:
        chunks = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
    finally:
        os.close(descriptor)
    if (
        int(metadata.st_dev) != source_binding.device
        or int(metadata.st_ino) != source_binding.inode
        or digest.hexdigest() != source_binding.sha256
    ):
        raise RuntimeError("materialized staging profile changed while being read")
    _assert_file_binding(source_binding)
    published = _write_exclusive_readonly_bytes(destination, b"".join(chunks))
    if published.sha256 != source_binding.sha256:
        raise RuntimeError("published derived profile differs from materialized staging")
    _assert_file_binding(source_binding)
    return published


def _recovery_command(args: argparse.Namespace, context: WorkflowContext) -> list[str]:
    del args
    return [
        str(ROOT / "scripts/recover_installed_rh56.sh"),
        "run",
        "--config",
        str(context.base_config_path),
        "--failed-evidence",
        str(context.failed_stage2),
        "--output",
        str(context.recovery_output),
        "--confirm-installed",
        INSTALLED_TOKEN,
        "--confirm-24v-cutoff",
        POWER_TOKEN,
        "--confirm-franka-stop",
        STOP_TOKEN,
        "--confirm-workspace-clear",
        CLEAR_TOKEN,
        "--confirm-no-contact",
        NO_CONTACT_TOKEN,
        "--confirm-recovery",
        RECOVERY_TOKEN,
    ]


def _fresh_stage1_command(
    args: argparse.Namespace, context: WorkflowContext
) -> list[str]:
    del args
    return [
        str(ROOT / "scripts/commission_installed_rh56.sh"),
        "run",
        "--config",
        str(context.base_config_path),
        "--output",
        str(context.fresh_stage1_output),
        "--target-q6",
        "900",
        "--q6-step",
        str(context.q6_step),
        "--confirm-installed",
        INSTALLED_TOKEN,
        "--confirm-24v-cutoff",
        POWER_TOKEN,
        "--confirm-franka-stop",
        STOP_TOKEN,
        "--confirm-workspace-clear",
        CLEAR_TOKEN,
        "--confirm-no-contact",
        NO_CONTACT_TOKEN,
        "--confirm-wide-q6",
        WIDE_Q6_TOKEN,
    ]


def _stage2_command(args: argparse.Namespace, context: WorkflowContext) -> list[str]:
    del args
    return [
        str(ROOT / "scripts/commission_installed_rh56.sh"),
        "run",
        "--config",
        str(context.base_config_path),
        "--output",
        str(context.stage2_output),
        "--snapshot",
        str(context.snapshot_path),
        "--candidate-index",
        str(context.candidate_index),
        "--stage1-evidence",
        str(context.fresh_stage1_output),
        "--q6-step",
        str(context.q6_step),
        "--coupled-air-close",
        "--confirm-installed",
        INSTALLED_TOKEN,
        "--confirm-24v-cutoff",
        POWER_TOKEN,
        "--confirm-franka-stop",
        STOP_TOKEN,
        "--confirm-workspace-clear",
        CLEAR_TOKEN,
        "--confirm-no-contact",
        NO_CONTACT_TOKEN,
        "--confirm-wide-q6",
        WIDE_Q6_TOKEN,
        "--confirm-coupled-closure",
        COUPLED_TOKEN,
        "--confirm-exact-air-target",
        EXACT_AIR_TARGET_TOKEN,
    ]


def run_workflow(
    args: argparse.Namespace,
    *,
    runner: ChildRunner = _run_child,
    context_loader: ContextLoader = _derive_context,
    recovery_validator: RecoveryValidator = _strict_recovery_evidence,
    stage1_validator: Stage1Validator = _strict_fresh_stage1_evidence,
    stage2_validator: Stage2Validator = _strict_stage2_evidence,
    applied_validator: AppliedValidator = _strict_applied_profile,
) -> int:
    _require_tokens(args)
    context = context_loader(args.failed_stage2, args.output_dir)
    _validate_context_layout(context)
    _reject_existing_leaf(context.output_dir, "workflow output directory")
    _reject_existing_leaf(context.derived_profile_path, "derived profile")
    context.output_dir.mkdir(parents=True, exist_ok=False)
    output_identity = _directory_identity(
        context.output_dir, "workflow output directory"
    )
    for path, name in (
        (context.recovery_output, "recovery evidence"),
        (context.fresh_stage1_output, "fresh Stage-1 evidence"),
        (context.stage2_output, "Stage-2 evidence"),
        (context.output_dir / "materialized_profile.json", "staging profile"),
        (context.receipt_output, "workflow receipt"),
    ):
        _reject_existing_leaf(path, name)
    started = _utc_now()
    stages: list[dict[str, Any]] = []
    sealed_outputs: list[FileBinding] = []
    validated_recovery: Optional[Mapping[str, Any]] = None
    status = "fail"
    failed_stage: Optional[str] = None
    error: Optional[str] = None
    returncode = 1
    try:
        _assert_runtime_state(context, output_identity)
        _execute_stage(
            "recovery",
            _recovery_command(args, context),
            stages,
            runner,
        )
        _assert_runtime_state(context, output_identity)
        recovery_binding = _require_readonly_regular_output(
            context.recovery_output, "recovery evidence"
        )
        validated_recovery = recovery_validator(context.recovery_output, context)
        _assert_runtime_state(context, output_identity)
        _assert_file_binding(recovery_binding)
        sealed_outputs.append(recovery_binding)

        _execute_stage(
            "fresh_stage1",
            _fresh_stage1_command(args, context),
            stages,
            runner,
        )
        _assert_runtime_state(context, output_identity)
        fresh_stage1_binding = _require_readonly_regular_output(
            context.fresh_stage1_output, "fresh Stage-1 evidence"
        )
        verify_stage1_command = [
            str(ROOT / "scripts/commission_installed_rh56.sh"),
            "verify",
            "--evidence",
            str(context.fresh_stage1_output),
            "--config",
            str(context.base_config_path),
        ]
        _execute_stage(
            "verify_fresh_stage1", verify_stage1_command, stages, runner
        )
        _assert_runtime_state(context, output_identity)
        _assert_file_binding(fresh_stage1_binding)
        stage1_validator(context.fresh_stage1_output, context)
        _assert_runtime_state(context, output_identity)
        _assert_file_binding(fresh_stage1_binding)
        sealed_outputs.append(fresh_stage1_binding)

        _execute_stage(
            "stage2",
            _stage2_command(args, context),
            stages,
            runner,
        )
        _assert_runtime_state(context, output_identity)
        _assert_file_binding(fresh_stage1_binding)
        stage2_binding = _require_readonly_regular_output(
            context.stage2_output, "Stage-2 evidence"
        )
        verify_command = [
            str(ROOT / "scripts/commission_installed_rh56.sh"),
            "verify",
            "--evidence",
            str(context.stage2_output),
            "--config",
            str(context.base_config_path),
            "--require-coupled",
        ]
        _execute_stage("verify_stage2", verify_command, stages, runner)
        _assert_runtime_state(context, output_identity)
        _assert_file_binding(fresh_stage1_binding)
        _assert_file_binding(stage2_binding)
        stage2_validator(context.stage2_output, context)
        _assert_runtime_state(context, output_identity)
        _assert_file_binding(fresh_stage1_binding)
        _assert_file_binding(stage2_binding)
        sealed_outputs.append(stage2_binding)
        _assert_runtime_state(context, output_identity)

        materialized_staging = context.output_dir / "materialized_profile.json"
        _reject_existing_leaf(materialized_staging, "staging profile")
        _reject_existing_leaf(context.derived_profile_path, "derived profile")
        materialize_command = [
            str(ROOT / "scripts/commission_installed_rh56.sh"),
            "materialize-config-update",
            "--evidence",
            str(context.stage2_output),
            "--config",
            str(context.base_config_path),
            "--output-config",
            str(materialized_staging),
        ]
        _execute_stage("materialize_profile", materialize_command, stages, runner)
        _assert_runtime_state(context, output_identity)
        _assert_file_binding(fresh_stage1_binding)
        staging_binding = _require_readonly_regular_output(
            materialized_staging, "materialized staging profile"
        )
        derived_binding = _publish_readonly_copy(
            materialized_staging, context.derived_profile_path
        )
        _assert_runtime_state(context, output_identity)
        _assert_file_binding(staging_binding)
        sealed_outputs.append(staging_binding)
        verify_applied_command = [
            str(ROOT / "scripts/commission_installed_rh56.sh"),
            "verify-applied",
            "--evidence",
            str(context.stage2_output),
            "--config",
            str(context.derived_profile_path),
            "--require-coupled",
        ]
        _execute_stage(
            "verify_applied", verify_applied_command, stages, runner
        )
        _assert_runtime_state(context, output_identity)
        _assert_file_binding(fresh_stage1_binding)
        _assert_file_binding(derived_binding)
        _assert_file_binding(stage2_binding)
        applied_validator(
            context.stage2_output, context.derived_profile_path, context
        )
        _assert_runtime_state(context, output_identity)
        _assert_file_binding(fresh_stage1_binding)
        _assert_file_binding(derived_binding)
        _assert_file_binding(stage2_binding)
        sealed_outputs.append(derived_binding)
        _assert_runtime_state(context, output_identity)
        status = "pass"
        returncode = 0
    except WorkflowInterrupted as exc:
        failed_stage = exc.stage
        error = str(exc)
        returncode = 130
    except StageFailure as exc:
        failed_stage = exc.stage
        error = str(exc)
        returncode = int(exc.returncode) if int(exc.returncode) != 0 else 1
    except BaseException as exc:
        failed_stage = stages[-1]["name"] if stages else "preflight"
        error = "{}: {}".format(type(exc).__name__, exc)
        returncode = 130 if isinstance(exc, KeyboardInterrupt) else 1

    try:
        _assert_runtime_state(context, output_identity)
        for binding in sealed_outputs:
            _assert_file_binding(binding)
    except BaseException as exc:
        integrity_error = "{}: {}".format(type(exc).__name__, exc)
        if status == "pass":
            failed_stage = "postflight_integrity"
            error = integrity_error
        else:
            error = "{}; postflight_integrity={}".format(error, integrity_error)
        status = "fail"
        returncode = 1

    receipt = _write_receipt(
        context,
        args,
        output_identity=output_identity,
        started_at=started,
        stages=stages,
        status=status,
        failed_stage=failed_stage,
        error=error,
        recovery_evidence=validated_recovery,
    )
    if status == "pass":
        print("PROFILE={}".format(context.derived_profile_path), flush=True)
        print("RECEIPT={}".format(receipt), flush=True)
    else:
        print(
            "[workflow] failed_stage={} error={}".format(failed_stage, error),
            file=sys.stderr,
            flush=True,
        )
        print("RECEIPT={}".format(receipt), file=sys.stderr, flush=True)
    return returncode


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command != "run":
            raise RuntimeError("unsupported command {}".format(args.command))
        return run_workflow(args)
    except KeyboardInterrupt:
        print(
            "[interrupted] no child is active; workflow stopped",
            file=sys.stderr,
        )
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print("[failed] {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
