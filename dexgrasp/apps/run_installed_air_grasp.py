#!/usr/bin/env python3
"""Prepare and run one immutable installed RH56 no-contact air-grasp session.

``prepare`` is the only motion-preparation entry point.  It delegates the two
existing reset commands unchanged, builds continuous telemetry before the
fresh D435 capture, reads one fresh idle Franka state, creates the schema-v2
air audit, runs the existing executor dry-run, and finally creates both the
native telemetry manifest and a sealed, non-overwritable session JSON.

``run`` replays every bound hash and freshness gate, starts the read-only live
viewer first, then runs the existing air-grasp executor in the foreground.  It
adds no contact or lift authority.  Ctrl+C is forwarded to the executor so its
reviewed stop/disable cleanup remains the sole hardware cleanup path; the
viewer is then stopped as well.

Importing this module opens no device.  ``pylibfranka`` is imported lazily only
after the reset subprocesses and all prepare confirmation gates have passed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# All imports above the prepare hardware gate are hardware-free.
from anydex_pipeline.control_config import (  # noqa: E402
    load_control_config,
    verify_adapter_assets,
)
from anydex_pipeline.host_network_preflight import (  # noqa: E402
    require_uncontended_franka_https_link,
)
from anydex_pipeline.installed_tool_audit import (  # noqa: E402
    load_installed_tool_audit,
)
from anydex_pipeline.telemetry_session_manifest import (  # noqa: E402
    canonical_json_sha256,
    load_telemetry_session_manifest,
    sha256_file,
)
from anydex_pipeline.rh56_commissioning import (  # noqa: E402
    build_stage1_prerequisite_binding,
    json_sha256,
    verify_applied_config,
    verify_evidence,
)
from anydex_pipeline.rh56_reset_open import (  # noqa: E402
    RESET_Q6_ARRIVAL_TOLERANCE_UNITS,
    RESET_Q6_DEADBAND_ESCAPE_UNITS,
    RESET_Q6_FIRST_STEP_UNITS,
    RESET_Q6_STEP_UNITS,
)
from anydex_pipeline.viewer_ready import (  # noqa: E402
    normalize_unfollowed_leaf,
    viewer_ready_is_published,
)


DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
DEFAULT_CAMERA_CONFIG = WORKSPACE / "perception/configs/d435_default.yaml"
DEFAULT_ANYDEX_ROOT = ROOT / "third_party/AnyDexGrasp"
DEFAULT_DYNAMIC_PYTHON = Path("/home/qiaoguanren/anaconda3/envs/dynamic/bin/python")
DEFAULT_READER_PYTHON_DIR = Path("/tmp/anydex-native-telemetry-viewer-py310/python")
DEFAULT_PRODUCER_PYTHON_DIR = Path("/tmp/anydex-franka-telemetry-producer-py39/python")

SESSION_SCHEMA_VERSION = 2
SESSION_ARTIFACT_TYPE = "fr3_rh56_installed_air_grasp_session"
SESSION_INTEGRITY_ALGORITHM = "sha256-canonical-json-without-integrity"
SESSION_FILENAME = "installed_air_grasp_session.json"
VIEWER_EXECUTE_TOKEN = "EXECUTE_SELECTED_GRASP"
AUDIT_DIRECTORY_NAME = "fresh_air_audit"
AUDIT_FILENAME = "installed_air_audit_v2.json"
MANIFEST_FILENAME = "continuous_telemetry.manifest.json"

COMMISSION_RECEIPT_SCHEMA_VERSION = 2
COMMISSION_RECEIPT_KIND = "recover_and_commission_rh56_workflow_receipt_v2"
COMMISSION_RECEIPT_INTEGRITY_ALGORITHM = (
    "sha256-canonical-json-without-integrity"
)
COMMISSION_RECEIPT_INPUT_NAMES = frozenset(
    (
        "failed_stage2",
        "base_config",
        "official_snapshot",
        "historical_stage1_evidence",
        "workflow_cli",
        "workflow_wrapper",
        "recovery_wrapper",
        "commission_wrapper",
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
COMMISSION_RECEIPT_SOURCE_NAMES = frozenset(
    (
        "workflow_cli",
        "workflow_wrapper",
        "recovery_wrapper",
        "commission_wrapper",
    )
)
COMMISSION_RECEIPT_RECOVERY_SOURCE_NAMES = frozenset(
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
COMMISSION_RECEIPT_HISTORICAL_SOURCE_NAMES = frozenset(
    (
        "commission_cli",
        "commission_evidence_module",
        "rh56_hand_path",
        "rh56_sequence_driver",
        "rh56_register_api",
        "franka_sequence_driver",
        "control_config_module",
        "adapter_mesh",
        "adapter_provenance",
        "actuator_to_joint_xlsx",
        "driver_to_angle_xls",
        "actuator_to_urdf_generator",
    )
)
COMMISSION_RECEIPT_OUTPUT_NAMES = frozenset(
    (
        "recovery_evidence",
        "stage1_evidence",
        "stage2_evidence",
        "materialized_staging_profile",
        "derived_profile",
    )
)
COMMISSION_RECEIPT_STAGE_NAMES = (
    "recovery",
    "fresh_stage1",
    "verify_fresh_stage1",
    "stage2",
    "verify_stage2",
    "materialize_profile",
    "verify_applied",
)
RECOVERY_MODE_COUPLED_CLOSE_PREFIX = "interrupted_coupled_close_prefix_v1"
RECOVERY_MODE_Q6_RETURN = "interrupted_stage2_q6_return_v1"
RECOVERY_ROUTE_Q6_RETURN = "sealed_stage2_q6_return_v1"
RECOVERY_ROUTE_NEAR_OPEN_RESET = "profile_near_open_reset_v1"
RECOVERY_ROUTE_COUPLED_PREFIX = "sealed_coupled_close_prefix_v1"
RECOVERY_IDLE_STATUSES = frozenset((2,))
RECOVERY_MAX_DISABLED_CURRENT_MA = 100
RECOVERY_BEND_OPEN_MIN_ANGLE = 980
RECOVERY_DEFAULT_SPEEDS = (1000,) * 6
RECOVERY_DEFAULT_FORCES = (500,) * 6
RECOVERY_EVIDENCE_KIND = "installed_rh56_interrupted_open_recovery_v1"
RECOVERY_EVIDENCE_INTEGRITY_ALGORITHM = (
    "sha256-canonical-json-without-integrity"
)

RH56_INSTALLED_TOKEN = "RH56_INSTALLED_ON_FR3"
RH56_POWER_TOKEN = "RH56_24V_CUTOFF_READY"
FRANKA_STOP_TOKEN = "FR3_STOP_READY"
RH56_CLEAR_TOKEN = "INSTALLED_AIR_WORKSPACE_CLEAR"
NO_CONTACT_TOKEN = "PLA_LOW_SPEED_NO_CONTACT"
RH56_RESET_TOKEN = "RH56_RESET_OPEN"
HAND_OPEN_TOKEN = "RH56_OPEN_DISABLED"
DEFAULT_SWEEP_TOKEN = "FR3_RH56_CURRENT_TO_DEFAULT_SWEEP_CLEAR"
PLA_LOW_SPEED_TOKEN = "PLA_LOW_SPEED_UNLOADED_ONLY"
STATIONARY_Q_TOKEN = "CURRENT_Q_READ_ONLY_AND_STATIONARY"

EXECUTOR_WORKSPACE_TOKEN = "FR3_RH56_WORKSPACE_CLEAR"
EXECUTOR_STOP_TOKEN = "IMMEDIATE_STOP_AND_24V_CUT_READY"
AIR_GRASP_TOKEN = "FR3_RH56_AIR_GRASP_NO_CONTACT_NO_LIFT"
Q6_PRESHAPE_TOKEN = "RH56_Q6_PRESHAPE_VERIFIED"
COLLISION_MODEL_TOKEN = "INSTALLED_TOOL_COLLISIONS_VERIFIED"

PREPARE_TOKEN_CONTRACT = {
    "confirm_installed": RH56_INSTALLED_TOKEN,
    "confirm_24v_cutoff": RH56_POWER_TOKEN,
    "confirm_franka_stop": FRANKA_STOP_TOKEN,
    "confirm_rh56_workspace_clear": RH56_CLEAR_TOKEN,
    "confirm_no_contact": NO_CONTACT_TOKEN,
    "confirm_rh56_reset_open": RH56_RESET_TOKEN,
    "confirm_hand_open": HAND_OPEN_TOKEN,
    "confirm_default_sweep_clear": DEFAULT_SWEEP_TOKEN,
    "confirm_pla_low_speed": PLA_LOW_SPEED_TOKEN,
    "confirm_stationary_q": STATIONARY_Q_TOKEN,
}

RUN_TOKEN_CONTRACT = {
    "confirm_workspace_clear": EXECUTOR_WORKSPACE_TOKEN,
    "confirm_immediate_stop": EXECUTOR_STOP_TOKEN,
    "confirm_air_grasp": AIR_GRASP_TOKEN,
    "confirm_q6_preshape": Q6_PRESHAPE_TOKEN,
    "confirm_installed_collision_model": COLLISION_MODEL_TOKEN,
}


class WorkflowError(RuntimeError):
    """A workflow gate or child stage failed before the next stage began."""


class StopUnconfirmedError(WorkflowError):
    """The executor did not finish its reviewed SIGINT cleanup in time."""


class StageError(WorkflowError):
    def __init__(self, stage: str, returncode: int) -> None:
        self.stage = str(stage)
        self.returncode = int(returncode)
        super().__init__(
            "stage {!r} failed with exit {}".format(self.stage, self.returncode)
        )


@dataclass(frozen=True)
class ValidatedSession:
    path: Path
    payload: Mapping[str, Any]
    expires_at_s: float


@dataclass(frozen=True)
class StaticFileBinding:
    """One opened, regular, final-component-non-symlink file identity."""

    path: Path
    sha256: str
    device: int
    inode: int
    size: int
    mode: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class CommissionReceiptAuthority:
    """Strictly re-derived prepare inputs; never a motion authority."""

    receipt_path: Path
    receipt_sha256: str
    config_path: Path
    snapshot_path: Path
    selected_index: int
    hand_targets: Tuple[int, int, int, int, int, int]
    stage2_evidence_path: Path
    derived_profile_path: Path


def _sha256(path: Path) -> str:
    return sha256_file(Path(path).expanduser().resolve())


def _binding(path: Path) -> Mapping[str, str]:
    source = Path(path).expanduser().resolve()
    return {"path": str(source), "sha256": _sha256(source)}


def _strict_json(path: Path) -> Mapping[str, Any]:
    def reject_pairs(pairs: Sequence[Tuple[str, Any]]) -> Mapping[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key {!r}".format(key))
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError("non-finite JSON constant {!r}".format(value))

    source = normalize_unfollowed_leaf(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(source), flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("air-grasp session is not a regular file")
            if stat.S_IMODE(metadata.st_mode) != 0o444:
                raise ValueError("air-grasp session mode must be exactly 0444")
            with os.fdopen(descriptor, "r", encoding="utf-8", closefd=True) as stream:
                descriptor = -1
                serialized = stream.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        value = json.loads(
            serialized,
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("cannot load air-grasp session {}: {}".format(source, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("air-grasp session must contain one JSON object")
    return value


def _require_exact_keys(value: Any, expected: Sequence[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be an object".format(name))
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        raise ValueError(
            "{} keys differ: missing={}, unknown={}".format(
                name,
                sorted(expected_set - actual_set),
                sorted(actual_set - expected_set),
            )
        )
    return value


def _read_static_file(
    value: Any,
    name: str,
    *,
    exact_mode: Optional[int] = None,
    include_content: bool = False,
) -> Tuple[StaticFileBinding, bytes]:
    """Read a canonical regular file without following its final component."""

    if not isinstance(value, (str, Path)):
        raise ValueError("{} path must be a string".format(name))
    supplied = Path(value).expanduser()
    if not supplied.is_absolute():
        raise ValueError("{} path must be absolute".format(name))
    source = normalize_unfollowed_leaf(supplied)
    if str(source) != str(supplied):
        raise ValueError("{} path must use its canonical absolute spelling".format(name))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(source), flags)
    except OSError as exc:
        raise ValueError(
            "{} cannot be opened without following a symlink: {}".format(name, exc)
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("{} is not a regular file".format(name))
        mode = stat.S_IMODE(metadata.st_mode)
        if exact_mode is not None and mode != int(exact_mode):
            raise ValueError(
                "{} mode must be exactly {:04o}".format(name, int(exact_mode))
            )
        digest = hashlib.sha256()
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if include_content:
                chunks.append(chunk)
        final_metadata = os.fstat(descriptor)
        identity = (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
            int(metadata.st_ctime_ns),
        )
        final_identity = (
            int(final_metadata.st_dev),
            int(final_metadata.st_ino),
            int(final_metadata.st_size),
            int(final_metadata.st_mtime_ns),
            int(final_metadata.st_ctime_ns),
        )
        if final_identity != identity:
            raise ValueError("{} changed while it was being read".format(name))
        binding = StaticFileBinding(
            path=source,
            sha256=digest.hexdigest(),
            device=int(metadata.st_dev),
            inode=int(metadata.st_ino),
            size=int(metadata.st_size),
            mode=mode,
            mtime_ns=int(metadata.st_mtime_ns),
            ctime_ns=int(metadata.st_ctime_ns),
        )
        return binding, b"".join(chunks)
    finally:
        os.close(descriptor)


def _assert_static_file_unchanged(expected: StaticFileBinding, name: str) -> None:
    actual, _content = _read_static_file(
        expected.path,
        name,
        exact_mode=expected.mode,
    )
    if actual != expected:
        raise ValueError("{} changed after it was validated".format(name))


def _strict_json_bytes(content: bytes, name: str) -> Mapping[str, Any]:
    def reject_pairs(pairs: Sequence[Tuple[str, Any]]) -> Mapping[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("{} has duplicate JSON key {!r}".format(name, key))
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError("{} has non-finite JSON constant {!r}".format(name, value))

    try:
        decoded = content.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("cannot parse {}: {}".format(name, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("{} must contain one JSON object".format(name))
    return value


def _validate_receipt_identity(
    value: Any,
    name: str,
    *,
    require_mode: Optional[int] = None,
) -> StaticFileBinding:
    item = _require_exact_keys(value, ("name", "path", "sha256", "identity"), name)
    if item["name"] != name.rsplit(" ", 1)[-1]:
        raise ValueError("{} name does not match its receipt slot".format(name))
    identity = _require_exact_keys(
        item["identity"],
        ("device", "inode", "size", "mode", "mtime_ns", "ctime_ns"),
        "{} identity".format(name),
    )
    actual, _content = _read_static_file(
        item["path"],
        name,
        exact_mode=require_mode,
    )
    expected_digest = item["sha256"]
    if not isinstance(expected_digest, str) or actual.sha256 != expected_digest:
        raise ValueError("{} SHA-256 mismatch".format(name))
    expected_identity = {
        "device": actual.device,
        "inode": actual.inode,
        "size": actual.size,
        "mode": actual.mode,
        "mtime_ns": actual.mtime_ns,
        "ctime_ns": actual.ctime_ns,
    }
    if dict(identity) != expected_identity:
        raise ValueError("{} file identity mismatch".format(name))
    return actual


def _indexed_named_objects(
    value: Any,
    expected_names: frozenset[str],
    name: str,
) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != len(expected_names):
        raise ValueError("{} must contain exactly {} entries".format(name, len(expected_names)))
    indexed = {}
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError("{}[{}] must be an object".format(name, index))
        item_name = item.get("name")
        if not isinstance(item_name, str) or item_name in indexed:
            raise ValueError("{} names must be unique strings".format(name))
        indexed[item_name] = item
    if set(indexed) != set(expected_names):
        raise ValueError("{} names differ from the reviewed contract".format(name))
    return indexed


def _strict_six_targets(value: Any, name: str) -> Tuple[int, int, int, int, int, int]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError("{} must contain six actuator targets".format(name))
    targets = []
    for index, item in enumerate(value):
        if type(item) is not int or not 0 <= item <= 1000:
            raise ValueError("{}[{}] must be an integer in 0..1000".format(name, index))
        targets.append(int(item))
    return tuple(targets)  # type: ignore[return-value]


def _strict_six_signed_registers(value: Any, name: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError("{} must contain six signed registers".format(name))
    registers = []
    for index, item in enumerate(value):
        if type(item) is not int or not -5000 <= item <= 5000:
            raise ValueError(
                "{}[{}] must be an integer in -5000..5000".format(name, index)
            )
        registers.append(int(item))
    return tuple(registers)


def _recovery_open_feedback_policy(
    config: Mapping[str, Any],
) -> tuple[Tuple[int, ...], Tuple[int, int]]:
    inspire = config.get("inspire")
    if not isinstance(inspire, Mapping):
        raise ValueError("commission receipt recovery profile has no inspire policy")
    targets = inspire.get("open_targets")
    validated_range = inspire.get("thumb_rotate_validated_realtime_range")
    tolerance = inspire.get("arrival_tolerance_units")
    if (
        not isinstance(targets, list)
        or len(targets) != 6
        or any(type(value) is not int or not 0 <= value <= 1000 for value in targets)
        or targets != [1000] * 6
    ):
        raise ValueError("commission receipt recovery open targets are malformed")
    if (
        not isinstance(validated_range, list)
        or len(validated_range) != 2
        or any(
            type(value) is not int or not 0 <= value <= 1000
            for value in validated_range
        )
        or validated_range[0] > validated_range[1]
    ):
        raise ValueError("commission receipt recovery q6 range is malformed")
    if type(tolerance) is not int or not 0 <= tolerance <= 30:
        raise ValueError("commission receipt recovery tolerance is malformed")
    if targets[5] != validated_range[1]:
        raise ValueError("commission receipt q6 open target is not the range endpoint")
    q6_minimum = targets[5] - tolerance
    if not validated_range[0] <= q6_minimum <= validated_range[1]:
        raise ValueError("commission receipt q6 open floor is outside its range")
    return (
        (RECOVERY_BEND_OPEN_MIN_ANGLE,) * 5 + (q6_minimum,),
        (validated_range[0], validated_range[1]),
    )


def _receipt_expected_recovery_routes(
    recovery_plan: Mapping[str, Any],
) -> Tuple[str, ...]:
    mode = recovery_plan.get("recovery_mode")
    if mode == RECOVERY_MODE_Q6_RETURN:
        return (RECOVERY_ROUTE_Q6_RETURN, RECOVERY_ROUTE_NEAR_OPEN_RESET)
    if mode == RECOVERY_MODE_COUPLED_CLOSE_PREFIX:
        return (RECOVERY_ROUTE_COUPLED_PREFIX,)
    raise ValueError("commission receipt recovery mode has no reviewed route")


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


def _validate_recovery_execution_contract(
    *,
    evidence: Mapping[str, Any],
    receipt_binding: Mapping[str, Any],
    recovery_plan: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    """Revalidate route choice, final feedback, and route-specific restore."""

    if evidence.get("schema_version") != 1:
        raise ValueError("recovery evidence schema_version is unsupported")
    if evidence.get("kind") != RECOVERY_EVIDENCE_KIND:
        raise ValueError("recovery evidence kind is unsupported")
    integrity = _require_exact_keys(
        evidence.get("integrity"),
        ("algorithm", "payload_sha256"),
        "recovery evidence integrity",
    )
    if integrity["algorithm"] != RECOVERY_EVIDENCE_INTEGRITY_ALGORITHM:
        raise ValueError("recovery evidence integrity algorithm is unsupported")
    unsigned_evidence = {
        key: value for key, value in evidence.items() if key != "integrity"
    }
    if integrity["payload_sha256"] != json_sha256(unsigned_evidence):
        raise ValueError("recovery evidence canonical payload SHA-256 mismatch")
    if evidence.get("failed_commissioning_binding") != recovery_plan:
        raise ValueError("recovery evidence binds another interrupted plan")

    receipt_route = _require_exact_keys(
        receipt_binding,
        (
            "route",
            "evidence_bound_path_executed",
            "allowed_recovery_routes",
            "route_dispatch_initial_q6",
            "reset_preflight_initial_q6",
            "executed_q6_return_waypoints",
            "executed_reset_q6_waypoints",
        ),
        "commission receipt recovery_execution",
    )
    request = evidence.get("request")
    result = evidence.get("result")
    final = evidence.get("final")
    if (
        not isinstance(request, Mapping)
        or not isinstance(result, Mapping)
        or not isinstance(final, Mapping)
    ):
        raise ValueError("recovery evidence route/result/final binding is missing")
    expected_routes = _receipt_expected_recovery_routes(recovery_plan)
    route = request.get("selected_recovery_route")
    historical_path_executed = result.get("evidence_bound_path_executed")
    if request.get("allowed_recovery_routes") != list(expected_routes):
        raise ValueError("recovery evidence allowed route list differs")
    if (
        not isinstance(route, str)
        or route not in expected_routes
        or result.get("recovery_route") != route
    ):
        raise ValueError("recovery evidence selected route is unsupported or inconsistent")
    expected_historical_path = route != RECOVERY_ROUTE_NEAR_OPEN_RESET
    if historical_path_executed is not expected_historical_path:
        raise ValueError("recovery evidence misstates sealed-path execution")
    minimums, q6_range = _recovery_open_feedback_policy(config)
    if recovery_plan.get("allowed_recovery_routes") != list(expected_routes):
        raise ValueError("commission receipt recovery plan route list differs")
    if recovery_plan.get("profile_near_open_q6_range") != list(q6_range):
        raise ValueError("commission receipt recovery plan near-open range differs")
    if recovery_plan.get("q6_open_min_angle") != minimums[5]:
        raise ValueError("commission receipt recovery plan q6 open floor differs")
    if request.get("profile_near_open_q6_range") != list(q6_range):
        raise ValueError("recovery evidence near-open range differs from its profile")
    if request.get("q6_open_min_angle") != minimums[5]:
        raise ValueError("recovery evidence q6 open threshold differs from its profile")
    route_initial_q6 = result.get("recovery_route_initial_q6")
    if (
        type(route_initial_q6) is not int
        or request.get("route_dispatch_initial_q6") != route_initial_q6
        or receipt_route["route_dispatch_initial_q6"] != route_initial_q6
    ):
        raise ValueError("recovery route dispatch q6 binding is inconsistent")
    route_dispatch = evidence.get("route_dispatch")
    if not isinstance(route_dispatch, Mapping):
        raise ValueError("recovery route dispatch boundary sample is missing")
    dispatch_targets = _strict_six_signed_registers(
        route_dispatch.get("angle_targets"), "recovery dispatch angle_targets"
    )
    dispatch_angles = _strict_six_targets(
        route_dispatch.get("angles"), "recovery dispatch angles"
    )
    dispatch_currents = _strict_six_signed_registers(
        route_dispatch.get("currents"), "recovery dispatch currents"
    )
    dispatch_errors = _strict_six_targets(
        route_dispatch.get("errors"), "recovery dispatch errors"
    )
    dispatch_statuses = _strict_six_targets(
        route_dispatch.get("statuses"), "recovery dispatch statuses"
    )
    dispatch_temperatures = _strict_six_targets(
        route_dispatch.get("temperatures"), "recovery dispatch temperatures"
    )
    if (
        route_dispatch.get("phase") != "boundary_state_snapshot"
        or dispatch_targets != (-1,) * 6
        or dispatch_angles[5] != route_initial_q6
        or any(
            abs(value) > RECOVERY_MAX_DISABLED_CURRENT_MA
            for value in dispatch_currents
        )
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
        raise ValueError("recovery evidence executed waypoint records are malformed")
    reset_preflight_q6 = None
    if route == RECOVERY_ROUTE_NEAR_OPEN_RESET:
        if executed_q6:
            raise ValueError("near-open recovery cannot claim the sealed q6 return path")
        reset_preflights = [
            sample
            for sample in telemetry
            if isinstance(sample, Mapping)
            and sample.get("phase") == "reset_open_preflight"
        ]
        if len(reset_preflights) != 1:
            raise ValueError("near-open recovery lacks one reset-open preflight")
        reset_preflight = reset_preflights[0]
        reset_targets = _strict_six_signed_registers(
            reset_preflight.get("angle_targets"),
            "recovery reset preflight angle_targets",
        )
        reset_angles = _strict_six_targets(
            reset_preflight.get("angles"), "recovery reset preflight angles"
        )
        reset_errors = _strict_six_targets(
            reset_preflight.get("errors"), "recovery reset preflight errors"
        )
        reset_statuses = _strict_six_targets(
            reset_preflight.get("statuses"), "recovery reset preflight statuses"
        )
        reset_preflight_q6 = reset_angles[5]
        if (
            reset_targets != (-1,) * 6
            or reset_errors != (0,) * 6
            or any(value not in (2, 0xFF) for value in reset_statuses)
            or any(
                value < RECOVERY_BEND_OPEN_MIN_ANGLE
                for value in reset_angles[:5]
            )
            or receipt_route["reset_preflight_initial_q6"]
            != reset_preflight_q6
        ):
            raise ValueError("near-open reset preflight is not disabled/open/idle")
        expected_reset = _canonical_near_open_reset_commands(
            reset_preflight_q6,
            q6_range,
            minimums[5],
        )
        if (
            not q6_range[0] <= route_initial_q6 <= q6_range[1]
            or any(
                not q6_range[0] <= value <= q6_range[1]
                for value in executed_reset
            )
            or any(
                value < RECOVERY_BEND_OPEN_MIN_ANGLE
                for value in dispatch_angles[:5]
            )
            or tuple(executed_reset) != expected_reset
        ):
            raise ValueError(
                "near-open dispatch or reset path differs from the canonical profile path"
            )
    else:
        if executed_reset:
            raise ValueError("sealed recovery cannot claim reset-driver waypoints")
        if receipt_route["reset_preflight_initial_q6"] is not None or any(
            isinstance(sample, Mapping)
            and sample.get("phase") == "reset_open_preflight"
            for sample in telemetry
        ):
            raise ValueError("sealed recovery contains reset-open proof")
    if route == RECOVERY_ROUTE_Q6_RETURN:
        sealed_range = recovery_plan.get("permitted_live_q6_range")
        if (
            not isinstance(sealed_range, list)
            or len(sealed_range) != 2
            or any(type(value) is not int for value in sealed_range)
            or not sealed_range[0] <= route_initial_q6 <= sealed_range[1]
        ):
            raise ValueError("sealed q6 dispatch is outside its evidence range")
    if route == RECOVERY_ROUTE_Q6_RETURN and (
        not executed_q6 or executed_q6[-1] != 1000
    ):
        raise ValueError("sealed q6 recovery did not record a complete return")
    if dict(receipt_route) != {
        "route": route,
        "evidence_bound_path_executed": historical_path_executed,
        "allowed_recovery_routes": list(expected_routes),
        "route_dispatch_initial_q6": route_initial_q6,
        "reset_preflight_initial_q6": reset_preflight_q6,
        "executed_q6_return_waypoints": evidence.get(
            "executed_q6_return_waypoints"
        ),
        "executed_reset_q6_waypoints": evidence.get(
            "executed_reset_q6_waypoints"
        ),
    }:
        raise ValueError("commission receipt recovery route differs from its evidence")

    if result.get("status") != "pass" or any(
        result.get(name) is not True
        for name in (
            "adopted_disabled_verified",
            "recovered_open_verified",
            "disabled_verified",
        )
    ):
        raise ValueError("recovery evidence is not an unqualified route PASS")
    if (
        result.get("operation_error") is not None
        or result.get("stop_error") is not None
        or result.get("cleanup_error") is not None
    ):
        raise ValueError("recovery evidence PASS contains an operation or stop error")
    if evidence.get("motion_authorized") is not False:
        raise ValueError("recovery evidence cannot be motion authority")
    if evidence.get("commissioning_unlock_claimed") is not False:
        raise ValueError("recovery evidence cannot claim commissioning unlock")

    targets = _strict_six_signed_registers(
        final.get("angle_targets"), "recovery final angle_targets"
    )
    angles = _strict_six_targets(final.get("angles"), "recovery final angles")
    currents = _strict_six_signed_registers(
        final.get("currents"), "recovery final currents"
    )
    errors = _strict_six_targets(final.get("errors"), "recovery final errors")
    statuses = _strict_six_targets(final.get("statuses"), "recovery final statuses")
    if targets != (-1,) * 6:
        raise ValueError("recovery final output is not all-six disabled")
    if any(value < minimum for value, minimum in zip(angles, minimums)):
        raise ValueError("recovery final feedback is not profile-open")
    if any(abs(value) > RECOVERY_MAX_DISABLED_CURRENT_MA for value in currents):
        raise ValueError("recovery final disabled current is too high")
    if errors != (0,) * 6 or any(
        value not in RECOVERY_IDLE_STATUSES for value in statuses
    ):
        raise ValueError("recovery final feedback is faulted or non-idle")

    snapshot = final.get("snapshot_after_disable")
    if not isinstance(snapshot, Mapping):
        raise ValueError("recovery final post-disable snapshot is missing")
    snapshot_targets = _strict_six_signed_registers(
        snapshot.get("angle_targets"), "recovery snapshot angle_targets"
    )
    snapshot_angles = _strict_six_targets(
        snapshot.get("angles"), "recovery snapshot angles"
    )
    snapshot_currents = _strict_six_signed_registers(
        snapshot.get("currents"), "recovery snapshot currents"
    )
    snapshot_errors = _strict_six_targets(
        snapshot.get("errors"), "recovery snapshot errors"
    )
    snapshot_statuses = _strict_six_targets(
        snapshot.get("statuses"), "recovery snapshot statuses"
    )
    if snapshot_targets != (-1,) * 6:
        raise ValueError("recovery snapshot output is not all-six disabled")
    if any(
        value < minimum for value, minimum in zip(snapshot_angles, minimums)
    ):
        raise ValueError("recovery snapshot feedback is not profile-open")
    if any(
        abs(value) > RECOVERY_MAX_DISABLED_CURRENT_MA
        for value in snapshot_currents
    ):
        raise ValueError("recovery snapshot disabled current is too high")
    if snapshot_errors != (0,) * 6 or any(
        value not in RECOVERY_IDLE_STATUSES for value in snapshot_statuses
    ):
        raise ValueError("recovery snapshot feedback is faulted or non-idle")

    expected_speeds = (
        RECOVERY_DEFAULT_SPEEDS
        if route == RECOVERY_ROUTE_NEAR_OPEN_RESET
        else tuple(recovery_plan.get("original_speeds", ()))
    )
    expected_forces = (
        RECOVERY_DEFAULT_FORCES
        if route == RECOVERY_ROUTE_NEAR_OPEN_RESET
        else tuple(recovery_plan.get("original_forces", ()))
    )
    expected_restore_target = (
        "reset_defaults"
        if route == RECOVERY_ROUTE_NEAR_OPEN_RESET
        else "failed_run_snapshot"
    )
    if (
        final.get("original_settings_restored") is not True
        or final.get("settings_restore_target") != expected_restore_target
        or tuple(snapshot.get("speeds", ())) != expected_speeds
        or tuple(snapshot.get("force_limits", ())) != expected_forces
    ):
        raise ValueError("recovery settings restore proof differs from its route")
    return route


def _validate_receipt_output(
    value: Mapping[str, Any],
    expected_name: str,
) -> StaticFileBinding:
    item = _require_exact_keys(
        value,
        (
            "name",
            "path",
            "exists",
            "safe_regular_file",
            "sha256",
            "read_only",
            "identity",
        ),
        "receipt output {}".format(expected_name),
    )
    if item["name"] != expected_name:
        raise ValueError("receipt output name differs from its slot")
    if (
        item["exists"] is not True
        or item["safe_regular_file"] is not True
        or item["read_only"] is not True
    ):
        raise ValueError("receipt output {} is not sealed read-only".format(expected_name))
    actual, _content = _read_static_file(
        item["path"],
        "receipt output {}".format(expected_name),
        exact_mode=0o444,
    )
    if item["sha256"] != actual.sha256:
        raise ValueError("receipt output {} SHA-256 mismatch".format(expected_name))
    identity = _require_exact_keys(
        item["identity"],
        ("device", "inode", "size", "mode", "mtime_ns", "ctime_ns"),
        "receipt output {} identity".format(expected_name),
    )
    if dict(identity) != {
        "device": actual.device,
        "inode": actual.inode,
        "size": actual.size,
        "mode": actual.mode,
        "mtime_ns": actual.mtime_ns,
        "ctime_ns": actual.ctime_ns,
    }:
        raise ValueError("receipt output {} file identity mismatch".format(expected_name))
    return actual


def _expected_commission_workflow_sources() -> Mapping[str, Path]:
    return {
        "workflow_cli": (ROOT / "apps/recover_and_commission_rh56.py").resolve(),
        "workflow_wrapper": (ROOT / "scripts/recover_and_commission_rh56.sh").resolve(),
        "recovery_wrapper": (ROOT / "scripts/recover_installed_rh56.sh").resolve(),
        "commission_wrapper": (ROOT / "scripts/commission_installed_rh56.sh").resolve(),
    }


def _expected_recovery_runtime_source_paths(assets: Any) -> Mapping[str, Path]:
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


def _validate_evidence_source_files(
    evidence: Mapping[str, Any],
    name: str,
) -> Mapping[str, StaticFileBinding]:
    values = evidence.get("source_bindings")
    if not isinstance(values, list) or not values:
        raise ValueError("{} source_bindings must be a non-empty array".format(name))
    names = set()
    bindings = {}
    for index, value in enumerate(values):
        item = _require_exact_keys(
            value,
            ("name", "path", "sha256"),
            "{} source_bindings[{}]".format(name, index),
        )
        source_name = item["name"]
        if not isinstance(source_name, str) or not source_name or source_name in names:
            raise ValueError("{} source binding names must be unique".format(name))
        names.add(source_name)
        binding, _content = _read_static_file(
            item["path"],
            "{} source {}".format(name, source_name),
        )
        if item["sha256"] != binding.sha256:
            raise ValueError("{} source {} SHA-256 mismatch".format(name, source_name))
        bindings[source_name] = binding
    return bindings


def load_commission_receipt_authority(path: Path) -> CommissionReceiptAuthority:
    """Strictly validate a PASS receipt and re-derive its three prepare inputs.

    The receipt is only a tamper-evident index into independently verified
    Stage-2 evidence and its exact applied profile.  It never grants motion.
    """

    receipt_binding, serialized = _read_static_file(
        Path(path).expanduser(),
        "commission receipt",
        exact_mode=0o444,
        include_content=True,
    )
    receipt = _strict_json_bytes(serialized, "commission receipt")
    root = _require_exact_keys(
        receipt,
        (
            "schema_version",
            "kind",
            "workflow_id",
            "started_at_utc",
            "completed_at_utc",
            "motion_authorized",
            "receipt_is_motion_authority",
            "inputs",
            "historical_commissioning_sources",
            "recovery_runtime_sources",
            "selection",
            "recovery_plan_binding",
            "recovery_execution",
            "operator_confirmations",
            "stages",
            "outputs",
            "workflow_sources",
            "result",
            "integrity",
        ),
        "commission receipt",
    )
    if root["schema_version"] != COMMISSION_RECEIPT_SCHEMA_VERSION:
        raise ValueError("commission receipt schema_version is unsupported")
    if root["kind"] != COMMISSION_RECEIPT_KIND:
        raise ValueError("commission receipt kind is unsupported")
    if root["motion_authorized"] is not False:
        raise ValueError("commission receipt must keep motion_authorized=false")
    if root["receipt_is_motion_authority"] is not False:
        raise ValueError("commission receipt cannot be motion authority")
    try:
        parsed_workflow_id = uuid.UUID(str(root["workflow_id"]))
    except (ValueError, AttributeError) as exc:
        raise ValueError("commission receipt workflow_id is malformed") from exc
    if str(parsed_workflow_id) != root["workflow_id"]:
        raise ValueError("commission receipt workflow_id is not canonical")
    if not isinstance(root["started_at_utc"], str) or not root["started_at_utc"]:
        raise ValueError("commission receipt start timestamp is malformed")
    if not isinstance(root["completed_at_utc"], str) or not root["completed_at_utc"]:
        raise ValueError("commission receipt completion timestamp is malformed")

    integrity = _require_exact_keys(
        root["integrity"],
        ("algorithm", "payload_sha256"),
        "commission receipt integrity",
    )
    if integrity["algorithm"] != COMMISSION_RECEIPT_INTEGRITY_ALGORITHM:
        raise ValueError("commission receipt integrity algorithm is unsupported")
    unsigned = dict(root)
    del unsigned["integrity"]
    if integrity["payload_sha256"] != json_sha256(unsigned):
        raise ValueError("commission receipt canonical payload SHA-256 mismatch")

    result = _require_exact_keys(
        root["result"],
        ("status", "failed_stage", "error", "final_profile_verified", "profile_path"),
        "commission receipt result",
    )
    if dict(result) != {
        "status": "pass",
        "failed_stage": None,
        "error": None,
        "final_profile_verified": True,
        "profile_path": result["profile_path"],
    }:
        raise ValueError("commission receipt result is not an unqualified PASS")

    stages = root["stages"]
    if not isinstance(stages, list) or tuple(
        item.get("name") if isinstance(item, Mapping) else None for item in stages
    ) != COMMISSION_RECEIPT_STAGE_NAMES:
        raise ValueError("commission receipt stage sequence differs from the reviewed workflow")
    for stage in stages:
        record = _require_exact_keys(
            stage,
            (
                "name",
                "argv",
                "started_at_utc",
                "completed_at_utc",
                "returncode",
                "interrupted",
                "error",
            ),
            "commission receipt stage",
        )
        if (
            record["returncode"] != 0
            or record["interrupted"] is not False
            or record["error"] is not None
            or not isinstance(record["argv"], list)
        ):
            raise ValueError("commission receipt contains a failed or interrupted stage")

    inputs = _indexed_named_objects(
        root["inputs"],
        COMMISSION_RECEIPT_INPUT_NAMES,
        "commission receipt inputs",
    )
    input_bindings = {
        name: _validate_receipt_identity(
            inputs[name], "receipt input {}".format(name)
        )
        for name in sorted(COMMISSION_RECEIPT_INPUT_NAMES)
    }
    historical_sources = _indexed_named_objects(
        root["historical_commissioning_sources"],
        COMMISSION_RECEIPT_HISTORICAL_SOURCE_NAMES,
        "commission receipt historical_commissioning_sources",
    )
    for name, value in historical_sources.items():
        item = _require_exact_keys(
            value,
            ("name", "path", "sha256"),
            "historical commissioning source {}".format(name),
        )
        if item["name"] != name:
            raise ValueError("historical commissioning source name differs")
        path = item["path"]
        digest = item["sha256"]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("historical commissioning source path is not absolute")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("historical commissioning source digest is malformed")

    recovery_runtime_sources = _indexed_named_objects(
        root["recovery_runtime_sources"],
        COMMISSION_RECEIPT_RECOVERY_SOURCE_NAMES,
        "commission receipt recovery_runtime_sources",
    )
    recovery_source_bindings = {
        name: _validate_receipt_identity(
            recovery_runtime_sources[name], "recovery runtime source {}".format(name)
        )
        for name in sorted(COMMISSION_RECEIPT_RECOVERY_SOURCE_NAMES)
    }
    for name, binding in recovery_source_bindings.items():
        if binding != input_bindings[name]:
            raise ValueError(
                "commission receipt recovery runtime source differs from its input binding"
            )
    workflow_sources = _indexed_named_objects(
        root["workflow_sources"],
        COMMISSION_RECEIPT_SOURCE_NAMES,
        "commission receipt workflow_sources",
    )
    source_bindings = {
        name: _validate_receipt_identity(
            workflow_sources[name], "workflow source {}".format(name)
        )
        for name in sorted(COMMISSION_RECEIPT_SOURCE_NAMES)
    }
    expected_sources = _expected_commission_workflow_sources()
    for name, expected_path in expected_sources.items():
        if source_bindings[name].path != expected_path:
            raise ValueError("commission receipt workflow source path changed: {}".format(name))
        if source_bindings[name] != input_bindings[name]:
            raise ValueError("commission receipt workflow source differs from its input binding")

    outputs = _indexed_named_objects(
        root["outputs"],
        COMMISSION_RECEIPT_OUTPUT_NAMES,
        "commission receipt outputs",
    )
    output_bindings = {
        name: _validate_receipt_output(outputs[name], name)
        for name in sorted(COMMISSION_RECEIPT_OUTPUT_NAMES)
    }

    selection = _require_exact_keys(
        root["selection"],
        (
            "source",
            "candidate_index",
            "official_score",
            "hand_targets",
            "q6_step",
            "manual_target_or_candidate_override_allowed",
        ),
        "commission receipt selection",
    )
    if selection["source"] != "failed_stage2_snapshot_candidate_binding":
        raise ValueError("commission receipt selection source is unsupported")
    candidate_index = selection["candidate_index"]
    if type(candidate_index) is not int or candidate_index < 0:
        raise ValueError("commission receipt candidate index is invalid")
    official_score = selection["official_score"]
    if (
        isinstance(official_score, bool)
        or not isinstance(official_score, (int, float))
        or not math.isfinite(float(official_score))
    ):
        raise ValueError("commission receipt candidate score is invalid")
    hand_targets = _strict_six_targets(
        selection["hand_targets"], "commission receipt hand_targets"
    )
    if type(selection["q6_step"]) is not int or not 1 <= selection["q6_step"] <= 100:
        raise ValueError("commission receipt q6 step is invalid")
    if selection["manual_target_or_candidate_override_allowed"] is not False:
        raise ValueError("commission receipt unexpectedly permits a manual override")

    receipt_confirmations = _require_exact_keys(
        root["operator_confirmations"],
        (
            "installed_on_fr3",
            "24v_cutoff_ready",
            "franka_stop_ready",
            "workspace_clear",
            "no_contact_PLA_scope",
            "interrupted_recovery",
            "wide_q6",
            "coupled_closure",
            "exact_air_target",
        ),
        "commission receipt confirmations",
    )
    if any(value is not True for value in receipt_confirmations.values()):
        raise ValueError("commission receipt lost a required commissioning confirmation")

    recovery_plan = root["recovery_plan_binding"]
    if not isinstance(recovery_plan, Mapping):
        raise ValueError("commission receipt recovery plan binding is malformed")
    if recovery_plan.get("path") != str(input_bindings["failed_stage2"].path):
        raise ValueError("commission receipt recovery plan binds another failed Stage-2")
    if recovery_plan.get("file_sha256") != input_bindings["failed_stage2"].sha256:
        raise ValueError("commission receipt failed Stage-2 hash differs from recovery plan")
    if recovery_plan.get("target_q6") != hand_targets[5]:
        raise ValueError("commission receipt q6 differs from its recovery plan")
    if recovery_plan.get("step_units") != selection["q6_step"]:
        raise ValueError("commission receipt q6 step differs from its recovery plan")
    if recovery_plan.get("coupled_targets") != list(hand_targets):
        raise ValueError("commission receipt targets differ from its recovery plan")
    historical_policy = recovery_plan.get("historical_source_provenance")
    if not isinstance(historical_policy, Mapping):
        raise ValueError("commission receipt lacks historical source provenance policy")
    if historical_policy.get("execution_authority") is not False:
        raise ValueError("historical commissioning sources cannot be execution authority")
    if historical_policy.get("source_bindings_sha256") != json_sha256(
        root["historical_commissioning_sources"]
    ):
        raise ValueError("historical source provenance digest differs")

    failed_binding_before, failed_bytes = _read_static_file(
        input_bindings["failed_stage2"].path,
        "historical failed Stage-2 evidence",
        include_content=True,
    )
    failed_stage2 = _strict_json_bytes(
        failed_bytes, "historical failed Stage-2 evidence"
    )
    failed_integrity = failed_stage2.get("integrity")
    if not isinstance(failed_integrity, Mapping):
        raise ValueError("historical failed Stage-2 integrity is missing")
    failed_unsigned = {
        key: value for key, value in failed_stage2.items() if key != "integrity"
    }
    if (
        failed_integrity.get("payload_sha256") != json_sha256(failed_unsigned)
        or recovery_plan.get("payload_sha256")
        != failed_integrity.get("payload_sha256")
        or recovery_plan.get("run_id") != failed_stage2.get("run_id")
    ):
        raise ValueError("historical failed Stage-2 payload differs from recovery plan")
    if failed_stage2.get("source_bindings") != root["historical_commissioning_sources"]:
        raise ValueError("historical source provenance differs from failed Stage-2")
    historical_stage1 = failed_stage2.get("stage1_prerequisite")
    if not isinstance(historical_stage1, Mapping) or (
        historical_stage1.get("path")
        != str(input_bindings["historical_stage1_evidence"].path)
        or historical_stage1.get("file_sha256")
        != input_bindings["historical_stage1_evidence"].sha256
    ):
        raise ValueError("historical Stage-1 provenance differs from failed Stage-2")
    _assert_static_file_unchanged(
        failed_binding_before, "historical failed Stage-2 evidence"
    )

    recovery_path = output_bindings["recovery_evidence"].path
    recovery_binding_before, recovery_bytes = _read_static_file(
        recovery_path,
        "recovery evidence",
        exact_mode=0o444,
        include_content=True,
    )
    recovery_evidence = _strict_json_bytes(recovery_bytes, "recovery evidence")
    recovery_evidence_sources = _validate_evidence_source_files(
        recovery_evidence, "recovery evidence"
    )
    if set(recovery_evidence_sources) != set(COMMISSION_RECEIPT_RECOVERY_SOURCE_NAMES):
        raise ValueError("recovery evidence source names differ from runtime authority")
    for name, binding in recovery_evidence_sources.items():
        if binding != recovery_source_bindings[name]:
            raise ValueError("recovery evidence source differs from runtime authority")
    _assert_static_file_unchanged(recovery_binding_before, "recovery evidence")

    stage2_path = output_bindings["stage2_evidence"].path
    profile_path = output_bindings["derived_profile"].path
    base_config_path = input_bindings["base_config"].path
    snapshot_path = input_bindings["official_snapshot"].path
    stage1_path = output_bindings["stage1_evidence"].path
    base_config, resolved_base_config = load_control_config(base_config_path)
    if resolved_base_config != base_config_path:
        raise ValueError("commission receipt base profile path changed")
    assets = verify_adapter_assets(base_config, base_config_path)
    expected_recovery_sources = _expected_recovery_runtime_source_paths(assets)
    for name, expected_path in expected_recovery_sources.items():
        if recovery_source_bindings[name].path != expected_path:
            raise ValueError(
                "recovery runtime source path changed: {}".format(name)
            )
    _validate_recovery_execution_contract(
        evidence=recovery_evidence,
        receipt_binding=root["recovery_execution"],
        recovery_plan=recovery_plan,
        config=base_config,
    )
    fresh_stage1_binding = build_stage1_prerequisite_binding(
        stage1_path,
        expected_config=base_config,
        expected_config_path=base_config_path,
    )
    if (
        fresh_stage1_binding.get("path") != str(stage1_path)
        or fresh_stage1_binding.get("file_sha256")
        != output_bindings["stage1_evidence"].sha256
        or fresh_stage1_binding.get("target_q6") != 900
    ):
        raise ValueError("fresh Stage-1 output binding is inconsistent")
    _fresh_stage1_file, fresh_stage1_bytes = _read_static_file(
        stage1_path,
        "fresh Stage-1 evidence",
        exact_mode=0o444,
        include_content=True,
    )
    fresh_stage1_sources = _validate_evidence_source_files(
        _strict_json_bytes(fresh_stage1_bytes, "fresh Stage-1 evidence"),
        "fresh Stage-1 evidence",
    )
    shared_runtime_names = set(COMMISSION_RECEIPT_RECOVERY_SOURCE_NAMES) - {
        "recovery_cli",
        "recovery_module",
    }
    if not shared_runtime_names <= set(fresh_stage1_sources):
        raise ValueError("fresh Stage-1 lost shared recovery runtime sources")
    for name in sorted(shared_runtime_names):
        if fresh_stage1_sources[name] != recovery_source_bindings[name]:
            raise ValueError(
                "fresh Stage-1 source differs from runtime authority: {}".format(name)
            )
    if (
        output_bindings["materialized_staging_profile"].sha256
        != output_bindings["derived_profile"].sha256
    ):
        raise ValueError("published profile differs from materialized staging profile")
    if result["profile_path"] != str(profile_path):
        raise ValueError("commission receipt result points to another profile")
    if profile_path.parent != base_config_path.parent:
        raise ValueError("commissioned profile is not beside its sealed base profile")

    stage2_binding_before, stage2_bytes = _read_static_file(
        stage2_path,
        "commissioned Stage-2 evidence",
        exact_mode=0o444,
        include_content=True,
    )
    stage2 = _strict_json_bytes(stage2_bytes, "commissioned Stage-2 evidence")
    stage2_source_bindings = _validate_evidence_source_files(
        stage2, "commissioned Stage-2 evidence"
    )
    if not shared_runtime_names <= set(stage2_source_bindings):
        raise ValueError("Stage-2 evidence lost shared recovery runtime sources")
    for name in sorted(shared_runtime_names):
        if stage2_source_bindings[name] != recovery_source_bindings[name]:
            raise ValueError(
                "Stage-2 source differs from recovery runtime authority: {}".format(name)
            )
    verification = verify_evidence(
        stage2_path,
        config_path=base_config_path,
        require_coupled=True,
    )
    _assert_static_file_unchanged(stage2_binding_before, "commissioned Stage-2 evidence")
    if not verification.passed:
        raise ValueError(
            "commissioned Stage-2 evidence is not coupled PASS: "
            + "; ".join(verification.blockers)
        )
    if stage2.get("motion_authorized") is not False:
        raise ValueError("Stage-2 evidence must keep motion_authorized=false")
    stage2_result = stage2.get("result")
    if not isinstance(stage2_result, Mapping) or any(
        stage2_result.get(field) is not True
        for field in (
            "coupled_closure_pass",
            "q6_return_pass",
            "reopened_and_verified",
            "disabled_verified",
        )
    ) or stage2_result.get("status") != "pass":
        raise ValueError("Stage-2 evidence lacks full coupled return/disable PASS")
    franka = stage2.get("franka_read_only")
    if (
        not isinstance(franka, Mapping)
        or franka.get("verified") is not True
        or franka.get("connection") != "read_once_only_no_controller_no_robot_write"
    ):
        raise ValueError("Stage-2 evidence lost its Franka read-only binding")

    profile = stage2.get("control_profile")
    snapshot_candidate = stage2.get("snapshot_candidate")
    candidate = (
        snapshot_candidate.get("candidate")
        if isinstance(snapshot_candidate, Mapping)
        else None
    )
    stage1 = stage2.get("stage1_prerequisite")
    request = stage2.get("request")
    if not isinstance(profile, Mapping) or (
        profile.get("path") != str(base_config_path)
        or profile.get("file_sha256") != input_bindings["base_config"].sha256
    ):
        raise ValueError("Stage-2 evidence binds another base profile")
    if not isinstance(snapshot_candidate, Mapping) or (
        snapshot_candidate.get("path") != str(snapshot_path)
        or snapshot_candidate.get("file_sha256")
        != input_bindings["official_snapshot"].sha256
    ):
        raise ValueError("Stage-2 evidence binds another official snapshot")
    if not isinstance(candidate, Mapping) or (
        candidate.get("index") != candidate_index
        or candidate.get("hand_targets") != list(hand_targets)
        or float(candidate.get("official_score", float("nan")))
        != float(official_score)
    ):
        raise ValueError("Stage-2 evidence binds another official candidate")
    if not isinstance(stage1, Mapping) or dict(stage1) != dict(fresh_stage1_binding):
        raise ValueError("Stage-2 evidence binds another Stage-1 prerequisite")
    if not isinstance(request, Mapping) or (
        request.get("target_source") != "official_snapshot_candidate"
        or request.get("coupled_closure_requested") is not True
        or request.get("coupled_targets") != list(hand_targets)
        or request.get("step_units") != selection["q6_step"]
    ):
        raise ValueError("Stage-2 request is not the exact official coupled target")

    profile_binding_before, _profile_bytes = _read_static_file(
        profile_path, "commissioned profile", exact_mode=0o444
    )
    applied = verify_applied_config(
        stage2_path,
        profile_path,
        require_coupled=True,
    )
    _assert_static_file_unchanged(stage2_binding_before, "commissioned Stage-2 evidence")
    _assert_static_file_unchanged(profile_binding_before, "commissioned profile")
    if not applied.passed:
        raise ValueError(
            "commissioned profile is not the exact coupled applied update: "
            + "; ".join(applied.blockers)
        )

    for name, binding in fresh_stage1_sources.items():
        _assert_static_file_unchanged(
            binding,
            "fresh Stage-1 source {}".format(name),
        )
    for name, binding in stage2_source_bindings.items():
        _assert_static_file_unchanged(
            binding,
            "commissioned Stage-2 source {}".format(name),
        )

    # Recheck all receipt-bound files after the two independent validators.
    for name, binding in input_bindings.items():
        _assert_static_file_unchanged(binding, "receipt input {}".format(name))
    for name, binding in source_bindings.items():
        _assert_static_file_unchanged(binding, "workflow source {}".format(name))
    for name, binding in output_bindings.items():
        _assert_static_file_unchanged(binding, "receipt output {}".format(name))
    _assert_static_file_unchanged(receipt_binding, "commission receipt")

    return CommissionReceiptAuthority(
        receipt_path=receipt_binding.path,
        receipt_sha256=receipt_binding.sha256,
        config_path=profile_path,
        snapshot_path=snapshot_path,
        selected_index=int(candidate_index),
        hand_targets=hand_targets,
        stage2_evidence_path=stage2_path,
        derived_profile_path=profile_path,
    )


def _require_tokens(args: argparse.Namespace, contract: Mapping[str, str]) -> None:
    wrong = []
    for name, expected in contract.items():
        actual = getattr(args, name, None)
        if actual != expected:
            wrong.append("--{} {}".format(name.replace("_", "-"), expected))
    if wrong:
        raise ValueError("exact confirmation tokens required: " + "; ".join(wrong))


def _finite_q(values: Sequence[float]) -> Tuple[float, ...]:
    q = np.asarray(values, dtype=np.float64)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError("fresh Franka q must contain seven finite radians")
    return tuple(float(value) for value in q)


def _resolve_prepare_authority(
    args: argparse.Namespace,
    *,
    receipt_loader: Callable[[Path], CommissionReceiptAuthority] = (
        load_commission_receipt_authority
    ),
) -> argparse.Namespace:
    """Select exactly one explicit or receipt-derived prepare input mode."""

    receipt = getattr(args, "commission_receipt", None)
    explicit_values = {
        "--config": getattr(args, "config", None),
        "--snapshot": getattr(args, "snapshot", None),
        "--selected-index": getattr(args, "selected_index", None),
    }
    values = dict(vars(args))
    if receipt is not None:
        mixed = [name for name, value in explicit_values.items() if value is not None]
        if mixed:
            raise ValueError(
                "--commission-receipt cannot be mixed with explicit overrides: "
                + ", ".join(mixed)
            )
        authority = receipt_loader(Path(receipt))
        values.update(
            {
                "config": authority.config_path,
                "snapshot": authority.snapshot_path,
                "selected_index": authority.selected_index,
                "commission_receipt": authority.receipt_path,
                "_commission_receipt_authority": authority,
            }
        )
    else:
        missing = [
            name
            for name in ("--snapshot", "--selected-index")
            if explicit_values[name] is None
        ]
        if missing:
            raise ValueError(
                "prepare requires either --commission-receipt or explicit "
                + ", ".join(missing)
            )
        values["config"] = (
            DEFAULT_CONFIG if explicit_values["--config"] is None else args.config
        )
        values["_commission_receipt_authority"] = None
    return argparse.Namespace(**values)


def _build_limits(config: Mapping[str, Any], limits_type: Any) -> Any:
    franka = config["franka"]
    dynamics = franka["expected_end_effector"]
    return limits_type(
        expected_F_T_EE=np.asarray(franka["expected_F_T_EE"], dtype=np.float64),
        expected_m_ee_kg=float(dynamics["mass_kg"]),
        expected_F_x_Cee_m=np.asarray(dynamics["F_x_Cee_m"], dtype=np.float64),
        expected_I_ee_kg_m2=np.asarray(dynamics["inertia_kg_m2"], dtype=np.float64),
        joint_limits_rad=np.asarray(franka["joint_limits_rad"], dtype=np.float64),
        joint_limit_margin_rad=float(franka["joint_limit_margin_rad"]),
        max_joint_speed_rad_s=float(franka["default_max_joint_velocity_rad_s"]),
        max_joint_segment_rad=float(franka["default_max_joint_segment_rad"]),
        min_joint_duration_s=float(franka["default_min_duration_s"]),
        joint_arrival_tolerance_rad=float(franka["default_arrival_tolerance_rad"]),
        settle_time_s=float(franka["settle_time_s"]),
        settle_timeout_s=float(franka["settle_timeout_s"]),
        settle_poll_s=float(franka["settle_poll_s"]),
    )


def _read_fresh_franka_q(config: Mapping[str, Any]) -> Tuple[float, ...]:
    """Read and validate one idle state without creating a control handle."""

    require_uncontended_franka_https_link(str(config["franka"]["ip"]))
    # This is intentionally the first pylibfranka import in this application.
    import pylibfranka
    from anydex_pipeline.franka_sequence_driver import (
        FrankaMotionLimits,
        FrankaSequenceDriver,
    )

    robot = pylibfranka.Robot(
        str(config["franka"]["ip"]), pylibfranka.RealtimeConfig.kIgnore
    )
    state = robot.read_once()
    validator = FrankaSequenceDriver(
        robot,
        pylibfranka,
        _build_limits(config, FrankaMotionLimits),
    )
    success = validator._validate_state(
        state, require_idle=True, enforce_success=False
    )
    q = np.asarray(state.q, dtype=np.float64)
    values = _finite_q(q)
    default_q = np.asarray(config["franka"]["default_q_rad"], dtype=np.float64)
    tolerance = float(config["franka"]["default_arrival_tolerance_rad"])
    error = float(np.max(np.abs(q - default_q)))
    if error > tolerance:
        raise RuntimeError(
            "fresh Franka q is not at configured default: Linf={:.9f}rad, "
            "tolerance={:.9f}rad".format(error, tolerance)
        )
    print(
        "[fresh-q] read_once only; idle/load/limits verified; "
        "control_success={:.6f}; q={}".format(success, list(values)),
        flush=True,
    )
    return values


def _run_stage(
    name: str,
    command: Sequence[str],
    *,
    runner: Callable[..., Any],
    env: Optional[Mapping[str, str]] = None,
) -> None:
    values = [str(item) for item in command]
    print("[prepare] stage={} command={}".format(name, " ".join(values)), flush=True)
    result = runner(
        values,
        cwd=str(ROOT),
        env=None if env is None else dict(env),
        check=False,
    )
    returncode = int(getattr(result, "returncode", 1))
    if returncode != 0:
        raise StageError(name, returncode)


def _single_native_module(directory: Path, prefix: str) -> Path:
    root = Path(directory).expanduser().resolve()
    matches = sorted(path.resolve() for path in root.glob(prefix + "*.so") if path.is_file())
    if len(matches) != 1:
        raise WorkflowError(
            "expected exactly one {}*.so in {}, found {}".format(
                prefix, root, len(matches)
            )
        )
    return matches[0]


def _audit_expiry(audit: Mapping[str, Any]) -> float:
    if audit.get("mode") != "air_grasp":
        raise ValueError("session audit is not mode=air_grasp")
    decision = audit.get("decision")
    if not isinstance(decision, Mapping) or decision.get("passed") is not True:
        raise ValueError("session audit is not EVIDENCE-PASS")
    if decision.get("motion_authorized") is not False:
        raise ValueError("session audit must keep motion_authorized=false")
    try:
        captured = float(audit["bindings"]["scene"]["captured_at_s"])
        max_age = float(audit["policies"]["max_scene_age_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("session audit freshness fields are malformed") from exc
    if not np.isfinite(captured) or not np.isfinite(max_age) or max_age <= 0.0:
        raise ValueError("session audit freshness fields are invalid")
    return captured + max_age


def _exclusive_json_write(path: Path, payload: Mapping[str, Any]) -> Path:
    output = normalize_unfollowed_leaf(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        str(output),
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        directory_fd = os.open(str(output.parent), os.O_RDONLY | os.O_DIRECTORY)
    except (AttributeError, OSError):
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return output


def _session_unsigned(
    *,
    args: argparse.Namespace,
    run_uuid: str,
    capture_q: Sequence[float],
    audit_path: Path,
    audit_expires_at_s: float,
    manifest_path: Path,
    producer_build: Path,
    reader_build: Path,
    telemetry_map: Path,
    created_at_s: float,
) -> Mapping[str, Any]:
    receipt_authority = getattr(args, "_commission_receipt_authority", None)
    if receipt_authority is not None and not isinstance(
        receipt_authority, CommissionReceiptAuthority
    ):
        raise ValueError("internal commission receipt authority is malformed")
    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "artifact_type": SESSION_ARTIFACT_TYPE,
        "created_at_s": float(created_at_s),
        "expires_at_s": float(audit_expires_at_s),
        "motion_authorized": False,
        "run_uuid": str(run_uuid),
        "selected_index": int(args.selected_index),
        "scope": {
            "mode": "air_grasp",
            "contact_allowed": False,
            "lift_allowed": False,
            "loaded_grasp_allowed": False,
        },
        "commissioning": {
            "source": (
                "verified_recover_and_commission_receipt"
                if receipt_authority is not None
                else "explicit_profile_snapshot_candidate"
            ),
            "receipt_is_motion_authority": False,
            "receipt": (
                {
                    "path": str(receipt_authority.receipt_path),
                    "sha256": receipt_authority.receipt_sha256,
                }
                if receipt_authority is not None
                else None
            ),
        },
        "capture": {
            "q_rad": list(_finite_q(capture_q)),
            "source": "fresh pylibfranka read_once after both reset stages",
            "stationary_confirmation": STATIONARY_Q_TOKEN,
        },
        "bindings": {
            "config": _binding(args.config),
            "camera_config": _binding(args.camera_config),
            "snapshot": _binding(args.snapshot),
            "audit": _binding(audit_path),
            "telemetry_manifest": _binding(manifest_path),
            "producer_build": _binding(producer_build),
            "reader_build": _binding(reader_build),
        },
        "telemetry": {
            "mapping_path": str(Path(telemetry_map).resolve()),
            "reader_python_dir": str(Path(args.reader_python_dir).resolve()),
            "producer_python_dir": str(Path(args.producer_python_dir).resolve()),
        },
        "execution": {
            "command": "air-grasp",
            "trajectory_mode": "audited-joint",
            "hold_seconds": float(args.hold_seconds),
        },
        "viewer": {
            "source": "realsense",
            "execution_mode": "air",
            "telemetry_wait_seconds": float(args.telemetry_wait_seconds),
            "arm_max_age_s": 0.25,
            "hand_max_age_s": 0.75,
            "show_current_hand_mesh": True,
            "hand_mesh_resolution": "simplified",
        },
        "confirmation_contract": dict(RUN_TOKEN_CONTRACT),
    }


def _seal_session(unsigned: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = dict(unsigned)
    payload["integrity"] = {
        "algorithm": SESSION_INTEGRITY_ALGORITHM,
        "payload_sha256": canonical_json_sha256(unsigned),
    }
    return payload


def _validate_file_binding(value: Any, name: str, *, verify_files: bool) -> Path:
    binding = _require_exact_keys(value, ("path", "sha256"), name)
    path = Path(str(binding["path"])).expanduser().resolve()
    digest = str(binding["sha256"])
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("{} SHA-256 is malformed".format(name))
    if verify_files and _sha256(path) != digest:
        raise ValueError("{} file hash changed: {}".format(name, path))
    return path


def _validate_session_payload(
    payload: Mapping[str, Any],
    *,
    verify_files: bool,
    now_s: float,
    audit_loader: Callable[..., Mapping[str, Any]] = load_installed_tool_audit,
    manifest_loader: Callable[..., Any] = load_telemetry_session_manifest,
    receipt_loader: Callable[[Path], CommissionReceiptAuthority] = (
        load_commission_receipt_authority
    ),
) -> float:
    root = _require_exact_keys(
        payload,
        (
            "schema_version",
            "artifact_type",
            "created_at_s",
            "expires_at_s",
            "motion_authorized",
            "run_uuid",
            "selected_index",
            "scope",
            "commissioning",
            "capture",
            "bindings",
            "telemetry",
            "execution",
            "viewer",
            "confirmation_contract",
            "integrity",
        ),
        "air-grasp session",
    )
    if root["schema_version"] != SESSION_SCHEMA_VERSION:
        raise ValueError("air-grasp session schema version is unsupported")
    if root["artifact_type"] != SESSION_ARTIFACT_TYPE:
        raise ValueError("air-grasp session artifact type is unsupported")
    if root["motion_authorized"] is not False:
        raise ValueError("air-grasp session must keep motion_authorized=false")
    try:
        parsed_uuid = uuid.UUID(str(root["run_uuid"]))
    except (ValueError, AttributeError) as exc:
        raise ValueError("air-grasp session run_uuid is malformed") from exc
    if str(parsed_uuid) != root["run_uuid"]:
        raise ValueError("air-grasp session run_uuid is not canonical")
    if type(root["selected_index"]) is not int or root["selected_index"] < 0:
        raise ValueError("air-grasp session selected_index is invalid")
    created_at = float(root["created_at_s"])
    expires_at = float(root["expires_at_s"])
    if not np.isfinite(created_at) or not np.isfinite(expires_at) or expires_at <= created_at:
        raise ValueError("air-grasp session timestamps are invalid")

    scope = _require_exact_keys(
        root["scope"],
        ("mode", "contact_allowed", "lift_allowed", "loaded_grasp_allowed"),
        "session scope",
    )
    if dict(scope) != {
        "mode": "air_grasp",
        "contact_allowed": False,
        "lift_allowed": False,
        "loaded_grasp_allowed": False,
    }:
        raise ValueError("session scope is not no-contact/no-lift air-grasp")
    commissioning = _require_exact_keys(
        root["commissioning"],
        ("source", "receipt_is_motion_authority", "receipt"),
        "session commissioning",
    )
    if commissioning["receipt_is_motion_authority"] is not False:
        raise ValueError("commission receipt cannot be session motion authority")
    receipt_binding = commissioning["receipt"]
    receipt_authority = None
    if commissioning["source"] == "explicit_profile_snapshot_candidate":
        if receipt_binding is not None:
            raise ValueError("explicit session must not carry a commission receipt")
    elif commissioning["source"] == "verified_recover_and_commission_receipt":
        binding_object = _require_exact_keys(
            receipt_binding,
            ("path", "sha256"),
            "session commission receipt binding",
        )
        digest = binding_object["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("session commission receipt SHA-256 is malformed")
        if verify_files:
            receipt_authority = receipt_loader(Path(str(binding_object["path"])))
            if receipt_authority.receipt_sha256 != digest:
                raise ValueError("session commission receipt file hash changed")
    else:
        raise ValueError("session commissioning source is unsupported")
    capture = _require_exact_keys(
        root["capture"], ("q_rad", "source", "stationary_confirmation"), "session capture"
    )
    _finite_q(capture["q_rad"])
    if capture["source"] != "fresh pylibfranka read_once after both reset stages":
        raise ValueError("session fresh-q source changed")
    if capture["stationary_confirmation"] != STATIONARY_Q_TOKEN:
        raise ValueError("session stationary-q confirmation changed")

    execution = _require_exact_keys(
        root["execution"], ("command", "trajectory_mode", "hold_seconds"), "session execution"
    )
    hold = float(execution["hold_seconds"])
    if (
        execution["command"] != "air-grasp"
        or execution["trajectory_mode"] != "audited-joint"
        or not np.isfinite(hold)
        or not 0.0 <= hold <= 10.0
    ):
        raise ValueError("session execution is not the bounded audited air-grasp contract")
    viewer = _require_exact_keys(
        root["viewer"],
        (
            "source",
            "execution_mode",
            "telemetry_wait_seconds",
            "arm_max_age_s",
            "hand_max_age_s",
            "show_current_hand_mesh",
            "hand_mesh_resolution",
        ),
        "session viewer",
    )
    if (
        viewer["source"] != "realsense"
        or viewer["execution_mode"] != "air"
        or viewer["show_current_hand_mesh"] is not True
        or viewer["hand_mesh_resolution"] != "simplified"
    ):
        raise ValueError("session viewer contract changed")
    for name in ("telemetry_wait_seconds", "arm_max_age_s", "hand_max_age_s"):
        number = float(viewer[name])
        if not np.isfinite(number) or number <= 0.0:
            raise ValueError("session viewer {} is invalid".format(name))
    confirmations = _require_exact_keys(
        root["confirmation_contract"], tuple(RUN_TOKEN_CONTRACT), "session confirmations"
    )
    if dict(confirmations) != RUN_TOKEN_CONTRACT:
        raise ValueError("session formal confirmation contract changed")

    integrity = _require_exact_keys(
        root["integrity"], ("algorithm", "payload_sha256"), "session integrity"
    )
    if integrity["algorithm"] != SESSION_INTEGRITY_ALGORITHM:
        raise ValueError("session integrity algorithm is unsupported")
    unsigned = dict(root)
    del unsigned["integrity"]
    if canonical_json_sha256(unsigned) != integrity["payload_sha256"]:
        raise ValueError("session integrity payload SHA-256 mismatch")

    bindings = _require_exact_keys(
        root["bindings"],
        (
            "config",
            "camera_config",
            "snapshot",
            "audit",
            "telemetry_manifest",
            "producer_build",
            "reader_build",
        ),
        "session bindings",
    )
    paths = {
        name: _validate_file_binding(value, "session binding {}".format(name), verify_files=verify_files)
        for name, value in bindings.items()
    }
    if verify_files and receipt_authority is not None:
        if receipt_authority.config_path != paths["config"]:
            raise ValueError("session config differs from its commission receipt")
        if receipt_authority.snapshot_path != paths["snapshot"]:
            raise ValueError("session snapshot differs from its commission receipt")
        if receipt_authority.selected_index != root["selected_index"]:
            raise ValueError("session candidate differs from its commission receipt")
    telemetry = _require_exact_keys(
        root["telemetry"],
        ("mapping_path", "reader_python_dir", "producer_python_dir"),
        "session telemetry",
    )
    mapping = normalize_unfollowed_leaf(Path(str(telemetry["mapping_path"])))
    if mapping.parent != Path("/tmp") or mapping.suffix != ".map":
        raise ValueError("session telemetry mapping must be one /tmp/*.map path")
    if verify_files:
        try:
            mapping.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError(
                "session telemetry mapping path already exists (including a symlink); "
                "session cannot be replayed"
            )
    reader_dir = Path(str(telemetry["reader_python_dir"])).expanduser().resolve()
    producer_dir = Path(str(telemetry["producer_python_dir"])).expanduser().resolve()
    if verify_files:
        if paths["reader_build"].parent != reader_dir:
            raise ValueError("reader build is outside the bound reader Python directory")
        if paths["producer_build"].parent != producer_dir:
            raise ValueError("producer build is outside the bound producer Python directory")

        audit = audit_loader(paths["audit"], verify_files=True, require_pass=True)
        actual_expiry = _audit_expiry(audit)
        if not np.isclose(expires_at, actual_expiry, atol=1.0e-6, rtol=0.0):
            raise ValueError("session expiry differs from its audit freshness contract")
        manifest = manifest_loader(paths["telemetry_manifest"], verify_files=True)
        identity = manifest.identity
        if identity.run_uuid != root["run_uuid"]:
            raise ValueError("session run UUID differs from telemetry manifest")
        if identity.source_snapshot_sha256 != bindings["snapshot"]["sha256"]:
            raise ValueError("session snapshot differs from telemetry manifest")
        if identity.control_config_sha256 != bindings["config"]["sha256"]:
            raise ValueError("session config differs from telemetry manifest")
        if identity.producer_build_sha256 != bindings["producer_build"]["sha256"]:
            raise ValueError("session producer differs from telemetry manifest")
        manifest_execution = manifest.payload["execution"]
        if (
            manifest_execution["command"] != "air-grasp"
            or manifest_execution["selected_index"] != root["selected_index"]
            or manifest.payload["sources"]["audit_artifact"]["file_sha256"]
            != bindings["audit"]["sha256"]
        ):
            raise ValueError("session execution differs from telemetry manifest")

    if not np.isfinite(float(now_s)) or float(now_s) >= expires_at:
        raise ValueError(
            "fresh air audit expired at {:.6f}; prepare a new session".format(expires_at)
        )
    return expires_at


def _prepare_commands(
    args: argparse.Namespace,
    capture_q: Sequence[float],
    *,
    audit_dir: Path,
    audit_path: Path,
    manifest_path: Path,
    producer_build: Path,
    run_uuid: str,
) -> Mapping[str, Sequence[str]]:
    config = str(Path(args.config).resolve())
    snapshot = str(Path(args.snapshot).resolve())
    commands = {
        "reset_rh56": [
            str(ROOT / "scripts/reset_installed_rh56_open.sh"),
            "run",
            "--config", config,
            "--confirm-installed", args.confirm_installed,
            "--confirm-24v-cutoff", args.confirm_24v_cutoff,
            "--confirm-franka-stop", args.confirm_franka_stop,
            "--confirm-workspace-clear", args.confirm_rh56_workspace_clear,
            "--confirm-no-contact", args.confirm_no_contact,
            "--confirm-reset-open", args.confirm_rh56_reset_open,
        ],
        "reset_franka": [
            str(ROOT / "scripts/reset_franka_default.sh"),
            "--config", config,
            "--confirm-installed", args.confirm_installed,
            "--confirm-hand-open", args.confirm_hand_open,
            "--confirm-workspace-clear", args.confirm_default_sweep_clear,
            "--confirm-stop-ready", args.confirm_franka_stop,
            "--confirm-pla-low-speed", args.confirm_pla_low_speed,
        ],
        "build_telemetry": [str(ROOT / "scripts/build_continuous_telemetry.sh")],
        "fresh_audit": [
            str(ROOT / "scripts/prepare_fresh_installed_air_audit.sh"),
            "--config", config,
            "--camera-config", str(Path(args.camera_config).resolve()),
            "--snapshot", snapshot,
            "--candidate-index", str(int(args.selected_index)),
            "--capture-q-rad", *["{:.17g}".format(value) for value in capture_q],
            "--confirm-stationary-q", args.confirm_stationary_q,
            "--output-dir", str(audit_dir),
            "--anydex-root", str(Path(args.anydex_root).resolve()),
            "--warmup-frames", str(int(args.warmup_frames)),
            "--capture-frames", str(int(args.capture_frames)),
        ],
        "executor_dry_run": [
            str(ROOT / "scripts/execute_control_sequence.sh"),
            "air-grasp",
            "--config", config,
            "--snapshot", snapshot,
            "--installed-tool-audit", str(audit_path),
            "--selected-index", str(int(args.selected_index)),
            "--trajectory-mode", "audited-joint",
            "--dry-run",
        ],
        "create_manifest": [
            str(ROOT / "scripts/telemetry_session_manifest.sh"),
            "create",
            "--snapshot", snapshot,
            "--config", config,
            "--audit-artifact", str(audit_path),
            "--producer-build", str(producer_build),
            "--command", "air-grasp",
            "--selected-index", str(int(args.selected_index)),
            "--run-uuid", run_uuid,
            "--output", str(manifest_path),
        ],
    }
    if args.port:
        commands["reset_rh56"] = list(commands["reset_rh56"]) + ["--port", str(args.port)]
        commands["reset_franka"] = list(commands["reset_franka"]) + ["--port", str(args.port)]
    return commands


def prepare_workflow(
    args: argparse.Namespace,
    *,
    runner: Callable[..., Any] = subprocess.run,
    q_reader: Callable[[Mapping[str, Any]], Sequence[float]] = _read_fresh_franka_q,
    audit_loader: Callable[..., Mapping[str, Any]] = load_installed_tool_audit,
    manifest_loader: Callable[..., Any] = load_telemetry_session_manifest,
    receipt_loader: Callable[[Path], CommissionReceiptAuthority] = (
        load_commission_receipt_authority
    ),
    clock: Callable[[], float] = time.time,
    uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4,
) -> Path:
    _require_tokens(args, PREPARE_TOKEN_CONTRACT)
    args = _resolve_prepare_authority(args, receipt_loader=receipt_loader)
    if not 0.0 <= float(args.hold_seconds) <= 10.0:
        raise ValueError("--hold-seconds must be in 0..10")
    if int(args.selected_index) < 0:
        raise ValueError("--selected-index must be nonnegative")
    if int(args.warmup_frames) < 0 or int(args.capture_frames) < 1:
        raise ValueError("capture frame counts are invalid")
    telemetry_wait_seconds = float(args.telemetry_wait_seconds)
    if not np.isfinite(telemetry_wait_seconds) or telemetry_wait_seconds <= 0.0:
        raise ValueError("--telemetry-wait-seconds must be finite and positive")
    for path, name, directory in (
        (args.config, "config", False),
        (args.camera_config, "camera config", False),
        (args.snapshot, "snapshot", False),
        (args.anydex_root, "AnyDex root", True),
        (args.dynamic_python, "dynamic Python", False),
    ):
        resolved = Path(path).expanduser().resolve()
        if (directory and not resolved.is_dir()) or (not directory and not resolved.is_file()):
            raise FileNotFoundError("{} does not exist: {}".format(name, resolved))
    config, _config_path = load_control_config(args.config)
    output_dir = normalize_unfollowed_leaf(args.output_dir)
    try:
        output_dir.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(
            "output directory path already exists (including a symlink): {}".format(
                output_dir
            )
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    run_uuid = str(uuid_factory())
    telemetry_map = Path("/tmp/fr3_rh56_air_{}.map".format(run_uuid))
    try:
        telemetry_map.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(
            "fresh telemetry mapping path already exists (including a symlink): {}".format(
                telemetry_map
            )
        )
    audit_dir = output_dir / AUDIT_DIRECTORY_NAME
    audit_path = audit_dir / AUDIT_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    session_path = output_dir / SESSION_FILENAME

    # Reset stages remain the existing reviewed CLIs.  The build follows both
    # resets but precedes the fresh q read and final D435 capture, so compilation
    # time never consumes the audit freshness window.
    initial_commands = _prepare_commands(
        args,
        (0.0,) * 7,
        audit_dir=audit_dir,
        audit_path=audit_path,
        manifest_path=manifest_path,
        producer_build=Path(args.producer_python_dir) / "pending.so",
        run_uuid=run_uuid,
    )
    _run_stage("reset_rh56", initial_commands["reset_rh56"], runner=runner)
    _run_stage("reset_franka", initial_commands["reset_franka"], runner=runner)
    build_env = os.environ.copy()
    build_env["DEXGRASP_DYNAMIC_PYTHON"] = str(Path(args.dynamic_python).resolve())
    build_env["ANYDEX_VIEWER_TELEMETRY_BUILD_DIR"] = str(
        Path(args.reader_python_dir).resolve().parent
    )
    build_env["ANYDEX_FRANKA_TAP_BUILD_DIR"] = str(
        Path(args.producer_python_dir).resolve().parent
    )
    _run_stage(
        "build_telemetry",
        initial_commands["build_telemetry"],
        runner=runner,
        env=build_env,
    )

    reader_build = _single_native_module(args.reader_python_dir, "_anydex_telemetry")
    producer_build = _single_native_module(
        args.producer_python_dir, "_anydex_franka_telemetry"
    )
    capture_q = _finite_q(q_reader(config))
    commands = _prepare_commands(
        args,
        capture_q,
        audit_dir=audit_dir,
        audit_path=audit_path,
        manifest_path=manifest_path,
        producer_build=producer_build,
        run_uuid=run_uuid,
    )
    _run_stage("fresh_audit", commands["fresh_audit"], runner=runner)
    _run_stage("executor_dry_run", commands["executor_dry_run"], runner=runner)
    if manifest_path.exists():
        raise FileExistsError("telemetry manifest output already exists: {}".format(manifest_path))
    _run_stage("create_manifest", commands["create_manifest"], runner=runner)

    audit = audit_loader(audit_path, verify_files=True, require_pass=True)
    expires_at = _audit_expiry(audit)
    loaded_manifest = manifest_loader(manifest_path, verify_files=True)
    if loaded_manifest.identity.run_uuid != run_uuid:
        raise WorkflowError("created telemetry manifest returned a different run UUID")
    created_at = float(clock())
    if created_at >= expires_at:
        raise WorkflowError("fresh audit expired before the immutable session was written")
    unsigned = _session_unsigned(
        args=args,
        run_uuid=run_uuid,
        capture_q=capture_q,
        audit_path=audit_path,
        audit_expires_at_s=expires_at,
        manifest_path=manifest_path,
        producer_build=producer_build,
        reader_build=reader_build,
        telemetry_map=telemetry_map,
        created_at_s=created_at,
    )
    payload = _seal_session(unsigned)
    # The two objects above were already loaded with full file replay.  Reuse
    # them for the final session schema/identity check so the finite freshness
    # window is not spent validating the same large audit a second time.
    _validate_session_payload(
        payload,
        verify_files=True,
        now_s=created_at,
        audit_loader=lambda _path, **_kwargs: audit,
        manifest_loader=lambda _path, **_kwargs: loaded_manifest,
        receipt_loader=receipt_loader,
    )
    output = _exclusive_json_write(session_path, payload)
    print(
        "[prepare] EVIDENCE-PASS remaining_s={:.2f}; "
        "motion_authorized=false; run still requires all five live confirmations".format(
            expires_at - float(clock())
        ),
        flush=True,
    )
    # Keep the final line directly reusable in the caller's shell, matching
    # the PROFILE=/RECEIPT= contract of the commissioning workflow.
    print("SESSION={}".format(output), flush=True)
    return output


def load_validated_session(
    path: Path,
    *,
    now_s: Optional[float] = None,
    audit_loader: Callable[..., Mapping[str, Any]] = load_installed_tool_audit,
    manifest_loader: Callable[..., Any] = load_telemetry_session_manifest,
    receipt_loader: Callable[[Path], CommissionReceiptAuthority] = (
        load_commission_receipt_authority
    ),
) -> ValidatedSession:
    source = normalize_unfollowed_leaf(path)
    payload = _strict_json(source)
    moment = time.time() if now_s is None else float(now_s)
    expires = _validate_session_payload(
        payload,
        verify_files=True,
        now_s=moment,
        audit_loader=audit_loader,
        manifest_loader=manifest_loader,
        receipt_loader=receipt_loader,
    )
    return ValidatedSession(source, payload, expires)


def _viewer_ready_path(session: ValidatedSession) -> Path:
    mapping = normalize_unfollowed_leaf(
        Path(session.payload["telemetry"]["mapping_path"])
    )
    return mapping.with_suffix(".viewer-ready")


def _viewer_command(
    session: ValidatedSession,
    *,
    ready_file: Path,
) -> Sequence[str]:
    payload = session.payload
    bindings = payload["bindings"]
    telemetry = payload["telemetry"]
    viewer = payload["viewer"]
    return [
        str(ROOT / "scripts/run_live_pipeline_preview.sh"),
        bindings["snapshot"]["path"],
        "--control-config", bindings["config"]["path"],
        "--camera-config", bindings["camera_config"]["path"],
        "--selected-index", str(payload["selected_index"]),
        "--source", "realsense",
        "--execution-mode", "air",
        "--continuous-telemetry", telemetry["mapping_path"],
        "--telemetry-session-manifest", bindings["telemetry_manifest"]["path"],
        "--continuous-telemetry-python-dir", telemetry["reader_python_dir"],
        "--telemetry-wait-seconds", str(viewer["telemetry_wait_seconds"]),
        "--arm-max-age-s", str(viewer["arm_max_age_s"]),
        "--hand-max-age-s", str(viewer["hand_max_age_s"]),
        "--error-target", "auto",
        "--show-current-hand-mesh",
        "--hand-mesh-resolution", viewer["hand_mesh_resolution"],
        "--ready-file", str(normalize_unfollowed_leaf(ready_file)),
    ]


def _executor_command(session: ValidatedSession, args: argparse.Namespace) -> Sequence[str]:
    payload = session.payload
    bindings = payload["bindings"]
    telemetry = payload["telemetry"]
    execution = payload["execution"]
    return [
        str(ROOT / "scripts/execute_control_sequence.sh"),
        "air-grasp",
        "--config", bindings["config"]["path"],
        "--snapshot", bindings["snapshot"]["path"],
        "--installed-tool-audit", bindings["audit"]["path"],
        "--selected-index", str(payload["selected_index"]),
        "--trajectory-mode", "audited-joint",
        "--hold-seconds", str(execution["hold_seconds"]),
        "--continuous-telemetry", telemetry["mapping_path"],
        "--telemetry-session-manifest", bindings["telemetry_manifest"]["path"],
        "--continuous-telemetry-python-dir", telemetry["producer_python_dir"],
        "--confirm-workspace-clear", args.confirm_workspace_clear,
        "--confirm-immediate-stop", args.confirm_immediate_stop,
        "--confirm-air-grasp", args.confirm_air_grasp,
        "--confirm-q6-preshape", args.confirm_q6_preshape,
        "--confirm-installed-collision-model", args.confirm_installed_collision_model,
    ]


def _request_executor_cleanup(
    process: Any,
    *,
    timeout_s: float,
) -> bool:
    """Request only the executor's reviewed SIGINT cleanup; never escalate."""

    if process is None or process.poll() is not None:
        return True
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=float(timeout_s))
    except ProcessLookupError:
        return process.poll() is not None
    except (subprocess.TimeoutExpired, KeyboardInterrupt, OSError) as exc:
        print(
            "[hardware][STOP UNCONFIRMED] executor did not complete its "
            "reviewed SIGINT cleanup within {:.3f}s: {}; no terminate/kill "
            "was sent to the executor".format(float(timeout_s), exc),
            file=sys.stderr,
            flush=True,
        )
        return False
    if process.poll() is None:
        print(
            "[hardware][STOP UNCONFIRMED] executor wait returned but the "
            "process is still live; no terminate/kill was sent to the executor",
            file=sys.stderr,
            flush=True,
        )
        return False
    return True


def _stop_viewer_with_escalation(process: Any, *, timeout_s: float) -> None:
    """Stop only the perception child; escalation is permitted for the viewer."""

    if process is None or process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=float(timeout_s))
        return
    except (ProcessLookupError, OSError):
        if process.poll() is not None:
            return
    except subprocess.TimeoutExpired:
        pass
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2.0)


def _wait_for_viewer_ready(
    viewer: Any,
    ready_file: Path,
    *,
    timeout_s: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    ready_checker: Callable[[Path], bool],
) -> None:
    deadline = float(monotonic()) + float(timeout_s)
    while True:
        viewer_code = viewer.poll()
        if viewer_code is not None:
            raise WorkflowError(
                "viewer exited before publishing readiness with code {}; "
                "executor was not started".format(viewer_code)
            )
        if ready_checker(ready_file):
            if viewer.poll() is not None:
                raise WorkflowError(
                    "viewer exited immediately after publishing readiness; "
                    "executor was not started"
                )
            return
        now = float(monotonic())
        if now >= deadline:
            raise WorkflowError(
                "viewer readiness timed out after {:.3f}s; executor was not "
                "started".format(float(timeout_s))
            )
        sleep(min(0.10, max(0.0, deadline - now)))


def run_workflow(
    args: argparse.Namespace,
    *,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    session_loader: Callable[..., ValidatedSession] = load_validated_session,
    clock: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    ready_checker: Callable[[Path], bool] = viewer_ready_is_published,
) -> int:
    _require_tokens(args, RUN_TOKEN_CONTRACT)
    dynamic_python = Path(args.dynamic_python).expanduser().resolve()
    if not dynamic_python.is_file() or not os.access(dynamic_python, os.X_OK):
        raise FileNotFoundError(
            "dynamic Python is not executable: {}".format(dynamic_python)
        )
    minimum_freshness_seconds = float(args.minimum_freshness_seconds)
    if (
        not np.isfinite(minimum_freshness_seconds)
        or minimum_freshness_seconds < 0.0
    ):
        raise ValueError(
            "--minimum-freshness-seconds must be finite and nonnegative"
        )
    if (
        not np.isfinite(float(args.viewer_ready_timeout_seconds))
        or float(args.viewer_ready_timeout_seconds) <= 0.0
    ):
        raise ValueError("--viewer-ready-timeout-seconds must be finite and positive")
    cleanup_timeout_seconds = float(args.cleanup_timeout_seconds)
    if not np.isfinite(cleanup_timeout_seconds) or cleanup_timeout_seconds <= 0.0:
        raise ValueError("--cleanup-timeout-seconds must be finite and positive")
    session = session_loader(args.session, now_s=float(clock()))
    remaining = session.expires_at_s - float(clock())
    if remaining <= minimum_freshness_seconds:
        raise WorkflowError(
            "fresh audit has only {:.2f}s remaining; require >{:.2f}s".format(
                remaining, minimum_freshness_seconds
            )
        )
    print(
        "[run] session/hash/manifest/audit verified; remaining_freshness_s={:.2f}; "
        "scope=air/no-contact/no-lift".format(remaining),
        flush=True,
    )

    ready_file = _viewer_ready_path(session)
    try:
        ready_file.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(
            "viewer ready marker already exists; refusing stale readiness: {}".format(
                ready_file
            )
        )
    viewer_command = [
        str(item)
        for item in _viewer_command(session, ready_file=ready_file)
    ]
    executor_command = [str(item) for item in _executor_command(session, args)]
    viewer_env = os.environ.copy()
    viewer_env["DEXGRASP_SHELL_PYTHON"] = str(dynamic_python)
    print("[run] starting read-only viewer first", flush=True)
    viewer = popen_factory(
        viewer_command,
        cwd=str(ROOT),
        env=viewer_env,
        start_new_session=True,
    )
    executor = None
    executor_cleanup_requested = False
    try:
        _wait_for_viewer_ready(
            viewer,
            ready_file,
            timeout_s=float(args.viewer_ready_timeout_seconds),
            monotonic=monotonic,
            sleep=sleep,
            ready_checker=ready_checker,
        )
        remaining = session.expires_at_s - float(clock())
        if remaining <= minimum_freshness_seconds:
            raise WorkflowError(
                "fresh audit expired while waiting for viewer readiness; "
                "executor was not started"
            )
        if viewer.poll() is not None:
            raise WorkflowError(
                "viewer exited after readiness; executor was not started"
            )

        if bool(getattr(args, "interactive_execution_confirmation", False)):
            print(
                "[run] viewer READY; inspect the live cloud, target grasp/mesh, "
                "and current EE/hand before motion",
                flush=True,
            )
            try:
                confirmation = input(
                    "输入 {} 执行所选抓取：".format(VIEWER_EXECUTE_TOKEN)
                ).strip()
            except EOFError as exc:
                raise WorkflowError(
                    "interactive execution confirmation has no terminal input; "
                    "executor was not started"
                ) from exc
            if confirmation != VIEWER_EXECUTE_TOKEN:
                raise WorkflowError(
                    "interactive execution confirmation declined; executor was "
                    "not started"
                )

        print(
            "[run] viewer READY (Open3D+D435+reader); starting foreground "
            "air-grasp executor",
            flush=True,
        )
        executor = popen_factory(
            executor_command,
            cwd=str(ROOT),
            start_new_session=True,
        )
        while True:
            executor_code = executor.poll()
            viewer_code = viewer.poll()
            if executor_code is not None:
                return int(executor_code)
            if viewer_code is not None:
                executor_cleanup_requested = True
                confirmed = _request_executor_cleanup(
                    executor,
                    timeout_s=cleanup_timeout_seconds,
                )
                if not confirmed:
                    raise StopUnconfirmedError(
                        "viewer failed and executor cleanup is STOP UNCONFIRMED"
                    )
                raise WorkflowError(
                    "viewer exited during execution with code {}; reviewed "
                    "executor cleanup completed".format(
                        viewer_code
                    )
                )
            sleep(0.10)
    except KeyboardInterrupt:
        print(
            "[run] Ctrl+C: forwarding SIGINT to reviewed executor cleanup; "
            "viewer stops after cleanup",
            file=sys.stderr,
            flush=True,
        )
        if executor is not None and executor.poll() is None:
            executor_cleanup_requested = True
            if not _request_executor_cleanup(
                executor, timeout_s=cleanup_timeout_seconds
            ):
                raise StopUnconfirmedError(
                    "Ctrl+C executor cleanup is STOP UNCONFIRMED"
                )
        return 130
    except BaseException as exc:
        if (
            executor is not None
            and executor.poll() is None
            and not executor_cleanup_requested
        ):
            executor_cleanup_requested = True
            if not _request_executor_cleanup(
                executor, timeout_s=cleanup_timeout_seconds
            ):
                raise StopUnconfirmedError(
                    "workflow failed and executor cleanup is STOP UNCONFIRMED"
                ) from exc
        raise
    finally:
        _stop_viewer_with_escalation(
            viewer,
            timeout_s=cleanup_timeout_seconds,
        )
        try:
            ready_file.lstat()
        except OSError:
            pass
        else:
            print(
                "[run] immutable viewer ready marker retained as a one-shot "
                "session latch: {}".format(ready_file),
                flush=True,
            )


def _add_prepare_tokens(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--confirm-installed", required=True, metavar=RH56_INSTALLED_TOKEN)
    parser.add_argument("--confirm-24v-cutoff", required=True, metavar=RH56_POWER_TOKEN)
    parser.add_argument("--confirm-franka-stop", required=True, metavar=FRANKA_STOP_TOKEN)
    parser.add_argument("--confirm-rh56-workspace-clear", required=True, metavar=RH56_CLEAR_TOKEN)
    parser.add_argument("--confirm-no-contact", required=True, metavar=NO_CONTACT_TOKEN)
    parser.add_argument("--confirm-rh56-reset-open", required=True, metavar=RH56_RESET_TOKEN)
    parser.add_argument("--confirm-hand-open", required=True, metavar=HAND_OPEN_TOKEN)
    parser.add_argument("--confirm-default-sweep-clear", required=True, metavar=DEFAULT_SWEEP_TOKEN)
    parser.add_argument("--confirm-pla-low-speed", required=True, metavar=PLA_LOW_SPEED_TOKEN)
    parser.add_argument("--confirm-stationary-q", required=True, metavar=STATIONARY_Q_TOKEN)


def _add_run_tokens(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--confirm-workspace-clear", required=True, metavar=EXECUTOR_WORKSPACE_TOKEN)
    parser.add_argument("--confirm-immediate-stop", required=True, metavar=EXECUTOR_STOP_TOKEN)
    parser.add_argument("--confirm-air-grasp", required=True, metavar=AIR_GRASP_TOKEN)
    parser.add_argument("--confirm-q6-preshape", required=True, metavar=Q6_PRESHAPE_TOKEN)
    parser.add_argument(
        "--confirm-installed-collision-model",
        required=True,
        metavar=COLLISION_MODEL_TOKEN,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage installed RH56 air-grasp workflow. Only no-contact, "
            "no-load, no-lift air-grasp is supported."
        )
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="reset, build telemetry, capture/audit, dry-run, and write one session",
    )
    prepare.add_argument(
        "--commission-receipt",
        type=Path,
        help=(
            "PASS receipt from recover_and_commission_rh56; strictly derives "
            "the commissioned profile, official snapshot, and candidate index"
        ),
    )
    prepare.add_argument("--config", type=Path)
    prepare.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    prepare.add_argument("--snapshot", type=Path)
    prepare.add_argument("--selected-index", type=int)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--anydex-root", type=Path, default=DEFAULT_ANYDEX_ROOT)
    prepare.add_argument("--port")
    prepare.add_argument("--warmup-frames", type=int, default=5)
    prepare.add_argument("--capture-frames", type=int, default=5)
    prepare.add_argument("--hold-seconds", type=float, default=2.0)
    prepare.add_argument("--telemetry-wait-seconds", type=float, default=60.0)
    prepare.add_argument("--dynamic-python", type=Path, default=DEFAULT_DYNAMIC_PYTHON)
    prepare.add_argument(
        "--reader-python-dir", type=Path, default=DEFAULT_READER_PYTHON_DIR
    )
    prepare.add_argument(
        "--producer-python-dir", type=Path, default=DEFAULT_PRODUCER_PYTHON_DIR
    )
    _add_prepare_tokens(prepare)

    run = subparsers.add_parser(
        "run",
        help="verify one fresh session, start viewer, and foreground the executor",
    )
    run.add_argument("--session", type=Path, required=True)
    run.add_argument("--dynamic-python", type=Path, default=DEFAULT_DYNAMIC_PYTHON)
    run.add_argument("--minimum-freshness-seconds", type=float, default=5.0)
    run.add_argument("--viewer-ready-timeout-seconds", type=float, default=30.0)
    run.add_argument("--cleanup-timeout-seconds", type=float, default=20.0)
    run.add_argument(
        "--interactive-execution-confirmation",
        action="store_true",
        help=(
            "after the viewer is READY, require one terminal confirmation "
            "before starting the hardware executor"
        ),
    )
    _add_run_tokens(run)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.operation == "prepare":
            prepare_workflow(args)
            return 0
        return run_workflow(args)
    except StageError as exc:
        print("[air-grasp workflow] {}".format(exc), file=sys.stderr)
        return exc.returncode if exc.returncode != 0 else 2
    except KeyboardInterrupt:
        print(
            "[air-grasp workflow] interrupted before a managed child was active",
            file=sys.stderr,
        )
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print("[air-grasp workflow] rejected: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
