"""Tamper-evident records for installed RH56 q6/coupled-air commissioning.

This module is deliberately hardware-free.  Importing it cannot import
``pylibfranka`` or the serial RH56 API, discover devices, or issue writes.
The motion CLI creates hardware clients only after its exact operator tokens
have passed; this module then canonicalizes and verifies the resulting record.
The embedded SHA-256 is an integrity checksum, not a trusted signature: human
review and independently retained file/source hashes remain required.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .rh56_hand_path import (
    MIN_DIRECTION_REVERSAL_BOOTSTRAP_UNITS,
    Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS,
    build_rh56_no_contact_execution_path,
    require_rh56_feedback_in_interval,
    rh56_feedback_envelope_policy,
    validate_rh56_feedback_envelope_policy,
)
from .snapshot import has_complete_official_provenance, load_snapshot_npz


SCHEMA_VERSION = 1
EVIDENCE_KIND = "installed_rh56_q6_coupled_air_commissioning"
STAGE1_PREREQUISITE_KIND = "installed_rh56_q6_stage1_pass_v1"
STAGE1_Q6_TARGET = 900
DISABLED_TARGETS = [-1, -1, -1, -1, -1, -1]
OPEN_TARGETS = [1000, 1000, 1000, 1000, 1000, 1000]
SNAPSHOT_CANDIDATE_BINDING_KIND = "official_anydexgrasp_snapshot_candidate_v1"
SNAPSHOT_AUTO_SELECTION = "highest_official_score_then_lowest_index"
SNAPSHOT_EXPLICIT_SELECTION = "explicit_candidate_index"
COMMISSION_BEND_OPEN_MIN_ANGLE = 980
COMMISSION_Q6_OPEN_MAX_TOLERANCE_UNITS = 30


@dataclass(frozen=True)
class EvidenceVerification:
    blockers: Tuple[str, ...]
    proposal: Optional[dict[str, Any]]

    @property
    def passed(self) -> bool:
        return not self.blockers


def _profile_q6_open_min_angle(evidence: Mapping[str, Any]) -> int:
    """Derive the installed q6 feedback endpoint from the bound profile."""

    profile = evidence.get("control_profile")
    snapshot = profile.get("snapshot") if isinstance(profile, Mapping) else None
    inspire = snapshot.get("inspire") if isinstance(snapshot, Mapping) else None
    if not isinstance(inspire, Mapping):
        raise ValueError(
            "control profile inspire object is missing for q6 open verification"
        )
    open_targets = inspire.get("open_targets")
    q6_range = inspire.get("thumb_rotate_validated_realtime_range")
    tolerance = inspire.get("arrival_tolerance_units")
    if (
        not isinstance(open_targets, list)
        or len(open_targets) != 6
        or any(isinstance(value, bool) or not isinstance(value, int) for value in open_targets)
    ):
        raise ValueError("control profile open_targets must contain six integers")
    if (
        not isinstance(q6_range, list)
        or len(q6_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in q6_range)
    ):
        raise ValueError(
            "control profile q6 validated range must contain two integers"
        )
    if isinstance(tolerance, bool) or not isinstance(tolerance, int):
        raise ValueError(
            "control profile arrival_tolerance_units must be an integer"
        )
    lower, upper = (int(value) for value in q6_range)
    target = int(open_targets[5])
    if not 0 <= lower <= upper <= 1000 or target != upper:
        raise ValueError(
            "control profile q6 open target must equal its validated range endpoint"
        )
    if not 0 <= int(tolerance) <= COMMISSION_Q6_OPEN_MAX_TOLERANCE_UNITS:
        raise ValueError(
            "control profile arrival tolerance exceeds the reviewed q6 open limit"
        )
    minimum = target - int(tolerance)
    if not lower <= minimum <= upper:
        raise ValueError(
            "control profile q6 open feedback threshold is outside its validated range"
        )
    return minimum


def sha256_file(path: Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_snapshot_candidate_binding(
    snapshot_path: Path,
    *,
    config: Mapping[str, Any],
    candidate_index: Optional[int] = None,
) -> dict[str, Any]:
    """Load and bind one official AnyDex candidate without hardware imports.

    An omitted ``candidate_index`` deliberately means the greatest official
    score, with the lowest immutable snapshot index as the stable tie-break.
    Passing ``0`` remains an explicit request for the first candidate.
    """

    source = Path(snapshot_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"official snapshot is missing: {source}")
    digest_before = sha256_file(source)
    snapshot = load_snapshot_npz(source)
    digest_after = sha256_file(source)
    if digest_after != digest_before:
        raise ValueError("official snapshot changed while it was being loaded")

    if snapshot.reference_frame != config.get("reference_frame"):
        raise ValueError(
            "official snapshot reference_frame differs from the control profile"
        )
    calibration = config.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("control profile calibration binding is missing")
    if not snapshot.calibration_id.strip():
        raise ValueError("official snapshot calibration_id must be non-empty")
    if snapshot.calibration_id != calibration.get("id"):
        raise ValueError(
            "official snapshot calibration_id differs from the control profile"
        )
    if not snapshot.camera_serial.strip():
        raise ValueError("official snapshot camera_serial must be non-empty")
    if snapshot.camera_serial != calibration.get("camera_serial"):
        raise ValueError(
            "official snapshot camera_serial differs from the control profile"
        )
    if "AnyDexGrasp official" not in snapshot.model_name:
        raise ValueError(
            f"snapshot is not an official AnyDexGrasp result: {snapshot.model_name!r}"
        )
    if not has_complete_official_provenance(snapshot):
        raise ValueError(
            "official snapshot model/checkpoint/source provenance is incomplete"
        )

    grasps = snapshot.grasps
    if grasps.count < 1:
        raise ValueError("official snapshot contains no grasp candidates")
    if grasps.hand_poses is None or grasps.hand_angles is None:
        raise ValueError("official snapshot lacks Inspire hand poses/targets")
    hand_poses = np.asarray(grasps.hand_poses, dtype=np.float64)
    hand_angles = np.asarray(grasps.hand_angles, dtype=np.float64)
    if hand_poses.shape != (grasps.count, 4, 4):
        raise ValueError("official snapshot hand_poses must have shape [K,4,4]")
    if hand_angles.shape != (grasps.count, 6):
        raise ValueError("official snapshot hand_angles must have shape [K,6]")

    scores = np.asarray(grasps.scores, dtype=np.float64)
    if candidate_index is None:
        selected = min(
            range(grasps.count),
            key=lambda index: (-float(scores[index]), int(index)),
        )
        selection_method = SNAPSHOT_AUTO_SELECTION
    else:
        if isinstance(candidate_index, (bool, np.bool_)) or not isinstance(
            candidate_index, (int, np.integer)
        ):
            raise ValueError("--candidate-index must be an integer")
        selected = int(candidate_index)
        if not 0 <= selected < grasps.count:
            raise ValueError(
                f"--candidate-index {selected} is outside official candidates "
                f"0..{grasps.count - 1}"
            )
        selection_method = SNAPSHOT_EXPLICIT_SELECTION

    targets_float = hand_angles[selected]
    if (
        not np.all(np.isfinite(targets_float))
        or np.any(targets_float < 0.0)
        or np.any(targets_float > 1000.0)
        or not np.array_equal(targets_float, np.rint(targets_float))
    ):
        raise ValueError(
            "selected official hand targets must be six integer registers in 0..1000"
        )
    targets = np.rint(targets_float).astype(np.int64)
    grasp_type = int(np.asarray(grasps.type_ids, dtype=np.int64)[selected])
    if not 1 <= grasp_type <= 8:
        raise ValueError("selected official Inspire grasp type must be in 1..8")

    source_index = None
    if grasps.source_indices is not None:
        source_index = int(np.asarray(grasps.source_indices, dtype=np.int64)[selected])
    width_m = None
    if grasps.widths_m is not None:
        numeric = float(np.asarray(grasps.widths_m, dtype=np.float64)[selected])
        width_m = numeric if math.isfinite(numeric) else None
    depth_m = None
    if grasps.depths_m is not None:
        numeric = float(np.asarray(grasps.depths_m, dtype=np.float64)[selected])
        depth_m = numeric if math.isfinite(numeric) else None

    return {
        "kind": SNAPSHOT_CANDIDATE_BINDING_KIND,
        "path": str(source),
        "file_sha256": digest_before,
        "selection_method": selection_method,
        "snapshot_provenance": {
            "reference_frame": str(snapshot.reference_frame),
            "frame_id": int(snapshot.frame_id),
            "timestamp_s": float(snapshot.timestamp_s),
            "calibration_id": str(snapshot.calibration_id),
            "camera_serial": str(snapshot.camera_serial),
            "T_reference_camera": np.asarray(
                snapshot.T_reference_camera, dtype=np.float64
            ).tolist(),
            "model_name": str(snapshot.model_name),
            "representation_checkpoint_sha256": str(
                snapshot.representation_checkpoint_sha256
            ),
            "decision_checkpoint_sha256s": list(
                snapshot.decision_checkpoint_sha256s
            ),
            "official_source_commit": str(snapshot.official_source_commit),
            "snapshot_selected_index": int(grasps.selected_index),
        },
        "candidate": {
            "index": selected,
            "official_score": float(scores[selected]),
            "type_id": grasp_type,
            "source_index": source_index,
            "collision_checked": bool(grasps.collision_checked[selected]),
            "collision_free": bool(grasps.collision_free[selected]),
            "width_m": width_m,
            "depth_m": depth_m,
            "canonical_pose": np.asarray(
                grasps.canonical_poses[selected], dtype=np.float64
            ).tolist(),
            "hand_pose": hand_poses[selected].tolist(),
            "hand_targets": [int(value) for value in targets],
        },
    }


def _verify_snapshot_candidate_binding(evidence: Mapping[str, Any]) -> list[str]:
    """Reload a bound snapshot and cross-check it against the motion request."""

    blockers: list[str] = []
    request = evidence.get("request")
    if not isinstance(request, Mapping):
        return ["request object is missing for snapshot-target verification"]
    target_source = request.get("target_source")
    binding = evidence.get("snapshot_candidate")

    # Evidence created before snapshot-driven commissioning had neither field.
    # New manual records spell the mode explicitly while preserving the same
    # q6=900/[900]*5 defaults and verification contract.
    if target_source is None:
        if binding is not None:
            blockers.append("legacy manual evidence must not contain a snapshot binding")
        return blockers
    if target_source == "manual_cli":
        if binding is not None:
            blockers.append("manual target evidence must not contain a snapshot binding")
        return blockers
    if target_source != "official_snapshot_candidate":
        return ["request.target_source is invalid"]
    if not isinstance(binding, Mapping):
        return ["official snapshot candidate binding is missing"]
    if request.get("coupled_closure_requested") is not True:
        blockers.append("official snapshot targets require coupled closure")

    profile = evidence.get("control_profile")
    config = profile.get("snapshot") if isinstance(profile, Mapping) else None
    if not isinstance(config, Mapping):
        return blockers + ["control profile snapshot is missing for snapshot reload"]
    candidate = binding.get("candidate")
    if not isinstance(candidate, Mapping):
        return blockers + ["snapshot candidate object is missing"]
    index = candidate.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        return blockers + ["snapshot candidate index is invalid"]
    method = binding.get("selection_method")
    if method == SNAPSHOT_AUTO_SELECTION:
        requested_index: Optional[int] = None
    elif method == SNAPSHOT_EXPLICIT_SELECTION:
        requested_index = index
    else:
        return blockers + ["snapshot candidate selection method is invalid"]

    try:
        rebuilt = build_snapshot_candidate_binding(
            Path(str(binding.get("path", ""))),
            config=config,
            candidate_index=requested_index,
        )
    except (OSError, ValueError) as exc:
        blockers.append(f"bound official snapshot reload failed: {exc}")
        return blockers
    if dict(binding) != rebuilt:
        blockers.append("bound official snapshot/candidate data changed or is inconsistent")

    targets = candidate.get("hand_targets")
    try:
        target_six = _six_ints(
            targets, "snapshot_candidate.candidate.hand_targets", maximum=1000
        )
    except ValueError as exc:
        blockers.append(str(exc))
    else:
        if request.get("coupled_targets") != list(target_six):
            blockers.append(
                "request.coupled_targets differs from the bound official candidate"
            )
        if request.get("target_q6") != target_six[5]:
            blockers.append(
                "request.target_q6 differs from the bound official candidate q6"
            )
    return blockers


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"evidence is not canonical JSON data: {exc}") from exc
    return text.encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def seal_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "integrity" in payload:
        raise ValueError("unsealed evidence payload must not contain integrity")
    result = copy.deepcopy(dict(payload))
    result["integrity"] = {
        "algorithm": "sha256-canonical-json-without-integrity",
        "payload_sha256": json_sha256(result),
    }
    return result


def atomic_write_evidence(path: Path, evidence: Mapping[str, Any]) -> str:
    """Atomically create a read-only evidence file; never overwrite a path."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"evidence output already exists: {output}")
    encoded = json.dumps(
        evidence,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        # A same-filesystem hard link supplies atomic no-replace semantics.
        os.link(temporary, output)
        temporary.unlink()
        temporary = None
        directory_fd = os.open(str(output.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256_file(output)


def load_evidence(path: Path) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load RH56 commissioning evidence {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("RH56 commissioning evidence root must be an object")
    return value, source


def _six_ints(
    value: Any,
    name: str,
    *,
    allow_disabled: bool = False,
    allow_signed: bool = False,
    maximum: int = 5000,
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError(f"{name} must contain exactly six integer values")
    lower = -5000 if allow_signed else (-1 if allow_disabled else 0)
    output = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{name}[{index}] must be an integer")
        if not lower <= item <= int(maximum):
            raise ValueError(f"{name}[{index}] is outside the evidence register range")
        output.append(item)
    return tuple(output)


def _feedback(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    phase = value.get("phase")
    if not isinstance(phase, str) or not phase:
        raise ValueError(f"{name}.phase must be a non-empty string")
    _six_ints(
        value.get("angle_targets"),
        f"{name}.angle_targets",
        allow_disabled=True,
        maximum=1000,
    )
    _six_ints(value.get("angles"), f"{name}.angles", maximum=1000)
    _six_ints(value.get("currents"), f"{name}.currents", allow_signed=True)
    _six_ints(value.get("errors"), f"{name}.errors", maximum=255)
    _six_ints(value.get("statuses"), f"{name}.statuses", maximum=255)
    _six_ints(value.get("temperatures"), f"{name}.temperatures", maximum=255)
    for optional_name in ("positions", "forces"):
        optional = value.get(optional_name)
        if optional is not None:
            _six_ints(
                optional,
                f"{name}.{optional_name}",
                allow_signed=(optional_name == "forces"),
            )
    elapsed = value.get("elapsed_s")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or elapsed < 0
    ):
        raise ValueError(f"{name}.elapsed_s must be finite and non-negative")
    return value


def _strict_request_int(
    request: Mapping[str, Any],
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    value = request.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"request.{name} must be an integer")
    if not int(minimum) <= value <= int(maximum):
        raise ValueError(
            f"request.{name} must be in {int(minimum)}..{int(maximum)}"
        )
    return value


def _strict_request_float(
    request: Mapping[str, Any],
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    value = request.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"request.{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or not float(minimum) <= numeric <= float(maximum):
        raise ValueError(
            f"request.{name} must be in {float(minimum):g}..{float(maximum):g}"
        )
    return numeric


def _waypoints(target: int, step: int) -> list[int]:
    points = list(range(1000 - step, target, -step))
    if not points or points[-1] != target:
        points.append(target)
    return points


def _coupled_waypoints(
    bend_targets: Sequence[int], target_q6: int, step: int
) -> list[list[int]]:
    current = [1000] * 5
    points: list[list[int]] = []
    for axis, target in enumerate(int(value) for value in bend_targets):
        while current[axis] != target:
            current[axis] = max(target, current[axis] - int(step))
            points.append(list(current) + [int(target_q6)])
    return points


def _coupled_return_waypoints(
    forward: Sequence[Sequence[int]], target_q6: int, step: int
) -> list[list[int]]:
    points = [list(int(value) for value in item) for item in forward]
    if not points:
        return []
    path = build_rh56_no_contact_execution_path(points[-1], step_units=int(step))
    return [
        list(item.command_targets)
        for item in path.waypoints
        if item.phase.startswith("bend_reverse_")
    ]


def _q6_return_waypoints(target_q6: int, step: int) -> list[int]:
    """Return path accepted by the installed commissioning entrypoint.

    q6=900 retains the separately observed direct-open Stage-1 exception.
    Wider ranges must use the canonical reversal bootstrap and small steps.
    """

    target = int(target_q6)
    if target == 900:
        return [1000]
    path = build_rh56_no_contact_execution_path(
        (1000, 1000, 1000, 1000, 1000, target),
        step_units=int(step),
    )
    return [
        int(item.command_targets[5])
        for item in path.waypoints
        if item.phase.startswith("q6_reverse_")
    ]


def _base_integrity_blockers(evidence: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if evidence.get("schema_version") != SCHEMA_VERSION:
        blockers.append(f"schema_version must be {SCHEMA_VERSION}")
    if evidence.get("kind") != EVIDENCE_KIND:
        blockers.append(f"kind must be {EVIDENCE_KIND}")
    integrity = evidence.get("integrity")
    if not isinstance(integrity, Mapping):
        blockers.append("integrity object is missing")
    else:
        expected = integrity.get("payload_sha256")
        unsigned = {key: value for key, value in evidence.items() if key != "integrity"}
        actual = json_sha256(unsigned)
        if expected != actual:
            blockers.append("canonical payload SHA-256 mismatch")
    return blockers


def _validate_bound_files(bindings: Any) -> list[str]:
    blockers: list[str] = []
    if not isinstance(bindings, list) or not bindings:
        return ["source_bindings must be a non-empty array"]
    names = set()
    for index, item in enumerate(bindings):
        if not isinstance(item, Mapping):
            blockers.append(f"source_bindings[{index}] is not an object")
            continue
        names.add(str(item.get("name", "")))
        source = Path(str(item.get("path", ""))).expanduser().resolve()
        expected = str(item.get("sha256", ""))
        if len(expected) != 64:
            blockers.append(f"source_bindings[{index}] has malformed SHA-256")
        elif not source.is_file():
            blockers.append(f"bound source file is missing: {source}")
        elif sha256_file(source) != expected:
            blockers.append(f"bound source file changed: {source}")
    required = {
        "commission_cli",
        "commission_evidence_module",
        "rh56_hand_path",
        "rh56_reset_open",
        "rh56_sequence_driver",
        "rh56_register_api",
        "franka_sequence_driver",
        "control_config_module",
        "adapter_mesh",
        "adapter_provenance",
        "actuator_to_joint_xlsx",
        "driver_to_angle_xls",
        "actuator_to_urdf_generator",
    }
    missing = sorted(required - names)
    if missing:
        blockers.append("required source bindings are missing: " + ", ".join(missing))
    return blockers


def _configs_equal_except_commissioning_fields(
    stage1_config: Mapping[str, Any], expected_config: Mapping[str, Any]
) -> bool:
    first = copy.deepcopy(dict(stage1_config))
    second = copy.deepcopy(dict(expected_config))
    try:
        first_inspire = first["inspire"]
        second_inspire = second["inspire"]
        for name in (
            "thumb_rotate_validated_realtime_range",
            "six_axis_coupled_closure_commissioned",
            "commissioned_air_closure_targets",
        ):
            first_inspire.pop(name)
            second_inspire.pop(name)
    except (KeyError, AttributeError):
        return False
    return first == second


def _stage1_binding_payload(
    evidence: Mapping[str, Any], evidence_path: Path
) -> dict[str, Any]:
    return {
        "kind": STAGE1_PREREQUISITE_KIND,
        "path": str(Path(evidence_path).expanduser().resolve()),
        "file_sha256": sha256_file(evidence_path),
        "payload_sha256": evidence["integrity"]["payload_sha256"],
        "run_id": evidence["run_id"],
        "completed_at_utc": evidence["completed_at_utc"],
        "target_q6": STAGE1_Q6_TARGET,
        "control_profile_parsed_sha256": evidence["control_profile"][
            "parsed_sha256"
        ],
    }


def _stage1_record_blockers(
    evidence: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any],
    expected_config_path: Path,
) -> list[str]:
    """Verify a historical Stage-1 record against the current code and profile lineage."""

    blockers = _base_integrity_blockers(evidence)
    blockers.extend(_validate_bound_files(evidence.get("source_bindings")))
    blockers.extend(_verify_franka_binding(evidence))

    profile = evidence.get("control_profile")
    if not isinstance(profile, Mapping):
        return blockers + ["Stage1 control_profile binding is missing"]
    snapshot = profile.get("snapshot")
    if not isinstance(snapshot, Mapping):
        return blockers + ["Stage1 control_profile snapshot is missing"]
    if json_sha256(snapshot) != profile.get("parsed_sha256"):
        blockers.append("Stage1 control_profile parsed snapshot hash mismatch")
    if Path(str(profile.get("path", ""))).expanduser().resolve() != Path(
        expected_config_path
    ).expanduser().resolve():
        blockers.append("Stage1 evidence belongs to a different control-profile path")
    if not _configs_equal_except_commissioning_fields(snapshot, expected_config):
        blockers.append(
            "Stage1 profile differs outside the three evidence-owned commissioning fields"
        )
    inspire = snapshot.get("inspire")
    if not isinstance(inspire, Mapping):
        blockers.append("Stage1 profile has no inspire object")
    else:
        if inspire.get("thumb_rotate_validated_realtime_range") != [900, 1000]:
            blockers.append("Stage1 evidence must start from q6 range [900,1000]")
        if inspire.get("six_axis_coupled_closure_commissioned") is not False:
            blockers.append("Stage1 evidence must predate coupled-closure commissioning")
        if inspire.get("commissioned_air_closure_targets") != []:
            blockers.append("Stage1 evidence must predate every exact air target")

    request = evidence.get("request")
    result = evidence.get("result")
    is_stage1_request = False
    if not isinstance(request, Mapping):
        blockers.append("Stage1 request object is missing")
    else:
        is_stage1_request = (
            request.get("target_q6") == STAGE1_Q6_TARGET
            and request.get("coupled_closure_requested") is False
            and request.get("coupled_targets") is None
        )
        if request.get("target_q6") != STAGE1_Q6_TARGET:
            blockers.append("Stage1 prerequisite target_q6 must be exactly 900")
        if request.get("coupled_closure_requested") is not False:
            blockers.append("Stage1 prerequisite must be q6-only")
        if request.get("coupled_targets") is not None:
            blockers.append("Stage1 prerequisite must not contain coupled targets")
        if request.get("q6_return_strategy") != "direct_stage1_v1":
            blockers.append("Stage1 prerequisite has the wrong return strategy")
    if not isinstance(result, Mapping) or result.get("coupled_closure_pass") is not False:
        blockers.append("Stage1 prerequisite must not claim a coupled-closure pass")
    if is_stage1_request:
        blockers.extend(_verify_motion_record(evidence))
    return list(dict.fromkeys(blockers))


def build_stage1_prerequisite_binding(
    evidence_path: Path,
    *,
    expected_config: Mapping[str, Any],
    expected_config_path: Path,
) -> dict[str, Any]:
    """Load and bind a genuine q6=900 q6-only PASS before any wide-range run."""

    evidence, resolved = load_evidence(evidence_path)
    blockers = _stage1_record_blockers(
        evidence,
        expected_config=expected_config,
        expected_config_path=expected_config_path,
    )
    if blockers:
        raise ValueError("Stage1 prerequisite is LOCKED: " + "; ".join(blockers))
    return _stage1_binding_payload(evidence, resolved)


def _verify_stage1_prerequisite_binding(
    evidence: Mapping[str, Any], target_q6: int
) -> list[str]:
    binding = evidence.get("stage1_prerequisite")
    if int(target_q6) >= STAGE1_Q6_TARGET:
        if binding is not None:
            return ["q6>=900 evidence must not claim a Stage1 prerequisite binding"]
        return []
    if not isinstance(binding, Mapping):
        return ["q6<900 requires a bound formal Stage1 PASS evidence"]

    profile = evidence.get("control_profile")
    if not isinstance(profile, Mapping) or not isinstance(
        profile.get("snapshot"), Mapping
    ):
        return ["wide-range evidence lacks its expected control profile"]
    expected_config = profile["snapshot"]
    expected_config_path = Path(str(profile.get("path", ""))).expanduser().resolve()
    source = Path(str(binding.get("path", ""))).expanduser().resolve()
    if not source.is_file():
        return [f"bound Stage1 evidence is missing: {source}"]
    if sha256_file(source) != binding.get("file_sha256"):
        return ["bound Stage1 evidence file SHA-256 changed"]
    try:
        stage1, resolved = load_evidence(source)
    except ValueError as exc:
        return [str(exc)]
    blockers: list[str] = []
    try:
        expected_binding = _stage1_binding_payload(stage1, resolved)
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append(f"bound Stage1 evidence cannot form a binding: {exc}")
    else:
        if dict(binding) != expected_binding:
            blockers.append("Stage1 prerequisite binding fields/hash do not match the file")
    blockers.extend(
        _stage1_record_blockers(
            stage1,
            expected_config=expected_config,
            expected_config_path=expected_config_path,
        )
    )
    completed = str(stage1.get("completed_at_utc", ""))
    started = str(evidence.get("started_at_utc", ""))
    if not completed or not started or completed > started:
        blockers.append("Stage1 prerequisite must complete before the wide-range run starts")
    return list(dict.fromkeys(blockers))


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    return numeric


def _finite_list(value: Any, size: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must contain {size} finite numbers")
    return [_finite_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _verify_franka_binding(evidence: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    profile = evidence.get("control_profile")
    franka_record = evidence.get("franka_read_only")
    if not isinstance(profile, Mapping) or not isinstance(profile.get("snapshot"), Mapping):
        return ["Franka verification lacks a bound control profile"]
    config_franka = profile["snapshot"].get("franka")
    if not isinstance(config_franka, Mapping):
        return ["bound control profile has no franka object"]
    if not isinstance(franka_record, Mapping):
        return ["franka_read_only evidence is missing"]
    if franka_record.get("connection") != "read_once_only_no_controller_no_robot_write":
        blockers.append("Franka evidence is not marked read_once-only")
    if franka_record.get("verified") is not True:
        blockers.append("Franka read-only gate was not verified")
    continuous_gate = franka_record.get("continuous_gate")
    if not isinstance(continuous_gate, Mapping):
        blockers.append("continuous Franka read-only gate evidence is missing")
        continuous_last = None
    else:
        check_count = continuous_gate.get("check_count")
        if (
            isinstance(check_count, bool)
            or not isinstance(check_count, int)
            or check_count <= 0
        ):
            blockers.append("continuous Franka gate check_count must be positive")
        if continuous_gate.get("failure") is not None:
            blockers.append("continuous Franka gate recorded a failure")
        continuous_last = continuous_gate.get("last_success")
    dynamics = config_franka.get("expected_end_effector")
    if not isinstance(dynamics, Mapping):
        return blockers + ["bound profile has no expected end-effector dynamics"]
    expected_F_T_EE = config_franka.get("expected_F_T_EE")
    bound_states = (
        ("initial", franka_record.get("initial")),
        ("final", franka_record.get("final")),
        ("continuous_last", continuous_last),
    )
    for endpoint, state in bound_states:
        if not isinstance(state, Mapping):
            blockers.append(f"Franka {endpoint} read_once state is missing")
            continue
        mode = str(state.get("robot_mode", "")).lower().rsplit(".", 1)[-1]
        if mode not in ("idle", "kidle"):
            blockers.append(f"Franka {endpoint} state is not Idle")
        try:
            actual_rows = state.get("F_T_EE")
            if not isinstance(actual_rows, list) or len(actual_rows) != 4:
                raise ValueError("F_T_EE must be 4x4")
            actual_transform = [
                _finite_number(item, f"Franka {endpoint}.F_T_EE")
                for row in actual_rows
                for item in row
            ]
            if not isinstance(expected_F_T_EE, list) or len(expected_F_T_EE) != 4:
                raise ValueError("profile F_T_EE must be 4x4")
            expected_transform = [
                _finite_number(item, "profile F_T_EE")
                for row in expected_F_T_EE
                for item in row
            ]
            if len(actual_transform) != 16 or len(expected_transform) != 16:
                raise ValueError("F_T_EE must be 4x4")
        except (TypeError, ValueError) as exc:
            blockers.append(str(exc))
        else:
            if any(
                abs(a - b) > 1.0e-8
                for a, b in zip(actual_transform, expected_transform)
            ):
                blockers.append(f"Franka {endpoint} F_T_EE differs from profile")
        comparisons = (
            ("m_ee_kg", state.get("m_ee_kg"), dynamics.get("mass_kg"), 1.0e-4),
            ("m_load_kg", state.get("m_load_kg"), 0.0, 1.0e-5),
            ("m_total_kg", state.get("m_total_kg"), dynamics.get("mass_kg"), 1.0e-4),
        )
        for name, actual_value, expected_value, tolerance in comparisons:
            try:
                actual = _finite_number(actual_value, f"Franka {endpoint}.{name}")
                expected = _finite_number(expected_value, f"profile {name}")
            except ValueError as exc:
                blockers.append(str(exc))
            else:
                if abs(actual - expected) > tolerance:
                    blockers.append(f"Franka {endpoint}.{name} differs from profile")
        try:
            actual_com = _finite_list(
                state.get("F_x_Cee_m"), 3, f"Franka {endpoint}.F_x_Cee_m"
            )
            expected_com = _finite_list(
                dynamics.get("F_x_Cee_m"), 3, "profile F_x_Cee_m"
            )
        except ValueError as exc:
            blockers.append(str(exc))
        else:
            if any(abs(a - b) > 1.0e-4 for a, b in zip(actual_com, expected_com)):
                blockers.append(f"Franka {endpoint} CoM differs from profile")
        try:
            actual_inertia = _finite_list(
                state.get("I_ee_kg_m2"), 9, f"Franka {endpoint}.I_ee_kg_m2"
            )
            expected_matrix = dynamics.get("inertia_kg_m2")
            if not isinstance(expected_matrix, list) or len(expected_matrix) != 3:
                raise ValueError("profile inertia_kg_m2 must be 3x3")
            expected_inertia = [
                _finite_number(item, "profile inertia_kg_m2")
                for row in expected_matrix
                for item in row
            ]
            if len(expected_inertia) != 9:
                raise ValueError("profile inertia_kg_m2 must be 3x3")
        except (TypeError, ValueError) as exc:
            blockers.append(str(exc))
        else:
            if any(
                abs(a - b) > 1.0e-5
                for a, b in zip(actual_inertia, expected_inertia)
            ):
                blockers.append(f"Franka {endpoint} inertia differs from profile")
    return blockers


def _verify_motion_record(evidence: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    result = evidence.get("result")
    request = evidence.get("request")
    observations = evidence.get("observations")
    final = evidence.get("final")
    if not isinstance(result, Mapping) or result.get("status") != "pass":
        blockers.append("commissioning result is not pass")
        return blockers
    if not isinstance(request, Mapping) or not isinstance(observations, Mapping):
        return blockers + ["request/observations objects are missing"]
    if not isinstance(final, Mapping):
        return blockers + ["final object is missing"]
    if evidence.get("motion_authorized") is not False:
        blockers.append("commissioning evidence motion_authorized must remain false")
    confirmations = evidence.get("operator_confirmations")
    required_confirmations = (
        "installed_on_fr3",
        "24v_cutoff_ready",
        "franka_stop_ready",
        "workspace_clear",
        "no_contact_PLA_scope",
    )
    if not isinstance(confirmations, Mapping) or any(
        confirmations.get(name) is not True for name in required_confirmations
    ):
        blockers.append("one or more mandatory operator confirmations are absent")
    if request.get("safety_scope") != "installed_on_FR3_PLA_low_speed_unloaded_free_air_only":
        blockers.append("commissioning safety scope is not the reviewed PLA free-air scope")
    if request.get("speed") != 40 or request.get("force_limit_g") != 80:
        blockers.append("commissioning speed/force differs from fixed 40/80g")
    if request.get("aggregate_current_policy") != "telemetry_only_device_limits_remain_active":
        blockers.append("aggregate current policy binding is invalid")
    if result.get("operation_error") is not None:
        blockers.append("passing result contains an operation_error")
    if result.get("stop_error") is not None:
        blockers.append("passing result contains a stop_error")
    for name in (
        "adopted_disabled_verified",
        "q6_sweep_pass",
        "q6_return_pass",
        "reopened_and_verified",
        "disabled_verified",
    ):
        if result.get(name) is not True:
            blockers.append(f"result.{name} is not true")

    try:
        target = _strict_request_int(request, "target_q6", 0, 1000)
        step = _strict_request_int(request, "step_units", 10, 50)
        tolerance = _strict_request_int(request, "angle_tolerance_units", 0, 100)
        q6_tolerance = _strict_request_int(
            request, "q6_endpoint_tolerance_units", 0, 20
        )
        q6_reverse_hysteresis_tolerance = _strict_request_int(
            request, "q6_reverse_hysteresis_tolerance_units", 30, 30
        )
        stable_required = _strict_request_int(
            request, "endpoint_stable_samples", 2, 10
        )
        current_cap = _strict_request_int(request, "max_axis_current_ma", 1, 1400)
        stop_current_cap = _strict_request_int(
            request, "stop_max_axis_current_ma", 1, 400
        )
        inactive_drift = _strict_request_int(
            request, "max_inactive_drift_units", 1, 20
        )
        motion_timeout = _strict_request_float(
            request, "motion_timeout_s_per_step", 5.0, 30.0
        )
    except ValueError as exc:
        return blockers + [str(exc)]
    del motion_timeout
    try:
        q6_open_min_angle = _profile_q6_open_min_angle(evidence)
    except ValueError as exc:
        return blockers + [str(exc)]
    recorded_q6_open_min = request.get("q6_open_min_angle")
    if recorded_q6_open_min is None:
        blockers.append("request.q6_open_min_angle is missing")
    elif (
        isinstance(recorded_q6_open_min, bool)
        or not isinstance(recorded_q6_open_min, int)
        or int(recorded_q6_open_min) != q6_open_min_angle
    ):
        blockers.append(
            "request.q6_open_min_angle differs from the bound profile"
        )
    if not any(
        isinstance(item, Mapping) and item.get("name") == "rh56_reset_open"
        for item in evidence.get("source_bindings", ())
    ):
        blockers.append(
            "profile-aware q6 open evidence is missing rh56_reset_open source binding"
        )
    feedback_envelope_policy = rh56_feedback_envelope_policy(tolerance)
    try:
        validate_rh56_feedback_envelope_policy(
            request.get("feedback_envelope_policy"), tolerance
        )
    except ValueError as exc:
        blockers.append(str(exc))
    if result.get("feedback_envelope_enforced") is not True:
        blockers.append("runtime feedback envelope was not enforced")
    if (
        result.get("feedback_envelope_policy_sha256")
        != feedback_envelope_policy["sha256"]
    ):
        blockers.append("runtime feedback-envelope policy hash differs from request")
    blockers.extend(_verify_stage1_prerequisite_binding(evidence, target))
    if q6_tolerance != min(20, tolerance):
        blockers.append(
            "q6 endpoint tolerance must equal min(20, angle_tolerance_units)"
        )
    if q6_reverse_hysteresis_tolerance != Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS:
        blockers.append("q6 reverse hysteresis tolerance must equal 30 units")
    expected_waypoints = _waypoints(target, step)
    expected_return_strategy = (
        "direct_stage1_v1"
        if target == 900
        else "canonical_reverse_bootstrap_v3"
    )
    if request.get("q6_return_strategy") != expected_return_strategy:
        blockers.append(
            "q6 return strategy does not match the target-specific safe path"
        )
    expected_return_waypoints = _q6_return_waypoints(target, step)
    if request.get("requested_q6_range") != [target, 1000]:
        blockers.append("requested_q6_range does not bind target_q6..1000")
    if request.get("q6_waypoints") != expected_waypoints:
        blockers.append("q6 waypoint list is not the exact descending step path")
    if request.get("q6_return_waypoints") != expected_return_waypoints:
        blockers.append("q6 return waypoint list is not the target-specific safe return path")

    all_feedback = observations.get("all_feedback")
    validated_all_feedback: list[Mapping[str, Any]] = []
    if not isinstance(all_feedback, list) or not all_feedback:
        blockers.append("all_feedback must be a non-empty array")
    else:
        try:
            validated_all_feedback = [
                _feedback(sample, f"all_feedback[{index}]")
                for index, sample in enumerate(all_feedback)
            ]
        except ValueError as exc:
            blockers.append(str(exc))
            validated_all_feedback = []

    adopt_samples = observations.get("adopt_disable_feedback")
    expected_adopt_samples = [
        sample
        for sample in validated_all_feedback
        if sample.get("phase") == "adopt_disable_verify"
    ]
    if (
        not isinstance(adopt_samples, list)
        or adopt_samples != expected_adopt_samples
        or len(adopt_samples) < 2
    ):
        blockers.append(
            "adopt-disable feedback is missing or differs from all_feedback"
        )
    else:
        for sample in adopt_samples:
            if sample["angle_targets"] != DISABLED_TARGETS:
                blockers.append("adopt-disable did not prove all-six ANGLE_SET=-1")
                break
            if any(sample["errors"]) or any(
                status not in (2, 255) for status in sample["statuses"]
            ):
                blockers.append("adopt-disable feedback is not fault-free and idle")
                break
            if max(sample["temperatures"]) >= 60 or any(
                abs(current) > stop_current_cap for current in sample["currents"]
            ):
                blockers.append("adopt-disable feedback is hot or still drawing current")
                break

    device = evidence.get("rh56_device")
    initial_snapshot = device.get("initial_snapshot") if isinstance(device, Mapping) else None
    try:
        initial_angles = _six_ints(
            initial_snapshot.get("angles")
            if isinstance(initial_snapshot, Mapping)
            else None,
            "rh56_device.initial_snapshot.angles",
        )
    except ValueError as exc:
        blockers.append(str(exc))
    else:
        if not q6_open_min_angle <= int(initial_angles[5]) <= 1000:
            blockers.append(
                "fresh commissioning initial q6 is outside the "
                "profile-derived open band"
            )
    if (
        recorded_q6_open_min is not None
        and observations.get("preopen_reset_q6_waypoints") != []
    ):
        blockers.append(
            "fresh commissioning performed an unauthorized pre-open q6 recovery"
        )
    try:
        device_current_limits = _six_ints(
            initial_snapshot.get("current_limits")
            if isinstance(initial_snapshot, Mapping)
            else None,
            "rh56_device.initial_snapshot.current_limits",
        )
    except ValueError as exc:
        blockers.append(str(exc))
        current_caps = (current_cap,) * 6
    else:
        if any(value <= 0 for value in device_current_limits):
            blockers.append("RH56 device CURRENT_LIMIT contains a non-positive value")
        current_caps = tuple(min(current_cap, value) for value in device_current_limits)

    groups = observations.get("q6_steps")
    observed_q6: list[int] = []
    preflight_q6 = None
    preflight_samples = [
        sample
        for sample in validated_all_feedback
        if sample.get("phase") == "q6_sweep_preflight"
    ]
    if len(preflight_samples) != 1:
        blockers.append("q6 sweep requires exactly one preflight feedback sample")
        inactive_reference = None
    else:
        preflight = preflight_samples[0]
        preflight_q6 = int(preflight["angles"][5])
        inactive_reference = tuple(int(v) for v in preflight["angles"][:5])
        if preflight["angle_targets"] != DISABLED_TARGETS:
            blockers.append("q6 sweep preflight does not have all-six ANGLE_SET=-1")
        if any(preflight["errors"]) or max(preflight["temperatures"]) >= 60:
            blockers.append("q6 sweep preflight contains fault/temperature feedback")
        if (
            any(
                angle < COMMISSION_BEND_OPEN_MIN_ANGLE
                for angle in preflight["angles"][:5]
            )
            or int(preflight["angles"][5]) < q6_open_min_angle
            or any(
            status not in (2, 255) for status in preflight["statuses"]
            )
        ):
            blockers.append("q6 sweep preflight does not prove six open idle axes")
        if any(
            abs(value) > cap
            for value, cap in zip(preflight["currents"], current_caps)
        ):
            blockers.append("q6 sweep preflight exceeded current cap")
    previous_actual = preflight_q6
    previous_command = 1000
    previous_configuration = tuple(OPEN_TARGETS)
    previous_feedback_phase = "start_open"
    if not isinstance(groups, list) or len(groups) != len(expected_waypoints):
        blockers.append("q6 step evidence count does not match requested waypoints")
    else:
        for group_index, (group, waypoint) in enumerate(zip(groups, expected_waypoints)):
            if not isinstance(group, Mapping) or group.get("target_q6") != waypoint:
                blockers.append(f"q6 step {group_index} target binding is invalid")
                continue
            samples = group.get("feedback")
            if not isinstance(samples, list) or len(samples) < stable_required:
                blockers.append(f"q6 step {waypoint} lacks stable feedback samples")
                continue
            phase = f"q6_step_{waypoint:04d}"
            phase_samples = [
                sample
                for sample in validated_all_feedback
                if sample.get("phase") == phase
            ]
            if samples != phase_samples:
                blockers.append(
                    f"q6 step {waypoint} feedback differs from all_feedback"
                )
                continue
            validated = []
            try:
                validated = [
                    _feedback(sample, f"q6 step {waypoint} feedback[{sample_index}]")
                    for sample_index, sample in enumerate(samples)
                ]
            except ValueError as exc:
                blockers.append(str(exc))
                continue
            expected_targets = [-1, -1, -1, -1, -1, waypoint]
            current_configuration = (1000, 1000, 1000, 1000, 1000, waypoint)
            current_feedback_phase = "q6_forward_{:04d}".format(group_index)
            opposite_slack = max(4, tolerance // 2)
            for sample in validated:
                observed_q6.append(int(sample["angles"][5]))
                if sample["phase"] != phase:
                    blockers.append(f"q6 step {waypoint} phase binding changed")
                    break
                if sample["angle_targets"] != expected_targets:
                    blockers.append(f"q6 step {waypoint} target readback changed")
                    break
                try:
                    require_rh56_feedback_in_interval(
                        sample["angles"],
                        previous_configuration,
                        current_configuration,
                        tolerance,
                        previous_phase=previous_feedback_phase,
                        current_phase=current_feedback_phase,
                        name=f"q6 step {waypoint} ANGLE_ACT",
                    )
                except ValueError as exc:
                    blockers.append(str(exc))
                    break
                if any(sample["errors"]):
                    blockers.append(f"q6 step {waypoint} contains actuator ERROR")
                    break
                if max(sample["temperatures"]) >= 60:
                    blockers.append(f"q6 step {waypoint} reached 60 C")
                    break
                if any(
                    abs(value) > cap
                    for value, cap in zip(sample["currents"], current_caps)
                ):
                    blockers.append(f"q6 step {waypoint} exceeded current cap")
                    break
                if any(angle < 980 for angle in sample["angles"][:5]) or any(
                    status not in (2, 255) for status in sample["statuses"][:5]
                ):
                    blockers.append(f"q6 step {waypoint} moved an inactive bend axis")
                    break
                if inactive_reference is None or any(
                    abs(int(angle) - int(reference)) > inactive_drift
                    for angle, reference in zip(
                        sample["angles"][:5], inactive_reference
                    )
                ):
                    blockers.append(
                        f"q6 step {waypoint} exceeded inactive-axis drift"
                    )
                    break
                if sample["statuses"][5] not in (0, 1, 2):
                    blockers.append(f"q6 step {waypoint} contains contact/fault status")
                    break
                if (
                    previous_actual is not None
                    and int(sample["angles"][5]) > previous_actual + opposite_slack
                ):
                    blockers.append(
                        f"q6 step {waypoint} moved opposite the descending command"
                    )
                    break
            for sample in validated[-stable_required:]:
                if (
                    sample["statuses"][5] != 2
                    or abs(sample["angles"][5] - waypoint) > q6_tolerance
                ):
                    blockers.append(f"q6 step {waypoint} has no stable endpoint")
                    break
            if validated and previous_actual is not None:
                endpoint_actual = int(validated[-1]["angles"][5])
                commanded_delta = previous_command - waypoint
                first_already_in_band = (
                    group_index == 0
                    and len(expected_waypoints) > 1
                    and abs(previous_actual - waypoint) <= q6_tolerance
                )
                required_progress = (
                    0
                    if first_already_in_band or commanded_delta < 10
                    else max(3, commanded_delta // 2)
                )
                if previous_actual - endpoint_actual < required_progress:
                    blockers.append(
                        f"q6 step {waypoint} lacks measurable directional progress"
                    )
                previous_actual = endpoint_actual
                previous_command = waypoint
                previous_configuration = current_configuration
                previous_feedback_phase = current_feedback_phase
    recorded_actual_range = observations.get("actual_q6_range")
    expected_actual_range = (
        [min(observed_q6), max(observed_q6)] if observed_q6 else None
    )
    if expected_actual_range is None:
        blockers.append("q6 step feedback is empty")
    if recorded_actual_range is None or recorded_actual_range != expected_actual_range:
        blockers.append("recorded actual_q6_range disagrees with step feedback")

    coupled_requested = bool(request.get("coupled_closure_requested"))
    coupled_pass = result.get("coupled_closure_pass")
    coupled_samples = observations.get("coupled_air_close_feedback")
    if coupled_requested:
        if not isinstance(confirmations, Mapping) or confirmations.get(
            "coupled_air_close_confirmed"
        ) is not True:
            blockers.append("coupled air-close confirmation is absent")
        if coupled_pass is not True:
            blockers.append("requested coupled air closure did not pass")
        targets = request.get("coupled_targets")
        try:
            expected = _six_ints(
                targets, "request.coupled_targets", maximum=1000
            )
            coupled_step = _strict_request_int(
                request, "coupled_step_units", 10, 50
            )
        except ValueError as exc:
            blockers.append(str(exc))
            expected = ()
            coupled_step = 0
        if expected and expected[5] != target:
            blockers.append("coupled target q6 does not equal target_q6")
        if expected and any(value < 800 or value > 950 for value in expected[:5]):
            if not isinstance(confirmations, Mapping) or confirmations.get(
                "exact_air_target_confirmed"
            ) is not True:
                blockers.append("exact candidate air-target confirmation is absent")

        expected_coupled_path = (
            _coupled_waypoints(expected[:5], target, coupled_step)
            if expected and coupled_step
            else []
        )
        expected_coupled_return = _coupled_return_waypoints(
            expected_coupled_path, target, coupled_step
        )
        if request.get("coupled_command_waypoints") != expected_coupled_path:
            blockers.append(
                "coupled command waypoints are not the exact one-axis-at-a-time path"
            )
        if (
            request.get("coupled_return_command_waypoints")
            != expected_coupled_return
        ):
            blockers.append("coupled return waypoints are not the canonical reversal-bootstrap path")

        close_groups = observations.get("coupled_air_close_steps")
        return_bend_groups = observations.get("coupled_air_return_steps")
        flattened_close: list[Mapping[str, Any]] = []

        def verify_coupled_groups(
            groups_value: Any,
            waypoints_value: Sequence[Sequence[int]],
            *,
            phase_prefix: str,
            opening: bool,
        ) -> None:
            if not isinstance(groups_value, list) or len(groups_value) != len(
                waypoints_value
            ):
                blockers.append(
                    f"{phase_prefix} evidence count does not match command waypoints"
                )
                return
            previous_endpoint = (
                list(expected) if opening and expected else [1000] * 5 + [target]
            )
            previous_endpoint_phase = (
                "bend_forward_complete" if opening else "q6_forward_complete"
            )
            reversal_bootstrapped_axes: set[int] = set()
            for index, (group, waypoint_value) in enumerate(
                zip(groups_value, waypoints_value)
            ):
                waypoint = [int(value) for value in waypoint_value]
                phase = f"{phase_prefix}_{index:04d}"
                current_endpoint_phase = (
                    "bend_reverse_{:04d}".format(index)
                    if opening
                    else "bend_forward_{:04d}".format(index)
                )
                if not isinstance(group, Mapping) or group.get("target") != waypoint:
                    blockers.append(f"{phase} target binding is invalid")
                    continue
                samples = group.get("feedback")
                exact_phase_samples = [
                    sample
                    for sample in validated_all_feedback
                    if sample.get("phase") == phase
                ]
                if (
                    not isinstance(samples, list)
                    or len(samples) < stable_required
                    or samples != exact_phase_samples
                ):
                    blockers.append(f"{phase} lacks bound stable feedback samples")
                    continue
                try:
                    validated = [
                        _feedback(sample, f"{phase}.feedback[{sample_index}]")
                        for sample_index, sample in enumerate(samples)
                    ]
                except ValueError as exc:
                    blockers.append(str(exc))
                    continue
                if not opening:
                    flattened_close.extend(validated)
                changed = [
                    axis
                    for axis, (before, after) in enumerate(
                        zip(previous_endpoint[:5], waypoint[:5])
                    )
                    if before != after
                ]
                if len(changed) != 1:
                    blockers.append(f"{phase} does not move exactly one bend axis")
                elif changed:
                    axis = changed[0]
                    allowed_step = coupled_step
                    if opening and axis not in reversal_bootstrapped_axes:
                        allowed_step = max(
                            MIN_DIRECTION_REVERSAL_BOOTSTRAP_UNITS,
                            coupled_step,
                        )
                        reversal_bootstrapped_axes.add(axis)
                    if abs(waypoint[axis] - previous_endpoint[axis]) > allowed_step:
                        blockers.append(f"{phase} exceeds the coupled step bound")
                if waypoint[5] != target:
                    blockers.append(f"{phase} changed q6 during bend motion")
                for sample in validated:
                    if sample["angle_targets"] != waypoint:
                        blockers.append(f"{phase} target readback changed")
                        break
                    try:
                        require_rh56_feedback_in_interval(
                            sample["angles"],
                            previous_endpoint,
                            waypoint,
                            tolerance,
                            previous_phase=previous_endpoint_phase,
                            current_phase=current_endpoint_phase,
                            name=f"{phase} ANGLE_ACT",
                        )
                    except ValueError as exc:
                        blockers.append(str(exc))
                        break
                    if any(sample["errors"]) or any(
                        status == 3 for status in sample["statuses"]
                    ):
                        blockers.append(f"{phase} contains fault/contact feedback")
                        break
                    if max(sample["temperatures"]) >= 60 or any(
                        status not in (0, 1, 2) for status in sample["statuses"]
                    ):
                        blockers.append(f"{phase} contains temperature/fault status")
                        break
                    if any(
                        abs(value) > cap
                        for value, cap in zip(sample["currents"], current_caps)
                    ):
                        blockers.append(f"{phase} exceeded current cap")
                        break
                for sample in validated[-stable_required:]:
                    if any(
                        status != 2 or abs(actual - wanted) > tolerance
                        for actual, wanted, status in zip(
                            sample["angles"], waypoint, sample["statuses"]
                        )
                    ):
                        blockers.append(f"{phase} has no stable no-contact endpoint")
                        break
                previous_endpoint = waypoint
                previous_endpoint_phase = current_endpoint_phase

        verify_coupled_groups(
            close_groups,
            expected_coupled_path,
            phase_prefix="coupled_air_close_step",
            opening=False,
        )
        verify_coupled_groups(
            return_bend_groups,
            expected_coupled_return,
            phase_prefix="coupled_air_return_step",
            opening=True,
        )
        if not isinstance(coupled_samples, list) or coupled_samples != flattened_close:
            blockers.append("coupled closure feedback differs from grouped all_feedback")
    else:
        if coupled_pass not in (False, None):
            blockers.append("coupled_closure_pass cannot be true when it was not requested")
        if request.get("coupled_targets") is not None:
            blockers.append("coupled_targets must be null when closure was not requested")
        for field in (
            "coupled_step_units",
            "coupled_command_waypoints",
            "coupled_return_command_waypoints",
        ):
            if request.get(field) is not None:
                blockers.append(f"{field} must be null when closure was not requested")
        if coupled_samples not in ([], None):
            blockers.append("coupled feedback must be empty when closure was not requested")
        if observations.get("coupled_air_close_steps") not in ([], None):
            blockers.append("coupled close steps must be empty when not requested")
        if observations.get("coupled_air_return_steps") not in ([], None):
            blockers.append("coupled return steps must be empty when not requested")

    return_preflight_samples = [
        sample
        for sample in validated_all_feedback
        if sample.get("phase") == "q6_return_preflight"
    ]
    if len(return_preflight_samples) != 1:
        blockers.append("q6 return requires exactly one preflight feedback sample")
        return_previous_actual = None
        return_inactive_reference = None
    else:
        return_preflight = return_preflight_samples[0]
        return_previous_actual = int(return_preflight["angles"][5])
        return_inactive_reference = tuple(
            int(value) for value in return_preflight["angles"][:5]
        )
        if return_preflight["angle_targets"] != DISABLED_TARGETS:
            blockers.append("q6 return preflight does not have all-six ANGLE_SET=-1")
        if any(return_preflight["errors"]) or max(
            return_preflight["temperatures"]
        ) >= 60:
            blockers.append("q6 return preflight contains fault/temperature feedback")
        if any(angle < 980 for angle in return_preflight["angles"][:5]) or any(
            status not in (2, 255) for status in return_preflight["statuses"]
        ):
            blockers.append("q6 return preflight does not prove open idle axes")
        if abs(return_previous_actual - target) > q6_tolerance:
            blockers.append("q6 return preflight is not at the commissioned target")
        if any(
            abs(value) > cap
            for value, cap in zip(return_preflight["currents"], current_caps)
        ):
            blockers.append("q6 return preflight exceeded current cap")

    return_groups = observations.get("q6_return_steps")
    observed_return_q6: list[int] = []
    return_previous_command = target
    return_previous_configuration = (1000, 1000, 1000, 1000, 1000, target)
    return_previous_feedback_phase = "bend_reverse_complete"
    if not isinstance(return_groups, list) or len(return_groups) != len(
        expected_return_waypoints
    ):
        blockers.append("q6 return evidence count does not match requested waypoints")
    else:
        for group_index, (group, waypoint) in enumerate(
            zip(return_groups, expected_return_waypoints)
        ):
            if not isinstance(group, Mapping) or group.get("target_q6") != waypoint:
                blockers.append(f"q6 return step {group_index} target binding is invalid")
                continue
            samples = group.get("feedback")
            if not isinstance(samples, list) or len(samples) < stable_required:
                blockers.append(
                    f"q6 return step {waypoint} lacks stable feedback samples"
                )
                continue
            phase = f"q6_return_{waypoint:04d}"
            if samples != [
                sample
                for sample in validated_all_feedback
                if sample.get("phase") == phase
            ]:
                blockers.append(
                    f"q6 return step {waypoint} feedback differs from all_feedback"
                )
                continue
            try:
                validated_return = [
                    _feedback(
                        sample,
                        f"q6 return step {waypoint} feedback[{sample_index}]",
                    )
                    for sample_index, sample in enumerate(samples)
                ]
            except ValueError as exc:
                blockers.append(str(exc))
                continue
            expected_targets = [-1, -1, -1, -1, -1, waypoint]
            return_current_configuration = (
                1000,
                1000,
                1000,
                1000,
                1000,
                waypoint,
            )
            return_current_feedback_phase = "q6_reverse_{:04d}".format(
                group_index
            )
            opposite_slack = max(4, tolerance // 2)
            for sample in validated_return:
                observed_return_q6.append(int(sample["angles"][5]))
                if sample["phase"] != phase or sample["angle_targets"] != expected_targets:
                    blockers.append(f"q6 return step {waypoint} target/phase changed")
                    break
                try:
                    require_rh56_feedback_in_interval(
                        sample["angles"],
                        return_previous_configuration,
                        return_current_configuration,
                        tolerance,
                        previous_phase=return_previous_feedback_phase,
                        current_phase=return_current_feedback_phase,
                        name=f"q6 return step {waypoint} ANGLE_ACT",
                    )
                except ValueError as exc:
                    blockers.append(str(exc))
                    break
                if any(sample["errors"]) or max(sample["temperatures"]) >= 60:
                    blockers.append(f"q6 return step {waypoint} contains a fault")
                    break
                if any(
                    abs(value) > cap
                    for value, cap in zip(sample["currents"], current_caps)
                ):
                    blockers.append(f"q6 return step {waypoint} exceeded current cap")
                    break
                if any(angle < 980 for angle in sample["angles"][:5]) or any(
                    status not in (2, 255) for status in sample["statuses"][:5]
                ):
                    blockers.append(
                        f"q6 return step {waypoint} moved an inactive bend axis"
                    )
                    break
                if return_inactive_reference is None or any(
                    abs(int(angle) - int(reference)) > inactive_drift
                    for angle, reference in zip(
                        sample["angles"][:5], return_inactive_reference
                    )
                ):
                    blockers.append(
                        f"q6 return step {waypoint} exceeded inactive-axis drift"
                    )
                    break
                if sample["statuses"][5] not in (0, 1, 2):
                    blockers.append(
                        f"q6 return step {waypoint} contains contact/fault status"
                    )
                    break
                if (
                    return_previous_actual is not None
                    and int(sample["angles"][5])
                    < return_previous_actual - opposite_slack
                ):
                    blockers.append(
                        f"q6 return step {waypoint} moved opposite the ascending command"
                    )
                    break
            for sample in validated_return[-stable_required:]:
                endpoint_ok = (
                    int(sample["angles"][5]) >= q6_open_min_angle
                    if waypoint == 1000
                    else abs(int(sample["angles"][5]) - waypoint)
                    <= q6_reverse_hysteresis_tolerance
                )
                if (
                    sample["statuses"][5] != 2
                    or not endpoint_ok
                    or any(
                        abs(int(current)) > stop_current_cap
                        for current in sample["currents"]
                    )
                ):
                    blockers.append(
                        f"q6 return step {waypoint} has no stable endpoint"
                    )
                    break
            if validated_return and return_previous_actual is not None:
                endpoint_actual = int(validated_return[-1]["angles"][5])
                commanded_delta = waypoint - return_previous_command
                required_progress = (
                    0
                    if waypoint == 1000 or commanded_delta < 10
                    else max(
                        3,
                        commanded_delta - q6_reverse_hysteresis_tolerance,
                    )
                )
                if endpoint_actual - return_previous_actual < required_progress:
                    blockers.append(
                        f"q6 return step {waypoint} lacks measurable directional progress"
                    )
                return_previous_actual = endpoint_actual
                return_previous_command = waypoint
                return_previous_configuration = return_current_configuration
                return_previous_feedback_phase = return_current_feedback_phase

    recorded_return_range = observations.get("actual_q6_return_range")
    expected_return_range = (
        [min(observed_return_q6), max(observed_return_q6)]
        if observed_return_q6
        else None
    )
    if expected_return_range is None:
        blockers.append("q6 return feedback is empty")
    if recorded_return_range is None or recorded_return_range != expected_return_range:
        blockers.append("recorded actual_q6_return_range disagrees with feedback")

    profile_snapshot = evidence.get("control_profile", {}).get("snapshot", {})
    try:
        previous_lower = int(
            profile_snapshot["inspire"]["thumb_rotate_validated_realtime_range"][0]
        )
    except (KeyError, TypeError, ValueError, IndexError):
        blockers.append("bound profile has no prior q6 range")
    else:
        if target < previous_lower and (
            not isinstance(confirmations, Mapping)
            or confirmations.get("wide_q6_confirmed") is not True
        ):
            blockers.append("wide q6 confirmation is absent for range expansion")

    try:
        final_targets = _six_ints(
            final.get("angle_targets"),
            "final.angle_targets",
            allow_disabled=True,
            maximum=1000,
        )
        final_errors = _six_ints(final.get("errors"), "final.errors", maximum=255)
        final_statuses = _six_ints(
            final.get("statuses"), "final.statuses", maximum=255
        )
        final_currents = _six_ints(
            final.get("currents"), "final.currents", allow_signed=True
        )
    except ValueError as exc:
        blockers.append(str(exc))
    else:
        if list(final_targets) != DISABLED_TARGETS:
            blockers.append("final all-six ANGLE_SET=-1 is not proven")
        if any(final_errors):
            blockers.append("final RH56 feedback contains actuator ERROR")
        if any(status not in (2, 255) for status in final_statuses):
            blockers.append("final RH56 feedback is not idle")
        if any(
            abs(value) > cap
            for value, cap in zip(final_currents, current_caps)
        ):
            blockers.append("final RH56 feedback exceeds current cap")
        if final.get("disabled_verified") is not True:
            blockers.append("final disabled_verified is not true")
        if final.get("reopened_and_verified") is not True:
            blockers.append("final six-axis reopen was not verified")

    final_open = observations.get("final_open_feedback")
    try:
        opened = _feedback(final_open, "observations.final_open_feedback")
    except ValueError as exc:
        blockers.append(str(exc))
    else:
        return_open_samples = [
            sample
            for sample in validated_all_feedback
            if sample.get("phase") == "q6_return_open_verify"
        ]
        if len(return_open_samples) != 1 or final_open != return_open_samples[0]:
            blockers.append(
                "final open feedback differs from the exact q6 return record"
            )
        if opened["angle_targets"] != DISABLED_TARGETS:
            blockers.append("final open feedback does not have ANGLE_SET=-1")
        if (
            any(
                angle < COMMISSION_BEND_OPEN_MIN_ANGLE
                for angle in opened["angles"][:5]
            )
            or int(opened["angles"][5]) < q6_open_min_angle
        ):
            blockers.append("final open feedback does not prove all six axes open")
        if any(opened["errors"]):
            blockers.append("final open feedback contains actuator ERROR")
        if any(status not in (2, 255) for status in opened["statuses"]):
            blockers.append("final open feedback is not idle")
        if max(opened["temperatures"]) >= 60 or any(
            abs(current) > stop_current_cap for current in opened["currents"]
        ):
            blockers.append("final open feedback is hot or still drawing current")

    disable_samples = observations.get("disable_feedback")
    exact_disable_samples = [
        sample
        for sample in validated_all_feedback
        if sample.get("phase") == "disable_verify"
    ]
    if (
        not isinstance(disable_samples, list)
        or len(disable_samples) < 2
        or disable_samples != exact_disable_samples
    ):
        blockers.append("post-disable feedback has fewer than two samples")
    else:
        try:
            disabled = [
                _feedback(sample, f"disable feedback[{index}]")
                for index, sample in enumerate(disable_samples)
            ]
        except ValueError as exc:
            blockers.append(str(exc))
        else:
            for sample in disabled:
                if sample["angle_targets"] != DISABLED_TARGETS:
                    blockers.append("post-disable feedback target changed from all -1")
                    break
                if any(sample["errors"]):
                    blockers.append("post-disable feedback contains actuator ERROR")
                    break
                if any(status not in (2, 255) for status in sample["statuses"]):
                    blockers.append("post-disable feedback is not idle")
                    break
                if max(sample["temperatures"]) >= 60 or any(
                    abs(current) > stop_current_cap
                    for current in sample["currents"]
                ):
                    blockers.append("post-disable feedback is hot or drawing current")
                    break
            for axis in range(6):
                angles = [sample["angles"][axis] for sample in disabled]
                if max(angles) - min(angles) > 2:
                    blockers.append("post-disable ANGLE_ACT still changes")
                    break
                positions = [
                    sample.get("positions") for sample in disabled
                ]
                if all(isinstance(value, list) and len(value) == 6 for value in positions):
                    axis_positions = [int(value[axis]) for value in positions]
                    if max(axis_positions) - min(axis_positions) > 3:
                        blockers.append("post-disable POS_ACT still changes")
                        break
            if disabled:
                last = disabled[-1]
                final_fields = (
                    ("angle_targets", final.get("angle_targets")),
                    ("angles", final.get("angles")),
                    ("currents", final.get("currents")),
                    ("errors", final.get("errors")),
                    ("statuses", final.get("statuses")),
                    ("temperatures", final.get("temperatures")),
                )
                for field_name, recorded in final_fields:
                    if recorded != last[field_name]:
                        blockers.append(
                            f"final.{field_name} differs from last disable feedback"
                        )
    return blockers


def derived_config_proposal(evidence: Mapping[str, Any]) -> dict[str, Any]:
    request = evidence["request"]
    result = evidence["result"]
    profile = evidence["control_profile"]["snapshot"]
    previous_range = profile["inspire"]["thumb_rotate_validated_realtime_range"]
    previous_coupled = bool(
        profile["inspire"]["six_axis_coupled_closure_commissioned"]
    )
    previous_targets = [
        [int(value) for value in target]
        for target in profile["inspire"].get("commissioned_air_closure_targets", [])
    ]
    commissioned_target = request.get("coupled_targets")
    if result["coupled_closure_pass"] and commissioned_target is not None:
        exact_target = [int(value) for value in commissioned_target]
        if exact_target not in previous_targets:
            previous_targets.append(exact_target)
    return {
        "inspire.thumb_rotate_validated_realtime_range": [
            min(int(previous_range[0]), int(request["target_q6"])),
            1000,
        ],
        "inspire.six_axis_coupled_closure_commissioned": bool(
            previous_coupled or result["coupled_closure_pass"]
        ),
        "inspire.commissioned_air_closure_targets": previous_targets,
        "scope": "low_speed_unloaded_no_contact_PLA",
        "evidence_payload_sha256": evidence["integrity"]["payload_sha256"],
    }


def verify_evidence(
    evidence_path: Path,
    *,
    config_path: Optional[Path] = None,
    require_coupled: bool = False,
) -> EvidenceVerification:
    evidence, _ = load_evidence(evidence_path)
    blockers = _base_integrity_blockers(evidence)
    blockers.extend(_validate_bound_files(evidence.get("source_bindings")))
    blockers.extend(_verify_franka_binding(evidence))
    blockers.extend(_verify_motion_record(evidence))
    blockers.extend(_verify_snapshot_candidate_binding(evidence))

    profile = evidence.get("control_profile")
    if not isinstance(profile, Mapping):
        blockers.append("control_profile binding is missing")
    else:
        snapshot = profile.get("snapshot")
        if not isinstance(snapshot, Mapping):
            blockers.append("control_profile snapshot is missing")
        elif json_sha256(snapshot) != profile.get("parsed_sha256"):
            blockers.append("control_profile parsed snapshot hash mismatch")
        expected_path = Path(str(profile.get("path", ""))).expanduser().resolve()
        active_path = (
            expected_path
            if config_path is None
            else Path(config_path).expanduser().resolve()
        )
        if active_path != expected_path:
            blockers.append("verified config path differs from evidence binding")
        elif not active_path.is_file():
            blockers.append(f"bound control profile is missing: {active_path}")
        elif sha256_file(active_path) != profile.get("file_sha256"):
            blockers.append("bound control profile file SHA-256 changed")
    if require_coupled and evidence.get("result", {}).get("coupled_closure_pass") is not True:
        blockers.append("coupled closure pass was explicitly required")
    proposal = None if blockers else derived_config_proposal(evidence)
    return EvidenceVerification(tuple(dict.fromkeys(blockers)), proposal)


def verify_applied_config(
    evidence_path: Path,
    config_path: Path,
    *,
    require_coupled: bool = False,
) -> EvidenceVerification:
    """Verify an explicitly, manually patched profile against exact evidence."""

    evidence, _ = load_evidence(evidence_path)
    blockers = _base_integrity_blockers(evidence)
    blockers.extend(_validate_bound_files(evidence.get("source_bindings")))
    blockers.extend(_verify_franka_binding(evidence))
    blockers.extend(_verify_motion_record(evidence))
    blockers.extend(_verify_snapshot_candidate_binding(evidence))
    if require_coupled and evidence.get("result", {}).get("coupled_closure_pass") is not True:
        blockers.append("coupled closure pass was explicitly required")
    profile = evidence.get("control_profile")
    if not isinstance(profile, Mapping) or not isinstance(profile.get("snapshot"), Mapping):
        blockers.append("control_profile snapshot is missing")
    else:
        if json_sha256(profile["snapshot"]) != profile.get("parsed_sha256"):
            blockers.append("control_profile parsed snapshot hash mismatch")
        expected = copy.deepcopy(dict(profile["snapshot"]))
        proposal = derived_config_proposal(evidence)
        expected["inspire"]["thumb_rotate_validated_realtime_range"] = proposal[
            "inspire.thumb_rotate_validated_realtime_range"
        ]
        expected["inspire"]["six_axis_coupled_closure_commissioned"] = proposal[
            "inspire.six_axis_coupled_closure_commissioned"
        ]
        expected["inspire"]["commissioned_air_closure_targets"] = proposal[
            "inspire.commissioned_air_closure_targets"
        ]
        try:
            actual = json.loads(
                Path(config_path).expanduser().resolve().read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(f"cannot load applied control profile: {exc}")
        else:
            if actual != expected:
                blockers.append(
                    "applied profile is not the exact evidence-derived three-field update"
                )
    proposal = None if blockers else derived_config_proposal(evidence)
    return EvidenceVerification(tuple(dict.fromkeys(blockers)), proposal)


def telemetry_to_json(sample: Any) -> dict[str, Any]:
    return {
        "phase": str(sample.phase),
        "elapsed_s": float(sample.elapsed_s),
        "angle_targets": [int(value) for value in sample.angle_targets],
        "angles": [int(value) for value in sample.angles],
        "positions": (
            None if sample.positions is None else [int(value) for value in sample.positions]
        ),
        "forces": (
            None if sample.forces is None else [int(value) for value in sample.forces]
        ),
        "currents": [int(value) for value in sample.currents],
        "errors": [int(value) for value in sample.errors],
        "statuses": [int(value) for value in sample.statuses],
        "temperatures": [int(value) for value in sample.temperatures],
    }


def group_telemetry(
    telemetry: Iterable[Any],
    q6_waypoints: Sequence[int],
    *,
    q6_return_waypoints: Sequence[int] = (),
    coupled_waypoints: Sequence[Sequence[int]] = (),
    coupled_return_waypoints: Sequence[Sequence[int]] = (),
) -> dict[str, Any]:
    serialized = [telemetry_to_json(sample) for sample in telemetry]
    q6_steps = []
    for target in q6_waypoints:
        phase = f"q6_step_{int(target):04d}"
        q6_steps.append(
            {
                "target_q6": int(target),
                "feedback": [item for item in serialized if item["phase"] == phase],
            }
        )
    q6_return_steps = []
    for target in q6_return_waypoints:
        phase = f"q6_return_{int(target):04d}"
        q6_return_steps.append(
            {
                "target_q6": int(target),
                "feedback": [item for item in serialized if item["phase"] == phase],
            }
        )
    coupled_steps = []
    for index, target in enumerate(coupled_waypoints):
        phase = f"coupled_air_close_step_{index:04d}"
        coupled_steps.append(
            {
                "target": [int(value) for value in target],
                "feedback": [item for item in serialized if item["phase"] == phase],
            }
        )
    coupled_return_steps = []
    for index, target in enumerate(coupled_return_waypoints):
        phase = f"coupled_air_return_step_{index:04d}"
        coupled_return_steps.append(
            {
                "target": [int(value) for value in target],
                "feedback": [item for item in serialized if item["phase"] == phase],
            }
        )
    final_open = [
        item
        for item in serialized
        if item["phase"] == "q6_return_open_verify"
    ]
    disable = [item for item in serialized if item["phase"] == "disable_verify"]
    q6_observed = [
        item["angles"][5]
        for group in q6_steps
        for item in group["feedback"]
    ]
    q6_return_observed = [
        item["angles"][5]
        for group in q6_return_steps
        for item in group["feedback"]
    ]
    return {
        "all_feedback": serialized,
        "adopt_disable_feedback": [
            item for item in serialized if item["phase"] == "adopt_disable_verify"
        ],
        "q6_steps": q6_steps,
        "q6_return_steps": q6_return_steps,
        "coupled_air_close_steps": coupled_steps,
        "coupled_air_return_steps": coupled_return_steps,
        "coupled_air_close_feedback": [
            item
            for item in serialized
            if item["phase"].startswith("coupled_air_close_step_")
        ],
        "final_open_feedback": final_open[-1] if final_open else None,
        "disable_feedback": disable,
        "actual_q6_range": (
            [min(q6_observed), max(q6_observed)] if q6_observed else None
        ),
        "actual_q6_return_range": (
            [min(q6_return_observed), max(q6_return_observed)]
            if q6_return_observed
            else None
        ),
    }


__all__ = [
    "EVIDENCE_KIND",
    "EvidenceVerification",
    "SCHEMA_VERSION",
    "atomic_write_evidence",
    "derived_config_proposal",
    "group_telemetry",
    "json_sha256",
    "load_evidence",
    "seal_evidence",
    "sha256_file",
    "telemetry_to_json",
    "verify_applied_config",
    "verify_evidence",
]
