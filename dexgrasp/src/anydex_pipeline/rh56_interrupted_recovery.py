"""Fail-closed recovery for an interrupted installed RH56 air-close run.

This module deliberately does not discover or open hardware.  It first binds a
sealed failed commissioning record to its reviewed source/profile lineage.  The
hardware entrypoint may then use :class:`InterruptedRecoveryDriver` to reopen
only the already-commanded bend prefix, followed by the canonical q6 reverse
path.

Recovery records are never commissioning passes and cannot extend a validated
q6 range or unlock coupled closure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence, Tuple

from .inspire_sequence_driver import (
    DISABLED_TARGETS,
    RH56MotionStopped,
    RH56SequenceDriver,
    RH56SequenceDriverError,
    _six_integral,
)
from .rh56_commissioning import (
    EVIDENCE_KIND,
    SCHEMA_VERSION,
    _base_integrity_blockers,
    _stage1_binding_payload,
    _stage1_record_blockers,
    _validate_bound_files,
    _verify_franka_binding,
    _verify_snapshot_candidate_binding,
    build_stage1_prerequisite_binding,
    json_sha256,
    load_evidence,
    sha256_file,
)
from .rh56_hand_path import (
    Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS,
    build_rh56_no_contact_execution_path,
)
from .rh56_reset_open import (
    RESET_BEND_OPEN_MIN_ANGLE,
    RESET_Q6_ARRIVAL_TOLERANCE_UNITS,
    RH56ResetOpenDriver,
)


RECOVERY_EVIDENCE_KIND = "installed_rh56_interrupted_open_recovery_v1"
RECOVERY_MODE_COUPLED_CLOSE_PREFIX = "interrupted_coupled_close_prefix_v1"
RECOVERY_MODE_Q6_RETURN = "interrupted_stage2_q6_return_v1"
RECOVERY_ROUTE_COUPLED_CLOSE_PREFIX = "sealed_coupled_close_prefix_v1"
RECOVERY_ROUTE_SEALED_Q6_RETURN = "sealed_stage2_q6_return_v1"
RECOVERY_ROUTE_NEAR_OPEN_RESET = "profile_near_open_reset_v1"
LIVE_Q6_SETTLE_SLACK_UNITS = 4
LIVE_Q6_REACQUIRE_TOLERANCE_UNITS = 8
AUTHORIZED_STAGE2_FILE_SHA256 = (
    "335fb62ff3db22fd0cabec964aefa11fec2ae3c9f1b84ff229ec440366e72d44"
)
AUTHORIZED_STAGE2_PAYLOAD_SHA256 = (
    "b5d80a26da4ec0f9a22d877da0f23ed7b29a5483f2ca90cc09fab3da8226422e"
)
AUTHORIZED_STAGE2_RUN_ID = "f2be3cc5-6045-43eb-bfab-450d13c52454"

# This is not a general "accept stale evidence" policy.  The one interrupted
# Stage-2 record above and its exact Stage-1 prerequisite were created before
# two deliberate stop/settling safety changes.  Preserve their historical
# source list as provenance, while requiring each deliberately migrated live
# file to be the reviewed post-change bytes below and every other source to
# remain byte-for-byte identical to the historical record.
AUTHORIZED_STAGE1_FILE_SHA256 = (
    "2333cc7127060f3a8fa8844dd74a955f8e2282bff6018e84ad0ea5174a877b7d"
)
AUTHORIZED_STAGE1_PAYLOAD_SHA256 = (
    "5fa2f6629d4d8efff46c54f59cf66b1d20383f9c17ccb43dcc1eea62a6b0ba37"
)
AUTHORIZED_STAGE1_RUN_ID = "b4788166-0e29-4c7d-b8a5-ebd4ef815f4a"
AUTHORIZED_HISTORICAL_SOURCE_POLICY = (
    "exact_151314_historical_source_provenance_v1"
)
AUTHORIZED_HISTORICAL_SOURCE_BINDINGS_SHA256 = (
    "d8bb8f26d105ec63a73bb84db48e75c79fdfaa09111d7be987e657a6dbf2b83a"
)
AUTHORIZED_HISTORICAL_CONTROL_PROFILE_MIGRATION_POLICY = (
    "exact_v7_fr3_current_official_joint_limits_only_v1"
)
_AUTHORIZED_HISTORICAL_PROFILE_FILE_SHA256 = (
    "132c69a63fd1264b8f49582ac090acb91c11f9ae9a04f71e476e9e220b0e610b"
)
_AUTHORIZED_HISTORICAL_PROFILE_PARSED_SHA256 = (
    "862862f7064a1cbcbec2b82d41251e6acdfd400ca9e5b9a53cfbdf5d44a44c63"
)
_AUTHORIZED_MIGRATED_PROFILE_FILE_SHA256 = (
    "08e1af98c005ee1724b6f805aea213d7400efecc98ff751ff7d1b31bf0f06d1f"
)
_AUTHORIZED_MIGRATED_PROFILE_PARSED_SHA256 = (
    "9deb1d8392764292084fc6bccedf7dcc324e447b54a874a9830c5e2d9ee1e38e"
)
_HISTORICAL_FR3_JOINT_LIMITS = (
    (-2.7437, 2.7437),
    (-1.7837, 1.7837),
    (-2.9007, 2.9007),
    (-3.0421, -0.1518),
    (-2.8065, 2.8065),
    (0.5445, 4.5169),
    (-3.0159, 3.0159),
)
_CURRENT_FR3_JOINT_LIMITS = (
    (-2.9007, 2.9007),
    (-1.8361, 1.8361),
    (-2.9007, 2.9007),
    (-3.0770, -0.1169),
    (-2.8763, 2.8763),
    (0.4398, 4.6216),
    (-3.0508, 3.0508),
)
_HISTORICAL_ROOT = "/home/qiaoguanren/code/franka"
_AUTHORIZED_PROFILE_PATH = (
    _HISTORICAL_ROOT + "/dexgrasp/configs/fr3_rh56_v7_commissioning.json"
)
_AUTHORIZED_HISTORICAL_SOURCE_BINDINGS = (
    (
        "commission_cli",
        _HISTORICAL_ROOT + "/dexgrasp/apps/commission_installed_rh56.py",
        "9c980f6c21f0eec2c274ed6819175428bbdac99d3f81eb087eb16f036860a8a9",
    ),
    (
        "commission_evidence_module",
        _HISTORICAL_ROOT
        + "/dexgrasp/src/anydex_pipeline/rh56_commissioning.py",
        "3c9508f3f8cfa7c1bc9ee5809802ad6146e10ea617a7d40390e8a257a3b73531",
    ),
    (
        "rh56_hand_path",
        _HISTORICAL_ROOT + "/dexgrasp/src/anydex_pipeline/rh56_hand_path.py",
        "9096769434160e9329cea738825964e8248d17e51e6d4e6b1bcf56cb0dcff68a",
    ),
    (
        "rh56_sequence_driver",
        _HISTORICAL_ROOT
        + "/dexgrasp/src/anydex_pipeline/inspire_sequence_driver.py",
        "7a612b70c0c1252e0c2a2cc0c0a859699a351e5e10d4f714fefd74e221390f52",
    ),
    (
        "rh56_register_api",
        _HISTORICAL_ROOT + "/examples/inspire_rh56_test.py",
        "b80eb4e4224d4d9c5012ebc0b0ae876638692c706252ea55b8f7326ce6a1975d",
    ),
    (
        "franka_sequence_driver",
        _HISTORICAL_ROOT
        + "/dexgrasp/src/anydex_pipeline/franka_sequence_driver.py",
        "8a997f6b6a2880786a967a9877ee9aa4ac602d0c3f0bbd10b070231db413d15a",
    ),
    (
        "control_config_module",
        _HISTORICAL_ROOT + "/dexgrasp/src/anydex_pipeline/control_config.py",
        "8dc57b116fd11f1a4a40c07e03b84c54796829fff9b28fea45a2603f1ff99084",
    ),
    (
        "adapter_mesh",
        _HISTORICAL_ROOT
        + "/dexgrasp/assets/adapter/V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.stl",
        "7fdd3dd06bd8dafed445f6a6910315edd3073bd1b9cd9015a951b6e415b947e1",
    ),
    (
        "adapter_provenance",
        _HISTORICAL_ROOT
        + "/dexgrasp/assets/adapter/V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.provenance.json",
        "f2daad6cb446773b9110d8585499235409e12aae2efb642a8eb4e86f64883d5f",
    ),
    (
        "actuator_to_joint_xlsx",
        _HISTORICAL_ROOT
        + "/dexgrasp/third_party/AnyDexGrasp/generate_mesh_and_pointcloud/"
        "inspire_urdf/inspire_hand_routine_to_angle-use.xlsx",
        "432f6bc67fe307e184ab18e48fc6e85e0ee19e714b6064c3e8f06ad031ad4b5e",
    ),
    (
        "driver_to_angle_xls",
        _HISTORICAL_ROOT
        + "/dexgrasp/third_party/AnyDexGrasp/generate_mesh_and_pointcloud/"
        "inspire_urdf/driver_routine_to_angle.xls",
        "23ca934b1092ce1a42e46f7bd7edc1ac0e3cacb98efe397ab9d9db1c938e7bdb",
    ),
    (
        "actuator_to_urdf_generator",
        _HISTORICAL_ROOT
        + "/dexgrasp/third_party/AnyDexGrasp/generate_mesh_and_pointcloud/"
        "recover_inspire_hand_to_stl.py",
        "48688a875b4b6b121040c16548ae4635974113bd44aaf8c9585a93db5efc7942",
    ),
)
_AUTHORIZED_HISTORICAL_SOURCE_MIGRATIONS = {
    "commission_cli": (
        "5d8d23470e509b1fa2997326d84910806061a0ec3af85d750cec82cf65ece6ca"
    ),
    "commission_evidence_module": (
        "cfdc2db3070645ca17dd42a7e9e1f75f24d70cd047f35dea27c2da880d8912a3"
    ),
    "rh56_sequence_driver": (
        "fb9e354d0b73f112e70ad121aa7c35f316a847820df786e97fccc7d0fcf47864"
    ),
    "rh56_register_api": (
        "4caf3d41bf936d59a4fa55e4f6e7ceb5a13d57f677a9c98d0f79d146eb4555a4"
    ),
    "franka_sequence_driver": (
        "c7e392045f6e59a9efb433521fa2f444f780f699e621c2e9187510a5b0468a23"
    ),
    "control_config_module": (
        "c9366249378e7d98dc652fa216d0ff075dc7ef0d5d6dc8037b9256137a69ba78"
    ),
}
_AUTHORIZED_STAGE1_BINDING = {
    "completed_at_utc": "2026-07-22T06:53:47.158671Z",
    "control_profile_parsed_sha256": (
        "862862f7064a1cbcbec2b82d41251e6acdfd400ca9e5b9a53cfbdf5d44a44c63"
    ),
    "file_sha256": AUTHORIZED_STAGE1_FILE_SHA256,
    "kind": "installed_rh56_q6_stage1_pass_v1",
    "path": _HISTORICAL_ROOT
    + "/dexgrasp/runs/rh56_q6_900_stage1_20260722_145312.json",
    "payload_sha256": AUTHORIZED_STAGE1_PAYLOAD_SHA256,
    "run_id": AUTHORIZED_STAGE1_RUN_ID,
    "target_q6": 900,
}


def _is_exact_authorized_stage2(
    evidence: Mapping[str, Any], resolved: Path
) -> bool:
    integrity = evidence.get("integrity")
    request = evidence.get("request")
    return (
        sha256_file(resolved) == AUTHORIZED_STAGE2_FILE_SHA256
        and isinstance(integrity, Mapping)
        and integrity.get("payload_sha256") == AUTHORIZED_STAGE2_PAYLOAD_SHA256
        and evidence.get("run_id") == AUTHORIZED_STAGE2_RUN_ID
        and isinstance(request, Mapping)
        and request.get("target_source") == "official_snapshot_candidate"
    )


def _is_exact_authorized_control_profile_migration(
    profile: Any,
    *,
    expected_config: Mapping[str, Any],
    expected_config_path: Path,
) -> bool:
    """Accept only the reviewed old->current FR3 joint-limit table update."""

    if not isinstance(profile, Mapping):
        return False
    snapshot = profile.get("snapshot")
    if not isinstance(snapshot, Mapping):
        return False
    resolved = Path(expected_config_path).expanduser().resolve()
    if str(resolved) != _AUTHORIZED_PROFILE_PATH:
        return False
    if Path(str(profile.get("path", ""))).expanduser().resolve() != resolved:
        return False
    if (
        profile.get("file_sha256")
        != _AUTHORIZED_HISTORICAL_PROFILE_FILE_SHA256
        or profile.get("parsed_sha256")
        != _AUTHORIZED_HISTORICAL_PROFILE_PARSED_SHA256
        or json_sha256(snapshot)
        != _AUTHORIZED_HISTORICAL_PROFILE_PARSED_SHA256
        or sha256_file(resolved) != _AUTHORIZED_MIGRATED_PROFILE_FILE_SHA256
        or json_sha256(expected_config)
        != _AUTHORIZED_MIGRATED_PROFILE_PARSED_SHA256
    ):
        return False
    historical_franka = snapshot.get("franka")
    current_franka = expected_config.get("franka")
    if not isinstance(historical_franka, Mapping) or not isinstance(
        current_franka, Mapping
    ):
        return False
    if historical_franka.get("joint_limits_rad") != [
        list(values) for values in _HISTORICAL_FR3_JOINT_LIMITS
    ]:
        return False
    if current_franka.get("joint_limits_rad") != [
        list(values) for values in _CURRENT_FR3_JOINT_LIMITS
    ]:
        return False
    migrated = dict(snapshot)
    migrated_franka = dict(historical_franka)
    migrated_franka["joint_limits_rad"] = [
        list(values) for values in _CURRENT_FR3_JOINT_LIMITS
    ]
    migrated["franka"] = migrated_franka
    return migrated == dict(expected_config)


def _validate_exact_historical_source_bindings(bindings: Any) -> list[str]:
    """Validate the exact historical list and reviewed live migrations."""

    if not isinstance(bindings, list):
        return ["authorized historical source_bindings must be an array"]
    actual = []
    for index, item in enumerate(bindings):
        if not isinstance(item, Mapping) or set(item) != {"name", "path", "sha256"}:
            return [
                f"authorized historical source_bindings[{index}] schema differs"
            ]
        name = item.get("name")
        path = item.get("path")
        digest = item.get("sha256")
        if (
            not isinstance(name, str)
            or not isinstance(path, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or digest.lower() != digest
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return [
                f"authorized historical source_bindings[{index}] is malformed"
            ]
        actual.append((name, path, digest))
    if tuple(actual) != _AUTHORIZED_HISTORICAL_SOURCE_BINDINGS:
        return ["authorized historical source_bindings differ from exact 151314"]
    if json_sha256(bindings) != AUTHORIZED_HISTORICAL_SOURCE_BINDINGS_SHA256:
        return ["authorized historical source_bindings digest differs"]

    blockers = []
    for name, path_value, historical_digest in actual:
        source = Path(path_value)
        try:
            metadata = source.lstat()
        except FileNotFoundError:
            blockers.append(f"bound source file is missing: {source}")
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            blockers.append(
                "authorized historical source is not a real regular file: "
                + str(source)
            )
            continue
        if source.resolve() != source:
            blockers.append(
                "authorized historical source path is not canonical: " + str(source)
            )
            continue
        expected_live = _AUTHORIZED_HISTORICAL_SOURCE_MIGRATIONS.get(
            name, historical_digest
        )
        if sha256_file(source) != expected_live:
            blockers.append(
                "authorized historical source live hash differs: " + str(source)
            )
    return blockers


def _exact_historical_stage1_binding(
    binding: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any],
    expected_config_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Revalidate the exact Stage-1 parent without granting generic staleness."""

    blockers: list[str] = []
    if dict(binding) != _AUTHORIZED_STAGE1_BINDING:
        return {}, ["failed evidence Stage1 binding differs from exact 151314"]
    source = Path(str(binding["path"]))
    try:
        stage1, resolved = load_evidence(source)
    except ValueError as exc:
        return {}, [str(exc)]
    integrity = stage1.get("integrity")
    if (
        sha256_file(resolved) != AUTHORIZED_STAGE1_FILE_SHA256
        or not isinstance(integrity, Mapping)
        or integrity.get("payload_sha256") != AUTHORIZED_STAGE1_PAYLOAD_SHA256
        or stage1.get("run_id") != AUTHORIZED_STAGE1_RUN_ID
    ):
        blockers.append("historical Stage1 is not the exact 151314 prerequisite")
    source_blockers = set(
        _validate_exact_historical_source_bindings(stage1.get("source_bindings"))
    )
    blockers.extend(sorted(source_blockers))

    # The shared semantic verifier remains strict.  Predictable old-file and
    # pre-profile-aware-q6 schema messages are removed only for this exact
    # retained Stage-1 artifact, after the exact-list validator above has
    # required every reviewed old->new byte migration.
    allowed_stale_messages = {
        "bound source file changed: "
        + path
        for name, path, _digest in _AUTHORIZED_HISTORICAL_SOURCE_BINDINGS
        if name in _AUTHORIZED_HISTORICAL_SOURCE_MIGRATIONS
    }
    allowed_stale_messages.update(
        {
            "required source bindings are missing: rh56_reset_open",
            "request.q6_open_min_angle is missing",
            "profile-aware q6 open evidence is missing "
            "rh56_reset_open source binding",
        }
    )
    semantic_blockers = _stage1_record_blockers(
        stage1,
        expected_config=expected_config,
        expected_config_path=expected_config_path,
    )
    if _is_exact_authorized_control_profile_migration(
        stage1.get("control_profile"),
        expected_config=expected_config,
        expected_config_path=expected_config_path,
    ):
        allowed_stale_messages.add(
            "Stage1 profile differs outside the three evidence-owned "
            "commissioning fields"
        )
    blockers.extend(
        item for item in semantic_blockers if item not in allowed_stale_messages
    )
    try:
        payload = _stage1_binding_payload(stage1, resolved)
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append(f"historical Stage1 cannot form its binding: {exc}")
        payload = {}
    if payload != dict(binding):
        blockers.append("failed evidence Stage1 binding does not match its exact file")
    return payload, list(dict.fromkeys(blockers))


def _q6_forward_waypoints(target_q6: int, step_units: int) -> Tuple[int, ...]:
    points = list(range(1000 - int(step_units), int(target_q6), -int(step_units)))
    if not points or points[-1] != int(target_q6):
        points.append(int(target_q6))
    return tuple(points)


def _profile_near_open_policy(
    config: Mapping[str, Any],
) -> tuple[Tuple[int, int], int]:
    """Bind the superseded-recovery route to the reviewed profile endpoint.

    This is deliberately not a generic escape from the historical 640..656
    evidence gate.  It recognizes only the separately commissioned realtime
    q6 band 900..1000 and derives the physical open endpoint from the profile's
    ordinary arrival tolerance.
    """

    inspire = config.get("inspire")
    if not isinstance(inspire, Mapping):
        raise ValueError("control profile has no inspire object")
    raw_range = inspire.get("thumb_rotate_validated_realtime_range")
    if (
        not isinstance(raw_range, list)
        or len(raw_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_range)
    ):
        raise ValueError(
            "inspire.thumb_rotate_validated_realtime_range must be two integers"
        )
    q6_range = (int(raw_range[0]), int(raw_range[1]))
    if q6_range != (900, 1000):
        raise ValueError(
            "superseded recovery requires the reviewed profile q6 range "
            f"900..1000; actual={q6_range[0]}..{q6_range[1]}"
        )
    raw_open = inspire.get("open_targets")
    if (
        not isinstance(raw_open, list)
        or len(raw_open) != 6
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_open)
    ):
        raise ValueError("inspire.open_targets must be six integers")
    tolerance = inspire.get("arrival_tolerance_units")
    if isinstance(tolerance, bool) or not isinstance(tolerance, int):
        raise ValueError("inspire.arrival_tolerance_units must be an integer")
    if not 0 <= int(tolerance) <= RESET_Q6_ARRIVAL_TOLERANCE_UNITS:
        raise ValueError(
            "inspire.arrival_tolerance_units exceeds the reviewed reset-open "
            f"limit {RESET_Q6_ARRIVAL_TOLERANCE_UNITS}"
        )
    if int(raw_open[5]) != q6_range[1]:
        raise ValueError(
            "inspire q6 open target must equal the validated range endpoint"
        )
    q6_open_min = int(raw_open[5]) - int(tolerance)
    if not q6_range[0] <= q6_open_min <= q6_range[1]:
        raise ValueError(
            "profile-derived q6 open feedback threshold is outside its "
            "validated realtime range"
        )
    return q6_range, q6_open_min


def _coupled_forward_waypoints(
    targets: Sequence[int], step_units: int
) -> Tuple[Tuple[int, ...], ...]:
    requested = _six_integral(
        targets, "recorded coupled targets", allow_disabled=False
    )
    current = [1000] * 5
    output = []
    for axis, wanted in enumerate(requested[:5]):
        while current[axis] != wanted:
            current[axis] = max(wanted, current[axis] - int(step_units))
            output.append(tuple(current) + (requested[5],))
    return tuple(output)


def _canonical_return_waypoints(
    interrupted_targets: Sequence[int], step_units: int
) -> tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...]]:
    path = build_rh56_no_contact_execution_path(
        interrupted_targets, step_units=int(step_units)
    )
    bends = tuple(
        item.command_targets
        for item in path.waypoints
        if item.phase.startswith("bend_reverse_")
    )
    q6 = tuple(
        item.command_targets[5]
        for item in path.waypoints
        if item.phase.startswith("q6_reverse_")
    )
    return bends, q6


def _six_from_json(value: Any, name: str) -> Tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 6
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must be a six-integer JSON array")
    return _six_integral(value, name, allow_disabled=False)


def _feedback_six(value: Mapping[str, Any], name: str, *, allow_disabled: bool = False) -> Tuple[int, ...]:
    raw = value.get(name)
    if (
        not isinstance(raw, list)
        or len(raw) != 6
        or any(isinstance(item, bool) or not isinstance(item, int) for item in raw)
    ):
        raise ValueError(f"feedback.{name} must be a six-integer JSON array")
    return _six_integral(raw, f"feedback.{name}", allow_disabled=allow_disabled)


def _feedback_signed_six(value: Mapping[str, Any], name: str) -> Tuple[int, ...]:
    raw = value.get(name)
    if not isinstance(raw, list) or len(raw) != 6:
        raise ValueError(f"feedback.{name} must be a six-value JSON array")
    output = []
    for index, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, int) or not -5000 <= item <= 5000:
            raise ValueError(f"feedback.{name}[{index}] is not a valid signed register")
        output.append(int(item))
    return tuple(output)


def _require_stable_endpoint_tail(
    feedback: Any,
    command_target: Sequence[int],
    expected_angles: Sequence[int],
    *,
    stable_required: int,
    tolerance: int,
    current_cap: int,
    name: str,
) -> None:
    if not isinstance(feedback, list) or len(feedback) < int(stable_required):
        raise ValueError(f"{name} lacks the required stable endpoint samples")
    expected_command = tuple(int(item) for item in command_target)
    expected_actual = tuple(int(item) for item in expected_angles)
    for sample in feedback[-int(stable_required):]:
        if not isinstance(sample, Mapping):
            raise ValueError(f"{name} endpoint sample is not an object")
        if _feedback_six(sample, "angle_targets", allow_disabled=True) != expected_command:
            raise ValueError(f"{name} endpoint ANGLE_SET differs from its waypoint")
        angles = _feedback_six(sample, "angles")
        if any(abs(actual - wanted) > int(tolerance) for actual, wanted in zip(angles, expected_actual)):
            raise ValueError(f"{name} endpoint ANGLE_ACT did not reach its waypoint")
        if any(_feedback_six(sample, "errors")):
            raise ValueError(f"{name} endpoint contains a device fault")
        if any(status != 2 for status in _feedback_six(sample, "statuses")):
            raise ValueError(f"{name} endpoint is not idle/status-2")
        if any(
            abs(current) > int(current_cap)
            for current in _feedback_signed_six(sample, "currents")
        ):
            raise ValueError(f"{name} endpoint exceeds the recorded current cap")


def _derive_interrupted_waypoint(
    groups: Any,
    expected_waypoints: Sequence[Sequence[int]],
) -> Tuple[int, ...]:
    """Return the last commanded waypoint only for one contiguous prefix.

    A hole followed by later telemetry is ambiguous and is rejected.  The last
    active group's ANGLE_SET must still equal its recorded command; physical
    ANGLE_ACT is checked freshly by the recovery driver before any numeric
    write.
    """

    expected = tuple(tuple(int(item) for item in row) for row in expected_waypoints)
    if not expected:
        raise ValueError("failed evidence contains no coupled-air command path")
    if not isinstance(groups, list) or len(groups) != len(expected):
        raise ValueError("coupled telemetry groups do not match the recorded command path")
    active_indices = []
    for index, (group, target) in enumerate(zip(groups, expected)):
        if not isinstance(group, Mapping):
            raise ValueError(f"coupled telemetry group {index} is not an object")
        if _six_from_json(group.get("target"), f"coupled group {index} target") != target:
            raise ValueError(f"coupled telemetry group {index} target is inconsistent")
        feedback = group.get("feedback")
        if not isinstance(feedback, list):
            raise ValueError(f"coupled telemetry group {index} feedback is not an array")
        if feedback:
            active_indices.append(index)
    if not active_indices:
        raise ValueError("failed evidence contains no started coupled-air waypoint")
    if active_indices != list(range(active_indices[-1] + 1)):
        raise ValueError("coupled telemetry is not a contiguous command prefix")
    last_index = active_indices[-1]
    last_feedback = groups[last_index]["feedback"][-1]
    if not isinstance(last_feedback, Mapping):
        raise ValueError("last coupled feedback sample is not an object")
    angle_targets = _feedback_six(
        last_feedback, "angle_targets", allow_disabled=True
    )
    if angle_targets != expected[last_index]:
        raise ValueError(
            "last coupled feedback ANGLE_SET does not match its recorded waypoint"
        )
    errors = _feedback_six(last_feedback, "errors")
    if any(errors):
        raise ValueError("last coupled feedback contains a device fault")
    return expected[last_index]


def _require_complete_waypoint_groups(
    groups: Any,
    expected_waypoints: Sequence[Sequence[int]],
    *,
    stable_required: int,
    tolerance: int,
    current_cap: int,
    name: str,
) -> None:
    expected = tuple(tuple(int(value) for value in item) for item in expected_waypoints)
    if not isinstance(groups, list) or len(groups) != len(expected):
        raise ValueError(f"{name} groups do not match the canonical path")
    for index, (group, target) in enumerate(zip(groups, expected)):
        if not isinstance(group, Mapping):
            raise ValueError(f"{name} group {index} is not an object")
        if _six_from_json(group.get("target"), f"{name} group {index} target") != target:
            raise ValueError(f"{name} group {index} target is inconsistent")
        _require_stable_endpoint_tail(
            group.get("feedback"),
            target,
            target,
            stable_required=stable_required,
            tolerance=tolerance,
            current_cap=current_cap,
            name=f"{name} group {index}",
        )


def _require_unstarted_return_groups(
    groups: Any,
    expected_waypoints: Sequence[Any],
    *,
    q6_only: bool,
    name: str,
) -> None:
    """Prove that an older prefix recovery never entered either return phase."""

    expected = tuple(expected_waypoints)
    if not isinstance(groups, list) or len(groups) != len(expected):
        raise ValueError(f"{name} groups do not match the canonical path")
    for index, (group, waypoint) in enumerate(zip(groups, expected)):
        if not isinstance(group, Mapping):
            raise ValueError(f"{name} group {index} is not an object")
        if q6_only:
            if group.get("target_q6") != int(waypoint):
                raise ValueError(f"{name} group {index} target is inconsistent")
        elif _six_from_json(
            group.get("target"), f"{name} group {index} target"
        ) != tuple(int(value) for value in waypoint):
            raise ValueError(f"{name} group {index} target is inconsistent")
        feedback = group.get("feedback")
        if not isinstance(feedback, list):
            raise ValueError(f"{name} group {index} feedback is not an array")
        if feedback:
            raise ValueError(
                f"{name} contains started return telemetry and is not a "
                "coupled-close-prefix interruption"
            )


def _require_group_feedback_binding(
    groups: Any,
    all_feedback: Sequence[Mapping[str, Any]],
    phases: Sequence[str],
    *,
    name: str,
) -> None:
    """Require grouped samples to be exact views of the sealed telemetry log."""

    expected_phases = tuple(str(value) for value in phases)
    if not isinstance(groups, list) or len(groups) != len(expected_phases):
        raise ValueError(f"{name} groups do not match the canonical path")
    for index, (group, phase) in enumerate(zip(groups, expected_phases)):
        if not isinstance(group, Mapping):
            raise ValueError(f"{name} group {index} is not an object")
        grouped = group.get("feedback")
        exact = [sample for sample in all_feedback if sample.get("phase") == phase]
        if not isinstance(grouped, list) or grouped != exact:
            raise ValueError(
                f"{name} group {index} feedback differs from sealed all_feedback"
            )


def _derive_stage2_q6_return_anchor(
    evidence: Mapping[str, Any],
    *,
    target_q6: int,
    q6_forward: Sequence[int],
    full_forward: Sequence[Sequence[int]],
    full_bends_return: Sequence[Sequence[int]],
    full_q6_return: Sequence[int],
    stable_required: int,
    angle_tolerance: int,
    current_cap: int,
) -> tuple[int, tuple[int, int]]:
    """Prove Stage2 return progress and bind live q6 to its sealed final sample."""

    observations = evidence["observations"]
    final = evidence["final"]
    _require_complete_waypoint_groups(
        observations.get("coupled_air_close_steps"),
        full_forward,
        stable_required=stable_required,
        tolerance=angle_tolerance,
        current_cap=current_cap,
        name="coupled close",
    )
    _require_complete_waypoint_groups(
        observations.get("coupled_air_return_steps"),
        full_bends_return,
        stable_required=stable_required,
        tolerance=angle_tolerance,
        current_cap=current_cap,
        name="bend return",
    )

    all_feedback = observations.get("all_feedback")
    if not isinstance(all_feedback, list) or not all_feedback or any(
        not isinstance(sample, Mapping) for sample in all_feedback
    ):
        raise ValueError("Stage2 all_feedback is missing")
    _require_group_feedback_binding(
        observations.get("q6_steps"),
        all_feedback,
        [f"q6_step_{int(value):04d}" for value in q6_forward],
        name="q6 forward",
    )
    _require_group_feedback_binding(
        observations.get("coupled_air_close_steps"),
        all_feedback,
        [f"coupled_air_close_step_{index:04d}" for index in range(len(full_forward))],
        name="coupled close",
    )
    _require_group_feedback_binding(
        observations.get("coupled_air_return_steps"),
        all_feedback,
        [
            f"coupled_air_return_step_{index:04d}"
            for index in range(len(full_bends_return))
        ],
        name="bend return",
    )
    _require_group_feedback_binding(
        observations.get("q6_return_steps"),
        all_feedback,
        [f"q6_return_{int(value):04d}" for value in full_q6_return],
        name="q6 return",
    )
    close_feedback = [
        sample
        for phase in (
            f"coupled_air_close_step_{index:04d}"
            for index in range(len(full_forward))
        )
        for sample in all_feedback
        if sample.get("phase") == phase
    ]
    if observations.get("coupled_air_close_feedback") != close_feedback:
        raise ValueError("coupled close aggregate differs from sealed all_feedback")
    preflights = [
        sample
        for sample in all_feedback
        if isinstance(sample, Mapping) and sample.get("phase") == "q6_return_preflight"
    ]
    if len(preflights) != 1:
        raise ValueError("Stage2 q6 return must contain exactly one preflight")
    preflight = preflights[0]
    if _feedback_six(preflight, "angle_targets", allow_disabled=True) != DISABLED_TARGETS:
        raise ValueError("Stage2 q6 return preflight is not all-six disabled")
    preflight_angles = _feedback_six(preflight, "angles")
    preflight_statuses = _feedback_six(preflight, "statuses")
    if any(value < 980 for value in preflight_angles[:5]) or any(
        value != 2 for value in preflight_statuses
    ):
        raise ValueError("Stage2 q6 return preflight does not prove six idle/open axes")
    if abs(preflight_angles[5] - int(target_q6)) > min(int(angle_tolerance), 20):
        raise ValueError("Stage2 q6 return preflight is not at the commissioned q6 target")
    if any(_feedback_six(preflight, "errors")) or any(
        abs(value) > int(current_cap)
        for value in _feedback_signed_six(preflight, "currents")
    ):
        raise ValueError("Stage2 q6 return preflight contains fault/current evidence")

    groups = observations.get("q6_return_steps")
    expected = tuple(int(value) for value in full_q6_return)
    if not isinstance(groups, list) or len(groups) != len(expected):
        raise ValueError("Stage2 q6 return groups do not match the canonical path")
    active_indices: list[int] = []
    completed_indices: list[int] = []
    previous_target = int(target_q6)
    for index, (group, waypoint) in enumerate(zip(groups, expected)):
        if not isinstance(group, Mapping) or group.get("target_q6") != waypoint:
            raise ValueError(f"Stage2 q6 return group {index} target is inconsistent")
        samples = group.get("feedback")
        if not isinstance(samples, list):
            raise ValueError(f"Stage2 q6 return group {index} feedback is not an array")
        if not samples:
            continue
        active_indices.append(index)
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise ValueError(f"Stage2 q6 return group {index} sample is not an object")
            expected_target = (-1, -1, -1, -1, -1, waypoint)
            if _feedback_six(sample, "angle_targets", allow_disabled=True) != expected_target:
                raise ValueError(f"Stage2 q6 return group {index} ANGLE_SET changed")
            angles = _feedback_six(sample, "angles")
            statuses = _feedback_six(sample, "statuses")
            if any(value < 980 for value in angles[:5]) or any(
                value != 2 for value in statuses[:5]
            ):
                raise ValueError(f"Stage2 q6 return group {index} moved a bend axis")
            if statuses[5] not in (0, 1, 2):
                raise ValueError(f"Stage2 q6 return group {index} has contact/fault status")
            if any(_feedback_six(sample, "errors")) or any(
                abs(value) > int(current_cap)
                for value in _feedback_signed_six(sample, "currents")
            ):
                raise ValueError(f"Stage2 q6 return group {index} has fault/current evidence")
            if not (
                previous_target - Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS
                <= angles[5]
                <= waypoint + Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS
            ):
                raise ValueError(f"Stage2 q6 return group {index} escaped its command interval")
        try:
            _require_stable_endpoint_tail(
                samples,
                (-1, -1, -1, -1, -1, waypoint),
                (1000, 1000, 1000, 1000, 1000, waypoint),
                stable_required=stable_required,
                tolerance=Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS,
                current_cap=current_cap,
                name=f"Stage2 q6 return group {index}",
            )
        except ValueError:
            pass
        else:
            completed_indices.append(index)
            previous_target = waypoint
    if not active_indices or active_indices != list(range(active_indices[-1] + 1)):
        raise ValueError("Stage2 q6 return telemetry is not a non-empty contiguous prefix")
    if completed_indices and completed_indices != list(range(completed_indices[-1] + 1)):
        raise ValueError("Stage2 completed q6 return groups are not a contiguous prefix")
    if any(index < active_indices[-1] and index not in completed_indices for index in active_indices):
        raise ValueError("Stage2 q6 return continued after an incomplete waypoint")

    last_completed_target = (
        int(target_q6)
        if not completed_indices
        else expected[completed_indices[-1]]
    )
    last_attempted_target = expected[active_indices[-1]]
    telemetry_segment = (
        max(
            0,
            last_completed_target
            - Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS
            - LIVE_Q6_SETTLE_SLACK_UNITS,
        ),
        min(
            1000,
            last_attempted_target
            + Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS
            + LIVE_Q6_SETTLE_SLACK_UNITS,
        ),
    )
    if _feedback_six(final, "angle_targets", allow_disabled=True) != DISABLED_TARGETS:
        raise ValueError("Stage2 final ANGLE_SET is not all-six disabled")
    final_angles = _feedback_six(final, "angles")
    final_statuses = _feedback_six(final, "statuses")
    if any(value < 980 for value in final_angles[:5]) or any(
        value != 2 for value in final_statuses[:5]
    ):
        raise ValueError("Stage2 final feedback does not prove five open/idle bends")
    if final_statuses[5] not in (0, 1, 2):
        raise ValueError("Stage2 final q6 status is a contact/fault state")
    if not telemetry_segment[0] <= final_angles[5] <= telemetry_segment[1]:
        raise ValueError("Stage2 final q6 lies outside its evidence-bound live segment")
    if any(_feedback_six(final, "errors")) or any(
        abs(value) > int(current_cap)
        for value in _feedback_signed_six(final, "currents")
    ):
        raise ValueError("Stage2 final feedback contains fault/current evidence")
    last_feedback = all_feedback[-1]
    for key in (
        "angle_targets",
        "angles",
        "currents",
        "errors",
        "statuses",
        "temperatures",
    ):
        if final.get(key) != last_feedback.get(key):
            raise ValueError("Stage2 final feedback differs from sealed all_feedback")
    # Runtime recovery may reacquire only the sealed final actuator position,
    # with the same small inactive-drift allowance used elsewhere.  The wider
    # telemetry segment above proves that the final sample is physically
    # consistent with the interrupted canonical return, but it must not turn
    # into authority to adopt an arbitrary position elsewhere in that segment.
    anchor = int(final_angles[5])
    permitted_live = (
        max(0, anchor - LIVE_Q6_REACQUIRE_TOLERANCE_UNITS),
        min(1000, anchor + LIVE_Q6_REACQUIRE_TOLERANCE_UNITS),
    )
    return anchor, permitted_live


@dataclass(frozen=True)
class InterruptedRecoveryPlan:
    failed_evidence_path: Path
    failed_file_sha256: str
    failed_payload_sha256: str
    failed_run_id: str
    target_q6: int
    step_units: int
    coupled_targets: Tuple[int, ...]
    interrupted_targets: Tuple[int, ...]
    q6_forward_waypoints: Tuple[int, ...]
    bend_return_waypoints: Tuple[Tuple[int, ...], ...]
    q6_return_waypoints: Tuple[int, ...]
    original_speeds: Tuple[int, ...]
    original_forces: Tuple[int, ...]
    recovery_mode: str = RECOVERY_MODE_COUPLED_CLOSE_PREFIX
    permitted_live_q6_range: Tuple[int, int] = (0, 1000)
    historical_source_policy: str | None = None
    historical_source_bindings_sha256: str | None = None
    allowed_recovery_routes: Tuple[str, ...] = ()
    profile_near_open_q6_range: Tuple[int, int] = (900, 1000)
    q6_open_min_angle: int = 975

    def as_binding(self) -> dict[str, Any]:
        allowed_routes = self.allowed_recovery_routes
        if not allowed_routes:
            allowed_routes = (
                (
                    RECOVERY_ROUTE_SEALED_Q6_RETURN,
                    RECOVERY_ROUTE_NEAR_OPEN_RESET,
                )
                if self.recovery_mode == RECOVERY_MODE_Q6_RETURN
                else (RECOVERY_ROUTE_COUPLED_CLOSE_PREFIX,)
            )
        binding = {
            "kind": "interrupted_installed_rh56_commissioning_failure_v1",
            "path": str(self.failed_evidence_path),
            "file_sha256": self.failed_file_sha256,
            "payload_sha256": self.failed_payload_sha256,
            "run_id": self.failed_run_id,
            "target_q6": self.target_q6,
            "step_units": self.step_units,
            "coupled_targets": list(self.coupled_targets),
            "interrupted_targets": list(self.interrupted_targets),
            "bend_return_waypoints": [list(item) for item in self.bend_return_waypoints],
            "q6_return_waypoints": list(self.q6_return_waypoints),
            "original_speeds": list(self.original_speeds),
            "original_forces": list(self.original_forces),
            "recovery_mode": self.recovery_mode,
            "permitted_live_q6_range": list(self.permitted_live_q6_range),
            "allowed_recovery_routes": list(allowed_routes),
            "profile_near_open_q6_range": list(
                self.profile_near_open_q6_range
            ),
            "q6_open_min_angle": int(self.q6_open_min_angle),
        }
        if self.historical_source_policy is not None:
            binding["historical_source_provenance"] = {
                "policy": self.historical_source_policy,
                "source_bindings_sha256": self.historical_source_bindings_sha256,
                "control_profile_migration": (
                    AUTHORIZED_HISTORICAL_CONTROL_PROFILE_MIGRATION_POLICY
                ),
                "execution_authority": False,
                "live_migrations": sorted(
                    _AUTHORIZED_HISTORICAL_SOURCE_MIGRATIONS
                ),
            }
        return binding


def build_interrupted_recovery_plan(
    evidence_path: Path,
    *,
    expected_config: Mapping[str, Any],
    expected_config_path: Path,
) -> InterruptedRecoveryPlan:
    """Validate a genuine interrupted coupled run and derive its only return.

    This deliberately accepts no successful commissioning record and no
    arbitrary hand pose.  The record must prove a complete q6 sweep, a started
    but interrupted coupled-air prefix, an all-six disable attempt, and intact
    source/profile provenance.
    """

    evidence, resolved = load_evidence(evidence_path)
    exact_authorized_stage2 = _is_exact_authorized_stage2(evidence, resolved)
    blockers = _base_integrity_blockers(evidence)
    if exact_authorized_stage2:
        blockers.extend(
            _validate_exact_historical_source_bindings(
                evidence.get("source_bindings")
            )
        )
    else:
        blockers.extend(_validate_bound_files(evidence.get("source_bindings")))
    blockers.extend(_verify_franka_binding(evidence))
    blockers.extend(_verify_snapshot_candidate_binding(evidence))

    if evidence.get("schema_version") != SCHEMA_VERSION:
        blockers.append(f"schema_version must be {SCHEMA_VERSION}")
    if evidence.get("kind") != EVIDENCE_KIND:
        blockers.append(f"kind must be {EVIDENCE_KIND}")
    if evidence.get("motion_authorized") is not False:
        blockers.append("failed evidence motion_authorized must remain false")

    profile = evidence.get("control_profile")
    if not isinstance(profile, Mapping):
        blockers.append("failed evidence control_profile binding is missing")
    else:
        exact_profile_migration = (
            exact_authorized_stage2
            and _is_exact_authorized_control_profile_migration(
                profile,
                expected_config=expected_config,
                expected_config_path=expected_config_path,
            )
        )
        snapshot = profile.get("snapshot")
        if not isinstance(snapshot, Mapping):
            blockers.append("failed evidence control-profile snapshot is missing")
        else:
            if json_sha256(snapshot) != profile.get("parsed_sha256"):
                blockers.append("failed evidence control-profile snapshot hash mismatch")
            if (
                dict(snapshot) != dict(expected_config)
                and not exact_profile_migration
            ):
                blockers.append("failed evidence belongs to a different control profile")
        expected_path = Path(expected_config_path).expanduser().resolve()
        if Path(str(profile.get("path", ""))).expanduser().resolve() != expected_path:
            blockers.append("failed evidence belongs to a different profile path")
        if (
            profile.get("file_sha256") != sha256_file(expected_path)
            and not exact_profile_migration
        ):
            blockers.append("control profile file changed after the interrupted run")

    request = evidence.get("request")
    result = evidence.get("result")
    observations = evidence.get("observations")
    final = evidence.get("final")
    if not isinstance(request, Mapping):
        blockers.append("failed evidence request object is missing")
    if not isinstance(result, Mapping):
        blockers.append("failed evidence result object is missing")
    if not isinstance(observations, Mapping):
        blockers.append("failed evidence observations object is missing")
    if not isinstance(final, Mapping):
        blockers.append("failed evidence final object is missing")
    if blockers:
        raise ValueError("interrupted recovery is LOCKED: " + "; ".join(dict.fromkeys(blockers)))

    assert isinstance(request, Mapping)
    assert isinstance(result, Mapping)
    assert isinstance(observations, Mapping)
    assert isinstance(final, Mapping)
    try:
        target_q6 = int(request.get("target_q6"))
        step_units = int(request.get("step_units"))
    except (TypeError, ValueError) as exc:
        raise ValueError("interrupted recovery is LOCKED: invalid q6 request") from exc
    if (
        isinstance(request.get("target_q6"), bool)
        or not isinstance(request.get("target_q6"), int)
        or not 0 <= target_q6 < 900
    ):
        blockers.append("recovery requires a wide-range q6 target in 0..899")
    if (
        isinstance(request.get("step_units"), bool)
        or not isinstance(request.get("step_units"), int)
        or not 10 <= step_units <= 50
    ):
        blockers.append("recorded step_units must be an integer in 10..50")
    if request.get("speed") != 40 or request.get("force_limit_g") != 80:
        blockers.append("recorded motion is outside fixed 40/80g commissioning settings")
    if request.get("coupled_closure_requested") is not True:
        blockers.append("failed record did not request coupled-air closure")
    if request.get("safety_scope") != "installed_on_FR3_PLA_low_speed_unloaded_free_air_only":
        blockers.append("failed record has the wrong installed free-air safety scope")
    confirmations = evidence.get("operator_confirmations")
    if not isinstance(confirmations, Mapping) or any(
        confirmations.get(name) is not True
        for name in (
            "installed_on_fr3",
            "24v_cutoff_ready",
            "franka_stop_ready",
            "workspace_clear",
            "no_contact_PLA_scope",
            "wide_q6_confirmed",
            "coupled_air_close_confirmed",
        )
    ):
        blockers.append("failed record lacks one or more original operator confirmations")

    coupled_targets = _six_from_json(
        request.get("coupled_targets"), "request.coupled_targets"
    )
    if coupled_targets[5] != target_q6:
        blockers.append("coupled q6 target differs from request.target_q6")
    q6_forward = _q6_forward_waypoints(target_q6, step_units)
    if request.get("q6_waypoints") != list(q6_forward):
        blockers.append("recorded q6 forward path is not canonical")
    full_forward = _coupled_forward_waypoints(coupled_targets, step_units)
    if request.get("coupled_command_waypoints") != [list(item) for item in full_forward]:
        blockers.append("recorded coupled command path is not canonical")
    if request.get("coupled_step_units") != step_units:
        blockers.append("recorded coupled step size differs from the q6 step size")
    full_bends_return, full_q6_return = _canonical_return_waypoints(
        coupled_targets, step_units
    )
    if request.get("coupled_return_command_waypoints") != [
        list(item) for item in full_bends_return
    ]:
        blockers.append("recorded coupled return path is not canonical")
    if request.get("q6_return_waypoints") != list(full_q6_return):
        blockers.append("recorded q6 return path is not canonical")
    if request.get("q6_return_strategy") != "canonical_reverse_bootstrap_v3":
        blockers.append("recorded q6 return strategy is not the wide-range canonical path")

    if result.get("status") != "fail":
        blockers.append("recovery requires a failed commissioning result")
    if result.get("adopted_disabled_verified") is not True:
        blockers.append("failed run did not verify initial all-six disable")
    if result.get("q6_sweep_pass") is not True:
        blockers.append("failed run did not complete its q6 sweep")
    coupled_closure_pass = result.get("coupled_closure_pass")
    if coupled_closure_pass not in (False, True):
        blockers.append("failed run has an invalid coupled-closure result")
    if result.get("q6_return_pass") is not False:
        blockers.append("failed run already claims a q6 return pass")
    operation_error = result.get("operation_error")
    if operation_error != "KeyboardInterrupt: ":
        blockers.append("failed run was not interrupted by the operator")
    if final.get("angle_targets") != list(DISABLED_TARGETS):
        blockers.append("failed run did not read back the final all-six disable target")
    if final.get("errors") != [0] * 6:
        blockers.append("failed run ended with a device error")
    if coupled_closure_pass is True:
        if request.get("target_source") != "official_snapshot_candidate":
            blockers.append(
                "interrupted Stage2 recovery requires an official snapshot candidate"
            )
        if not exact_authorized_stage2:
            blockers.append(
                "failed evidence is not the authorized interrupted Stage2 artifact"
            )
        if result.get("reopened_and_verified") is not False:
            blockers.append("interrupted Stage2 record already claims complete reopen")
        stop_error = result.get("stop_error")
        if result.get("disabled_verified") is not True and (
            not isinstance(stop_error, str)
            or "RH56StopUnconfirmed" not in stop_error
            or "post-disable status is not idle" not in stop_error
        ):
            blockers.append(
                "interrupted Stage2 record lacks the reviewed post-disable idle-stop result"
            )

    device = evidence.get("rh56_device")
    initial_snapshot = device.get("initial_snapshot") if isinstance(device, Mapping) else None
    if not isinstance(initial_snapshot, Mapping):
        blockers.append("failed evidence lacks its initial RH56 settings snapshot")
        original_speeds = (1000,) * 6
        original_forces = (500,) * 6
    else:
        try:
            original_speeds = _six_from_json(
                initial_snapshot.get("speeds"), "initial_snapshot.speeds"
            )
            original_forces = _six_from_json(
                initial_snapshot.get("force_limits"), "initial_snapshot.force_limits"
            )
        except ValueError as exc:
            blockers.append(str(exc))
            original_speeds = (1000,) * 6
            original_forces = (500,) * 6
        if initial_snapshot.get("angle_targets") != list(DISABLED_TARGETS):
            blockers.append("failed run did not begin from all-six disabled targets")
        if initial_snapshot.get("errors") != [0] * 6:
            blockers.append("failed run initial snapshot contains a device error")

    stable_required = request.get("endpoint_stable_samples")
    angle_tolerance = request.get("angle_tolerance_units")
    current_cap = request.get("max_axis_current_ma")
    if (
        isinstance(stable_required, bool)
        or not isinstance(stable_required, int)
        or not 2 <= stable_required <= 10
        or isinstance(angle_tolerance, bool)
        or not isinstance(angle_tolerance, int)
        or not 0 <= angle_tolerance <= 100
        or isinstance(current_cap, bool)
        or not isinstance(current_cap, int)
        or not 50 <= current_cap <= 1400
    ):
        blockers.append("failed record has invalid endpoint/current acceptance settings")
    else:
        q6_groups = observations.get("q6_steps")
        if not isinstance(q6_groups, list) or len(q6_groups) != len(q6_forward):
            blockers.append("q6 telemetry groups do not match the canonical sweep")
        else:
            for index, (group, waypoint) in enumerate(zip(q6_groups, q6_forward)):
                if not isinstance(group, Mapping) or group.get("target_q6") != waypoint:
                    blockers.append(f"q6 telemetry group {index} target is inconsistent")
                    continue
                try:
                    _require_stable_endpoint_tail(
                        group.get("feedback"),
                        (-1, -1, -1, -1, -1, waypoint),
                        (1000, 1000, 1000, 1000, 1000, waypoint),
                        stable_required=stable_required,
                        tolerance=min(angle_tolerance, 20),
                        current_cap=current_cap,
                        name=f"q6 telemetry group {index}",
                    )
                except ValueError as exc:
                    blockers.append(str(exc))

    binding = evidence.get("stage1_prerequisite")
    if not isinstance(binding, Mapping):
        blockers.append("wide-range failed evidence lacks its Stage1 prerequisite")
    else:
        expected_stage1: dict[str, Any] = {}
        if exact_authorized_stage2:
            expected_stage1, stage1_blockers = _exact_historical_stage1_binding(
                binding,
                expected_config=expected_config,
                expected_config_path=expected_config_path,
            )
            blockers.extend(stage1_blockers)
        else:
            try:
                expected_stage1 = build_stage1_prerequisite_binding(
                    Path(str(binding.get("path", ""))),
                    expected_config=expected_config,
                    expected_config_path=expected_config_path,
                )
            except ValueError as exc:
                blockers.append(str(exc))
        if expected_stage1 and dict(binding) != expected_stage1:
            blockers.append("failed evidence Stage1 binding does not match its file")

    recovery_mode = RECOVERY_MODE_COUPLED_CLOSE_PREFIX
    permitted_live_q6_range = (target_q6, target_q6)
    allowed_recovery_routes = (RECOVERY_ROUTE_COUPLED_CLOSE_PREFIX,)
    profile_near_open_q6_range = (900, 1000)
    q6_open_min_angle = 975
    if coupled_closure_pass is True:
        recovery_mode = RECOVERY_MODE_Q6_RETURN
        allowed_recovery_routes = (
            RECOVERY_ROUTE_SEALED_Q6_RETURN,
            RECOVERY_ROUTE_NEAR_OPEN_RESET,
        )
        try:
            (
                profile_near_open_q6_range,
                q6_open_min_angle,
            ) = _profile_near_open_policy(expected_config)
        except ValueError as exc:
            blockers.append(str(exc))
        if (
            isinstance(stable_required, int)
            and not isinstance(stable_required, bool)
            and 2 <= stable_required <= 10
            and isinstance(angle_tolerance, int)
            and not isinstance(angle_tolerance, bool)
            and 0 <= angle_tolerance <= 100
            and isinstance(current_cap, int)
            and not isinstance(current_cap, bool)
            and 50 <= current_cap <= 1400
        ):
            try:
                anchor_q6, permitted_live_q6_range = _derive_stage2_q6_return_anchor(
                    evidence,
                    target_q6=target_q6,
                    q6_forward=q6_forward,
                    full_forward=full_forward,
                    full_bends_return=full_bends_return,
                    full_q6_return=full_q6_return,
                    stable_required=stable_required,
                    angle_tolerance=angle_tolerance,
                    current_cap=current_cap,
                )
            except ValueError as exc:
                blockers.append(str(exc))
                anchor_q6 = target_q6
        else:
            anchor_q6 = target_q6
        interrupted = (1000, 1000, 1000, 1000, 1000, int(anchor_q6))
        bends_return: Tuple[Tuple[int, ...], ...] = ()
        q6_return: Tuple[int, ...] = ()
    else:
        try:
            _require_unstarted_return_groups(
                observations.get("coupled_air_return_steps"),
                full_bends_return,
                q6_only=False,
                name="bend return",
            )
            _require_unstarted_return_groups(
                observations.get("q6_return_steps"),
                full_q6_return,
                q6_only=True,
                name="q6 return",
            )
        except ValueError as exc:
            blockers.append(str(exc))
        try:
            interrupted = _derive_interrupted_waypoint(
                observations.get("coupled_air_close_steps"), full_forward
            )
        except ValueError as exc:
            blockers.append(str(exc))
            interrupted = coupled_targets
        if interrupted[5] != target_q6:
            blockers.append("interrupted command did not hold the commissioned q6 endpoint")
        coupled_groups = observations.get("coupled_air_close_steps")
        if (
            isinstance(coupled_groups, list)
            and isinstance(stable_required, int)
            and isinstance(angle_tolerance, int)
            and isinstance(current_cap, int)
        ):
            active = [
                index
                for index, group in enumerate(coupled_groups)
                if isinstance(group, Mapping) and group.get("feedback")
            ]
            for index in active[:-1]:
                try:
                    _require_stable_endpoint_tail(
                        coupled_groups[index].get("feedback"),
                        full_forward[index],
                        full_forward[index],
                        stable_required=stable_required,
                        tolerance=angle_tolerance,
                        current_cap=current_cap,
                        name=f"coupled telemetry group {index}",
                    )
                except ValueError as exc:
                    blockers.append(str(exc))
        bends_return, q6_return = _canonical_return_waypoints(interrupted, step_units)

    if blockers:
        raise ValueError("interrupted recovery is LOCKED: " + "; ".join(dict.fromkeys(blockers)))
    integrity = evidence["integrity"]
    return InterruptedRecoveryPlan(
        failed_evidence_path=resolved,
        failed_file_sha256=sha256_file(resolved),
        failed_payload_sha256=str(integrity["payload_sha256"]),
        failed_run_id=str(evidence.get("run_id", "")),
        target_q6=target_q6,
        step_units=step_units,
        coupled_targets=coupled_targets,
        interrupted_targets=interrupted,
        q6_forward_waypoints=q6_forward,
        bend_return_waypoints=bends_return,
        q6_return_waypoints=q6_return,
        original_speeds=original_speeds,
        original_forces=original_forces,
        recovery_mode=recovery_mode,
        permitted_live_q6_range=permitted_live_q6_range,
        historical_source_policy=(
            AUTHORIZED_HISTORICAL_SOURCE_POLICY
            if exact_authorized_stage2
            else None
        ),
        historical_source_bindings_sha256=(
            AUTHORIZED_HISTORICAL_SOURCE_BINDINGS_SHA256
            if exact_authorized_stage2
            else None
        ),
        allowed_recovery_routes=allowed_recovery_routes,
        profile_near_open_q6_range=profile_near_open_q6_range,
        q6_open_min_angle=q6_open_min_angle,
    )


class InterruptedRecoveryDriver(RH56SequenceDriver):
    """RH56 driver that can adopt one evidence-bound interrupted waypoint."""

    last_recovery_route: str | None = None
    last_recovery_initial_q6: int | None = None

    def disable_and_verify(self) -> None:
        """Preserve route-specific stop versus settings-cleanup semantics."""

        if self.last_recovery_route == RECOVERY_ROUTE_NEAR_OPEN_RESET:
            # The borrowed reset path intentionally restores reusable defaults.
            # Its cleanup distinguishes a proven physical stop from a later
            # speed/force restore failure; do not collapse that distinction into
            # the base driver's RH56StopUnconfirmed.
            return RH56ResetOpenDriver.disable_and_verify(self)
        return super().disable_and_verify()

    def close(self) -> None:
        """Close using the same route-specific stop classification."""

        if self.last_recovery_route == RECOVERY_ROUTE_NEAR_OPEN_RESET:
            return RH56ResetOpenDriver.close(self)
        return super().close()

    def _write_six_verified(
        self,
        address: int,
        values: Sequence[int],
        *,
        numeric_motion: bool,
    ) -> Tuple[int, ...]:
        # Recovery's Franka gate is not merely observational telemetry: no
        # numeric RH56 register write may occur after Franka has left Idle.
        # Keep this stricter behavior local to the evidence-bound recovery
        # driver so ordinary reset/open start gates remain unchanged.
        if numeric_motion:
            self._run_external_safety_check(
                f"recovery register {int(address)}", "before numeric write"
            )
        return super()._write_six_verified(
            address, values, numeric_motion=numeric_motion
        )

    def _read_feedback(self, phase: str, *args: Any, **kwargs: Any) -> Any:
        # The inherited commissioning return predates asynchronous stop polling
        # in its bend-reverse loop.  Recovery must honor request_stop at every
        # feedback boundary, not only after all bend waypoints have completed.
        if str(phase).startswith("coupled_air_return_step_") and self._stop_requested.is_set():
            raise RH56MotionStopped("interrupted bend recovery stopped by request")
        return super()._read_feedback(phase, *args, **kwargs)

    def _return_stage2_q6_to_open(
        self,
        *,
        permitted_live_q6_range: Tuple[int, int],
        max_axis_current_ma: int,
        endpoint_stable_samples: int,
        max_inactive_drift_units: int,
    ) -> Tuple[int, ...]:
        """Reacquire disabled q6 at the last pre-write sample, then reopen it."""

        stable_required = int(endpoint_stable_samples)
        inactive_drift = int(max_inactive_drift_units)
        if (
            isinstance(endpoint_stable_samples, bool)
            or stable_required != endpoint_stable_samples
            or not 2 <= stable_required <= 10
        ):
            raise ValueError("endpoint_stable_samples must be an integer in 2..10")
        if (
            isinstance(max_inactive_drift_units, bool)
            or inactive_drift != max_inactive_drift_units
            or not 1 <= inactive_drift <= 20
        ):
            raise ValueError("max_inactive_drift_units must be an integer in 1..20")
        lower, upper = (int(value) for value in permitted_live_q6_range)

        with self._operation_scope():
            try:
                self._ensure_motion_allowed()
                # Prove all outputs disabled again immediately before applying
                # the fixed recovery settings.  No bend helper is called: the
                # evidence and fresh feedback must already prove five open axes.
                for _pass_index in (1, 2):
                    self._write_disable_pass()
                self._verify_disabled_feedback(phase="stage2_q6_return_adopt_verify")
                self._configure_commissioning_settings()

                started = self._monotonic()
                preflight = self._read_feedback(
                    "stage2_q6_return_reacquire_preflight", started
                )
                self._check_fault_feedback(preflight, "Stage2 q6 return preflight")
                if preflight.angle_targets != DISABLED_TARGETS:
                    raise RH56SequenceDriverError(
                        "Stage2 q6 return requires all-six ANGLE_SET=-1"
                    )
                if any(
                    int(angle) < self.open_min_angle
                    for angle in preflight.angles[:5]
                ):
                    raise RH56SequenceDriverError(
                        "Stage2 q6 return requires all five bend axes fully open"
                    )
                if any(int(status) != 2 for status in preflight.statuses):
                    raise RH56SequenceDriverError(
                        "Stage2 q6 return requires all six actuators idle/status-2"
                    )
                latest_q6 = int(preflight.angles[5])
                if not lower <= latest_q6 <= upper:
                    raise RH56SequenceDriverError(
                        "Stage2 q6 return preflight is outside the sealed final "
                        f"reacquisition range {lower}..{upper}: actual={latest_q6}"
                    )
                caps = self._commissioning_current_caps(max_axis_current_ma)
                self._check_commissioning_currents(
                    preflight, caps, "Stage2 q6 return preflight"
                )
                if any(
                    abs(int(current)) > self.stop_max_axis_current_ma
                    for current in preflight.currents
                ):
                    raise RH56SequenceDriverError(
                        "Stage2 q6 return preflight current is not idle"
                    )
                self._observe_validated_feedback(
                    preflight, "stage2_q6_return_reacquire_preflight"
                )

                # This path is generated only after the final disabled-state
                # feedback above.  Its first command is the reviewed +50
                # direction-reversal bootstrap from the latest live ACT value.
                canonical = build_rh56_no_contact_execution_path(
                    (1000, 1000, 1000, 1000, 1000, latest_q6),
                    step_units=self.thumb_preshape_step_units,
                )
                return_waypoints = tuple(
                    item.command_targets[5]
                    for item in canonical.waypoints
                    if item.phase.startswith("q6_reverse_")
                )
                inactive_reference = tuple(int(value) for value in preflight.angles[:5])
                previous_endpoint = latest_q6
                previous_command = latest_q6
                previous_configuration = (
                    1000,
                    1000,
                    1000,
                    1000,
                    1000,
                    latest_q6,
                )
                previous_phase = "stage2_reacquired_disabled"

                for waypoint_index, waypoint in enumerate(return_waypoints):
                    phase = f"stage2_q6_return_{waypoint:04d}"
                    expected_targets = (-1, -1, -1, -1, -1, waypoint)
                    current_configuration = (
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        waypoint,
                    )
                    current_phase = f"q6_reverse_{waypoint_index:04d}"
                    self._write_six_verified(
                        self._constant("REG_ANGLE_SET"),
                        expected_targets,
                        numeric_motion=True,
                    )
                    step_started = self._monotonic()
                    deadline = step_started + self.motion_timeout_s
                    stable = 0
                    commanded_delta = waypoint - previous_command
                    required_progress = (
                        0
                        if waypoint == 1000 or commanded_delta < 10
                        else max(
                            3,
                            commanded_delta
                            - Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS,
                        )
                    )
                    opposite_slack = max(4, self.angle_tolerance // 2)
                    while True:
                        if self._stop_requested.is_set():
                            raise RH56MotionStopped(
                                f"Stage2 q6 return interrupted at target {waypoint}"
                            )
                        feedback = self._read_feedback(phase, step_started)
                        self._check_fault_feedback(feedback, phase)
                        self._check_commissioning_currents(feedback, caps, phase)
                        if feedback.angle_targets != expected_targets:
                            raise RH56SequenceDriverError(
                                f"{phase}: ANGLE_SET changed unexpectedly"
                            )
                        self._check_feedback_envelope(
                            feedback,
                            previous_configuration,
                            current_configuration,
                            phase,
                            previous_phase=previous_phase,
                            current_phase=current_phase,
                        )
                        for index, (reference, actual) in enumerate(
                            zip(inactive_reference, feedback.angles[:5])
                        ):
                            if (
                                int(actual) < self.open_min_angle
                                or abs(int(actual) - int(reference)) > inactive_drift
                                or int(feedback.statuses[index]) != 2
                            ):
                                raise RH56SequenceDriverError(
                                    f"{phase}: inactive bend axis {index} moved or left idle"
                                )
                        q6_status = int(feedback.statuses[5])
                        if q6_status == 3:
                            raise RH56SequenceDriverError(
                                f"{phase}: q6 reported force contact in free air"
                            )
                        if q6_status not in (0, 1, 2):
                            raise RH56SequenceDriverError(
                                f"{phase}: q6 returned unsupported status {q6_status}"
                            )
                        if int(feedback.angles[5]) < previous_endpoint - opposite_slack:
                            raise RH56SequenceDriverError(
                                f"{phase}: q6 moved opposite the ascending command"
                            )
                        endpoint_ok = (
                            int(feedback.angles[5]) >= self.open_min_angle
                            if waypoint == 1000
                            else abs(int(feedback.angles[5]) - waypoint)
                            <= Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS
                        )
                        stopped_current_ok = all(
                            abs(int(current)) <= self.stop_max_axis_current_ma
                            for current in feedback.currents
                        )
                        self._observe_validated_feedback(feedback, phase)
                        reached = (
                            q6_status == 2
                            and endpoint_ok
                            and stopped_current_ok
                            and int(feedback.angles[5]) - previous_endpoint
                            >= required_progress
                        )
                        stable = stable + 1 if reached else 0
                        if stable >= stable_required:
                            previous_endpoint = int(feedback.angles[5])
                            previous_command = waypoint
                            previous_configuration = current_configuration
                            previous_phase = current_phase
                            break
                        if self._monotonic() >= deadline:
                            raise RH56SequenceDriverError(
                                f"{phase}: timed out at ANGLE_ACT={feedback.angles[5]}"
                            )
                        self._sleep(self.poll_interval_s)

                for _pass_index in (1, 2):
                    self._write_disable_pass()
                self._verify_disabled_feedback(phase="stage2_q6_return_disable_verify")
                final_feedback = self._read_feedback(
                    "stage2_q6_return_open_verify", started
                )
                self._check_fault_feedback(
                    final_feedback, "Stage2 q6 return open verification"
                )
                if final_feedback.angle_targets != DISABLED_TARGETS or any(
                    int(angle) < self.open_min_angle
                    for angle in final_feedback.angles
                ):
                    raise RH56SequenceDriverError(
                        "Stage2 q6 return did not finish six-axis open and disabled"
                    )
                if any(int(status) != 2 for status in final_feedback.statuses):
                    raise RH56SequenceDriverError(
                        "Stage2 q6 return final feedback is not idle/status-2"
                    )
                if any(
                    abs(int(current)) > self.stop_max_axis_current_ma
                    for current in final_feedback.currents
                ):
                    raise RH56SequenceDriverError(
                        "Stage2 q6 return final current did not return to idle"
                    )
                self._observe_validated_feedback(
                    final_feedback, "stage2_q6_return_open_verify"
                )
                self._preshaped_q6 = None
                self._commissioned_q6_waypoints = None
                self._commissioned_bend_waypoints = None
                self._numeric_hold_targets = None
                self._numeric_hold_current_caps = None
                self._disabled_verified = True
                self.last_contact_axes = ()
                return return_waypoints
            except BaseException as exc:
                self._fail_and_latch(exc)
                raise

    def recover_interrupted_coupled_air_close_to_open(
        self,
        plan: InterruptedRecoveryPlan,
        *,
        max_axis_current_ma: int = 400,
        endpoint_stable_samples: int = 3,
        max_inactive_drift_units: int = 8,
    ) -> Tuple[int, ...]:
        if not isinstance(plan, InterruptedRecoveryPlan):
            raise TypeError("plan must be an InterruptedRecoveryPlan")
        self.last_recovery_route = None
        self.last_recovery_initial_q6 = None
        expected_q6 = _q6_forward_waypoints(plan.target_q6, plan.step_units)
        expected_coupled = _coupled_forward_waypoints(
            plan.coupled_targets, plan.step_units
        )
        if plan.q6_forward_waypoints != expected_q6:
            raise ValueError("recovery plan q6 forward path is not canonical")
        if plan.recovery_mode == RECOVERY_MODE_COUPLED_CLOSE_PREFIX:
            expected_bends, expected_return = _canonical_return_waypoints(
                plan.interrupted_targets, plan.step_units
            )
            if plan.interrupted_targets not in expected_coupled:
                raise ValueError("recovery start is not a canonical coupled-air waypoint")
            if plan.bend_return_waypoints != expected_bends:
                raise ValueError("recovery plan bend return path is not canonical")
            if plan.q6_return_waypoints != expected_return:
                raise ValueError("recovery plan q6 return path is not canonical")
        elif plan.recovery_mode == RECOVERY_MODE_Q6_RETURN:
            if plan.interrupted_targets[:5] != (1000, 1000, 1000, 1000, 1000):
                raise ValueError("Stage2 q6 recovery plan does not start with open bends")
            if plan.bend_return_waypoints or plan.q6_return_waypoints:
                raise ValueError("Stage2 q6 recovery path must be anchored from fresh feedback")
            lower, upper = plan.permitted_live_q6_range
            if not 0 <= lower <= plan.interrupted_targets[5] <= upper <= 1000:
                raise ValueError("Stage2 permitted live q6 range is invalid")
            if self._external_safety_check is None:
                raise RH56SequenceDriverError(
                    "Stage2 recovery requires the continuous Franka read-only Idle gate"
                )
            expected_routes = (
                RECOVERY_ROUTE_SEALED_Q6_RETURN,
                RECOVERY_ROUTE_NEAR_OPEN_RESET,
            )
            actual_routes = plan.allowed_recovery_routes or expected_routes
            if tuple(actual_routes) != expected_routes:
                raise ValueError("Stage2 recovery route policy is not canonical")
            near_lower, near_upper = plan.profile_near_open_q6_range
            if (int(near_lower), int(near_upper)) != (900, 1000):
                raise ValueError(
                    "Stage2 near-open recovery range must be exactly 900..1000"
                )
            if not (
                int(near_lower)
                <= int(plan.q6_open_min_angle)
                <= int(near_upper)
                and int(near_upper) - int(plan.q6_open_min_angle)
                <= RESET_Q6_ARRIVAL_TOLERANCE_UNITS
            ):
                raise ValueError(
                    "Stage2 near-open physical endpoint is not profile-derived"
                )
        else:
            raise ValueError("unsupported interrupted recovery mode")
        if plan.step_units != self.thumb_preshape_step_units:
            raise ValueError("recovery plan step size differs from the driver")
        original_speeds = _six_integral(
            plan.original_speeds, "recovery original speeds", allow_disabled=False
        )
        original_forces = _six_integral(
            plan.original_forces, "recovery original forces", allow_disabled=False
        )
        if original_speeds != plan.original_speeds or original_forces != plan.original_forces:
            raise ValueError("recovery original settings are not canonical integer tuples")

        with self._operation_lock:
            try:
                self._ensure_motion_allowed()
                if not self._disabled_verified or self._numeric_hold_targets is not None:
                    raise RH56SequenceDriverError(
                        "recovery requires a fresh adopt_disabled_state_and_verify"
                    )
                started = self._monotonic()
                # Route classification is a fresh, external-gated, read-only
                # boundary snapshot.  Never enter the strict sealed recovery,
                # latch a mismatch, and then attempt a fallback.
                preflight = self.read_state_snapshot()
                if preflight.angle_targets != DISABLED_TARGETS:
                    raise RH56SequenceDriverError(
                        "recovery preflight requires all-six ANGLE_SET=-1"
                    )
                if not all(status in (2, 0xFF) for status in preflight.statuses):
                    raise RH56SequenceDriverError(
                        "recovery preflight requires every actuator idle"
                    )
                caps = self._commissioning_current_caps(max_axis_current_ma)
                self._check_commissioning_currents(
                    preflight, caps, "interrupted recovery preflight"
                )
                if any(
                    abs(int(current)) > self.stop_max_axis_current_ma
                    for current in preflight.currents
                ):
                    raise RH56SequenceDriverError(
                        "recovery preflight current exceeds the disabled/idle bound"
                    )
                if max(preflight.temperatures) >= 50:
                    raise RH56SequenceDriverError(
                        "recovery preflight requires every actuator below 50C"
                    )
                if plan.recovery_mode == RECOVERY_MODE_Q6_RETURN:
                    lower, upper = plan.permitted_live_q6_range
                    live_q6 = int(preflight.angles[5])
                    near_lower, near_upper = (
                        int(value)
                        for value in plan.profile_near_open_q6_range
                    )
                    if int(lower) <= live_q6 <= int(upper):
                        self.last_recovery_route = (
                            RECOVERY_ROUTE_SEALED_Q6_RETURN
                        )
                        self.last_recovery_initial_q6 = live_q6
                        mismatches = [
                            (index, actual, "open")
                            for index, actual in enumerate(preflight.angles[:5])
                            if int(actual) < self.open_min_angle
                        ]
                        mismatches.extend(
                            (index, int(status), "idle/status-2")
                            for index, status in enumerate(preflight.statuses)
                            if int(status) != 2
                        )
                    elif near_lower <= live_q6 <= near_upper:
                        # This is a separate, profile-authorized state reached
                        # after the original interrupted return was superseded.
                        # It is not evidence that the historical 640..656 path
                        # executed successfully.
                        self._run_external_safety_check(
                            "profile near-open recovery route",
                            "before route selection",
                        )
                        self.last_recovery_route = (
                            RECOVERY_ROUTE_NEAR_OPEN_RESET
                        )
                        self.last_recovery_initial_q6 = live_q6
                        mismatches = [
                            (index, actual, "bend-open>=980")
                            for index, actual in enumerate(preflight.angles[:5])
                            if int(actual) < RESET_BEND_OPEN_MIN_ANGLE
                        ]
                        mismatches.extend(
                            (index, int(status), "idle/status-2")
                            for index, status in enumerate(preflight.statuses)
                            if int(status) != 2
                        )
                        mismatches.extend(
                            (
                                index,
                                int(current),
                                f"idle-current<=±{self.stop_max_axis_current_ma}mA",
                            )
                            for index, current in enumerate(preflight.currents)
                            if abs(int(current)) > self.stop_max_axis_current_ma
                        )
                    else:
                        raise RH56SequenceDriverError(
                            "live q6 is authorized by neither recovery route: "
                            f"actual={live_q6}, sealed_range={int(lower)}.."
                            f"{int(upper)}, profile_near_open_range="
                            f"{near_lower}..{near_upper}"
                        )
                else:
                    self.last_recovery_route = (
                        RECOVERY_ROUTE_COUPLED_CLOSE_PREFIX
                    )
                    self.last_recovery_initial_q6 = int(preflight.angles[5])
                    mismatches = [
                        (index, actual, expected)
                        for index, (actual, expected) in enumerate(
                            zip(preflight.angles, plan.interrupted_targets)
                        )
                        if (
                            (
                                index == 5
                                and abs(int(actual) - int(expected))
                                > min(self.angle_tolerance, 20)
                            )
                            or (
                                index < 5
                                and int(expected) == 1000
                                and int(actual) < self.open_min_angle
                            )
                            or (
                                index < 5
                                and int(expected) != 1000
                                and abs(int(actual) - int(expected))
                                > self.angle_tolerance
                            )
                        )
                    ]
                if mismatches:
                    if (
                        self.last_recovery_route
                        == RECOVERY_ROUTE_SEALED_Q6_RETURN
                    ):
                        raise RH56SequenceDriverError(
                            "live feedback does not satisfy the sealed Stage2 "
                            f"recovery route {int(lower)}..{int(upper)}: "
                            f"{mismatches}"
                        )
                    if (
                        self.last_recovery_route
                        == RECOVERY_ROUTE_NEAR_OPEN_RESET
                    ):
                        raise RH56SequenceDriverError(
                            "live feedback does not satisfy the profile near-open "
                            f"reset route {near_lower}..{near_upper}: {mismatches}"
                        )
                    raise RH56SequenceDriverError(
                        "live ANGLE_ACT does not match the evidence-bound "
                        f"interrupted waypoint: {mismatches}"
                    )

                # Seed only the state already proven by the sealed failed record
                # and the fresh disabled/idle feedback above.  The inherited
                # return method immediately disables again, then executes the
                # canonical bend reverse followed by the q6 reverse path.
                if (
                    self.last_recovery_route
                    == RECOVERY_ROUTE_SEALED_Q6_RETURN
                ):
                    live_q6 = int(preflight.angles[5])
                    live_targets = (1000, 1000, 1000, 1000, 1000, live_q6)
                    self._preshaped_q6 = live_q6
                    self._commissioned_q6_waypoints = (live_q6,)
                    # A non-empty marker takes the already-proven open-bend
                    # branch without invoking the ordinary reset/open helper.
                    self._commissioned_bend_waypoints = (live_targets,)
                    self._numeric_hold_targets = live_targets
                elif (
                    self.last_recovery_route
                    == RECOVERY_ROUTE_COUPLED_CLOSE_PREFIX
                ):
                    self._preshaped_q6 = plan.target_q6
                    self._commissioned_q6_waypoints = plan.q6_forward_waypoints
                    self._commissioned_bend_waypoints = (plan.interrupted_targets,)
                    self._numeric_hold_targets = plan.interrupted_targets
                # Restore the settings sealed before the interrupted run, not
                # whatever temporary 40/80 values that process may have left.
                self._original_speeds = original_speeds
                self._original_forces = original_forces
                self._disabled_verified = True
            except BaseException as exc:
                self._fail_and_latch(exc)
                raise

        if self.last_recovery_route == RECOVERY_ROUTE_NEAR_OPEN_RESET:
            # Reuse the reviewed, actual-anchored deadband escape implementation
            # on this same driver/session.  Narrowing thumb_rotate_range here is
            # what makes every generated target and every accepted q6 feedback
            # remain inside the profile-authorized 900..1000 band.
            original_range = self.thumb_rotate_range
            had_q6_open_min = hasattr(self, "q6_open_min_angle")
            original_q6_open_min = getattr(self, "q6_open_min_angle", None)
            self.thumb_rotate_range = tuple(
                int(value) for value in plan.profile_near_open_q6_range
            )
            self.q6_open_min_angle = int(plan.q6_open_min_angle)
            try:
                returned = RH56ResetOpenDriver.reset_to_open(
                    self,
                    max_axis_current_ma=max_axis_current_ma,
                    endpoint_stable_samples=endpoint_stable_samples,
                    max_inactive_drift_units=max_inactive_drift_units,
                )
            finally:
                self.thumb_rotate_range = original_range
                if had_q6_open_min:
                    self.q6_open_min_angle = original_q6_open_min
                else:
                    del self.q6_open_min_angle
        elif plan.recovery_mode == RECOVERY_MODE_Q6_RETURN:
            returned = self._return_stage2_q6_to_open(
                permitted_live_q6_range=plan.permitted_live_q6_range,
                max_axis_current_ma=max_axis_current_ma,
                endpoint_stable_samples=endpoint_stable_samples,
                max_inactive_drift_units=max_inactive_drift_units,
            )
        else:
            returned = self.return_commissioned_thumb_to_open(
                max_axis_current_ma=max_axis_current_ma,
                endpoint_stable_samples=endpoint_stable_samples,
                max_inactive_drift_units=max_inactive_drift_units,
                direct_q6_return_to_open=False,
            )
        if self.last_recovery_route == RECOVERY_ROUTE_NEAR_OPEN_RESET:
            near_lower, near_upper = (
                int(value) for value in plan.profile_near_open_q6_range
            )
            runtime_path_matches = (
                all(near_lower <= int(value) <= near_upper for value in returned)
                and (
                    not returned
                    or int(returned[-1]) == near_upper
                )
            )
        elif plan.recovery_mode == RECOVERY_MODE_Q6_RETURN:
            lower, upper = plan.permitted_live_q6_range
            permitted_runtime_returns = {
                tuple(
                    item.command_targets[5]
                    for item in build_rh56_no_contact_execution_path(
                        (1000, 1000, 1000, 1000, 1000, anchor),
                        step_units=plan.step_units,
                    ).waypoints
                    if item.phase.startswith("q6_reverse_")
                )
                for anchor in range(int(lower), int(upper) + 1)
            }
            runtime_path_matches = tuple(returned) in permitted_runtime_returns
        else:
            runtime_path_matches = tuple(returned) == plan.q6_return_waypoints
        if not runtime_path_matches:
            error = RH56SequenceDriverError(
                "runtime q6 return path differs from the evidence-bound recovery plan"
            )
            self._fail_and_latch(error)
            raise error
        final_feedback = self.telemetry[-1] if self.telemetry else None
        final_q6_min = (
            max(
                int(plan.profile_near_open_q6_range[0]),
                int(plan.profile_near_open_q6_range[1])
                - RESET_Q6_ARRIVAL_TOLERANCE_UNITS,
            )
            if self.last_recovery_route == RECOVERY_ROUTE_NEAR_OPEN_RESET
            else int(self.open_min_angle)
        )
        if (
            final_feedback is None
            or final_feedback.angle_targets != DISABLED_TARGETS
            or any(
                int(angle) < RESET_BEND_OPEN_MIN_ANGLE
                for angle in final_feedback.angles[:5]
            )
            or int(final_feedback.angles[5]) < final_q6_min
            or any(int(status) != 2 for status in final_feedback.statuses)
            or any(int(error) != 0 for error in final_feedback.errors)
            or any(
                abs(int(current)) > self.stop_max_axis_current_ma
                for current in final_feedback.currents
            )
        ):
            error = RH56SequenceDriverError(
                "recovery final feedback is not six-axis open, disabled, and idle/status-2"
            )
            self._fail_and_latch(error)
            raise error
        return tuple(returned)


__all__ = [
    "InterruptedRecoveryDriver",
    "InterruptedRecoveryPlan",
    "RECOVERY_EVIDENCE_KIND",
    "RECOVERY_MODE_COUPLED_CLOSE_PREFIX",
    "RECOVERY_MODE_Q6_RETURN",
    "RECOVERY_ROUTE_COUPLED_CLOSE_PREFIX",
    "RECOVERY_ROUTE_NEAR_OPEN_RESET",
    "RECOVERY_ROUTE_SEALED_Q6_RETURN",
    "build_interrupted_recovery_plan",
]
