"""Canonical deterministic RH56 no-contact execution waypoint path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence, Tuple

import numpy as np


HAND_PATH_ALGORITHM = (
    "rh56_q6_then_axes0_to4_stepwise_with_empirical_q6_reverse_hysteresis_v3"
)
MIN_DIRECTION_REVERSAL_BOOTSTRAP_UNITS = 50
Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS = 30
FEEDBACK_ENVELOPE_OPEN_MIN_ANGLE = 980
FEEDBACK_ENVELOPE_ALGORITHM = (
    "rh56_post_write_angle_act_adjacent_endpoint_hull_v1"
)
LOADED_HAND_PATH_NOT_APPLICABLE_ALGORITHM = "not_applicable_loaded_grasp_v1"


def _six(value: Sequence[int], name: str, *, allow_disabled: bool = False) -> Tuple[int, ...]:
    raw = np.asarray(tuple(value), dtype=np.float64)
    minimum = -1 if allow_disabled else 0
    if (
        raw.shape != (6,)
        or not np.all(np.isfinite(raw))
        or not np.array_equal(raw, np.rint(raw))
        or np.any(raw < minimum)
        or np.any(raw > 1000)
    ):
        raise ValueError("{} must be six integer registers in [{},1000]".format(name, minimum))
    return tuple(int(item) for item in raw)


def _arrival_tolerance(value: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or int(value) != float(value)
        or not 0 <= int(value) <= 100
    ):
        raise ValueError("arrival_tolerance_units must be an integer in [0,100]")
    return int(value)


def rh56_feedback_envelope_policy(arrival_tolerance_units: int) -> dict:
    """Return the canonical runtime/geometry feedback-envelope contract.

    A numeric command is allowed to lag at its preceding accepted endpoint and
    then traverse towards the new endpoint.  Consequently every post-write
    ANGLE_ACT sample is checked against the inclusive per-axis interval hull of
    the two endpoint acceptance bands.  This is deliberately not a check
    against the new target alone: doing that would reject normal actuator lag
    immediately after a write.

    The q6 reverse path retains the empirically commissioned 30-unit endpoint
    hysteresis.  Open endpoints retain the driver's 980-unit acceptance.  Both
    widen the *audited* interval where necessary; they are never hidden behind
    the usually smaller generic arrival tolerance.
    """

    tolerance = _arrival_tolerance(arrival_tolerance_units)
    payload = {
        "algorithm": FEEDBACK_ENVELOPE_ALGORITHM,
        "arrival_tolerance_units": tolerance,
        "register_domain": [0, 1000],
        "bounds": (
            "inclusive component-wise hull of previous and current endpoint "
            "acceptance bands, clamped to register domain"
        ),
        "sample_scope": (
            "every ANGLE_ACT feedback sample after verified numeric ANGLE_SET "
            "write and before the next numeric or disable write"
        ),
        "normal_lag_policy": (
            "previous endpoint, continuous in-range transit, and current "
            "endpoint are accepted"
        ),
        "inactive_axis_policy": (
            "equal adjacent endpoint targets reduce to one target band; any "
            "stricter live-reference drift gate remains additionally active"
        ),
        "open_min_angle": FEEDBACK_ENVELOPE_OPEN_MIN_ANGLE,
        "q6_reverse_hysteresis_tolerance_units": (
            Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS
        ),
        "violation_policy": (
            "fail-latch and request immediate all-six ANGLE_SET disable"
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def validate_rh56_feedback_envelope_policy(
    value: Mapping[str, object], arrival_tolerance_units: int
) -> Mapping[str, object]:
    expected = rh56_feedback_envelope_policy(arrival_tolerance_units)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError(
            "RH56 feedback-envelope policy/hash differs from the canonical contract"
        )
    return expected


def _waypoint_feedback_tolerances(
    phase: str,
    configuration: Sequence[int],
    arrival_tolerance_units: int,
) -> np.ndarray:
    configuration6 = np.asarray(
        _six(configuration, "feedback endpoint configuration"), dtype=np.int64
    )
    tolerance = _arrival_tolerance(arrival_tolerance_units)
    result = np.full(6, tolerance, dtype=np.int64)
    phase_name = str(phase)
    if phase_name == "start_open" or phase_name.startswith("q6_reverse_"):
        result[configuration6 == 1000] = np.maximum(
            result[configuration6 == 1000],
            1000 - FEEDBACK_ENVELOPE_OPEN_MIN_ANGLE,
        )
    if phase_name.startswith("q6_reverse_") and configuration6[5] < 1000:
        result[5] = max(
            int(result[5]), Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS
        )
    return result


def rh56_feedback_interval_bounds(
    previous_configuration: Sequence[int],
    current_configuration: Sequence[int],
    arrival_tolerance_units: int,
    *,
    previous_phase: str = "",
    current_phase: str = "",
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return the inclusive six-register envelope for one numeric interval."""

    previous = np.asarray(
        _six(previous_configuration, "previous feedback endpoint"), dtype=np.int64
    )
    current = np.asarray(
        _six(current_configuration, "current feedback endpoint"), dtype=np.int64
    )
    previous_tolerance = _waypoint_feedback_tolerances(
        previous_phase, previous, arrival_tolerance_units
    )
    current_tolerance = _waypoint_feedback_tolerances(
        current_phase, current, arrival_tolerance_units
    )
    lower = np.maximum(
        0,
        np.minimum(
            previous - previous_tolerance,
            current - current_tolerance,
        ),
    )
    upper = np.minimum(
        1000,
        np.maximum(
            previous + previous_tolerance,
            current + current_tolerance,
        ),
    )
    return tuple(int(item) for item in lower), tuple(int(item) for item in upper)


def require_rh56_feedback_in_interval(
    angles: Sequence[int],
    previous_configuration: Sequence[int],
    current_configuration: Sequence[int],
    arrival_tolerance_units: int,
    *,
    previous_phase: str = "",
    current_phase: str = "",
    name: str = "RH56 feedback",
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Reject an ANGLE_ACT sample outside its audited adjacent-endpoint hull."""

    actual = _six(angles, "{} ANGLE_ACT".format(name))
    lower, upper = rh56_feedback_interval_bounds(
        previous_configuration,
        current_configuration,
        arrival_tolerance_units,
        previous_phase=previous_phase,
        current_phase=current_phase,
    )
    violations = [
        {
            "axis": index,
            "actual": int(value),
            "lower": int(low),
            "upper": int(high),
        }
        for index, (value, low, high) in enumerate(zip(actual, lower, upper))
        if value < low or value > high
    ]
    if violations:
        raise ValueError(
            "{} escaped {}: violations={}; previous={}; current={}".format(
                name,
                FEEDBACK_ENVELOPE_ALGORITHM,
                violations,
                tuple(int(value) for value in previous_configuration),
                tuple(int(value) for value in current_configuration),
            )
        )
    return lower, upper


@dataclass(frozen=True)
class RH56HandWaypoint:
    phase: str
    command_targets: Tuple[int, ...]
    actuator_configuration: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str) or not self.phase:
            raise ValueError("hand waypoint phase must be non-empty")
        command = _six(self.command_targets, "command_targets", allow_disabled=True)
        configuration = _six(self.actuator_configuration, "actuator_configuration")
        object.__setattr__(self, "command_targets", command)
        object.__setattr__(self, "actuator_configuration", configuration)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "phase": self.phase,
            "command_targets": list(self.command_targets),
            "actuator_configuration": list(self.actuator_configuration),
        }


@dataclass(frozen=True)
class RH56HandExecutionPath:
    target: Tuple[int, ...]
    step_units: int
    waypoints: Tuple[RH56HandWaypoint, ...]

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.as_dict(include_sha256=False),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self, *, include_sha256: bool = True) -> dict:
        payload = {
            "algorithm": HAND_PATH_ALGORITHM,
            "target": list(self.target),
            "step_units": self.step_units,
            "waypoints": [dict(item.as_dict()) for item in self.waypoints],
            "waypoint_count": len(self.waypoints),
            "direction_reversal_bootstrap_min_units": (
                MIN_DIRECTION_REVERSAL_BOOTSTRAP_UNITS
            ),
            "final_open_acceptance": (
                "status2_idle_and_angle_at_least_driver_open_min; "
                "no_minimum_progress_at_target_1000"
            ),
            "q6_reverse_hysteresis_tolerance_units": (
                Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS
            ),
            "continuous_inter_waypoint_collision_claimed": False,
        }
        if include_sha256:
            payload["sha256"] = self.sha256
        return payload


def build_rh56_no_contact_execution_path(
    target: Sequence[int], *, step_units: int = 25
) -> RH56HandExecutionPath:
    """Mirror the reviewed driver including its proven reversal bootstrap.

    RH56 firmware may report idle with zero current and make no motion when
    the first command after a direction reversal is only 25 units.  The first
    reopen command for each active axis is therefore at least 50 units away;
    later commands return to the configured small step.
    """

    targets = _six(target, "target")
    if isinstance(step_units, (bool, np.bool_)) or int(step_units) != float(step_units):
        raise ValueError("step_units must be an integer in 10..50")
    step = int(step_units)
    if not 10 <= step <= 50:
        raise ValueError("step_units must be an integer in 10..50")

    q6_forward = list(range(1000 - step, targets[5], -step))
    if not q6_forward or q6_forward[-1] != targets[5]:
        q6_forward.append(targets[5])
    result = [
        RH56HandWaypoint(
            "start_open",
            (-1, -1, -1, -1, -1, -1),
            (1000, 1000, 1000, 1000, 1000, 1000),
        )
    ]
    for index, value in enumerate(q6_forward):
        result.append(
            RH56HandWaypoint(
                "q6_forward_{:04d}".format(index),
                (-1, -1, -1, -1, -1, value),
                (1000, 1000, 1000, 1000, 1000, value),
            )
        )

    current = [1000] * 5
    bend_forward = []
    for axis, wanted in enumerate(targets[:5]):
        while current[axis] != wanted:
            current[axis] = max(wanted, current[axis] - step)
            bend_forward.append(tuple(current) + (targets[5],))
    for index, value in enumerate(bend_forward):
        result.append(RH56HandWaypoint("bend_forward_{:04d}".format(index), value, value))

    reverse_bends = []
    current = list(targets[:5])
    bootstrap = max(MIN_DIRECTION_REVERSAL_BOOTSTRAP_UNITS, step)
    for axis in reversed(range(5)):
        if current[axis] == 1000:
            continue
        current[axis] = min(1000, current[axis] + bootstrap)
        reverse_bends.append(tuple(current) + (targets[5],))
        while current[axis] != 1000:
            current[axis] = min(1000, current[axis] + step)
            reverse_bends.append(tuple(current) + (targets[5],))
    for index, value in enumerate(reverse_bends):
        result.append(RH56HandWaypoint("bend_reverse_{:04d}".format(index), value, value))

    first_q6_reverse = min(1000, targets[5] + bootstrap)
    q6_reverse_values = [first_q6_reverse]
    while q6_reverse_values[-1] != 1000:
        q6_reverse_values.append(min(1000, q6_reverse_values[-1] + step))
    q6_reverse = tuple(q6_reverse_values)
    for index, value in enumerate(q6_reverse):
        result.append(
            RH56HandWaypoint(
                "q6_reverse_{:04d}".format(index),
                (-1, -1, -1, -1, -1, value),
                (1000, 1000, 1000, 1000, 1000, value),
            )
        )
    return RH56HandExecutionPath(targets, step, tuple(result))


def validate_rh56_hand_execution_path(value: Mapping[str, object]) -> RH56HandExecutionPath:
    if not isinstance(value, Mapping):
        raise ValueError("hand execution path must be an object")
    if value.get("algorithm") != HAND_PATH_ALGORITHM:
        raise ValueError("unsupported RH56 hand-path algorithm")
    rebuilt = build_rh56_no_contact_execution_path(
        value.get("target", ()), step_units=value.get("step_units", 0)
    )
    expected = rebuilt.as_dict()
    if dict(value) != expected:
        raise ValueError("RH56 hand execution path differs from canonical path/hash")
    return rebuilt


def loaded_hand_execution_path_not_applicable() -> dict:
    """Explicit schema for loaded grasps, whose contact path is not this air path."""

    payload = {
        "algorithm": LOADED_HAND_PATH_NOT_APPLICABLE_ALGORITHM,
        "target": [],
        "step_units": None,
        "waypoints": [],
        "waypoint_count": 0,
        "direction_reversal_bootstrap_min_units": None,
        "final_open_acceptance": "not_applicable",
        "q6_reverse_hysteresis_tolerance_units": None,
        "continuous_inter_waypoint_collision_claimed": False,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def validate_loaded_hand_execution_path_not_applicable(value: Mapping[str, object]) -> None:
    if not isinstance(value, Mapping) or dict(value) != loaded_hand_execution_path_not_applicable():
        raise ValueError("loaded grasp hand-path binding must be explicitly not applicable")


def dense_rh56_hand_interval_joint_paths(
    path: RH56HandExecutionPath, mapper: object
) -> Tuple[np.ndarray, ...]:
    """Enumerate official q12 at every integer register in each command interval."""

    converter = getattr(mapper, "to_joint_positions_rad", None)
    if not callable(converter):
        raise TypeError("mapper must provide to_joint_positions_rad")
    result = []
    for index, (first, second) in enumerate(zip(path.waypoints[:-1], path.waypoints[1:])):
        start = np.asarray(first.actuator_configuration, dtype=np.int64)
        end = np.asarray(second.actuator_configuration, dtype=np.int64)
        changed = np.flatnonzero(start != end)
        if len(changed) > 1:
            raise ValueError("hand path interval {} changes more than one axis".format(index))
        if len(changed) == 0:
            registers = [start.copy()]
        else:
            axis = int(changed[0])
            direction = 1 if end[axis] > start[axis] else -1
            registers = []
            for value in range(int(start[axis]), int(end[axis]) + direction, direction):
                item = start.copy()
                item[axis] = value
                registers.append(item)
        q12 = np.stack([converter(item.tolist()) for item in registers], axis=0)
        if q12.ndim != 2 or q12.shape[1:] != (12,) or not np.all(np.isfinite(q12)):
            raise RuntimeError("official hand interval mapping produced invalid q12")
        result.append(q12)
    return tuple(result)


def rh56_hand_interval_q12_paths_and_feedback_tubes(
    path: RH56HandExecutionPath,
    mapper: object,
    arrival_tolerance_units: int,
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    """Map nominal 1-register paths and conservative all-six feedback tubes."""

    if (
        isinstance(arrival_tolerance_units, (bool, np.bool_))
        or int(arrival_tolerance_units) != float(arrival_tolerance_units)
        or not 0 <= int(arrival_tolerance_units) <= 100
    ):
        raise ValueError("arrival_tolerance_units must be an integer in [0,100]")
    tolerance = int(arrival_tolerance_units)
    converter = getattr(mapper, "to_joint_positions_rad", None)
    if not callable(converter):
        raise TypeError("mapper must provide to_joint_positions_rad")
    q12_paths = dense_rh56_hand_interval_joint_paths(path, mapper)
    tubes = []
    for first, second in zip(path.waypoints[:-1], path.waypoints[1:]):
        start = np.asarray(first.actuator_configuration, dtype=np.int64)
        end = np.asarray(second.actuator_configuration, dtype=np.int64)
        axis_tolerances = np.maximum(
            _waypoint_feedback_tolerances(
                first.phase, start, tolerance
            ),
            _waypoint_feedback_tolerances(
                second.phase, end, tolerance
            ),
        )
        changed = np.flatnonzero(start != end)
        if len(changed) == 0:
            registers = [start.copy()]
        else:
            axis = int(changed[0])
            direction = 1 if end[axis] > start[axis] else -1
            registers = []
            for value in range(int(start[axis]), int(end[axis]) + direction, direction):
                item = start.copy()
                item[axis] = value
                registers.append(item)
        interval_tube = np.zeros(12, dtype=np.float64)
        for registers_at_sample in registers:
            nominal = np.asarray(converter(registers_at_sample.tolist()), dtype=np.float64)
            summed_axis_uncertainty = np.zeros(12, dtype=np.float64)
            for axis in range(6):
                axis_uncertainty = np.zeros(12, dtype=np.float64)
                for signed in (-int(axis_tolerances[axis]), int(axis_tolerances[axis])):
                    perturbed = registers_at_sample.copy()
                    perturbed[axis] = int(np.clip(perturbed[axis] + signed, 0, 1000))
                    mapped = np.asarray(converter(perturbed.tolist()), dtype=np.float64)
                    axis_uncertainty = np.maximum(axis_uncertainty, np.abs(mapped - nominal))
                summed_axis_uncertainty += axis_uncertainty
            interval_tube = np.maximum(interval_tube, summed_axis_uncertainty)
        tubes.append(interval_tube)
    return q12_paths, tuple(tubes)


def rh56_hand_interval_feedback_q12_envelopes(
    path: RH56HandExecutionPath,
    mapper: object,
    arrival_tolerance_units: int,
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    """Return exact component-wise q12 feedback boxes for every interval.

    At an accepted waypoint every actuator may differ from its numeric target
    by ``arrival_tolerance_units``.  During a one-axis command interval the
    moving actuator can consequently occupy the union of the two endpoint
    tolerance bands, while each unchanged actuator remains in its own target
    band.  The official mapping is evaluated at *every integer register* in
    those ranges, including the special 1000-open row.

    The released RH56 mapping assigns disjoint q12 joints to the six actuator
    registers.  We verify that property rather than assuming it, then combine
    the six exact per-actuator extrema.  The resulting component boxes are a
    conservative superset: they retain correlations only by enlargement and
    therefore cannot create a false clearance PASS.
    """

    if (
        isinstance(arrival_tolerance_units, (bool, np.bool_))
        or int(arrival_tolerance_units) != float(arrival_tolerance_units)
        or not 0 <= int(arrival_tolerance_units) <= 100
    ):
        raise ValueError("arrival_tolerance_units must be an integer in [0,100]")
    tolerance = _arrival_tolerance(arrival_tolerance_units)
    converter = getattr(mapper, "to_joint_positions_rad", None)
    if not callable(converter):
        raise TypeError("mapper must provide to_joint_positions_rad")

    lowers = []
    uppers = []
    for interval_index, (first, second) in enumerate(
        zip(path.waypoints[:-1], path.waypoints[1:])
    ):
        start = np.asarray(first.actuator_configuration, dtype=np.int64)
        end = np.asarray(second.actuator_configuration, dtype=np.int64)
        changed = np.flatnonzero(start != end)
        if len(changed) > 1:
            raise ValueError(
                "hand path interval {} changes more than one axis".format(
                    interval_index
                )
            )
        reference = np.asarray(converter(start.tolist()), dtype=np.float64)
        if reference.shape != (12,) or not np.all(np.isfinite(reference)):
            raise RuntimeError("official hand mapping produced invalid q12")
        lower = reference.copy()
        upper = reference.copy()
        joint_contributors = np.zeros(12, dtype=np.int64)
        register_lower, register_upper = rh56_feedback_interval_bounds(
            start,
            end,
            tolerance,
            previous_phase=first.phase,
            current_phase=second.phase,
        )
        for axis in range(6):
            register_low = int(register_lower[axis])
            register_high = int(register_upper[axis])
            mapped = []
            for register in range(register_low, register_high + 1):
                sample = start.copy()
                sample[axis] = register
                value = np.asarray(converter(sample.tolist()), dtype=np.float64)
                if value.shape != (12,) or not np.all(np.isfinite(value)):
                    raise RuntimeError("official hand mapping produced invalid q12")
                mapped.append(value)
            values = np.stack(mapped, axis=0)
            axis_lower = np.min(values, axis=0)
            axis_upper = np.max(values, axis=0)
            affected = np.maximum(
                np.abs(axis_lower - reference), np.abs(axis_upper - reference)
            ) > 1.0e-12
            joint_contributors += affected.astype(np.int64)
            lower = np.minimum(lower, axis_lower)
            upper = np.maximum(upper, axis_upper)
        if np.any(joint_contributors > 1):
            raise RuntimeError(
                "official RH56 mapping is not actuator-separable at interval {}"
                .format(interval_index)
            )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise RuntimeError("RH56 feedback q12 envelope is non-finite")
        if np.any(lower > upper):
            raise RuntimeError("RH56 feedback q12 envelope is inverted")
        lowers.append(lower)
        uppers.append(upper)
    return tuple(lowers), tuple(uppers)


def dense_hand_interval_tubes_sha256(tubes: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(b"rh56_dense_q12_feedback_tubes_float64_v1;")
    for index, value in enumerate(tubes):
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update("{}:{};".format(index, array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def feedback_q12_envelopes_sha256(
    lowers: Sequence[np.ndarray], uppers: Sequence[np.ndarray]
) -> str:
    if len(lowers) != len(uppers):
        raise ValueError("RH56 feedback q12 lower/upper envelope counts differ")
    digest = hashlib.sha256()
    digest.update(b"rh56_feedback_q12_interval_boxes_float64_v1;")
    for index, (lower, upper) in enumerate(zip(lowers, uppers)):
        lower_array = np.ascontiguousarray(np.asarray(lower, dtype="<f8"))
        upper_array = np.ascontiguousarray(np.asarray(upper, dtype="<f8"))
        if lower_array.shape != (12,) or upper_array.shape != (12,):
            raise ValueError("RH56 feedback q12 envelope must contain q12 vectors")
        if not np.all(np.isfinite(lower_array)) or not np.all(
            np.isfinite(upper_array)
        ):
            raise ValueError("RH56 feedback q12 envelope is non-finite")
        if np.any(lower_array > upper_array):
            raise ValueError("RH56 feedback q12 envelope is inverted")
        digest.update("{}:{};".format(index, lower_array.shape).encode("ascii"))
        digest.update(lower_array.tobytes(order="C"))
        digest.update(upper_array.tobytes(order="C"))
    return digest.hexdigest()


def dense_hand_interval_sha256(paths: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(b"rh56_dense_q12_intervals_float64_v1;")
    for index, value in enumerate(paths):
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update("{}:{};".format(index, array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "HAND_PATH_ALGORITHM",
    "FEEDBACK_ENVELOPE_ALGORITHM",
    "FEEDBACK_ENVELOPE_OPEN_MIN_ANGLE",
    "LOADED_HAND_PATH_NOT_APPLICABLE_ALGORITHM",
    "MIN_DIRECTION_REVERSAL_BOOTSTRAP_UNITS",
    "Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS",
    "RH56HandExecutionPath",
    "RH56HandWaypoint",
    "build_rh56_no_contact_execution_path",
    "dense_hand_interval_sha256",
    "dense_hand_interval_tubes_sha256",
    "feedback_q12_envelopes_sha256",
    "dense_rh56_hand_interval_joint_paths",
    "rh56_hand_interval_q12_paths_and_feedback_tubes",
    "rh56_hand_interval_feedback_q12_envelopes",
    "require_rh56_feedback_in_interval",
    "rh56_feedback_envelope_policy",
    "rh56_feedback_interval_bounds",
    "validate_rh56_feedback_envelope_policy",
    "loaded_hand_execution_path_not_applicable",
    "validate_loaded_hand_execution_path_not_applicable",
    "validate_rh56_hand_execution_path",
]
