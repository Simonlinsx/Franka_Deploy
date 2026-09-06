"""Hardware-free eligibility checks for one installed RH56 air candidate.

These checks deliberately run before a fresh camera capture.  A scene should
not be captured (and its short freshness window consumed) when the selected
six-axis command has not first been commissioned on the installed hand.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import numpy as np


EXACT_TARGET_ACCEPTANCE = "exact_targets_only"
SUPERVISED_RANGE_ACCEPTANCE = "operator_supervised_register_ranges_v1"
SUPERVISED_RANGE_SCOPE = "low_speed_no_contact_no_lift"


def validate_air_target_acceptance_policy(
    inspire: Mapping[str, Any],
) -> tuple[str, Tuple[Tuple[int, int], ...] | None]:
    """Validate the optional operator-supervised register-range policy.

    Existing profiles retain exact-target-only behavior.  The supervised mode
    is deliberately scoped to low-speed, no-contact, no-lift air execution;
    it records the already validated per-register envelope without pretending
    that every new AnyDex vector was separately commissioned.
    """

    mode = inspire.get("air_closure_target_acceptance", EXACT_TARGET_ACCEPTANCE)
    if mode == EXACT_TARGET_ACCEPTANCE:
        return EXACT_TARGET_ACCEPTANCE, None
    if mode != SUPERVISED_RANGE_ACCEPTANCE:
        raise ValueError("inspire.air_closure_target_acceptance is unsupported")
    if inspire.get("supervised_air_closure_scope") != SUPERVISED_RANGE_SCOPE:
        raise ValueError(
            "supervised air-closure scope must be low_speed_no_contact_no_lift"
        )
    raw_ranges = inspire.get("supervised_air_closure_axis_ranges")
    try:
        values = np.asarray(raw_ranges, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "supervised air-closure axis ranges must be six integer [min,max] "
            "pairs inside 0..1000"
        ) from exc
    if (
        values.shape != (6, 2)
        or not np.all(np.isfinite(values))
        or not np.array_equal(values, np.rint(values))
        or np.any(values[:, 0] < 0)
        or np.any(values[:, 1] > 1000)
        or np.any(values[:, 0] > values[:, 1])
    ):
        raise ValueError(
            "supervised air-closure axis ranges must be six integer [min,max] "
            "pairs inside 0..1000"
        )
    try:
        q6_range = np.asarray(
            inspire.get("thumb_rotate_validated_realtime_range"), dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "supervised q6 range must equal thumb_rotate_validated_realtime_range"
        ) from exc
    if q6_range.shape != (2,) or not np.array_equal(values[5], q6_range):
        raise ValueError(
            "supervised q6 range must equal thumb_rotate_validated_realtime_range"
        )
    return (
        SUPERVISED_RANGE_ACCEPTANCE,
        tuple((int(item[0]), int(item[1])) for item in values),
    )


def validate_air_candidate_commissioning(
    config: Mapping[str, Any], selected_targets: Sequence[float]
) -> Tuple[int, int, int, int, int, int]:
    """Return the exact target or raise when its installed evidence is absent.

    This is intentionally narrower than collision-audit validation.  It only
    establishes that the complete actuator vector, including thumb rotation,
    is already covered by the profile's commissioning evidence.  Geometry,
    scene freshness, live-q matching, and runtime confirmations remain
    independent downstream gates.
    """

    inspire = config["inspire"]
    values = np.asarray(selected_targets, dtype=np.float64)
    if (
        values.shape != (6,)
        or not np.all(np.isfinite(values))
        or not np.array_equal(values, np.rint(values))
        or np.any(values < 0)
        or np.any(values > 1000)
    ):
        raise ValueError(
            "selected air candidate must contain six integer registers in 0..1000"
        )
    validated = np.asarray(
        inspire["thumb_rotate_validated_realtime_range"], dtype=np.float64
    )
    if (
        validated.shape != (2,)
        or not np.all(np.isfinite(validated))
        or not np.array_equal(validated, np.rint(validated))
        or not 0 <= validated[0] < validated[1] <= 1000
    ):
        raise ValueError("thumb-rotate validated range is malformed")
    target = tuple(int(value) for value in values)
    q6 = target[5]
    if q6 < int(validated[0]) or q6 > int(validated[1]):
        raise ValueError(
            "candidate thumb-rotate target {} is outside validated range "
            "[{},{}]".format(q6, int(validated[0]), int(validated[1]))
        )
    if inspire.get("six_axis_coupled_closure_commissioned") is not True:
        raise ValueError(
            "inspire.six_axis_coupled_closure_commissioned is not true"
        )
    exact_targets = {
        tuple(int(value) for value in item)
        for item in inspire.get("commissioned_air_closure_targets", [])
    }
    if target not in exact_targets:
        mode, supervised_ranges = validate_air_target_acceptance_policy(inspire)
        if mode != SUPERVISED_RANGE_ACCEPTANCE or supervised_ranges is None:
            raise ValueError(
                "candidate six-axis target {} has no exact commissioned "
                "air-closure evidence".format(list(target))
            )
        if any(
            value < limits[0] or value > limits[1]
            for value, limits in zip(target, supervised_ranges)
        ):
            raise ValueError(
                "candidate six-axis target {} is outside the supervised "
                "register ranges {}".format(
                    list(target), [list(item) for item in supervised_ranges]
                )
            )
    return target  # type: ignore[return-value]


__all__ = [
    "EXACT_TARGET_ACCEPTANCE",
    "SUPERVISED_RANGE_ACCEPTANCE",
    "SUPERVISED_RANGE_SCOPE",
    "validate_air_candidate_commissioning",
    "validate_air_target_acceptance_policy",
]
