"""Bounded, pickle-free replay payload loading."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Optional
import zipfile

import numpy as np

from .config import (
    _optional_rate,
    _readonly_optional_array,
    _tabletop_intercept_config,
    _tabletop_online_planner_config,
    _validate_action_order,
)
from .models import (
    CANONICAL_ACTION_ORDER,
    MAX_REPLAY_ACTION_BYTES,
    MAX_REPLAY_ACTIONS,
    ReplayActionSequence,
)


def _decode_v205_zip(
    payload: bytes,
) -> tuple[
    object,
    Optional[float],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    str,
    object,
    object,
]:
    """Read the bounded v205 replay interchange bundle without extracting it."""

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
            members = [
                item
                for item in bundle.infolist()
                if not item.is_dir()
                and not item.filename.startswith("__MACOSX/")
                and "/._" not in item.filename
            ]
            if not members:
                raise ValueError("replay ZIP contains no usable files")
            for item in members:
                path = Path(item.filename)
                if path.is_absolute() or ".." in path.parts or item.flag_bits & 0x1:
                    raise ValueError("replay ZIP contains an unsafe member")
                if item.file_size > MAX_REPLAY_ACTION_BYTES:
                    raise ValueError("replay ZIP member exceeds the size limit")

            metadata_candidates: list[tuple[str, Mapping[str, Any]]] = []
            for item in members:
                if not item.filename.lower().endswith(".json"):
                    continue
                try:
                    candidate = json.loads(bundle.read(item).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(candidate, Mapping) and {
                    "control_hz",
                    "action_contract",
                    "recommended_replay_fields",
                }.issubset(candidate):
                    metadata_candidates.append((item.filename, candidate))
            if len(metadata_candidates) != 1:
                raise ValueError(
                    "replay ZIP must contain exactly one replay metadata JSON"
                )
            _, metadata = metadata_candidates[0]
            declared_rate = _optional_rate(
                metadata.get("control_hz"), "replay control_hz"
            )
            policy_order = metadata.get("inspire_policy_order")
            register_order = metadata.get("inspire_register_order")
            if policy_order != list(CANONICAL_ACTION_ORDER[7:]):
                raise ValueError("replay ZIP Inspire policy order is not canonical")
            if register_order != [
                "little",
                "ring",
                "middle",
                "index",
                "thumb_bending",
                "thumb_rotation",
            ]:
                raise ValueError("replay ZIP Inspire register order is not canonical")
            recommended = metadata.get("recommended_replay_fields")
            if recommended != {
                "franka": "franka_joint_target_rad",
                "inspire": "inspire_angle_set_register_order",
            }:
                raise ValueError("replay ZIP recommended target fields are unsupported")

            npz_members = [
                item
                for item in members
                if item.filename.lower().endswith(".npz")
            ]
            if len(npz_members) != 1:
                raise ValueError("replay ZIP must contain exactly one NPZ payload")
            npz_payload = bundle.read(npz_members[0])
            with np.load(io.BytesIO(npz_payload), allow_pickle=False) as archive:
                required = {
                    "time_s",
                    "policy_action",
                    "franka_joint_target_rad",
                    "inspire_angle_set_register_order",
                }
                missing = required - set(archive.files)
                if missing:
                    raise ValueError(
                        f"replay ZIP NPZ is missing arrays: {sorted(missing)}"
                    )
                raw_actions = archive["policy_action"].copy()
                time_s = archive["time_s"].copy()
                franka_targets = archive["franka_joint_target_rad"].copy()
                rh56_targets = archive[
                    "inspire_angle_set_register_order"
                ].copy()
            try:
                expected_frames = int(metadata.get("frames"))
                control_dt_s = float(metadata.get("control_dt_s"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "replay ZIP metadata frames/control_dt_s are invalid"
                ) from exc
            if expected_frames != int(np.asarray(raw_actions).shape[0]):
                raise ValueError("replay ZIP metadata frame count differs from NPZ")
            if (
                declared_rate is None
                or not np.isfinite(control_dt_s)
                or control_dt_s <= 0.0
                or not np.isclose(
                    control_dt_s,
                    1.0 / declared_rate,
                    atol=1.0e-9,
                    rtol=0.0,
                )
            ):
                raise ValueError("replay ZIP control_hz/control_dt_s disagree")
            return (
                raw_actions,
                declared_rate,
                time_s,
                franka_targets,
                rh56_targets,
                str(metadata.get("action_contract", "")).strip(),
                metadata.get("tabletop_intercept"),
                metadata.get("tabletop_online_planner"),
            )
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError(f"invalid replay ZIP: {exc}") from exc


def load_replay_actions_payload(
    payload: bytes,
    *,
    suffix: str,
    expected_policy_rate_hz: object,
    selected_steps: Optional[int] = None,
) -> ReplayActionSequence:
    """Decode and validate JSON/NPY/NPZ/CSV action rows without pickle."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("replay action payload is empty")
    if len(payload) > MAX_REPLAY_ACTION_BYTES:
        raise ValueError(
            f"replay action payload exceeds {MAX_REPLAY_ACTION_BYTES} bytes"
        )
    extension = str(suffix).strip().lower()
    declared_rate: Optional[float] = None
    recorded_time: Optional[np.ndarray] = None
    recorded_franka_targets: Optional[np.ndarray] = None
    recorded_rh56_targets: Optional[np.ndarray] = None
    recorded_action_contract = ""
    raw_tabletop_intercept: object = None
    raw_tabletop_online_planner: object = None
    if extension == ".json":
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid replay JSON: {exc}") from exc
        if isinstance(decoded, Mapping):
            unknown = set(decoded) - {
                "actions",
                "action13",
                "policy_rate_hz",
                "action_order",
            }
            if unknown:
                raise ValueError(f"unknown replay JSON fields: {sorted(unknown)}")
            raw_actions = decoded.get("actions", decoded.get("action13"))
            declared_rate = _optional_rate(
                decoded.get("policy_rate_hz"), "replay policy_rate_hz"
            )
            _validate_action_order(decoded.get("action_order"))
        else:
            raw_actions = decoded
        source_format = "json"
    elif extension == ".npy":
        try:
            raw_actions = np.load(io.BytesIO(payload), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid replay NPY: {exc}") from exc
        source_format = "npy"
    elif extension == ".npz":
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                allowed = {
                    "actions",
                    "action13",
                    "policy_action",
                    "policy_rate_hz",
                    "action_order",
                    "time_s",
                    "franka_joint_target_rad",
                    "inspire_angle_set_register_order",
                    "policy_step",
                    "franka_joint_position_rad",
                    "franka_joint_velocity_rad_s",
                    "inspire_joint_position_rad_policy_order",
                    "inspire_joint_velocity_rad_s_policy_order",
                    "inspire_joint_target_rad_policy_order",
                    "success",
                    "stable_hold",
                }
                unknown = set(archive.files) - allowed
                if unknown:
                    raise ValueError(f"unknown replay NPZ arrays: {sorted(unknown)}")
                action_key = next(
                    (
                        key
                        for key in ("actions", "action13", "policy_action")
                        if key in archive
                    ),
                    "",
                )
                if action_key not in archive:
                    raise ValueError(
                        "replay NPZ must contain actions, action13, or policy_action"
                    )
                raw_actions = archive[action_key].copy()
                if "policy_rate_hz" in archive:
                    declared_rate = _optional_rate(
                        archive["policy_rate_hz"], "replay policy_rate_hz"
                    )
                if "action_order" in archive:
                    _validate_action_order(archive["action_order"].tolist())
                if "time_s" in archive:
                    recorded_time = archive["time_s"].copy()
                if "franka_joint_target_rad" in archive:
                    recorded_franka_targets = archive[
                        "franka_joint_target_rad"
                    ].copy()
                if "inspire_angle_set_register_order" in archive:
                    recorded_rh56_targets = archive[
                        "inspire_angle_set_register_order"
                    ].copy()
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith(
                ("unknown replay", "replay NPZ", "replay action_order")
            ):
                raise
            raise ValueError(f"invalid replay NPZ: {exc}") from exc
        source_format = "npz"
    elif extension in (".csv", ".txt"):
        try:
            text = payload.decode("utf-8")
            if extension == ".csv" and text.lstrip().startswith("time_s,"):
                rows = list(csv.DictReader(io.StringIO(text)))
                action_names = [f"action_{index}" for index in range(13)]
                hand_names = [
                    "rh56_angle_set_little",
                    "rh56_angle_set_ring",
                    "rh56_angle_set_middle",
                    "rh56_angle_set_index",
                    "rh56_angle_set_thumb_bending",
                    "rh56_angle_set_thumb_rotation",
                ]
                if not rows or not all(
                    name in rows[0]
                    for name in action_names
                    + [f"franka_target_rad_{index}" for index in range(7)]
                    + hand_names
                ):
                    raise ValueError("replay CSV header is incomplete")
                raw_actions = np.asarray(
                    [[row[name] for name in action_names] for row in rows],
                    dtype=np.float64,
                )
                recorded_time = np.asarray(
                    [row["time_s"] for row in rows], dtype=np.float64
                )
                recorded_franka_targets = np.asarray(
                    [
                        [row[f"franka_target_rad_{index}"] for index in range(7)]
                        for row in rows
                    ],
                    dtype=np.float64,
                )
                recorded_rh56_targets = np.asarray(
                    [[row[name] for name in hand_names] for row in rows],
                    dtype=np.float64,
                )
            else:
                raw_actions = np.loadtxt(
                    io.StringIO(text),
                    dtype=np.float64,
                    delimiter="," if extension == ".csv" else None,
                    comments="#",
                    ndmin=2,
                )
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"invalid replay {extension[1:].upper()}: {exc}") from exc
        source_format = extension[1:]
    elif extension == ".zip":
        (
            raw_actions,
            declared_rate,
            recorded_time,
            recorded_franka_targets,
            recorded_rh56_targets,
            recorded_action_contract,
            raw_tabletop_intercept,
            raw_tabletop_online_planner,
        ) = _decode_v205_zip(payload)
        source_format = "v205_replay_zip"
    else:
        raise ValueError(
            "replay actions must use .json, .npy, .npz, .csv, .txt, or .zip"
        )

    try:
        actions = np.asarray(raw_actions, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("replay actions must be a numeric [K,13] array") from exc
    if (
        actions.ndim != 2
        or actions.shape[1] != 13
        or not 1 <= actions.shape[0] <= MAX_REPLAY_ACTIONS
    ):
        raise ValueError(
            f"replay actions must have shape [K,13], K=1..{MAX_REPLAY_ACTIONS}; "
            f"actual={actions.shape}"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("replay actions contain NaN or infinity")
    if np.any(actions < -1.0) or np.any(actions > 1.0):
        index = np.argwhere((actions < -1.0) | (actions > 1.0))[0]
        value = float(actions[tuple(index)])
        raise ValueError(
            "replay actions must already lie in [-1,1] without clipping: "
            f"row={int(index[0])} axis={int(index[1])} value={value:.9g}"
        )
    if selected_steps is not None:
        if (
            isinstance(selected_steps, bool)
            or not isinstance(selected_steps, (int, np.integer))
            or not 1 <= int(selected_steps) <= actions.shape[0]
        ):
            raise ValueError(f"selected replay steps must be in 1..{actions.shape[0]}")
    actions = np.ascontiguousarray(actions)
    actions.setflags(write=False)
    row_count = int(actions.shape[0])
    tabletop_intercept = _tabletop_intercept_config(
        raw_tabletop_intercept,
        action_count=row_count,
    )
    tabletop_online_planner = _tabletop_online_planner_config(
        raw_tabletop_online_planner,
        action_count=row_count,
    )
    if tabletop_intercept is not None and tabletop_online_planner is not None:
        raise ValueError(
            "replay cannot contain both tabletop_intercept and "
            "tabletop_online_planner"
        )
    time_values = (
        None
        if recorded_time is None
        else _readonly_optional_array(
            recorded_time,
            shape=(row_count,),
            dtype=np.dtype(np.float64),
            name="recorded time_s",
        )
    )
    if time_values is not None:
        if np.any(np.diff(time_values) <= 0.0):
            raise ValueError("recorded time_s must be strictly increasing")
        if declared_rate is None and row_count > 1:
            periods = np.diff(time_values)
            if np.allclose(periods, periods[0], atol=1.0e-6, rtol=0.0):
                declared_rate = float(1.0 / periods[0])
    expected_rate = _optional_rate(expected_policy_rate_hz, "expected policy rate")
    assert expected_rate is not None
    if declared_rate is not None and not np.isclose(
        declared_rate, expected_rate, atol=1.0e-6, rtol=0.0
    ):
        raise ValueError(
            "replay policy_rate_hz disagrees with --policy-rate-hz: "
            f"file={declared_rate:g} command={expected_rate:g}"
        )
    franka_targets = (
        None
        if recorded_franka_targets is None
        else _readonly_optional_array(
            recorded_franka_targets,
            shape=(row_count, 7),
            dtype=np.dtype(np.float32),
            name="recorded Franka targets",
        )
    )
    rh56_targets = (
        None
        if recorded_rh56_targets is None
        else _readonly_optional_array(
            recorded_rh56_targets,
            shape=(row_count, 6),
            dtype=np.dtype(np.float64),
            name="recorded RH56 targets",
        )
    )
    if rh56_targets is not None:
        if not np.all(rh56_targets == np.rint(rh56_targets)):
            raise ValueError("recorded RH56 targets must be integer registers")
        if np.any(rh56_targets < 0.0) or np.any(rh56_targets > 1000.0):
            raise ValueError("recorded RH56 targets must lie in [0,1000]")
        rh56_targets = np.asarray(rh56_targets, dtype=np.int32)
        rh56_targets.setflags(write=False)
    return ReplayActionSequence(
        actions13=actions,
        sha256=hashlib.sha256(payload).hexdigest(),
        source_format=source_format,
        declared_policy_rate_hz=declared_rate,
        recorded_time_s=time_values,
        recorded_franka_target_q_rad=franka_targets,
        recorded_rh56_angle_set_register_order=rh56_targets,
        recorded_action_contract=recorded_action_contract,
        tabletop_intercept=tabletop_intercept,
        tabletop_online_planner=tabletop_online_planner,
    )


def load_replay_actions(
    path: str | Path,
    *,
    expected_policy_rate_hz: object,
    selected_steps: Optional[int] = None,
) -> ReplayActionSequence:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"replay action file not found: {resolved}")
    size = resolved.stat().st_size
    if size > MAX_REPLAY_ACTION_BYTES:
        raise ValueError(f"replay action file exceeds {MAX_REPLAY_ACTION_BYTES} bytes")
    payload = resolved.read_bytes()
    if len(payload) != size:
        raise ValueError("replay action file changed while it was being read")
    return load_replay_actions_payload(
        payload,
        suffix=resolved.suffix,
        expected_policy_rate_hz=expected_policy_rate_hz,
        selected_steps=selected_steps,
    )

__all__ = ["load_replay_actions", "load_replay_actions_payload"]
