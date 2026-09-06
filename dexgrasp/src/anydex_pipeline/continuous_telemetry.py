"""Strict, hardware-free contract for continuous FR3/RH56 viewer feedback.

This module is the trust boundary between a separately reviewed telemetry
producer and the Open3D viewer.  It deliberately contains no libfranka,
serial, camera, shared-memory, or Open3D imports.  A producer may adapt a
native fixed-POD snapshot to the JSON-compatible mapping documented here;
the viewer accepts the result only after provenance, envelope coherence,
stream semantics, stage epoch, replay, and independent freshness checks.

The contract does *not* authenticate a malicious producer.  Its hashes and
UUID prevent accidental cross-run/cross-artifact display.  They are not a
signature and never grant motion authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import time
from typing import Any, Mapping, NoReturn, Optional, Tuple, Union
import uuid

import numpy as np

from .control_plan import validate_rigid_transform


CONTINUOUS_TELEMETRY_SCHEMA_VERSION = 2
CONTINUOUS_TELEMETRY_CONTRACT = "fr3_rh56_continuous_viewer_v1"
# Exact digest exported by the reviewed native ABI binding.  ABI_MAJOR alone
# is insufficient: a module can keep major version 1 while changing field
# names or semantics in a way that would make a permissive viewer unsafe.
NATIVE_TELEMETRY_ABI_SCHEMA_SHA256 = (
    "22347780a4caac337192480aff948c65b01e9ce2ab4fd1d1ae97e8a4c0317bb5"
)
MAX_FUTURE_SKEW_NS = 50_000_000
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

ARM_MEASUREMENT_KIND = "measured_robot_state"
ARM_SOURCE = "franka_robot_state.O_T_EE"
HAND_MEASUREMENT_KIND = "measured_register_readback"
HAND_SOURCE = "inspire_rh56.ANGLE_ACT"

NATIVE_READ_OK = 0
NATIVE_READ_NO_DATA = 1
NATIVE_READ_CONTENDED = 2
NATIVE_SOURCE_ARM = 1
NATIVE_SOURCE_HAND = 2
NATIVE_MEASUREMENT_MEASURED = 1
FNV1A64_OFFSET_BASIS = 14695981039346656037
FNV1A64_PRIME = 1099511628211


class ContinuousTelemetryError(ValueError):
    """Base class for a sample that must not be displayed as current."""


class TornContinuousTelemetryError(ContinuousTelemetryError):
    """The envelope cannot prove that its fields belong to one publication."""


class ContinuousTelemetryIdentityError(ContinuousTelemetryError):
    """The sample belongs to a different run or execution artifact set."""


class ContinuousTelemetryReplayError(ContinuousTelemetryError):
    """A publication sequence moved backwards or was mutated in place."""


@dataclass(frozen=True)
class TelemetryIdentity:
    """Immutable identity bound to one executor run and artifact set."""

    run_uuid: str
    execution_contract_sha256: str
    source_snapshot_sha256: str
    control_config_sha256: str
    calibration_sha256: str
    producer_build_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_uuid", _canonical_uuid(self.run_uuid))
        for field_name in (
            "execution_contract_sha256",
            "source_snapshot_sha256",
            "control_config_sha256",
            "calibration_sha256",
            "producer_build_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class TelemetryStage:
    """Stage identity; ``epoch`` changes whenever stage authority changes."""

    name: str
    epoch: int
    target_T_reference_EE: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ContinuousTelemetryError("telemetry stage name must not be empty")
        epoch = _nonnegative_integer(self.epoch, "telemetry stage epoch")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "epoch", epoch)
        if self.target_T_reference_EE is not None:
            object.__setattr__(
                self,
                "target_T_reference_EE",
                validate_rigid_transform(
                    self.target_T_reference_EE,
                    "continuous telemetry commanded target T_reference_EE",
                ),
            )


@dataclass(frozen=True)
class ArmTelemetrySample:
    """One measured Franka ``RobotState.O_T_EE`` sample."""

    sample_sequence: int
    stage_epoch: int
    timestamp_unix_ns: int
    timestamp_monotonic_ns: int
    T_reference_EE: np.ndarray
    reference_frame: str = "robot_base"
    measurement_kind: str = ARM_MEASUREMENT_KIND
    source: str = ARM_SOURCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_sequence",
            _nonnegative_integer(self.sample_sequence, "arm sample_sequence"),
        )
        object.__setattr__(
            self,
            "stage_epoch",
            _nonnegative_integer(self.stage_epoch, "arm stage_epoch"),
        )
        object.__setattr__(
            self,
            "timestamp_unix_ns",
            _nonnegative_integer(self.timestamp_unix_ns, "arm timestamp_unix_ns"),
        )
        object.__setattr__(
            self,
            "timestamp_monotonic_ns",
            _nonnegative_integer(
                self.timestamp_monotonic_ns, "arm timestamp_monotonic_ns"
            ),
        )
        if self.reference_frame != "robot_base":
            raise ContinuousTelemetryError(
                "arm reference_frame must be 'robot_base'"
            )
        if self.measurement_kind != ARM_MEASUREMENT_KIND or self.source != ARM_SOURCE:
            raise ContinuousTelemetryError(
                "arm feedback is not measured Franka RobotState.O_T_EE"
            )
        object.__setattr__(
            self,
            "T_reference_EE",
            validate_rigid_transform(
                self.T_reference_EE, "continuous arm T_reference_EE"
            ),
        )


@dataclass(frozen=True)
class HandTelemetrySample:
    """One measured RH56 six-register ``ANGLE_ACT`` sample.

    These values can drive an official kinematic *model reconstruction*.
    They are not twelve independent joint measurements and are not a measured
    physical hand surface.
    """

    sample_sequence: int
    stage_epoch: int
    timestamp_unix_ns: int
    timestamp_monotonic_ns: int
    angles: Tuple[int, ...]
    measurement_kind: str = HAND_MEASUREMENT_KIND
    source: str = HAND_SOURCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_sequence",
            _nonnegative_integer(self.sample_sequence, "hand sample_sequence"),
        )
        object.__setattr__(
            self,
            "stage_epoch",
            _nonnegative_integer(self.stage_epoch, "hand stage_epoch"),
        )
        object.__setattr__(
            self,
            "timestamp_unix_ns",
            _nonnegative_integer(self.timestamp_unix_ns, "hand timestamp_unix_ns"),
        )
        object.__setattr__(
            self,
            "timestamp_monotonic_ns",
            _nonnegative_integer(
                self.timestamp_monotonic_ns, "hand timestamp_monotonic_ns"
            ),
        )
        if self.measurement_kind != HAND_MEASUREMENT_KIND or self.source != HAND_SOURCE:
            raise ContinuousTelemetryError(
                "hand feedback is not measured Inspire RH56 ANGLE_ACT readback"
            )
        object.__setattr__(self, "angles", _six_angle_act(self.angles))


@dataclass(frozen=True)
class ContinuousTelemetryBundle:
    """One coherently packaged, but asynchronously sampled, viewer bundle."""

    identity: TelemetryIdentity
    bundle_sequence: int
    published_unix_ns: int
    published_monotonic_ns: int
    stage: TelemetryStage
    arm: Optional[ArmTelemetrySample]
    hand: Optional[HandTelemetrySample]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, TelemetryIdentity):
            raise TypeError("continuous telemetry identity is invalid")
        object.__setattr__(
            self,
            "bundle_sequence",
            _nonnegative_integer(self.bundle_sequence, "bundle_sequence"),
        )
        object.__setattr__(
            self,
            "published_unix_ns",
            _nonnegative_integer(self.published_unix_ns, "published_unix_ns"),
        )
        object.__setattr__(
            self,
            "published_monotonic_ns",
            _nonnegative_integer(
                self.published_monotonic_ns, "published_monotonic_ns"
            ),
        )
        if not isinstance(self.stage, TelemetryStage):
            raise TypeError("continuous telemetry stage is invalid")
        if self.arm is None and self.hand is None:
            raise ContinuousTelemetryError(
                "continuous telemetry bundle must contain arm or hand feedback"
            )


@dataclass(frozen=True)
class TelemetryVisibility:
    """Independently gated feedback safe for viewer consumption."""

    bundle: ContinuousTelemetryBundle
    arm: Optional[ArmTelemetrySample]
    hand: Optional[HandTelemetrySample]
    arm_status: str
    hand_status: str

    @property
    def show_arm_geometry(self) -> bool:
        return self.arm is not None

    @property
    def show_hand_mesh(self) -> bool:
        # The articulated mesh needs both measured wrist pose and fresh ANGLE_ACT.
        return self.arm is not None and self.hand is not None


@dataclass(frozen=True)
class NativeTelemetryStage:
    """Stage tag copied from one stable native stream sample."""

    name: str
    epoch: int
    name_hash64: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ContinuousTelemetryError("native stage name must be text")
        name = self.name
        if not name:
            raise ContinuousTelemetryError("native stage name must not be empty")
        if "\x00" in name:
            raise ContinuousTelemetryError("native stage name contains NUL")
        if len(name.encode("utf-8")) > 31:
            raise ContinuousTelemetryError("native stage name exceeds char[32]")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "epoch",
            _nonnegative_integer(self.epoch, "native stage epoch"),
        )
        hash64 = _nonnegative_integer(self.name_hash64, "native stage name_hash64")
        if hash64 > (1 << 64) - 1:
            raise ContinuousTelemetryError("native stage name_hash64 exceeds uint64")
        expected_hash = fnv1a_stage_name_hash64(name)
        if hash64 != expected_hash:
            raise ContinuousTelemetryError(
                "native stage name_hash64 does not match UTF-8 FNV-1a"
            )
        object.__setattr__(self, "name_hash64", hash64)


def fnv1a_stage_name_hash64(name: str) -> int:
    """Return the native ABI's FNV-1a hash of exact, non-NUL UTF-8 bytes."""

    if not isinstance(name, str):
        raise ContinuousTelemetryError("native stage name must be text")
    if "\x00" in name:
        raise ContinuousTelemetryError("native stage name contains NUL")
    encoded = name.encode("utf-8")
    if len(encoded) > 31:
        raise ContinuousTelemetryError("native stage name exceeds char[32]")
    value = FNV1A64_OFFSET_BASIS
    for byte in encoded:
        value ^= byte
        value = (value * FNV1A64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


@dataclass(frozen=True)
class NativeArmRead:
    """Stable native arm read, or one explicit unavailable result."""

    code: int
    sequence: int
    attempts: int
    sample: Optional[ArmTelemetrySample]
    stage: Optional[NativeTelemetryStage]
    bundle_sequence: Optional[int]

    @property
    def available(self) -> bool:
        return self.code == NATIVE_READ_OK


@dataclass(frozen=True)
class NativeHandRead:
    """Stable native hand read, or one explicit unavailable result."""

    code: int
    sequence: int
    attempts: int
    sample: Optional[HandTelemetrySample]
    stage: Optional[NativeTelemetryStage]
    bundle_sequence: Optional[int]

    @property
    def available(self) -> bool:
        return self.code == NATIVE_READ_OK


@dataclass(frozen=True)
class NativeTelemetryVisibility:
    """Viewer-safe result after independent native-stream gating."""

    identity: TelemetryIdentity
    arm: Optional[ArmTelemetrySample]
    hand: Optional[HandTelemetrySample]
    arm_stage: Optional[NativeTelemetryStage]
    hand_stage: Optional[NativeTelemetryStage]
    arm_bundle_sequence: Optional[int]
    hand_bundle_sequence: Optional[int]
    arm_status: str
    hand_status: str
    streams_coherent: bool

    @property
    def show_arm_geometry(self) -> bool:
        return self.arm is not None

    @property
    def show_hand_mesh(self) -> bool:
        return (
            self.arm is not None
            and self.hand is not None
            and self.streams_coherent
        )

    @property
    def active_stage(self) -> Optional[NativeTelemetryStage]:
        # Cartesian geometry is anchored by the arm sample.  A hand-only stage
        # may be reported in status but must not steer an EE error target.
        return self.arm_stage if self.arm is not None else None


@dataclass(frozen=True)
class NativeViewerFeedback:
    """Semantically explicit feedback that the viewer may render.

    ``T_reference_EE`` is measured Franka state feedback, not external
    ground-truth tracking. ``hand_angles`` is present only when a fresh,
    stage/bundle-coherent RH56 readback may be combined with that wrist pose.
    Any mesh built from it remains a model reconstruction.
    """

    run_uuid: str
    stage: str
    stage_epoch: int
    arm_sequence: int
    hand_sequence: Optional[int]
    arm_timestamp_unix_ns: int
    hand_timestamp_unix_ns: Optional[int]
    T_reference_EE: np.ndarray
    hand_angles: Optional[Tuple[int, ...]]
    ee_semantics: str = "measured_franka_O_T_EE_feedback_not_ground_truth"
    hand_semantics: str = (
        "official_model_reconstruction_from_measured_ANGLE_ACT_not_measured_surface"
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_uuid", _canonical_uuid(self.run_uuid))
        if not str(self.stage).strip():
            raise ContinuousTelemetryError("viewer feedback stage must not be empty")
        object.__setattr__(
            self,
            "stage_epoch",
            _nonnegative_integer(self.stage_epoch, "viewer feedback stage_epoch"),
        )
        object.__setattr__(
            self,
            "arm_sequence",
            _nonnegative_integer(self.arm_sequence, "viewer feedback arm_sequence"),
        )
        if self.hand_sequence is not None:
            object.__setattr__(
                self,
                "hand_sequence",
                _nonnegative_integer(
                    self.hand_sequence, "viewer feedback hand_sequence"
                ),
            )
        object.__setattr__(
            self,
            "arm_timestamp_unix_ns",
            _nonnegative_integer(
                self.arm_timestamp_unix_ns,
                "viewer feedback arm_timestamp_unix_ns",
            ),
        )
        if self.hand_timestamp_unix_ns is not None:
            object.__setattr__(
                self,
                "hand_timestamp_unix_ns",
                _nonnegative_integer(
                    self.hand_timestamp_unix_ns,
                    "viewer feedback hand_timestamp_unix_ns",
                ),
            )
        object.__setattr__(
            self,
            "T_reference_EE",
            validate_rigid_transform(
                self.T_reference_EE, "native viewer feedback T_reference_EE"
            ),
        )
        if self.hand_angles is not None:
            object.__setattr__(self, "hand_angles", _six_angle_act(self.hand_angles))

    @property
    def update_key(self) -> Tuple[str, int, int, int]:
        return (
            self.run_uuid,
            self.stage_epoch,
            self.arm_sequence,
            -1 if self.hand_sequence is None else self.hand_sequence,
        )


def native_viewer_feedback(
    visibility: NativeTelemetryVisibility,
) -> Optional[NativeViewerFeedback]:
    """Convert gated native streams to renderable feedback, or hide geometry."""

    if not isinstance(visibility, NativeTelemetryVisibility):
        raise TypeError("visibility must be NativeTelemetryVisibility")
    if visibility.arm is None or visibility.arm_stage is None:
        return None
    hand = visibility.hand if visibility.show_hand_mesh else None
    return NativeViewerFeedback(
        run_uuid=visibility.identity.run_uuid,
        stage=visibility.arm_stage.name,
        stage_epoch=visibility.arm_stage.epoch,
        arm_sequence=visibility.arm.sample_sequence,
        hand_sequence=(None if hand is None else hand.sample_sequence),
        arm_timestamp_unix_ns=visibility.arm.timestamp_unix_ns,
        hand_timestamp_unix_ns=(None if hand is None else hand.timestamp_unix_ns),
        T_reference_EE=visibility.arm.T_reference_EE,
        hand_angles=(None if hand is None else hand.angles),
    )


def _canonical_uuid(value: Any) -> str:
    if not isinstance(value, str):
        raise ContinuousTelemetryError("run_uuid must be a canonical UUID string")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ContinuousTelemetryError(
            "run_uuid must be a canonical UUID string"
        ) from exc
    canonical = str(parsed)
    if value != canonical:
        raise ContinuousTelemetryError(
            "run_uuid must use canonical lowercase hyphenated form"
        )
    return canonical


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContinuousTelemetryError("{} must be lowercase SHA-256".format(name))
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ContinuousTelemetryError("{} must be an integer".format(name))
    result = int(value)
    if result < 0:
        raise ContinuousTelemetryError("{} must be non-negative".format(name))
    return result


def _six_angle_act(value: Any) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ContinuousTelemetryError("hand angles must contain six integers")
    result = []
    for item in value:
        numeric = _nonnegative_integer(item, "hand ANGLE_ACT")
        if numeric > 1000:
            raise ContinuousTelemetryError("hand ANGLE_ACT must be in [0,1000]")
        result.append(numeric)
    return tuple(result)


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContinuousTelemetryError("{} must be a JSON object".format(name))
    return value


def _exact_keys(
    value: Mapping[str, Any],
    name: str,
    *,
    required: Tuple[str, ...],
    optional: Tuple[str, ...] = (),
) -> None:
    missing = sorted(set(required).difference(value))
    extra = sorted(set(value).difference(set(required).union(optional)))
    if missing:
        raise ContinuousTelemetryError(
            "{} is missing {}".format(name, ", ".join(missing))
        )
    if extra:
        raise ContinuousTelemetryError(
            "{} has unsupported fields {}".format(name, ", ".join(extra))
        )


def _identity_from_mapping(value: Any, name: str) -> TelemetryIdentity:
    mapping = _object(value, name)
    fields = (
        "run_uuid",
        "execution_contract_sha256",
        "source_snapshot_sha256",
        "control_config_sha256",
        "calibration_sha256",
        "producer_build_sha256",
    )
    _exact_keys(mapping, name, required=fields)
    return TelemetryIdentity(**{field: mapping[field] for field in fields})


def _stage_from_mapping(value: Any) -> TelemetryStage:
    mapping = _object(value, "telemetry stage")
    _exact_keys(
        mapping,
        "telemetry stage",
        required=("name", "epoch"),
        optional=("target",),
    )
    target_pose = None
    if "target" in mapping:
        target = _object(mapping["target"], "telemetry stage target")
        _exact_keys(
            target,
            "telemetry stage target",
            required=("kind", "reference_frame", "T_reference_EE"),
        )
        if target["kind"] != "commanded_target":
            raise ContinuousTelemetryError(
                "telemetry stage target kind must be 'commanded_target'"
            )
        if target["reference_frame"] != "robot_base":
            raise ContinuousTelemetryError(
                "telemetry stage target reference_frame must be 'robot_base'"
            )
        target_pose = np.asarray(target["T_reference_EE"], dtype=np.float64)
    return TelemetryStage(
        name=str(mapping["name"]),
        epoch=mapping["epoch"],
        target_T_reference_EE=target_pose,
    )


_STREAM_ENVELOPE_FIELDS = (
    "identity",
    "bundle_sequence",
    "stage_epoch",
    "sample_sequence",
    "timestamp_unix_ns",
    "timestamp_monotonic_ns",
    "measurement_kind",
    "source",
)


def _arm_from_mapping(value: Any) -> Tuple[ArmTelemetrySample, TelemetryIdentity, int]:
    mapping = _object(value, "arm telemetry")
    _exact_keys(
        mapping,
        "arm telemetry",
        required=_STREAM_ENVELOPE_FIELDS + ("reference_frame", "T_reference_EE"),
    )
    return (
        ArmTelemetrySample(
            sample_sequence=mapping["sample_sequence"],
            stage_epoch=mapping["stage_epoch"],
            timestamp_unix_ns=mapping["timestamp_unix_ns"],
            timestamp_monotonic_ns=mapping["timestamp_monotonic_ns"],
            T_reference_EE=np.asarray(mapping["T_reference_EE"], dtype=np.float64),
            reference_frame=str(mapping["reference_frame"]),
            measurement_kind=str(mapping["measurement_kind"]),
            source=str(mapping["source"]),
        ),
        _identity_from_mapping(mapping["identity"], "arm telemetry identity"),
        _nonnegative_integer(mapping["bundle_sequence"], "arm bundle_sequence"),
    )


def _hand_from_mapping(value: Any) -> Tuple[HandTelemetrySample, TelemetryIdentity, int]:
    mapping = _object(value, "hand telemetry")
    _exact_keys(
        mapping,
        "hand telemetry",
        required=_STREAM_ENVELOPE_FIELDS + ("angles",),
    )
    return (
        HandTelemetrySample(
            sample_sequence=mapping["sample_sequence"],
            stage_epoch=mapping["stage_epoch"],
            timestamp_unix_ns=mapping["timestamp_unix_ns"],
            timestamp_monotonic_ns=mapping["timestamp_monotonic_ns"],
            angles=mapping["angles"],
            measurement_kind=str(mapping["measurement_kind"]),
            source=str(mapping["source"]),
        ),
        _identity_from_mapping(mapping["identity"], "hand telemetry identity"),
        _nonnegative_integer(mapping["bundle_sequence"], "hand bundle_sequence"),
    )


def continuous_telemetry_from_mapping(
    payload: Mapping[str, Any],
) -> ContinuousTelemetryBundle:
    """Validate one JSON-compatible native-reader snapshot.

    The duplicated identity/sequence/epoch fields are intentional.  A mismatch
    means the adapter cannot prove that the streams and commit record came from
    one publication, so the whole bundle is rejected as torn.
    """

    mapping = _object(payload, "continuous telemetry")
    _exact_keys(
        mapping,
        "continuous telemetry",
        required=(
            "schema_version",
            "contract",
            "identity",
            "bundle_sequence",
            "published_unix_ns",
            "published_monotonic_ns",
            "stage",
            "commit",
        ),
        optional=("arm", "hand"),
    )
    schema = mapping["schema_version"]
    if isinstance(schema, bool) or schema != CONTINUOUS_TELEMETRY_SCHEMA_VERSION:
        raise ContinuousTelemetryError(
            "continuous telemetry schema_version must be {}".format(
                CONTINUOUS_TELEMETRY_SCHEMA_VERSION
            )
        )
    if mapping["contract"] != CONTINUOUS_TELEMETRY_CONTRACT:
        raise ContinuousTelemetryError(
            "continuous telemetry contract must be {!r}".format(
                CONTINUOUS_TELEMETRY_CONTRACT
            )
        )

    identity = _identity_from_mapping(mapping["identity"], "telemetry identity")
    bundle_sequence = _nonnegative_integer(
        mapping["bundle_sequence"], "bundle_sequence"
    )
    stage = _stage_from_mapping(mapping["stage"])
    commit = _object(mapping["commit"], "telemetry commit")
    _exact_keys(
        commit,
        "telemetry commit",
        required=("identity", "bundle_sequence", "stage_epoch"),
    )
    commit_identity = _identity_from_mapping(
        commit["identity"], "telemetry commit identity"
    )
    commit_sequence = _nonnegative_integer(
        commit["bundle_sequence"], "telemetry commit bundle_sequence"
    )
    commit_epoch = _nonnegative_integer(
        commit["stage_epoch"], "telemetry commit stage_epoch"
    )
    if (
        commit_identity != identity
        or commit_sequence != bundle_sequence
        or commit_epoch != stage.epoch
    ):
        raise TornContinuousTelemetryError(
            "continuous telemetry commit does not match its envelope"
        )

    arm = None
    hand = None
    stream_envelopes = []
    if "arm" in mapping:
        arm, arm_identity, arm_bundle_sequence = _arm_from_mapping(mapping["arm"])
        stream_envelopes.append(
            ("arm", arm_identity, arm_bundle_sequence, arm.stage_epoch)
        )
    if "hand" in mapping:
        hand, hand_identity, hand_bundle_sequence = _hand_from_mapping(mapping["hand"])
        stream_envelopes.append(
            ("hand", hand_identity, hand_bundle_sequence, hand.stage_epoch)
        )
    for name, stream_identity, stream_sequence, stream_epoch in stream_envelopes:
        if (
            stream_identity != identity
            or stream_sequence != bundle_sequence
            or stream_epoch != stage.epoch
        ):
            raise TornContinuousTelemetryError(
                "{} telemetry does not match the committed envelope".format(name)
            )

    published_unix_ns = _nonnegative_integer(
        mapping["published_unix_ns"], "published_unix_ns"
    )
    published_monotonic_ns = _nonnegative_integer(
        mapping["published_monotonic_ns"], "published_monotonic_ns"
    )
    for name, sample in (("arm", arm), ("hand", hand)):
        if sample is None:
            continue
        if sample.timestamp_unix_ns > published_unix_ns + MAX_FUTURE_SKEW_NS:
            raise TornContinuousTelemetryError(
                "{} sample is newer than its publication wall clock".format(name)
            )
        if (
            sample.timestamp_monotonic_ns
            > published_monotonic_ns + MAX_FUTURE_SKEW_NS
        ):
            raise TornContinuousTelemetryError(
                "{} sample is newer than its publication monotonic clock".format(name)
            )

    return ContinuousTelemetryBundle(
        identity=identity,
        bundle_sequence=bundle_sequence,
        published_unix_ns=published_unix_ns,
        published_monotonic_ns=published_monotonic_ns,
        stage=stage,
        arm=arm,
        hand=hand,
    )


def _sample_fingerprint(sample: Union[ArmTelemetrySample, HandTelemetrySample]) -> str:
    if isinstance(sample, ArmTelemetrySample):
        value = {
            "kind": "arm",
            "sequence": sample.sample_sequence,
            "stage_epoch": sample.stage_epoch,
            "unix_ns": sample.timestamp_unix_ns,
            "monotonic_ns": sample.timestamp_monotonic_ns,
            "T_reference_EE": sample.T_reference_EE.tolist(),
        }
    else:
        value = {
            "kind": "hand",
            "sequence": sample.sample_sequence,
            "stage_epoch": sample.stage_epoch,
            "unix_ns": sample.timestamp_unix_ns,
            "monotonic_ns": sample.timestamp_monotonic_ns,
            "angles": list(sample.angles),
        }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _native_envelope_fingerprint(
    sample: Union[ArmTelemetrySample, HandTelemetrySample],
    stage: NativeTelemetryStage,
    bundle_sequence: int,
) -> str:
    """Bind replay detection to every field outside the native sample model.

    ``stage`` and ``bundle_sequence`` live beside the reduced arm/hand sample
    dataclasses.  Excluding either would let a malformed reader mutate stream
    coherence under an unchanged publication sequence and make an old pair
    newly eligible for a green hand overlay.
    """

    value = {
        "sample_sha256": _sample_fingerprint(sample),
        "stage_name": stage.name,
        "stage_epoch": stage.epoch,
        "stage_name_hash64": stage.name_hash64,
        "bundle_sequence": bundle_sequence,
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _freshness_status(
    sample: Union[ArmTelemetrySample, HandTelemetrySample],
    *,
    name: str,
    max_age_ns: int,
    now_unix_ns: int,
    now_monotonic_ns: int,
) -> Optional[str]:
    wall_delta = now_unix_ns - sample.timestamp_unix_ns
    monotonic_delta = now_monotonic_ns - sample.timestamp_monotonic_ns
    if wall_delta < -MAX_FUTURE_SKEW_NS or monotonic_delta < -MAX_FUTURE_SKEW_NS:
        return "{} timestamp is in the future".format(name)
    age_ns = max(0, wall_delta, monotonic_delta)
    if age_ns > max_age_ns:
        return "{} stale: age={:.3f}s > {:.3f}s".format(
            name, age_ns / 1.0e9, max_age_ns / 1.0e9
        )
    return None


class ContinuousTelemetryReader:
    """Stateful fail-closed reader for one expected run.

    Duplicate polls of an unchanged bundle are allowed.  Bundle/stage sequence
    rollback, or changing bytes under the same bundle/sample sequence, is
    rejected.  Arm and hand freshness/replay are then gated independently.
    """

    def __init__(
        self,
        expected_identity: TelemetryIdentity,
        *,
        arm_max_age_s: float,
        hand_max_age_s: float,
    ) -> None:
        if not isinstance(expected_identity, TelemetryIdentity):
            raise TypeError("expected_identity must be TelemetryIdentity")
        self.expected_identity = expected_identity
        self.arm_max_age_ns = _positive_age_ns(arm_max_age_s, "arm_max_age_s")
        self.hand_max_age_ns = _positive_age_ns(hand_max_age_s, "hand_max_age_s")
        self._last_bundle_sequence: Optional[int] = None
        self._last_bundle_digest: Optional[str] = None
        self._last_stage_epoch: Optional[int] = None
        self._stage_names: dict[int, str] = {}
        self._stream_state: dict[str, Tuple[int, str]] = {}

    def read_mapping(
        self,
        payload: Mapping[str, Any],
        *,
        now_unix_ns: Optional[int] = None,
        now_monotonic_ns: Optional[int] = None,
        raw_digest: Optional[str] = None,
    ) -> TelemetryVisibility:
        bundle = continuous_telemetry_from_mapping(payload)
        digest = raw_digest or hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return self._accept(
            bundle,
            bundle_digest=digest,
            now_unix_ns=(time.time_ns() if now_unix_ns is None else now_unix_ns),
            now_monotonic_ns=(
                time.monotonic_ns()
                if now_monotonic_ns is None
                else now_monotonic_ns
            ),
        )

    def _accept(
        self,
        bundle: ContinuousTelemetryBundle,
        *,
        bundle_digest: str,
        now_unix_ns: Any,
        now_monotonic_ns: Any,
    ) -> TelemetryVisibility:
        if bundle.identity != self.expected_identity:
            raise ContinuousTelemetryIdentityError(
                "continuous telemetry run UUID/artifact hashes do not match this viewer"
            )
        now_wall = _nonnegative_integer(now_unix_ns, "now_unix_ns")
        now_mono = _nonnegative_integer(now_monotonic_ns, "now_monotonic_ns")

        if self._last_bundle_sequence is not None:
            if bundle.bundle_sequence < self._last_bundle_sequence:
                raise ContinuousTelemetryReplayError(
                    "continuous telemetry bundle_sequence moved backwards"
                )
            if (
                bundle.bundle_sequence == self._last_bundle_sequence
                and bundle_digest != self._last_bundle_digest
            ):
                raise ContinuousTelemetryReplayError(
                    "continuous telemetry changed under the same bundle_sequence"
                )
        if self._last_stage_epoch is not None:
            if bundle.stage.epoch < self._last_stage_epoch:
                raise ContinuousTelemetryReplayError(
                    "continuous telemetry stage_epoch moved backwards"
                )
        known_name = self._stage_names.get(bundle.stage.epoch)
        if known_name is not None and known_name != bundle.stage.name:
            raise TornContinuousTelemetryError(
                "continuous telemetry stage name changed without a new stage_epoch"
            )

        arm, arm_status = self._gate_stream(
            "arm",
            bundle.arm,
            max_age_ns=self.arm_max_age_ns,
            now_unix_ns=now_wall,
            now_monotonic_ns=now_mono,
        )
        hand, hand_status = self._gate_stream(
            "hand",
            bundle.hand,
            max_age_ns=self.hand_max_age_ns,
            now_unix_ns=now_wall,
            now_monotonic_ns=now_mono,
        )

        self._last_bundle_sequence = bundle.bundle_sequence
        self._last_bundle_digest = bundle_digest
        self._last_stage_epoch = bundle.stage.epoch
        self._stage_names[bundle.stage.epoch] = bundle.stage.name
        return TelemetryVisibility(
            bundle=bundle,
            arm=arm,
            hand=hand,
            arm_status=arm_status,
            hand_status=hand_status,
        )

    def _gate_stream(
        self,
        name: str,
        sample: Optional[Union[ArmTelemetrySample, HandTelemetrySample]],
        *,
        max_age_ns: int,
        now_unix_ns: int,
        now_monotonic_ns: int,
    ) -> Tuple[Optional[Any], str]:
        if sample is None:
            return None, "{} unavailable".format(name)
        fingerprint = _sample_fingerprint(sample)
        previous = self._stream_state.get(name)
        if previous is not None:
            previous_sequence, previous_fingerprint = previous
            if sample.sample_sequence < previous_sequence:
                return None, "{} sample_sequence moved backwards".format(name)
            if (
                sample.sample_sequence == previous_sequence
                and fingerprint != previous_fingerprint
            ):
                return None, "{} changed under the same sample_sequence".format(name)
        if previous is None or sample.sample_sequence > previous[0]:
            self._stream_state[name] = (sample.sample_sequence, fingerprint)
        freshness_error = _freshness_status(
            sample,
            name=name,
            max_age_ns=max_age_ns,
            now_unix_ns=now_unix_ns,
            now_monotonic_ns=now_monotonic_ns,
        )
        if freshness_error is not None:
            return None, freshness_error
        return sample, "{} fresh measured feedback".format(name)


def telemetry_identity_from_native_header(
    value: Mapping[str, Any],
) -> Tuple[TelemetryIdentity, int, int]:
    """Parse the immutable header returned by the reviewed native reader."""

    mapping = _object(value, "native telemetry header")
    identity_fields = (
        "run_uuid",
        "execution_contract_sha256",
        "source_snapshot_sha256",
        "control_config_sha256",
        "calibration_sha256",
        "producer_build_sha256",
    )
    _exact_keys(
        mapping,
        "native telemetry header",
        required=identity_fields
        + (
            "created_unix_ns",
            "created_monotonic_ns",
            "producer_name",
            "robot_id",
        ),
    )
    for field_name, maximum_bytes in (("producer_name", 47), ("robot_id", 31)):
        field_value = mapping[field_name]
        if (
            not isinstance(field_value, str)
            or not field_value
            or "\x00" in field_value
            or len(field_value.encode("utf-8")) > maximum_bytes
        ):
            raise ContinuousTelemetryError(
                "native telemetry header {} is invalid".format(field_name)
            )
    identity = TelemetryIdentity(
        **{field: mapping[field] for field in identity_fields}
    )
    return (
        identity,
        _nonnegative_integer(mapping["created_unix_ns"], "created_unix_ns"),
        _nonnegative_integer(
            mapping["created_monotonic_ns"], "created_monotonic_ns"
        ),
    )


def _native_stage_from_mapping(value: Any, name: str) -> NativeTelemetryStage:
    mapping = _object(value, name)
    _exact_keys(
        mapping,
        name,
        required=("name", "epoch", "name_hash64"),
    )
    return NativeTelemetryStage(
        name=mapping["name"],
        epoch=mapping["epoch"],
        name_hash64=mapping["name_hash64"],
    )


def _native_read_envelope(
    value: Any, name: str
) -> Tuple[int, int, int, Optional[Mapping[str, Any]]]:
    mapping = _object(value, name)
    _exact_keys(
        mapping,
        name,
        required=("available", "code", "sequence", "attempts", "sample"),
    )
    available = mapping["available"]
    if not isinstance(available, bool):
        raise TornContinuousTelemetryError("{} available must be boolean".format(name))
    code = _nonnegative_integer(mapping["code"], "{} code".format(name))
    if code not in (NATIVE_READ_OK, NATIVE_READ_NO_DATA, NATIVE_READ_CONTENDED):
        raise TornContinuousTelemetryError("{} code is unknown".format(name))
    if available != (code == NATIVE_READ_OK):
        raise TornContinuousTelemetryError(
            "{} available/code fields disagree".format(name)
        )
    sequence = _nonnegative_integer(
        mapping["sequence"], "{} sequence".format(name)
    )
    attempts = _nonnegative_integer(
        mapping["attempts"], "{} attempts".format(name)
    )
    sample = mapping["sample"]
    if available:
        sample = _object(sample, "{} sample".format(name))
    elif sample is not None:
        raise TornContinuousTelemetryError(
            "{} unavailable read must not carry a sample".format(name)
        )
    return code, sequence, attempts, sample


_NATIVE_COMMON_SAMPLE_FIELDS = (
    "timestamp_unix_ns",
    "timestamp_monotonic_ns",
    "stage",
    "bundle_sequence",
    "source",
    "source_code",
    "measurement_kind",
    "measurement_kind_code",
)


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ContinuousTelemetryError(
            "{} must contain {} finite values".format(name, length)
        )
    return result


def _six_native_integers(
    value: Any,
    name: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ContinuousTelemetryError("{} must contain six integers".format(name))
    result = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, np.integer)
        ):
            raise ContinuousTelemetryError(
                "{} must contain six integers".format(name)
            )
        numeric = int(item)
        if minimum is not None and numeric < minimum:
            raise ContinuousTelemetryError("{} is below its minimum".format(name))
        if maximum is not None and numeric > maximum:
            raise ContinuousTelemetryError("{} exceeds its maximum".format(name))
        result.append(numeric)
    return tuple(result)


def native_arm_read_from_mapping(value: Any) -> NativeArmRead:
    """Adapt one reviewed native arm read without importing its extension."""

    code, sequence, attempts, sample = _native_read_envelope(value, "native arm read")
    if sample is None:
        return NativeArmRead(code, sequence, attempts, None, None, None)
    required = _NATIVE_COMMON_SAMPLE_FIELDS + (
        "O_T_EE",
        "q",
        "dq",
        "control_command_success_rate",
    )
    _exact_keys(sample, "native arm sample", required=required)
    source_code = _nonnegative_integer(
        sample["source_code"], "native arm source_code"
    )
    kind_code = _nonnegative_integer(
        sample["measurement_kind_code"], "native arm measurement_kind_code"
    )
    if (
        sample["source"] != ARM_SOURCE
        or source_code != NATIVE_SOURCE_ARM
        or sample["measurement_kind"] != "measured"
        or kind_code != NATIVE_MEASUREMENT_MEASURED
    ):
        raise ContinuousTelemetryError(
            "native arm sample is not measured Franka RobotState.O_T_EE"
        )
    flat_pose = _finite_vector(sample["O_T_EE"], 16, "native arm O_T_EE")
    pose = flat_pose.reshape((4, 4), order="F")
    if sample["q"] is not None:
        _finite_vector(sample["q"], 7, "native arm q")
    if sample["dq"] is not None:
        _finite_vector(sample["dq"], 7, "native arm dq")
    if sample["control_command_success_rate"] is not None:
        if isinstance(sample["control_command_success_rate"], (bool, np.bool_)):
            raise ContinuousTelemetryError(
                "native arm control_command_success_rate must be numeric"
            )
        success_rate = float(sample["control_command_success_rate"])
        if not np.isfinite(success_rate) or not 0.0 <= success_rate <= 1.0:
            raise ContinuousTelemetryError(
                "native arm control_command_success_rate must be in [0,1]"
            )
    stage = _native_stage_from_mapping(sample["stage"], "native arm stage")
    bundle_sequence = _nonnegative_integer(
        sample["bundle_sequence"], "native arm bundle_sequence"
    )
    arm = ArmTelemetrySample(
        sample_sequence=sequence,
        stage_epoch=stage.epoch,
        timestamp_unix_ns=sample["timestamp_unix_ns"],
        timestamp_monotonic_ns=sample["timestamp_monotonic_ns"],
        T_reference_EE=pose,
    )
    return NativeArmRead(code, sequence, attempts, arm, stage, bundle_sequence)


def native_hand_read_from_mapping(value: Any) -> NativeHandRead:
    """Adapt one reviewed native RH56 read without importing its extension."""

    code, sequence, attempts, sample = _native_read_envelope(value, "native hand read")
    if sample is None:
        return NativeHandRead(code, sequence, attempts, None, None, None)
    required = _NATIVE_COMMON_SAMPLE_FIELDS + (
        "angles",
        "angle_targets",
        "current_mA",
        "force_g",
        "temperature_c",
        "status",
        "errors",
    )
    _exact_keys(sample, "native hand sample", required=required)
    source_code = _nonnegative_integer(
        sample["source_code"], "native hand source_code"
    )
    kind_code = _nonnegative_integer(
        sample["measurement_kind_code"], "native hand measurement_kind_code"
    )
    if (
        sample["source"] != HAND_SOURCE
        or source_code != NATIVE_SOURCE_HAND
        or sample["measurement_kind"] != "measured"
        or kind_code != NATIVE_MEASUREMENT_MEASURED
    ):
        raise ContinuousTelemetryError(
            "native hand sample is not measured Inspire RH56 ANGLE_ACT readback"
        )
    angles = _six_native_integers(
        sample["angles"], "native hand ANGLE_ACT", minimum=0, maximum=1000
    )
    optional_vectors = (
        ("angle_targets", "native hand ANGLE_SET", -1, 1000),
        ("current_mA", "native hand current_mA", None, None),
        ("force_g", "native hand force_g", None, None),
        ("temperature_c", "native hand temperature_c", -32768, 32767),
        ("status", "native hand status", 0, 255),
        ("errors", "native hand errors", 0, 255),
    )
    for field_name, label, minimum, maximum in optional_vectors:
        if sample[field_name] is not None:
            _six_native_integers(
                sample[field_name], label, minimum=minimum, maximum=maximum
            )
    stage = _native_stage_from_mapping(sample["stage"], "native hand stage")
    bundle_sequence = _nonnegative_integer(
        sample["bundle_sequence"], "native hand bundle_sequence"
    )
    hand = HandTelemetrySample(
        sample_sequence=sequence,
        stage_epoch=stage.epoch,
        timestamp_unix_ns=sample["timestamp_unix_ns"],
        timestamp_monotonic_ns=sample["timestamp_monotonic_ns"],
        angles=angles,
    )
    return NativeHandRead(code, sequence, attempts, hand, stage, bundle_sequence)


class NativeContinuousTelemetryAdapter:
    """Stateful native-reader adapter with independent stream visibility.

    The caller supplies plain mappings returned by the native reader.  This
    class intentionally does not know how to open or control any device and
    does not import the native extension itself.
    """

    def __init__(
        self,
        expected_identity: TelemetryIdentity,
        *,
        arm_max_age_s: float,
        hand_max_age_s: float,
    ) -> None:
        if not isinstance(expected_identity, TelemetryIdentity):
            raise TypeError("expected_identity must be TelemetryIdentity")
        self.expected_identity = expected_identity
        self.arm_max_age_ns = _positive_age_ns(arm_max_age_s, "arm_max_age_s")
        self.hand_max_age_ns = _positive_age_ns(hand_max_age_s, "hand_max_age_s")
        self._header_creation: Optional[Tuple[int, int]] = None
        self._identity_fault: Optional[str] = None
        self._stream_state: dict[str, Tuple[int, str]] = {}
        self._stream_stage: dict[str, NativeTelemetryStage] = {}

    def _latch_identity_fault(self, message: str) -> NoReturn:
        # The native header is immutable for the lifetime of one mapping.  If it
        # changes or ceases to validate, accepting it again later could combine
        # streams across mappings/runs.  Keep the whole viewer session rejected.
        self._identity_fault = str(message)
        raise ContinuousTelemetryIdentityError(self._identity_fault)

    def accept(
        self,
        header: Mapping[str, Any],
        arm_read: Mapping[str, Any],
        hand_read: Mapping[str, Any],
        *,
        now_unix_ns: Optional[int] = None,
        now_monotonic_ns: Optional[int] = None,
    ) -> NativeTelemetryVisibility:
        if self._identity_fault is not None:
            raise ContinuousTelemetryIdentityError(self._identity_fault)
        try:
            identity, created_wall, created_mono = (
                telemetry_identity_from_native_header(header)
            )
        except (TypeError, ValueError) as exc:
            self._latch_identity_fault(
                "native telemetry immutable header is invalid: {}".format(exc)
            )
        if identity != self.expected_identity:
            self._latch_identity_fault(
                "native telemetry run UUID/artifact hashes do not match this viewer"
            )
        creation = (created_wall, created_mono)
        if self._header_creation is not None and creation != self._header_creation:
            self._latch_identity_fault(
                "native telemetry header creation changed within one viewer run"
            )
        self._header_creation = creation
        now_wall = _nonnegative_integer(
            time.time_ns() if now_unix_ns is None else now_unix_ns,
            "now_unix_ns",
        )
        now_mono = _nonnegative_integer(
            time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns,
            "now_monotonic_ns",
        )
        if (
            created_wall > now_wall + MAX_FUTURE_SKEW_NS
            or created_mono > now_mono + MAX_FUTURE_SKEW_NS
        ):
            self._latch_identity_fault(
                "native telemetry header creation timestamp is in the future"
            )

        arm_parsed, arm_parse_status = self._parse_native_stream(
            "arm", arm_read, native_arm_read_from_mapping
        )
        hand_parsed, hand_parse_status = self._parse_native_stream(
            "hand", hand_read, native_hand_read_from_mapping
        )
        arm, arm_stage, arm_bundle, arm_status = self._gate_native_stream(
            "arm",
            arm_parsed,
            arm_parse_status,
            max_age_ns=self.arm_max_age_ns,
            now_unix_ns=now_wall,
            now_monotonic_ns=now_mono,
        )
        hand, hand_stage, hand_bundle, hand_status = self._gate_native_stream(
            "hand",
            hand_parsed,
            hand_parse_status,
            max_age_ns=self.hand_max_age_ns,
            now_unix_ns=now_wall,
            now_monotonic_ns=now_mono,
        )

        coherent = False
        if arm is not None and hand is not None:
            assert arm_stage is not None and hand_stage is not None
            assert arm_bundle is not None and hand_bundle is not None
            same_stage = (
                arm_stage.epoch == hand_stage.epoch
                and arm_stage.name == hand_stage.name
                and arm_stage.name_hash64 == hand_stage.name_hash64
            )
            bundle_compatible = (
                arm_bundle == 0
                or hand_bundle == 0
                or arm_bundle == hand_bundle
            )
            coherent = same_stage and bundle_compatible
            if not coherent:
                hand_status += "; not combined with arm: stage/bundle mismatch"

        return NativeTelemetryVisibility(
            identity=identity,
            arm=arm,
            hand=hand,
            arm_stage=arm_stage,
            hand_stage=hand_stage,
            arm_bundle_sequence=arm_bundle,
            hand_bundle_sequence=hand_bundle,
            arm_status=arm_status,
            hand_status=hand_status,
            streams_coherent=coherent,
        )

    @staticmethod
    def _parse_native_stream(name: str, value: Any, parser: Any) -> Tuple[Any, str]:
        try:
            return parser(value), ""
        except (TypeError, ValueError) as exc:
            # A torn/malformed slot cannot contaminate the independently stable
            # other stream.  The viewer hides only this stream and reports why.
            return None, "{} hidden: {}".format(name, exc)

    def _gate_native_stream(
        self,
        name: str,
        read: Optional[Union[NativeArmRead, NativeHandRead]],
        parse_status: str,
        *,
        max_age_ns: int,
        now_unix_ns: int,
        now_monotonic_ns: int,
    ) -> Tuple[
        Optional[Union[ArmTelemetrySample, HandTelemetrySample]],
        Optional[NativeTelemetryStage],
        Optional[int],
        str,
    ]:
        if read is None:
            return None, None, None, parse_status
        if not read.available:
            label = (
                "NO_DATA" if read.code == NATIVE_READ_NO_DATA else "CONTENDED"
            )
            return None, None, None, "{} unavailable: {}".format(name, label)
        sample = read.sample
        stage = read.stage
        bundle = read.bundle_sequence
        assert sample is not None and stage is not None and bundle is not None
        fingerprint = _native_envelope_fingerprint(sample, stage, bundle)
        previous = self._stream_state.get(name)
        if previous is not None:
            if read.sequence < previous[0]:
                return None, None, None, "{} sequence moved backwards".format(name)
            if read.sequence == previous[0] and fingerprint != previous[1]:
                return (
                    None,
                    None,
                    None,
                    "{} changed under the same sequence".format(name),
                )
        previous_stage = self._stream_stage.get(name)
        if previous_stage is not None:
            if stage.epoch < previous_stage.epoch:
                return None, None, None, "{} stage_epoch moved backwards".format(name)
            if (
                stage.epoch == previous_stage.epoch
                and (
                    stage.name != previous_stage.name
                    or stage.name_hash64 != previous_stage.name_hash64
                )
            ):
                return (
                    None,
                    None,
                    None,
                    "{} stage changed without a new epoch".format(name),
                )
        if previous is None or read.sequence > previous[0]:
            self._stream_state[name] = (read.sequence, fingerprint)
            self._stream_stage[name] = stage
        freshness_error = _freshness_status(
            sample,
            name=name,
            max_age_ns=max_age_ns,
            now_unix_ns=now_unix_ns,
            now_monotonic_ns=now_monotonic_ns,
        )
        if freshness_error is not None:
            return None, None, None, freshness_error
        return sample, stage, bundle, "{} fresh measured feedback".format(name)


def _positive_age_ns(value: Any, name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ContinuousTelemetryError("{} must be finite and positive".format(name)) from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ContinuousTelemetryError("{} must be finite and positive".format(name))
    nanoseconds = int(round(numeric * 1.0e9))
    if nanoseconds <= 0:
        raise ContinuousTelemetryError("{} is below one nanosecond".format(name))
    return nanoseconds


__all__ = [
    "ARM_MEASUREMENT_KIND",
    "ARM_SOURCE",
    "CONTINUOUS_TELEMETRY_CONTRACT",
    "CONTINUOUS_TELEMETRY_SCHEMA_VERSION",
    "ContinuousTelemetryBundle",
    "ContinuousTelemetryError",
    "ContinuousTelemetryIdentityError",
    "ContinuousTelemetryReader",
    "ContinuousTelemetryReplayError",
    "FNV1A64_OFFSET_BASIS",
    "FNV1A64_PRIME",
    "HAND_MEASUREMENT_KIND",
    "HAND_SOURCE",
    "NATIVE_MEASUREMENT_MEASURED",
    "NATIVE_READ_CONTENDED",
    "NATIVE_READ_NO_DATA",
    "NATIVE_READ_OK",
    "NATIVE_SOURCE_ARM",
    "NATIVE_SOURCE_HAND",
    "NativeArmRead",
    "NativeContinuousTelemetryAdapter",
    "NativeHandRead",
    "NativeTelemetryStage",
    "NativeTelemetryVisibility",
    "NativeViewerFeedback",
    "TornContinuousTelemetryError",
    "TelemetryIdentity",
    "TelemetryStage",
    "TelemetryVisibility",
    "ArmTelemetrySample",
    "HandTelemetrySample",
    "continuous_telemetry_from_mapping",
    "fnv1a_stage_name_hash64",
    "native_arm_read_from_mapping",
    "native_hand_read_from_mapping",
    "native_viewer_feedback",
    "telemetry_identity_from_native_header",
]
