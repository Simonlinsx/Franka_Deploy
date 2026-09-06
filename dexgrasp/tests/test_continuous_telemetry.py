from __future__ import annotations

from copy import deepcopy
import numpy as np
import pytest

from anydex_pipeline.continuous_telemetry import (
    CONTINUOUS_TELEMETRY_CONTRACT,
    ContinuousTelemetryError,
    ContinuousTelemetryIdentityError,
    ContinuousTelemetryReader,
    ContinuousTelemetryReplayError,
    NativeContinuousTelemetryAdapter,
    TelemetryIdentity,
    TornContinuousTelemetryError,
    continuous_telemetry_from_mapping,
    fnv1a_stage_name_hash64,
    native_arm_read_from_mapping,
    native_hand_read_from_mapping,
    native_viewer_feedback,
)


RUN_UUID = "12345678-1234-5678-9234-567812345678"
OTHER_RUN_UUID = "87654321-4321-6789-a234-678943216789"
NS = 1_000_000_000


def _identity(run_uuid=RUN_UUID, contract="a" * 64):
    return {
        "run_uuid": run_uuid,
        "execution_contract_sha256": contract,
        "source_snapshot_sha256": "b" * 64,
        "control_config_sha256": "c" * 64,
        "calibration_sha256": "d" * 64,
        "producer_build_sha256": "f" * 64,
    }


def _pose(x=0.5):
    value = np.eye(4, dtype=np.float64)
    value[0, 3] = float(x)
    return value.tolist()


def _payload(
    *,
    bundle_sequence=7,
    stage_epoch=3,
    stage_name="moving_pregrasp",
    arm_sequence=100,
    hand_sequence=20,
    arm_time_ns=10 * NS,
    hand_time_ns=10 * NS,
    published_time_ns=10 * NS,
    include_arm=True,
    include_hand=True,
):
    identity = _identity()
    result = {
        "schema_version": 2,
        "contract": CONTINUOUS_TELEMETRY_CONTRACT,
        "identity": deepcopy(identity),
        "bundle_sequence": bundle_sequence,
        "published_unix_ns": published_time_ns,
        "published_monotonic_ns": published_time_ns,
        "stage": {
            "name": stage_name,
            "epoch": stage_epoch,
            "target": {
                "kind": "commanded_target",
                "reference_frame": "robot_base",
                "T_reference_EE": _pose(0.6),
            },
        },
        "commit": {
            "identity": deepcopy(identity),
            "bundle_sequence": bundle_sequence,
            "stage_epoch": stage_epoch,
        },
    }
    if include_arm:
        result["arm"] = {
            "identity": deepcopy(identity),
            "bundle_sequence": bundle_sequence,
            "stage_epoch": stage_epoch,
            "sample_sequence": arm_sequence,
            "timestamp_unix_ns": arm_time_ns,
            "timestamp_monotonic_ns": arm_time_ns,
            "measurement_kind": "measured_robot_state",
            "source": "franka_robot_state.O_T_EE",
            "reference_frame": "robot_base",
            "T_reference_EE": _pose(),
        }
    if include_hand:
        result["hand"] = {
            "identity": deepcopy(identity),
            "bundle_sequence": bundle_sequence,
            "stage_epoch": stage_epoch,
            "sample_sequence": hand_sequence,
            "timestamp_unix_ns": hand_time_ns,
            "timestamp_monotonic_ns": hand_time_ns,
            "measurement_kind": "measured_register_readback",
            "source": "inspire_rh56.ANGLE_ACT",
            "angles": [1000, 990, 980, 970, 960, 950],
        }
    return result


def _expected_identity():
    return TelemetryIdentity(**_identity())


def _reader(arm_age=0.25, hand_age=0.75):
    return ContinuousTelemetryReader(
        _expected_identity(), arm_max_age_s=arm_age, hand_max_age_s=hand_age
    )


def test_contract_parses_measured_sources_and_never_calls_mesh_actual():
    bundle = continuous_telemetry_from_mapping(_payload())

    assert bundle.identity == _expected_identity()
    assert bundle.stage.epoch == 3
    assert bundle.stage.name == "moving_pregrasp"
    assert bundle.arm.source == "franka_robot_state.O_T_EE"
    assert bundle.arm.measurement_kind == "measured_robot_state"
    assert bundle.hand.source == "inspire_rh56.ANGLE_ACT"
    assert bundle.hand.measurement_kind == "measured_register_readback"
    assert bundle.hand.angles == (1000, 990, 980, 970, 960, 950)


@pytest.mark.parametrize(
    "stream,field,value,message",
    [
        ("arm", "measurement_kind", "commanded", "not measured Franka"),
        ("arm", "source", "estimated_fk", "not measured Franka"),
        ("hand", "measurement_kind", "model_estimate", "not measured Inspire"),
        ("hand", "source", "command_target", "not measured Inspire"),
    ],
)
def test_nonmeasurement_sources_cannot_masquerade_as_current(
    stream, field, value, message
):
    payload = _payload()
    payload[stream][field] = value

    with pytest.raises(ContinuousTelemetryError, match=message):
        continuous_telemetry_from_mapping(payload)


def test_run_uuid_and_artifact_hash_mismatch_are_rejected_before_visibility():
    reader = _reader()
    wrong_run = _payload()
    for location in (wrong_run["identity"], wrong_run["commit"]["identity"]):
        location["run_uuid"] = OTHER_RUN_UUID
    for stream in ("arm", "hand"):
        wrong_run[stream]["identity"]["run_uuid"] = OTHER_RUN_UUID

    with pytest.raises(ContinuousTelemetryIdentityError, match="UUID/artifact hashes"):
        reader.read_mapping(
            wrong_run, now_unix_ns=10 * NS, now_monotonic_ns=10 * NS
        )

    wrong_hash = _payload()
    for location in (wrong_hash["identity"], wrong_hash["commit"]["identity"]):
        location["execution_contract_sha256"] = "e" * 64
    for stream in ("arm", "hand"):
        wrong_hash[stream]["identity"]["execution_contract_sha256"] = "e" * 64

    with pytest.raises(ContinuousTelemetryIdentityError, match="UUID/artifact hashes"):
        reader.read_mapping(
            wrong_hash, now_unix_ns=10 * NS, now_monotonic_ns=10 * NS
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["commit"].__setitem__("bundle_sequence", 6),
        lambda p: p["commit"].__setitem__("stage_epoch", 2),
        lambda p: p["hand"].__setitem__("bundle_sequence", 6),
        lambda p: p["arm"].__setitem__("stage_epoch", 2),
        lambda p: p["hand"].__setitem__("identity", _identity(OTHER_RUN_UUID)),
    ],
)
def test_repeated_envelope_mismatch_is_torn_and_must_hide_everything(mutate):
    payload = _payload()
    mutate(payload)

    with pytest.raises(TornContinuousTelemetryError, match="commit|envelope"):
        continuous_telemetry_from_mapping(payload)


def test_arm_and_hand_freshness_are_independent():
    reader = _reader(arm_age=0.25, hand_age=0.75)
    arm_fresh_hand_stale = _payload(
        arm_time_ns=9_900_000_000,
        hand_time_ns=9_000_000_000,
        published_time_ns=10 * NS,
    )

    view = reader.read_mapping(
        arm_fresh_hand_stale,
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )
    assert view.show_arm_geometry is True
    assert view.arm is not None
    assert view.hand is None
    assert view.show_hand_mesh is False
    assert "hand stale" in view.hand_status

    reader = _reader(arm_age=0.25, hand_age=0.75)
    arm_stale_hand_fresh = _payload(
        arm_time_ns=9_000_000_000,
        hand_time_ns=9_900_000_000,
        published_time_ns=10 * NS,
    )
    view = reader.read_mapping(
        arm_stale_hand_fresh,
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )
    assert view.arm is None
    assert view.hand is not None
    assert view.show_arm_geometry is False
    assert view.show_hand_mesh is False
    assert "arm stale" in view.arm_status
    assert view.hand_status == "hand fresh measured feedback"


def test_wall_or_monotonic_staleness_hides_only_that_stream():
    payload = _payload(
        arm_time_ns=10 * NS,
        hand_time_ns=10 * NS,
        published_time_ns=11 * NS,
    )
    payload["hand"]["timestamp_monotonic_ns"] = 9 * NS
    reader = _reader(arm_age=0.25, hand_age=0.75)

    view = reader.read_mapping(
        payload,
        now_unix_ns=10_100_000_000,
        now_monotonic_ns=10_100_000_000,
    )

    assert view.arm is not None
    assert view.hand is None
    assert "hand stale" in view.hand_status


def test_stage_epoch_is_part_of_every_committed_stream():
    payload = _payload()
    payload["hand"]["stage_epoch"] = 2

    with pytest.raises(TornContinuousTelemetryError, match="hand telemetry"):
        continuous_telemetry_from_mapping(payload)


def test_stage_epoch_rollback_and_name_reuse_are_rejected():
    reader = _reader()
    reader.read_mapping(
        _payload(stage_epoch=3),
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )
    rollback = _payload(bundle_sequence=8, stage_epoch=2)
    with pytest.raises(ContinuousTelemetryReplayError, match="stage_epoch"):
        reader.read_mapping(
            rollback, now_unix_ns=10 * NS, now_monotonic_ns=10 * NS
        )

    reader = _reader()
    reader.read_mapping(
        _payload(stage_epoch=3, stage_name="moving_pregrasp"),
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )
    renamed = _payload(
        bundle_sequence=8, stage_epoch=3, stage_name="moving_grasp"
    )
    with pytest.raises(TornContinuousTelemetryError, match="stage name"):
        reader.read_mapping(
            renamed, now_unix_ns=10 * NS, now_monotonic_ns=10 * NS
        )


def test_bundle_replay_or_same_sequence_mutation_hides_whole_bundle():
    reader = _reader()
    first = _payload(bundle_sequence=7)
    reader.read_mapping(first, now_unix_ns=10 * NS, now_monotonic_ns=10 * NS)

    rollback = _payload(bundle_sequence=6)
    with pytest.raises(ContinuousTelemetryReplayError, match="moved backwards"):
        reader.read_mapping(
            rollback, now_unix_ns=10 * NS, now_monotonic_ns=10 * NS
        )

    changed = deepcopy(first)
    changed["arm"]["T_reference_EE"] = _pose(0.7)
    with pytest.raises(ContinuousTelemetryReplayError, match="same bundle_sequence"):
        reader.read_mapping(
            changed, now_unix_ns=10 * NS, now_monotonic_ns=10 * NS
        )


def test_stream_sequence_mutation_hides_only_that_stream():
    reader = _reader()
    first = _payload(bundle_sequence=7, arm_sequence=100, hand_sequence=20)
    reader.read_mapping(first, now_unix_ns=10 * NS, now_monotonic_ns=10 * NS)

    second = _payload(bundle_sequence=8, arm_sequence=100, hand_sequence=21)
    second["arm"]["T_reference_EE"] = _pose(0.7)
    second["arm"]["bundle_sequence"] = 8
    view = reader.read_mapping(
        second, now_unix_ns=10 * NS, now_monotonic_ns=10 * NS
    )

    assert view.arm is None
    assert "same sample_sequence" in view.arm_status
    assert view.hand is not None
    assert view.show_hand_mesh is False


def test_missing_hand_keeps_measured_arm_but_hides_reconstructed_mesh():
    view = _reader().read_mapping(
        _payload(include_hand=False),
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )

    assert view.arm is not None
    assert view.hand is None
    assert view.show_arm_geometry is True
    assert view.show_hand_mesh is False
    assert view.hand_status == "hand unavailable"


def test_future_stream_timestamp_is_hidden_independently():
    payload = _payload(
        arm_time_ns=10 * NS,
        hand_time_ns=10_100_000_000,
        published_time_ns=10_100_000_000,
    )
    view = _reader().read_mapping(
        payload,
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )

    assert view.arm is not None
    assert view.hand is None
    assert view.hand_status == "hand timestamp is in the future"


def _native_header(**changes):
    value = {
        **_identity(),
        "created_unix_ns": 9 * NS,
        "created_monotonic_ns": 9 * NS,
        "producer_name": "offline-test-producer",
        "robot_id": "fr3-test",
    }
    value.update(changes)
    return value


def _native_stage(name="moving_pregrasp", epoch=3):
    return {
        "name": name,
        "epoch": epoch,
        "name_hash64": fnv1a_stage_name_hash64(name),
    }


def _native_arm_read(
    *,
    sequence=100,
    timestamp_ns=10 * NS,
    stage_name="moving_pregrasp",
    stage_epoch=3,
    bundle_sequence=40,
):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (0.51, -0.02, 0.37)
    return {
        "available": True,
        "code": 0,
        "sequence": sequence,
        "attempts": 1,
        "sample": {
            "timestamp_unix_ns": timestamp_ns,
            "timestamp_monotonic_ns": timestamp_ns,
            "stage": _native_stage(stage_name, stage_epoch),
            "bundle_sequence": bundle_sequence,
            "source": "franka_robot_state.O_T_EE",
            "source_code": 1,
            "measurement_kind": "measured",
            "measurement_kind_code": 1,
            "O_T_EE": pose.reshape(-1, order="F").tolist(),
            "q": [0.0] * 7,
            "dq": [0.0] * 7,
            "control_command_success_rate": 0.99,
        },
    }


def _native_hand_read(
    *,
    sequence=20,
    timestamp_ns=10 * NS,
    stage_name="moving_pregrasp",
    stage_epoch=3,
    bundle_sequence=40,
):
    return {
        "available": True,
        "code": 0,
        "sequence": sequence,
        "attempts": 1,
        "sample": {
            "timestamp_unix_ns": timestamp_ns,
            "timestamp_monotonic_ns": timestamp_ns,
            "stage": _native_stage(stage_name, stage_epoch),
            "bundle_sequence": bundle_sequence,
            "source": "inspire_rh56.ANGLE_ACT",
            "source_code": 2,
            "measurement_kind": "measured",
            "measurement_kind_code": 1,
            "angles": [1000, 990, 980, 970, 960, 950],
            "angle_targets": [-1, -1, -1, -1, -1, -1],
            "current_mA": [0, 0, 0, 0, 0, 0],
            "force_g": [-10, -11, -12, -13, -14, -15],
            "temperature_c": [30, 30, 31, 31, 29, 30],
            "status": [2, 2, 2, 2, 2, 2],
            "errors": [0, 0, 0, 0, 0, 0],
        },
    }


def _native_adapter(arm_age=0.25, hand_age=0.75):
    return NativeContinuousTelemetryAdapter(
        _expected_identity(), arm_max_age_s=arm_age, hand_max_age_s=hand_age
    )


def test_native_stage_fnv1a_matches_canonical_vector_and_utf8_is_not_truncated():
    assert fnv1a_stage_name_hash64("hello") == 0xA430D84680AABD0B
    assert fnv1a_stage_name_hash64("闭合") == 0x82D7CA0F60AB9F8B
    with pytest.raises(ContinuousTelemetryError, match="NUL"):
        fnv1a_stage_name_hash64("close\x00old")
    with pytest.raises(ContinuousTelemetryError, match=r"char\[32\]"):
        fnv1a_stage_name_hash64("x" * 32)


def test_native_adapter_uses_column_major_pose_and_exact_measured_sources():
    arm = native_arm_read_from_mapping(_native_arm_read())
    hand = native_hand_read_from_mapping(_native_hand_read())

    np.testing.assert_allclose(arm.sample.T_reference_EE[:3, 3], [0.51, -0.02, 0.37])
    assert arm.stage.name_hash64 == fnv1a_stage_name_hash64("moving_pregrasp")
    assert hand.sample.angles == (1000, 990, 980, 970, 960, 950)


def test_native_optional_diagnostics_may_be_absent_without_hiding_pose_or_angles():
    arm_mapping = _native_arm_read()
    for name in ("q", "dq", "control_command_success_rate"):
        arm_mapping["sample"][name] = None
    hand_mapping = _native_hand_read()
    for name in (
        "angle_targets",
        "current_mA",
        "force_g",
        "temperature_c",
        "status",
        "errors",
    ):
        hand_mapping["sample"][name] = None

    view = _native_adapter().accept(
        _native_header(),
        arm_mapping,
        hand_mapping,
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )

    assert view.show_arm_geometry is True
    assert view.show_hand_mesh is True


def test_native_stage_hash_is_recomputed_and_torn_stream_is_hidden_only():
    arm = _native_arm_read()
    arm["sample"]["stage"]["name_hash64"] ^= 1

    view = _native_adapter().accept(
        _native_header(),
        arm,
        _native_hand_read(),
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )

    assert view.arm is None
    assert "name_hash64" in view.arm_status
    assert view.hand is not None
    assert view.show_arm_geometry is False
    assert view.show_hand_mesh is False


@pytest.mark.parametrize(
    "read_factory,field,value,message",
    [
        (_native_arm_read, "source_code", 32767, "not measured Franka"),
        (_native_arm_read, "source_code", True, "must be an integer"),
        (_native_arm_read, "measurement_kind", "derived", "not measured Franka"),
        (_native_hand_read, "source", "synthetic_test", "not measured Inspire"),
        (_native_hand_read, "measurement_kind_code", True, "must be an integer"),
        (_native_hand_read, "measurement_kind_code", 2, "not measured Inspire"),
    ],
)
def test_native_nonmeasurement_codes_cannot_masquerade_as_feedback(
    read_factory, field, value, message
):
    read = read_factory()
    read["sample"][field] = value
    parser = (
        native_arm_read_from_mapping
        if read_factory is _native_arm_read
        else native_hand_read_from_mapping
    )
    with pytest.raises(ContinuousTelemetryError, match=message):
        parser(read)


def test_native_available_code_disagreement_hides_one_stream():
    arm = _native_arm_read()
    arm["available"] = False

    view = _native_adapter().accept(
        _native_header(),
        arm,
        _native_hand_read(),
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )

    assert view.arm is None
    assert "available/code" in view.arm_status
    assert view.hand is not None


def test_native_arm_hand_freshness_and_geometry_are_independent():
    view = _native_adapter(arm_age=0.25, hand_age=0.75).accept(
        _native_header(),
        _native_arm_read(timestamp_ns=9_900_000_000),
        _native_hand_read(timestamp_ns=9_000_000_000),
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )

    assert view.arm is not None
    assert view.hand is None
    assert view.show_arm_geometry is True
    assert view.show_hand_mesh is False
    assert "hand stale" in view.hand_status


def test_native_cross_stage_or_nonzero_bundle_mismatch_never_combines_mesh():
    adapter = _native_adapter()
    cross_stage = adapter.accept(
        _native_header(),
        _native_arm_read(stage_epoch=3, bundle_sequence=40),
        _native_hand_read(stage_epoch=4, bundle_sequence=41),
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )
    assert cross_stage.arm is not None and cross_stage.hand is not None
    assert cross_stage.streams_coherent is False
    assert cross_stage.show_hand_mesh is False
    assert "not combined" in cross_stage.hand_status
    feedback = native_viewer_feedback(cross_stage)
    assert feedback is not None
    assert feedback.hand_angles is None
    assert feedback.hand_sequence is None
    assert "not_ground_truth" in feedback.ee_semantics
    assert "not_measured_surface" in feedback.hand_semantics

    adapter = _native_adapter()
    bundle_mismatch = adapter.accept(
        _native_header(),
        _native_arm_read(bundle_sequence=40),
        _native_hand_read(bundle_sequence=41),
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )
    assert bundle_mismatch.show_arm_geometry is True
    assert bundle_mismatch.show_hand_mesh is False


def test_native_async_bundle_zero_can_combine_when_stage_matches():
    view = _native_adapter().accept(
        _native_header(),
        _native_arm_read(bundle_sequence=0),
        _native_hand_read(bundle_sequence=41),
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )

    assert view.streams_coherent is True
    assert view.show_hand_mesh is True
    feedback = native_viewer_feedback(view)
    assert feedback.hand_angles == (1000, 990, 980, 970, 960, 950)
    assert feedback.update_key == (RUN_UUID, 3, 100, 20)


def test_native_bundle_cannot_change_under_the_same_stream_sequence():
    adapter = _native_adapter()
    first = adapter.accept(
        _native_header(),
        _native_arm_read(sequence=100, bundle_sequence=40),
        _native_hand_read(sequence=20, bundle_sequence=41),
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )
    assert first.streams_coherent is False

    mutated = adapter.accept(
        _native_header(),
        _native_arm_read(sequence=100, bundle_sequence=41),
        _native_hand_read(sequence=20, bundle_sequence=41),
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )
    assert mutated.arm is None
    assert mutated.show_hand_mesh is False
    assert "changed under the same sequence" in mutated.arm_status


def test_native_no_data_or_contended_hides_only_corresponding_stream():
    arm = {
        "available": False,
        "code": 2,
        "sequence": 0,
        "attempts": 8,
        "sample": None,
    }
    view = _native_adapter().accept(
        _native_header(),
        arm,
        _native_hand_read(),
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )
    assert view.arm is None
    assert view.hand is not None
    assert view.arm_status == "arm unavailable: CONTENDED"
    assert native_viewer_feedback(view) is None


def test_native_stream_replay_is_local_but_header_identity_change_is_global():
    adapter = _native_adapter()
    adapter.accept(
        _native_header(),
        _native_arm_read(sequence=100),
        _native_hand_read(sequence=20),
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )
    replay = adapter.accept(
        _native_header(),
        _native_arm_read(sequence=99),
        _native_hand_read(sequence=21),
        now_unix_ns=10 * NS,
        now_monotonic_ns=10 * NS,
    )
    assert replay.arm is None
    assert replay.hand is not None
    assert "moved backwards" in replay.arm_status

    with pytest.raises(ContinuousTelemetryIdentityError, match="creation changed"):
        adapter.accept(
            _native_header(created_unix_ns=8 * NS),
            _native_arm_read(sequence=101),
            _native_hand_read(sequence=22),
            now_unix_ns=10 * NS,
            now_monotonic_ns=10 * NS,
        )
    # Immutable-header faults are session-global and remain latched even if the
    # bytes appear to revert on a later poll.
    with pytest.raises(ContinuousTelemetryIdentityError, match="creation changed"):
        adapter.accept(
            _native_header(),
            _native_arm_read(sequence=102),
            _native_hand_read(sequence=23),
            now_unix_ns=10 * NS,
            now_monotonic_ns=10 * NS,
        )


def test_native_wrong_producer_build_hash_rejects_every_stream():
    adapter = _native_adapter()
    with pytest.raises(ContinuousTelemetryIdentityError, match="UUID/artifact hashes"):
        adapter.accept(
            _native_header(producer_build_sha256="0" * 64),
            _native_arm_read(),
            _native_hand_read(),
            now_unix_ns=10 * NS,
            now_monotonic_ns=10 * NS,
        )
    with pytest.raises(ContinuousTelemetryIdentityError, match="UUID/artifact hashes"):
        adapter.accept(
            _native_header(),
            _native_arm_read(sequence=101),
            _native_hand_read(sequence=21),
            now_unix_ns=10 * NS,
            now_monotonic_ns=10 * NS,
        )
