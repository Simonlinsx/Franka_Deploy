from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from anydex_pipeline.continuous_telemetry_runtime import (
    ContinuousTelemetryRuntime,
    ContinuousTelemetryStartError,
    _verify_native_dependencies,
    load_execution_telemetry_request,
)
from anydex_pipeline.snapshot import (
    GraspCandidates,
    VisualizationSnapshot,
    save_snapshot_npz,
)
from anydex_pipeline.telemetry_session_manifest import (
    build_telemetry_session_manifest,
    canonical_json_sha256,
    write_telemetry_session_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
RUN_UUID = "12345678-1234-5678-9234-567812345678"


def _pose(xyz=(0.0, 0.0, 0.0)):
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return result


def _snapshot():
    canonical = np.stack((_pose((0.5, 0.1, 0.25)), _pose((0.6, 0.2, 0.3))))
    hands = np.stack((_pose((0.52, 0.1, 0.28)), _pose((0.62, 0.2, 0.33))))
    grasps = GraspCandidates(
        canonical_poses=canonical,
        scores=np.asarray([0.7, 0.9], dtype=np.float32),
        type_ids=np.asarray([1, 2], dtype=np.int32),
        collision_free=np.asarray([True, True]),
        collision_checked=np.asarray([True, True]),
        selected_index=-1,
        approach_axis_local=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        hand_poses=hands,
        hand_angles=np.asarray(
            [[900, 800, 700, 600, 500, 950], [850, 750, 650, 550, 450, 925]],
            dtype=np.float32,
        ),
        widths_m=np.asarray([0.05, 0.06], dtype=np.float32),
        depths_m=np.asarray([0.02, 0.03], dtype=np.float32),
        source_indices=np.asarray([10, 11], dtype=np.int64),
    )
    return VisualizationSnapshot(
        scene_points=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
        scene_colors=np.asarray([[0.2, 0.3, 0.4]], dtype=np.float32),
        object_points=np.asarray([[0.5, 0.1, 0.2]], dtype=np.float32),
        object_colors=np.asarray([[0.9, 0.1, 0.2]], dtype=np.float32),
        grasps=grasps,
        reference_frame="robot_base",
        T_reference_camera=_pose((1.2, 0.35, 0.64)),
        frame_id=12,
        timestamp_s=1234.5,
        calibration_id="eye-to-hand-b722bce10485c8a3",
        camera_serial="337322072188",
        model_name="offline",
        representation_checkpoint_sha256="a" * 64,
        decision_checkpoint_sha256s=tuple(
            "{:064x}".format(index + 1) for index in range(8)
        ),
        official_source_commit="b" * 40,
    )


@pytest.fixture
def telemetry_files(tmp_path):
    snapshot_path = save_snapshot_npz(tmp_path / "snapshot.npz", _snapshot())
    config_path = tmp_path / "control.json"
    config_path.write_bytes(CONFIG.read_bytes())
    audit_path = tmp_path / "pregrasp-audit.json"
    audit = {
        "schema_version": 2,
        "artifact_type": "fr3_rh56_pregrasp_only_collision_audit",
        "execution_scope": "open_hand_current_to_pregrasp_only",
        "motion_authorized": False,
        "test_evidence": {"passed": True},
    }
    audit["artifact_sha256"] = canonical_json_sha256(audit)
    audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")
    producer_path = tmp_path / "_anydex_franka_telemetry.so"
    producer_path.write_bytes(b"offline fake producer binary\x00")
    payload = build_telemetry_session_manifest(
        snapshot_path=snapshot_path,
        control_config_path=config_path,
        audit_artifact_path=audit_path,
        producer_build_path=producer_path,
        command="pregrasp",
        selected_index=1,
        run_uuid=RUN_UUID,
    )
    manifest_path = write_telemetry_session_manifest(
        tmp_path / "session.json", payload
    )
    return SimpleNamespace(
        snapshot=snapshot_path,
        config=config_path,
        audit=audit_path,
        producer=producer_path,
        manifest=manifest_path,
        mapping=tmp_path / "telemetry.map",
    )


def _request(files):
    return load_execution_telemetry_request(
        mapping_path=files.mapping,
        manifest_path=files.manifest,
        python_dir=files.producer.parent,
        expected_command="pregrasp",
        expected_control_config_path=files.config,
        expected_snapshot_path=files.snapshot,
        expected_pregrasp_only_audit_path=files.audit,
        expected_selected_index=1,
    )


class _FakeProducer:
    def __init__(self, log):
        self.log = log
        self.closed = False

    def set_stage(self, name, epoch, bundle):
        self.log.append(("producer.stage", name, epoch, bundle))

    def synchronize_and_publish_arm(self, state, unix_ns, monotonic_ns):
        self.log.append(("producer.arm", state, unix_ns, monotonic_ns))
        return 1

    def publish_hand(self, *args):
        self.log.append(("producer.hand", args))
        return 7

    def close(self):
        self.log.append(("producer.close",))
        self.closed = True


class _FakeArm:
    def __init__(self, log, fail_remove_once=False):
        self.log = log
        self.tap = None
        self.fail_remove_once = fail_remove_once

    def install_native_telemetry_tap(self, tap):
        assert self.tap is None
        self.tap = tap
        self.log.append(("arm.install", tap))

    def remove_native_telemetry_tap(self, tap):
        assert tap is self.tap
        self.log.append(("arm.remove", tap))
        if self.fail_remove_once:
            self.fail_remove_once = False
            raise RuntimeError("fake arm detach failed once")
        self.tap = None


class _FakeHand:
    def __init__(self, log, fail_install=False):
        self.log = log
        self.callback = None
        self.identity = None
        self.fail_install = fail_install

    def install_validated_feedback_observer(self, callback, *, identity):
        self.log.append(("hand.install", identity))
        if self.fail_install:
            raise RuntimeError("fake hand observer install failed")
        self.callback = callback
        self.identity = identity

    def remove_validated_feedback_observer(self, *, identity):
        assert identity == self.identity
        self.log.append(("hand.remove", identity))
        self.callback = None
        self.identity = None


def _module(files, log):
    producer = _FakeProducer(log)

    class ProducerType:
        @staticmethod
        def create(path, provenance, arm_token, hand_token, decimation):
            log.append(
                (
                    "producer.create",
                    path,
                    provenance,
                    arm_token,
                    hand_token,
                    decimation,
                )
            )
            return producer

    return SimpleNamespace(
        __file__=str(files.producer),
        NativeTelemetryProducer=ProducerType,
        producer=producer,
    )


def test_offline_request_replays_exact_execution_without_native_import(
    telemetry_files, monkeypatch
):
    import anydex_pipeline.continuous_telemetry_runtime as runtime_module

    monkeypatch.setattr(
        runtime_module.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("offline request imported native producer")
        ),
    )
    request = _request(telemetry_files)

    assert request.mapping_path == telemetry_files.mapping.resolve()
    assert request.producer_build_path == telemetry_files.producer.resolve()
    assert not request.mapping_path.exists()

    with pytest.raises(ValueError, match="selected_index"):
        load_execution_telemetry_request(
            mapping_path=telemetry_files.mapping,
            manifest_path=telemetry_files.manifest,
            python_dir=telemetry_files.producer.parent,
            expected_command="pregrasp",
            expected_control_config_path=telemetry_files.config,
            expected_snapshot_path=telemetry_files.snapshot,
            expected_pregrasp_only_audit_path=telemetry_files.audit,
            expected_selected_index=0,
        )


def test_native_runtime_installs_exact_owners_and_adapts_validated_hand_feedback(
    telemetry_files
):
    request = _request(telemetry_files)
    log = []
    module = _module(telemetry_files, log)
    arm = _FakeArm(log)
    hand = _FakeHand(log)
    initial_state = object()
    tokens = iter((11, 22))

    runtime = ContinuousTelemetryRuntime.start(
        request,
        arm=arm,
        hand=hand,
        initial_validated_arm_state=initial_state,
        initial_arm_timestamp_unix_ns=900,
        initial_arm_timestamp_monotonic_ns=1900,
        robot_id="fr3:fake;rh56:1",
        native_module_loader=lambda: module,
        dependency_checker=lambda _module: None,
        token_source=lambda _bits: next(tokens),
        unix_time_ns=lambda: 1000,
        monotonic_time_ns=lambda: 2000,
    )

    create = next(item for item in log if item[0] == "producer.create")
    assert create[1] == str(telemetry_files.mapping.resolve())
    assert set(create[2]) == {
        "run_uuid",
        "execution_contract_sha256",
        "source_snapshot_sha256",
        "control_config_sha256",
        "calibration_sha256",
        "producer_build_sha256",
        "created_monotonic_ns",
        "created_unix_ns",
        "producer_name",
        "robot_id",
    }
    assert create[3:] == (11, 22, 20)
    assert ("producer.stage", "disarmed", 1, 0) in log
    assert ("producer.arm", initial_state, 900, 1900) in log
    assert arm.tap is module.producer
    assert hand.callback is not None

    runtime.observe_transition(SimpleNamespace(value="opening_hand"))
    assert ("producer.stage", "opening_hand", 2, 0) in log
    event = SimpleNamespace(
        observer_identity=runtime.observer_identity,
        phase="open_wait",
        timestamp_unix_ns=3000,
        timestamp_monotonic_ns=4000,
        angles=(1000, 999, 998, 997, 996, 995),
        angle_targets=(1000, 1000, 1000, 1000, 1000, -1),
        positions=(1, 2, 3, 4, 5, 6),
        forces=None,
        currents=(10, 11, 12, 13, 14, 15),
        errors=(0, 0, 0, 0, 0, 0),
        statuses=(2, 2, 2, 2, 2, 2),
        temperatures=(30, 31, 32, 33, 34, 35),
    )
    assert hand.callback(event) == 7
    hand_args = next(item[1] for item in log if item[0] == "producer.hand")
    assert hand_args == (
        event.angles,
        event.angle_targets,
        event.currents,
        None,
        event.temperatures,
        event.statuses,
        event.errors,
        3000,
        4000,
    )

    runtime.close()
    assert runtime.closed
    assert arm.tap is None
    assert hand.callback is None
    assert log.index(next(item for item in log if item[0] == "hand.remove")) < log.index(
        next(item for item in log if item[0] == "producer.close")
    )


def test_start_rolls_back_arm_tap_and_closes_producer_if_hand_install_fails(
    telemetry_files
):
    request = _request(telemetry_files)
    log = []
    module = _module(telemetry_files, log)
    arm = _FakeArm(log)
    hand = _FakeHand(log, fail_install=True)
    tokens = iter((101, 202))

    with pytest.raises(RuntimeError, match="observer install failed"):
        ContinuousTelemetryRuntime.start(
            request,
            arm=arm,
            hand=hand,
            initial_validated_arm_state=object(),
            initial_arm_timestamp_unix_ns=900,
            initial_arm_timestamp_monotonic_ns=1900,
            robot_id="fake",
            native_module_loader=lambda: module,
            dependency_checker=lambda _module: None,
            token_source=lambda _bits: next(tokens),
            unix_time_ns=lambda: 1000,
            monotonic_time_ns=lambda: 2000,
        )

    assert arm.tap is None
    assert module.producer.closed
    assert any(item[0] == "arm.remove" for item in log)


def test_failed_detach_never_closes_driver_referenced_producer_and_is_retryable(
    telemetry_files
):
    request = _request(telemetry_files)
    log = []
    module = _module(telemetry_files, log)
    arm = _FakeArm(log, fail_remove_once=True)
    hand = _FakeHand(log)
    tokens = iter((303, 404))
    runtime = ContinuousTelemetryRuntime.start(
        request,
        arm=arm,
        hand=hand,
        initial_validated_arm_state=object(),
        initial_arm_timestamp_unix_ns=900,
        initial_arm_timestamp_monotonic_ns=1900,
        robot_id="fake",
        native_module_loader=lambda: module,
        dependency_checker=lambda _module: None,
        token_source=lambda _bits: next(tokens),
        unix_time_ns=lambda: 1000,
        monotonic_time_ns=lambda: 2000,
    )

    with pytest.raises(RuntimeError, match="Franka tap detach"):
        runtime.close()

    assert arm.tap is module.producer
    assert not module.producer.closed
    assert not runtime.closed

    runtime.close()
    assert arm.tap is None
    assert module.producer.closed
    assert runtime.closed


def test_incomplete_start_rollback_exposes_live_runtime_for_cleanup_retry(
    telemetry_files
):
    request = _request(telemetry_files)
    log = []
    module = _module(telemetry_files, log)
    arm = _FakeArm(log, fail_remove_once=True)
    hand = _FakeHand(log, fail_install=True)
    tokens = iter((505, 606))

    with pytest.raises(ContinuousTelemetryStartError) as raised:
        ContinuousTelemetryRuntime.start(
            request,
            arm=arm,
            hand=hand,
            initial_validated_arm_state=object(),
            initial_arm_timestamp_unix_ns=900,
            initial_arm_timestamp_monotonic_ns=1900,
            robot_id="fake",
            native_module_loader=lambda: module,
            dependency_checker=lambda _module: None,
            token_source=lambda _bits: next(tokens),
            unix_time_ns=lambda: 1000,
            monotonic_time_ns=lambda: 2000,
        )

    runtime = raised.value.runtime
    assert arm.tap is module.producer
    assert not module.producer.closed
    assert not runtime.closed
    runtime.close()
    assert arm.tap is None
    assert module.producer.closed


def test_runtime_import_is_hardware_and_native_free():
    script = r'''
import builtins
original = builtins.__import__
forbidden = {"pylibfranka", "franka", "serial", "pyrealsense2", "_anydex_franka_telemetry"}
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise RuntimeError("forbidden import: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import anydex_pipeline.continuous_telemetry_runtime
print("offline-runtime-import-ok")
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "offline-runtime-import-ok"


def test_dependency_audit_requires_exact_wheel_and_single_libfranka_mapping(
    tmp_path, monkeypatch
):
    import anydex_pipeline.continuous_telemetry_runtime as runtime_module
    from anydex_pipeline.telemetry_session_manifest import sha256_file

    wheel = tmp_path / "_pylibfranka.so"
    wheel.write_bytes(b"exact fake wheel")
    library = tmp_path / "libfranka-abcdef.so.0.21.2"
    library.write_bytes(b"exact fake libfranka")
    module = SimpleNamespace(
        BUILD_DEPENDENCIES={
            "schema_version": 1,
            "pylibfranka": {"path": str(wheel), "sha256": sha256_file(wheel)},
            "libfranka": {
                "path": str(library),
                "sha256": sha256_file(library),
            },
        }
    )
    monkeypatch.setattr(
        runtime_module.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(__file__=str(wheel))
            if name == "pylibfranka._pylibfranka"
            else (_ for _ in ()).throw(AssertionError(name))
        ),
    )
    maps = tmp_path / "maps"
    maps.write_text(
        "1000-2000 r--p 0 00:00 0 {}\n"
        "2000-3000 r-xp 0 00:00 0 {}\n".format(library, library),
        encoding="utf-8",
    )

    _verify_native_dependencies(module, proc_maps_path=maps)

    maps.write_text(
        maps.read_text(encoding="utf-8")
        + "3000-4000 r-xp 0 00:00 0 /tmp/libfranka-other.so\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        _verify_native_dependencies(module, proc_maps_path=maps)
