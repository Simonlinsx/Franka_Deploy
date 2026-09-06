"""End-to-end guard regressions for the supervised V94 runtime.

All device boundaries are fakes.  In particular, these tests never import a
serial driver, RealSense provider, or pylibfranka backend.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from sim2real.runtime.bounded_c2_orchestrator import BoundedC2StopProof
from sim2real.runtime.bounded_c2_runtime import BoundedV94C2Runtime
from sim2real.closed_loop_core import (
    ClosedLoopCommand,
    ClosedLoopSafetyGate,
    ExecutedActionLedger,
    MotionAuthorization,
)
from robot_control.rh56.actuator import (
    RH56SafetyFeedback,
    RH56TransactionalActuator,
    _issue_supervised_rh56_preflight,
)
from robot_control.rh56.watchdog import RH56OwnedSession, RH56WatchdogOwner


@pytest.mark.parametrize("steps", (True, 720.5, "720", 721))
def test_runtime_rejects_non_integer_or_out_of_range_steps_before_filesystem(
    steps,
) -> None:
    import sim2real.runtime.supervised_v94_runtime as module

    request = SimpleNamespace(
        run_id="strict-step-boundary",
        steps=steps,
        bundle=Path("/definitely/missing/bundle.zip"),
        profile=Path("/definitely/missing/profile.json"),
        pcd_config=Path("/definitely/missing/pcd.yaml"),
    )
    with pytest.raises(module.SupervisedV94RuntimeError, match="1..720"):
        module._validate_request(request)


def test_online_visual_replay_completion_uses_committed_template_frames() -> None:
    import sim2real.runtime.supervised_v94_runtime as module

    result = {
        "completed_policy_steps": 88,
        "last_dual_ack_sequence": 88,
        "completed_replay_frames": 98,
        "replay_source_complete": True,
        "stopped_early": False,
    }
    assert module._supervised_runtime_completed_requested_work(
        result,
        requested_steps=98,
        replay_source_completion_is_authoritative=True,
    )
    assert not module._supervised_runtime_completed_requested_work(
        result,
        requested_steps=98,
        replay_source_completion_is_authoritative=False,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("completed_replay_frames", 97),
        ("replay_source_complete", False),
        ("last_dual_ack_sequence", 87),
        ("stopped_early", True),
    ),
)
def test_online_visual_replay_completion_rejects_incomplete_proof(
    field, value
) -> None:
    import sim2real.runtime.supervised_v94_runtime as module

    result = {
        "completed_policy_steps": 88,
        "last_dual_ack_sequence": 88,
        "completed_replay_frames": 98,
        "replay_source_complete": True,
        "stopped_early": False,
    }
    result[field] = value
    assert not module._supervised_runtime_completed_requested_work(
        result,
        requested_steps=98,
        replay_source_completion_is_authoritative=True,
    )


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self._lock = threading.Lock()
        self._value = value

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._value += seconds


class _Transport:
    hardware_backed = False
    baud_rate = 115200
    transport_name = "supervised-owner-fake"

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.targets = (-1,) * 6
        self.writes: list[tuple[int, ...]] = []
        self.feedback_reads = 0

    def write_angle_set(self, values, *, deadline_monotonic_s):
        assert self.clock() <= deadline_monotonic_s
        self.targets = tuple(int(value) for value in values)
        self.writes.append(self.targets)
        self.clock.advance(0.0002)

    def read_angle_set(self, *, deadline_monotonic_s):
        assert self.clock() <= deadline_monotonic_s
        self.clock.advance(0.0002)
        return self.targets

    def read_safety_feedback(self, *, deadline_monotonic_s):
        assert self.clock() <= deadline_monotonic_s
        self.clock.advance(0.0002)
        self.feedback_reads += 1
        return RH56SafetyFeedback(
            captured_monotonic_s=self.clock(),
            positions=(500,) * 6,
            angles=(1000,) * 6,
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=(2,) * 6,
            temperatures_c=(25,) * 6,
        )


class _SupervisedSessionFactory:
    def __init__(
        self,
        *,
        clock: _Clock,
        transport: _Transport,
        authorization: MotionAuthorization,
        gate: ClosedLoopSafetyGate,
    ) -> None:
        self.clock = clock
        self.transport = transport
        self.authorization = authorization
        self.gate = gate
        self.actuator: RH56TransactionalActuator | None = None

    def __call__(self) -> RH56OwnedSession:
        actuator = RH56TransactionalActuator(
            self.transport,
            command_watchdog_timeout_s=0.05,
            maximum_feedback_age_s=0.025,
            stop_timeout_s=0.25,
            stop_verify_samples=2,
            stop_verify_interval_s=0.0,
            monotonic=self.clock,
            sleep=self.clock.advance,
        )
        actuator.claim_for_current_thread()
        actuator.arm_supervised(
            preflight=_issue_supervised_rh56_preflight(
                run_id="supervised-execution",
                confirmed_permit_sha256="a" * 64,
                commissioning_profile_sha256="b" * 64,
                watchdog_timeout_s=0.05,
            ),
            authorization=self.authorization,
            safety_gate=self.gate,
            run_id="supervised-execution",
            now_monotonic_s=self.clock(),
        )
        self.actuator = actuator
        return RH56OwnedSession(actuator=actuator, close=lambda: None)


def _supervised_owner():
    clock = _Clock()
    authorization = MotionAuthorization(
        run_id="supervised-execution",
        authorization_id="supervised-authorization",
        issued_monotonic_s=9.0,
        expires_monotonic_s=100.0,
    )
    gate = ClosedLoopSafetyGate()
    gate.update_interlocks(deadman_asserted=True, estop_healthy=True)
    gate.arm(
        authorization,
        run_id=authorization.run_id,
        now_monotonic_s=clock(),
        franka_rest_verified=True,
        rh56_disabled_verified=True,
    )
    transport = _Transport(clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    factory = _SupervisedSessionFactory(
        clock=clock,
        transport=transport,
        authorization=authorization,
        gate=gate,
    )
    fault_seen = threading.Event()
    owner = RH56WatchdogOwner(
        factory,
        ledger,
        fault_callback=lambda _fault: fault_seen.set(),
        monotonic=clock,
        realtime=lambda: 1_750_000_000.0 + clock(),
        startup_timeout_s=0.5,
        response_timeout_s=0.5,
        join_timeout_s=0.5,
    )
    return owner, factory, transport, ledger, clock, fault_seen


def _first_command(ledger: ExecutedActionLedger, clock: _Clock) -> ClosedLoopCommand:
    action = np.full(13, 0.25, dtype=np.float32)
    return ClosedLoopCommand(
        sequence=1,
        produced_monotonic_s=clock(),
        observation_realtime_s=1_750_000_000.0 + clock(),
        previous_executed_action13_used=ledger.previous_executed_action13(),
        raw_policy_action13=action,
        executed_policy_action13=action,
        franka_target_q_rad=np.zeros(7),
        rh56_angle_set_register_order=np.full(6, 960, dtype=np.int32),
    )


def test_rh56_watchdog_starts_only_after_first_numeric_target() -> None:
    owner, factory, transport, ledger, clock, fault_seen = _supervised_owner()
    owner.start()
    assert factory.actuator is not None
    assert factory.actuator.awaiting_first_numeric_command is True
    assert factory.actuator.policy_watchdog_deadline_monotonic_s is None
    bootstrap_reads = transport.feedback_reads

    # Simulate inference taking far longer than 50 ms.  The owner continues
    # bounded safety reads, but ANGLE_SET remains untouched and no policy
    # watchdog exists until a numeric action is actually accepted.
    clock.advance(0.200)
    time.sleep(0.040)
    assert fault_seen.is_set() is False
    assert transport.writes == []
    assert transport.feedback_reads > bootstrap_reads

    command = _first_command(ledger, clock)
    ledger.stage(command, now_monotonic_s=clock())
    assert ledger.acknowledge("franka", sequence=1, now_monotonic_s=clock()) is False
    receipt = owner.execute(command)
    assert receipt.sequence == 1
    assert ledger.last_committed_sequence == 1
    assert factory.actuator.awaiting_first_numeric_command is False
    assert factory.actuator.policy_watchdog_deadline_monotonic_s is not None
    assert transport.writes == [(960,) * 6]

    clock.advance(0.051)
    assert fault_seen.wait(0.25)
    stopped = owner.stop_and_close()
    assert stopped.rh56_disabled_verified is True
    assert transport.writes[-2:] == [(-1,) * 6, (-1,) * 6]


class _StopSession:
    def request_clean_stop(self) -> None:
        return None


class _OwnerWithCloseFailure:
    def request_stop(self) -> None:
        return None

    def stop_and_close(self, *, timeout_s):
        del timeout_s
        return SimpleNamespace(
            worker_stopped=True,
            rh56_disabled_verified=True,
            disable_attempted=True,
            close_attempted=True,
            serial_closed=False,
            stop_report=None,
            fault=SimpleNamespace(stop_error="SPEED_SET restore failed"),
        )


def test_owner_restore_or_close_failure_is_recorded_as_runtime_stop_error() -> None:
    runtime = object.__new__(BoundedV94C2Runtime)
    runtime._state_lock = threading.Lock()
    runtime._stop_proof = None
    runtime._stop_errors = ()
    runtime._stop_requested = threading.Event()
    runtime.franka_session = _StopSession()
    runtime._rh56_watchdog_owner = _OwnerWithCloseFailure()
    runtime._rh56_actuator = None
    runtime._franka_thread = None
    runtime._rh56_transport = None
    runtime._policy_tick_source_factory = None
    runtime._franka_stop_join_timeout_s = 0.1

    proof = runtime.stop_and_verify()

    assert proof.rh56_disabled_verified is False
    assert runtime.stop_errors
    assert any(
        "close" in error.lower() or "restore" in error.lower()
        for error in runtime.stop_errors
    )


class _FinishedThread:
    def join(self, *, timeout: float) -> None:
        assert timeout > 0.0

    def is_alive(self) -> bool:
        return False


class _FaultedButPhysicallyStoppedSession:
    def __init__(self) -> None:
        self.last_telemetry = SimpleNamespace(
            state="fault_latched",
            stop_requested=True,
            stop_verified=True,
        )

    def request_clean_stop(self) -> None:
        return None


def test_physical_franka_stop_proof_is_independent_of_run_fault_state() -> None:
    runtime = object.__new__(BoundedV94C2Runtime)
    runtime._state_lock = threading.Lock()
    runtime._stop_proof = None
    runtime._stop_errors = ()
    runtime._stop_requested = threading.Event()
    runtime.franka_session = _FaultedButPhysicallyStoppedSession()
    runtime._rh56_watchdog_owner = None
    runtime._rh56_actuator = None
    runtime._franka_thread = _FinishedThread()
    runtime._rh56_transport = None
    runtime._policy_tick_source_factory = None
    runtime._franka_stop_join_timeout_s = 0.1
    runtime._supervised_non_c2 = True

    proof = runtime.stop_and_verify()

    assert proof.franka_stop_verified is True
    assert proof.rh56_disabled_verified is True
    assert runtime.stop_errors == ()


@dataclass
class _FakeCombinedRuntime:
    run_error: BaseException | None = None
    stop_errors: tuple[str, ...] = ()
    request_stop_calls: int = 0
    stop_and_verify_calls: int = 0
    franka_telemetry: object = None
    failure_pending_command: object = None
    committed_commands_snapshot: tuple[object, ...] = ()
    rh56_tracking_verdict: str = "not_exercised"
    runtime_diagnostics_snapshot: object = None

    def run(self, **_kwargs):
        if self.run_error is not None:
            raise self.run_error
        return {
            "completed_policy_steps": 1,
            "last_dual_ack_sequence": 1,
            "requested_policy_steps": 1,
            "stopped_early": False,
            "rh56_physical_tracking": {
                "verdict": self.rh56_tracking_verdict,
                "challenged_axes": (
                    []
                    if self.rh56_tracking_verdict == "not_exercised"
                    else [True, False, False, False, False, False]
                ),
            },
        }

    def request_stop(self, _reason: str) -> None:
        self.request_stop_calls += 1

    def stop_and_verify(self) -> BoundedC2StopProof:
        self.stop_and_verify_calls += 1
        return BoundedC2StopProof(
            franka_stop_verified=True,
            rh56_disabled_verified=True,
        )


def _patch_supervised_world(monkeypatch, tmp_path: Path, runtime: _FakeCombinedRuntime):
    import sim2real.runtime.supervised_v94_runtime as module

    bundle = tmp_path / "deploy.zip"
    bundle.write_bytes(b"fake bundle")
    (tmp_path / "profile.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pcd.yaml").write_text("fake: true\n", encoding="utf-8")
    output = tmp_path / "audit.json"
    profile = {
        "profile_id": "fr3_rh56_v94_reset_locked_v1",
        "policy_contract": "v94_inspire_semantic_13d",
        "franka": {"ip": "fake-franka"},
        "inspire": {
            "port": "fake-rh56",
            "baud": 115200,
            "hand_id": 1,
            "force_limit_g": 80,
            "open_targets": [1000] * 6,
            "thumb_rotate_validated_realtime_range": [416, 1000],
            "six_axis_coupled_closure_commissioned": True,
            "commissioned_air_closure_targets": [
                [61, 17, 524, 758, 422, 416]
            ],
        },
    }
    contract = SimpleNamespace(q_home_rad=np.zeros(7))
    envelope = SimpleNamespace(
        profile_sha256="c" * 64,
        binding_sha256="f" * 64,
    )
    monkeypatch.setattr(
        module,
        "_validate_request",
        lambda _request: (output, profile, contract, envelope),
    )
    fixed_roi = module.VerifiedObjectROI(
        xywh=(414, 228, 64, 79),
        evidence_path="fake-read-only-evidence.npz",
        evidence_sha256="d" * 64,
        camera_serial="fake-camera",
        calibration_id="fake-calibration",
        checkpoint_sha256="e" * 64,
        valid_publications=1771,
        invalid_publications=0,
        acquisition_elapsed_s=61.8,
    )
    monkeypatch.setattr(
        module,
        "_load_verified_object_roi",
        lambda **_kwargs: fixed_roi,
    )
    native_build = module.NativeServoBuildIdentity(
        manifest_path=tmp_path / "manifest.json",
        manifest_sha256="a" * 64,
        binary_path=tmp_path / "v94_franka_servo",
        binary_sha256="b" * 64,
        producer_build_sha256="c" * 64,
        libfranka_path=tmp_path / "libfranka.so",
        libfranka_sha256="d" * 64,
        libfranka_source_commit="e" * 40,
        protocol_version=4,
        state_decimation=16,
        safety_limits_schema=3,
        compiled_profile_sha256=envelope.profile_sha256,
        compiled_envelope_sha256=envelope.binding_sha256,
    )
    monkeypatch.setattr(
        module,
        "_load_native_servo_build_identity",
        lambda *_args, **_kwargs: native_build,
    )
    from dexgrasp.apps.reset_installed_rh56_open import RH56ResetOpenProof
    from sim2real.deployment.execution_reset import (
        FrankaV94HomeResetProof,
        V94ExecutionResetProof,
    )

    reset_proof = V94ExecutionResetProof(
        rh56=RH56ResetOpenProof(
            q6_waypoints=(),
            final_angles=(1000,) * 6,
            final_targets=(-1,) * 6,
            final_currents_ma=(0,) * 6,
            final_statuses=(2,) * 6,
            final_speeds=(1000,) * 6,
            final_forces_g=(500,) * 6,
        ),
        franka=FrankaV94HomeResetProof(
            target_q_rad=(0.0,) * 7,
            target_sha256_f64_le=hashlib.sha256(
                np.ascontiguousarray(np.zeros(7, dtype="<f8")).tobytes()
            ).hexdigest(),
            initial_linf_delta_rad=0.0,
            maximum_start_delta_rad=1.21,
            final_linf_error_rad=0.0,
        ),
        dwell_requested_s=0.5,
        dwell_elapsed_s=0.5,
    )
    monkeypatch.setattr(
        module,
        "run_v94_execution_reset",
        lambda **_kwargs: reset_proof,
    )
    monkeypatch.setattr(
        module,
        "_fresh_franka_preflight",
        lambda **_kwargs: ((0.0,) * 7, (0.0,) * 7, 0.0, True),
    )
    monkeypatch.setattr(
        module,
        "_fresh_rh56_preflight",
        lambda **_kwargs: ((-1,) * 6, (1000,) * 6, 0),
    )
    monkeypatch.setattr(module, "_issue_supervised_franka_preflight_token", lambda **_kwargs: object())
    monkeypatch.setattr(module, "_issue_supervised_rh56_preflight", lambda **_kwargs: object())

    def seal(**kwargs):
        return SimpleNamespace(
            run_id=kwargs["run_id"],
            classification="experimental_operator_supervised_non_c2",
            hard_deadline_monotonic_s=kwargs["hard_deadline_monotonic_s"],
            safety_gate=kwargs["safety_gate"],
        )

    monkeypatch.setattr(module, "_seal_supervised_v94_admission", seal)

    class FakeBundle:
        def __init__(self, _path):
            self.manifest = {
                "primary_checkpoint": {
                    "sha256": hashlib.sha256(b"checkpoint").hexdigest()
                }
            }

        def checkpoint_bytes(self):
            return b"checkpoint"

    class RuntimeFactory:
        def __init__(self, **kwargs):
            assert kwargs["maximum_consecutive_no_stage_hold_s"] == 0.4

        def __call__(self, _admission):
            return runtime

    monkeypatch.setattr(module, "DeployBundle", FakeBundle)
    monkeypatch.setattr(module, "load_checkpoint_safely", lambda _bytes: object())
    monkeypatch.setattr(module, "RollingStudentPolicy", lambda _checkpoint: object())
    def fixed_roi_factory(**kwargs):
        assert kwargs["roi_xywh"] == fixed_roi.xywh
        assert kwargs["roi_xywh"] is not None
        assert kwargs["disable_online_sam2"] is False
        assert kwargs["required_runtime_frame_timeout_ms"] == 100
        return object()

    monkeypatch.setattr(module, "LiveD435ProviderFactory", fixed_roi_factory)
    def camera_owner(**kwargs):
        assert kwargs["maximum_publication_stall_s"] == 0.4
        return object()

    monkeypatch.setattr(module, "D435ObjectCameraOwner", camera_owner)
    def policy_source_factory(**kwargs):
        assert kwargs["requested_object_mask_mode"] == "guarded"
        assert kwargs["effective_object_mask_mode"] == "guarded_v2"
        assert kwargs["camera_owner_preopened"] is False
        return object()

    monkeypatch.setattr(
        module,
        "ProductionV94PolicyTickSourceFactory",
        policy_source_factory,
    )
    monkeypatch.setattr(
        module,
        "_make_native_franka_session_factory",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(module, "_make_rh56_owner_factory", lambda **_kwargs: object())
    monkeypatch.setattr(module, "SupervisedV94RuntimeFactory", RuntimeFactory)
    request = SimpleNamespace(
        run_id="supervised-fake",
        steps=1,
        bundle=bundle,
        profile=tmp_path / "profile.json",
        pcd_config=tmp_path / "pcd.yaml",
    )
    return module, request, output


def test_supervised_runtime_consumes_preflight_camera_without_reopening(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _FakeCombinedRuntime()
    module, request, _output = _patch_supervised_world(
        monkeypatch,
        tmp_path,
        runtime,
    )
    preopened_owner = object()
    consumed = []
    factory_kwargs = []

    class Handoff:
        def consume(self, **kwargs):
            consumed.append(kwargs)
            return preopened_owner

    monkeypatch.setattr(module, "PrewarmedD435CameraHandoff", Handoff)
    monkeypatch.setattr(
        module,
        "LiveD435ProviderFactory",
        lambda **_kwargs: pytest.fail("prewarmed path reopened D435 provider"),
    )
    monkeypatch.setattr(
        module,
        "D435ObjectCameraOwner",
        lambda **_kwargs: pytest.fail("prewarmed path created a second owner"),
    )

    def policy_source_factory(**kwargs):
        factory_kwargs.append(kwargs)
        return object()

    monkeypatch.setattr(
        module,
        "ProductionV94PolicyTickSourceFactory",
        policy_source_factory,
    )

    result = module.run_supervised_v94(
        request,
        prewarmed_camera_handoff=Handoff(),
    )

    assert result["completed_policy_steps"] == 1
    assert len(consumed) == 1
    assert consumed[0]["roi_xywh"] == (414, 228, 64, 79)
    assert consumed[0]["object_mask_mode"] == "guarded"
    assert len(factory_kwargs) == 1
    assert factory_kwargs[0]["camera_owner"] is preopened_owner
    assert factory_kwargs[0]["camera_owner_preopened"] is True


def test_changed_bundle_between_reset_and_inference_is_rejected_before_preflight(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _FakeCombinedRuntime()
    module, request, _output = _patch_supervised_world(
        monkeypatch, tmp_path, runtime
    )
    before_reset = module.prepare_supervised_v94(request)
    after_reset = replace(before_reset, bundle_sha256="0" * 64)
    reset_proof = module.run_v94_execution_reset()
    monkeypatch.setattr(
        module,
        "prepare_supervised_v94",
        lambda _request: after_reset,
    )
    monkeypatch.setattr(
        module,
        "_fresh_franka_preflight",
        lambda **_kwargs: pytest.fail(
            "changed deployment inputs reached fresh hardware preflight"
        ),
    )

    with pytest.raises(
        module.SupervisedV94RuntimeError,
        match="changed during automatic reset",
    ):
        module.run_supervised_v94(
            request,
            preparation=before_reset,
            execution_reset=reset_proof,
        )


def test_changed_checkpoint_between_reset_and_inference_is_rejected_before_preflight(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _FakeCombinedRuntime()
    module, request, _output = _patch_supervised_world(
        monkeypatch, tmp_path, runtime
    )
    before_reset = module.prepare_supervised_v94(request)
    after_reset = replace(
        before_reset,
        checkpoint_sha256="0" * 64,
        pinned_checkpoint_bytes=b"changed-checkpoint",
    )
    reset_proof = module.run_v94_execution_reset()
    monkeypatch.setattr(
        module,
        "prepare_supervised_v94",
        lambda _request: after_reset,
    )
    monkeypatch.setattr(
        module,
        "_fresh_franka_preflight",
        lambda **_kwargs: pytest.fail(
            "changed checkpoint reached fresh hardware preflight"
        ),
    )

    with pytest.raises(
        module.SupervisedV94RuntimeError,
        match="changed during automatic reset",
    ):
        module.run_supervised_v94(
            request,
            preparation=before_reset,
            execution_reset=reset_proof,
        )


@pytest.mark.parametrize("run_error", (KeyboardInterrupt(), RuntimeError("boom")))
def test_interrupt_and_exception_both_request_and_verify_dual_stop(
    monkeypatch, tmp_path: Path, run_error: BaseException
) -> None:
    runtime = _FakeCombinedRuntime(run_error=run_error)
    module, request, output = _patch_supervised_world(monkeypatch, tmp_path, runtime)

    expected = KeyboardInterrupt if isinstance(run_error, KeyboardInterrupt) else module.SupervisedV94RuntimeError
    with pytest.raises(expected):
        module.run_supervised_v94(request)

    assert runtime.request_stop_calls == 1
    assert runtime.stop_and_verify_calls == 1
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["result"] == "FAIL"
    perception_contract = audit["permit"]["object_perception_contract"]
    assert perception_contract["requested_object_mask_mode"] == "guarded"
    assert perception_contract["effective_object_mask_mode"] == "guarded_v2"
    assert perception_contract["stale_palm"] == (
        "recoverable_fail_closed_before_policy_history_mapper_stage"
    )
    assert perception_contract["stale_palm_legacy_compatibility_switch"] == (
        "explicit_--object-mask-mode=legacy_only"
    )


def test_native_franka_fault_replaces_raced_dual_ack_timeout(
    monkeypatch, tmp_path: Path
) -> None:
    @dataclass(frozen=True)
    class _Telemetry:
        fault_reason: str

    runtime = _FakeCombinedRuntime(
        run_error=RuntimeError(
            "command commit deadline expired; "
            "missing acknowledgements=franka,rh56"
        ),
        franka_telemetry=_Telemetry(
            fault_reason=(
                "active readOnce failed: libfranka: Move command aborted: "
                'motion aborted by reflex! ["cartesian_reflex"]'
            )
        ),
    )
    module, request, output = _patch_supervised_world(
        monkeypatch, tmp_path, runtime
    )

    with pytest.raises(
        module.SupervisedV94RuntimeError,
        match="Franka owner fault.*cartesian_reflex",
    ):
        module.run_supervised_v94(request)

    audit = json.loads(output.read_text(encoding="utf-8"))
    assert "Franka owner fault" in audit["failure"]
    assert "cartesian_reflex" in audit["failure"]
    assert "missing acknowledgements" not in audit["failure"]


def test_native_franka_fault_replaces_raced_stale_state_symptom(
    monkeypatch, tmp_path: Path
) -> None:
    @dataclass(frozen=True)
    class _Telemetry:
        fault_reason: str

    runtime = _FakeCombinedRuntime(
        run_error=RuntimeError(
            "Franka sample is stale: age=0.050170s, limit=0.050000s"
        ),
        franka_telemetry=_Telemetry(
            fault_reason=(
                "active readOnce failed: libfranka: Move command aborted: "
                'motion aborted by reflex! ["cartesian_reflex"]'
            )
        ),
    )
    module, request, output = _patch_supervised_world(
        monkeypatch, tmp_path, runtime
    )

    with pytest.raises(
        module.SupervisedV94RuntimeError,
        match="Franka owner fault.*cartesian_reflex",
    ):
        module.run_supervised_v94(request)

    audit = json.loads(output.read_text(encoding="utf-8"))
    assert "Franka owner fault" in audit["failure"]
    assert "cartesian_reflex" in audit["failure"]
    assert "sample is stale" not in audit["failure"]


def test_nonempty_cleanup_errors_can_never_produce_pass(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _FakeCombinedRuntime(stop_errors=("RH56 restore failed",))
    module, request, output = _patch_supervised_world(monkeypatch, tmp_path, runtime)

    with pytest.raises(module.SupervisedV94RuntimeError, match="verified stop/restore failed"):
        module.run_supervised_v94(request)

    assert runtime.request_stop_calls == 1
    assert runtime.stop_and_verify_calls == 1
    assert '"result": "FAIL"' in output.read_text(encoding="utf-8")


def test_challenged_but_unverified_rh56_tracking_can_never_produce_pass(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _FakeCombinedRuntime(
        rh56_tracking_verdict="challenged_unverified"
    )
    module, request, output = _patch_supervised_world(
        monkeypatch, tmp_path, runtime
    )

    with pytest.raises(
        module.SupervisedV94RuntimeError,
        match="physical tracking",
    ):
        module.run_supervised_v94(request)

    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["result"] == "FAIL"
    assert audit["runtime"]["rh56_physical_tracking"]["verdict"] == (
        "challenged_unverified"
    )
    assert runtime.request_stop_calls == 1
    assert runtime.stop_and_verify_calls == 1


def test_failure_audit_retains_pending_command_and_policy_actions(
    monkeypatch, tmp_path: Path
) -> None:
    previous = np.linspace(-0.3, 0.3, 13, dtype=np.float32)
    raw = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
    executed = np.clip(raw, -0.8, 0.8).astype(np.float32)
    pending = ClosedLoopCommand(
        sequence=1,
        produced_monotonic_s=10.0,
        observation_realtime_s=1_750_000_010.0,
        previous_executed_action13_used=previous,
        raw_policy_action13=raw,
        executed_policy_action13=executed,
        franka_target_q_rad=np.linspace(0.0, 0.006, 7),
        rh56_angle_set_register_order=np.arange(901, 907, dtype=np.int32),
    )
    runtime_diagnostics = {
        "no_stage_policy_hold_count": 3,
        "no_stage_policy_hold_counts_by_reason": {
            "camera_frame_policy_reuse_limit": 3
        },
        "maximum_consecutive_no_stage_hold_duration_s": 0.126,
    }
    runtime = _FakeCombinedRuntime(
        run_error=RuntimeError("synthetic RH56 transport fault"),
        failure_pending_command=pending,
        runtime_diagnostics_snapshot=runtime_diagnostics,
    )
    module, request, output = _patch_supervised_world(monkeypatch, tmp_path, runtime)

    with pytest.raises(module.SupervisedV94RuntimeError):
        module.run_supervised_v94(request)

    audit = json.loads(output.read_text(encoding="utf-8"))
    recorded = audit["failure_pending_command"]
    assert recorded["sequence"] == 1
    assert recorded["raw_policy_action13"] == raw.tolist()
    assert recorded["executed_policy_action13"] == executed.tolist()
    assert recorded["franka_target_q_rad"] == pending.franka_target_q_rad.tolist()
    assert recorded["rh56_angle_set_register_order"] == list(range(901, 907))
    assert audit["runtime_diagnostics"] == runtime_diagnostics


def test_failure_audit_retains_already_dual_acked_command(
    monkeypatch, tmp_path: Path
) -> None:
    # Kept separate from pending-command evidence: this action crossed both
    # hardware ACKs before a later runtime failure.
    executed = {
        "sequence": 1,
        "executed_policy_action13": [0.25] * 13,
        "franka_target_q_rad": [0.003] * 7,
        "rh56_angle_set_register_order": [901] * 6,
        "dual_ack_completed": True,
        "policy_source_commit_completed": True,
    }
    runtime = _FakeCombinedRuntime(
        run_error=RuntimeError("synthetic failure after final dual ACK"),
        committed_commands_snapshot=(executed,),
    )
    module, request, output = _patch_supervised_world(monkeypatch, tmp_path, runtime)

    with pytest.raises(module.SupervisedV94RuntimeError):
        module.run_supervised_v94(request)

    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["result"] == "FAIL"
    assert audit["last_dual_ack_sequence"] == 1
    assert audit["failure_pending_command"] is None
    assert audit["committed_commands"] == [executed]


class _RH56StatusTransport:
    def __init__(
        self,
        statuses: tuple[int, ...],
        *,
        angles: tuple[int, ...] = (1000,) * 6,
    ) -> None:
        self.statuses = statuses
        self.angles = angles
        self.targets = (-1,) * 6
        self.speeds = (1000,) * 6
        self.forces = (500,) * 6
        self.closed = False
        self.speed_reads = 0
        self.speed_writes = 0

    def open(self):
        return self

    def close(self) -> None:
        self.closed = True

    def read_angle_set(self, *, deadline_monotonic_s):
        assert deadline_monotonic_s > time.monotonic()
        return self.targets

    def write_angle_set(self, values, *, deadline_monotonic_s) -> None:
        assert deadline_monotonic_s > time.monotonic()
        self.targets = tuple(int(value) for value in values)

    def read_safety_feedback(self, *, deadline_monotonic_s):
        assert deadline_monotonic_s > time.monotonic()
        return SimpleNamespace(
            angles=self.angles,
            positions=(1000,) * 6,
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=self.statuses,
            temperatures_c=(25,) * 6,
        )

    def read_speed_set(self, *, deadline_monotonic_s):
        assert deadline_monotonic_s > time.monotonic()
        self.speed_reads += 1
        return self.speeds

    def write_speed_set(self, values, *, deadline_monotonic_s) -> None:
        assert deadline_monotonic_s > time.monotonic()
        self.speed_writes += 1
        self.speeds = tuple(int(value) for value in values)

    def read_force_set(self, *, deadline_monotonic_s):
        assert deadline_monotonic_s > time.monotonic()
        return self.forces

    def write_force_set(self, values, *, deadline_monotonic_s) -> None:
        assert deadline_monotonic_s > time.monotonic()
        self.forces = tuple(int(value) for value in values)


@pytest.mark.parametrize(
    ("status", "accepted"),
    ((1, False), (2, True), (0xFF, True)),
)
def test_fresh_rh56_preflight_requires_every_status_idle(
    monkeypatch, status: int, accepted: bool
) -> None:
    import sim2real.runtime.supervised_v94_runtime as module

    transport = _RH56StatusTransport((status,) * 6)
    monkeypatch.setattr(
        module,
        "LinuxRH56TransactionalTransport",
        lambda *_args, **_kwargs: transport,
    )

    if accepted:
        targets, angles, current = module._fresh_rh56_preflight(
            port="fake",
            baud=115200,
            hand_id=1,
            open_targets=(1000,) * 6,
        )
        assert targets == (-1,) * 6
        assert angles == (1000,) * 6
        assert current == 0
    else:
        with pytest.raises(module.SupervisedV94RuntimeError, match="not idle"):
            module._fresh_rh56_preflight(
                port="fake",
                baud=115200,
                hand_id=1,
                open_targets=(1000,) * 6,
            )
    assert transport.closed is True


@pytest.mark.parametrize(
    ("q6", "accepted"),
    ((973, True), (970, True), (969, False)),
)
def test_fresh_rh56_preflight_uses_commissioned_q6_release_tolerance(
    monkeypatch, q6: int, accepted: bool
) -> None:
    import sim2real.runtime.supervised_v94_runtime as module

    angles = (1000, 1000, 1000, 1000, 1000, q6)
    transport = _RH56StatusTransport((2,) * 6, angles=angles)
    monkeypatch.setattr(
        module,
        "LinuxRH56TransactionalTransport",
        lambda *_args, **_kwargs: transport,
    )

    if accepted:
        _targets, actual, _current = module._fresh_rh56_preflight(
            port="fake",
            baud=115200,
            hand_id=1,
            open_targets=(1000,) * 6,
        )
        assert actual == angles
    else:
        with pytest.raises(
            module.SupervisedV94RuntimeError,
            match="not physically open",
        ):
            module._fresh_rh56_preflight(
                port="fake",
                baud=115200,
                hand_id=1,
                open_targets=(1000,) * 6,
            )
    assert transport.closed is True


def test_owner_wait_covers_fault_cleanup_and_reopen_rejects_status_one(
    monkeypatch,
) -> None:
    import sim2real.runtime.supervised_v94_runtime as module

    captured = {}

    class CapturingOwner:
        def __init__(self, session_factory, action_ledger, **kwargs):
            captured["session_factory"] = session_factory
            captured["action_ledger"] = action_ledger
            captured.update(kwargs)

    transport = _RH56StatusTransport((1,) * 6)
    monkeypatch.setattr(module, "RH56WatchdogOwner", CapturingOwner)
    monkeypatch.setattr(
        module,
        "LinuxRH56TransactionalTransport",
        lambda *_args, **_kwargs: transport,
    )
    factory = module._make_rh56_owner_factory(
        port="fake",
        baud=115200,
        hand_id=1,
        speed=module.RH56_SPEED_SET,
        force=80,
        open_targets=(1000,) * 6,
        minimum_angle_set_register_order=(0, 0, 0, 0, 0, 416),
        maximum_angle_set_register_order=(1000,) * 6,
    )

    factory(
        admission=SimpleNamespace(run_id="status-gate"),
        action_ledger=object(),
        fault_callback=lambda _fault: None,
    )

    assert captured["response_timeout_s"] == module.RH56_OWNER_RESPONSE_TIMEOUT_S
    assert captured["response_timeout_s"] > (
        module.RH56_WATCHDOG_S + module.RH56_STOP_TIMEOUT_S
    )
    assert captured["join_timeout_s"] == pytest.approx(
        module.RH56_OWNER_JOIN_TIMEOUT_S
    )
    assert captured["join_timeout_s"] > 2.0 * module.RH56_STOP_TIMEOUT_S
    assert captured["supervised_target_only"] is True
    assert captured["feedback_period_s"] == pytest.approx(
        module.RH56_FEEDBACK_POLL_TRIGGER_S
    )
    assert captured["feedback_hard_age_s"] == pytest.approx(
        module.RH56_FEEDBACK_HARD_AGE_S
    )
    assert captured["tracking_significant_gap_units"] == 50
    assert captured["tracking_min_progress_units"] == 3
    assert captured["tracking_timeout_s"] == pytest.approx(0.750)
    with pytest.raises(module.SupervisedV94RuntimeError):
        captured["session_factory"]()
    assert transport.speed_reads == 0
    assert transport.speed_writes == 0
    assert transport.closed is True


def test_rh56_owner_factory_wires_proven_stream_and_q6_stop_calibration(
    monkeypatch,
) -> None:
    import sim2real.runtime.supervised_v94_runtime as module

    captured = {}

    class CapturingOwner:
        def __init__(self, session_factory, action_ledger, **kwargs):
            captured["session_factory"] = session_factory
            captured["action_ledger"] = action_ledger
            captured["owner_kwargs"] = kwargs

    class ConstructorReached(RuntimeError):
        pass

    def capture_actuator(_transport, **kwargs):
        captured["actuator_kwargs"] = kwargs
        raise ConstructorReached("captured production actuator parameters")

    transport = _RH56StatusTransport(
        (2,) * 6,
        angles=(1000, 1000, 1000, 1000, 1000, 974),
    )
    monkeypatch.setattr(module, "RH56WatchdogOwner", CapturingOwner)
    monkeypatch.setattr(module, "RH56TransactionalActuator", capture_actuator)
    monkeypatch.setattr(
        module,
        "LinuxRH56TransactionalTransport",
        lambda *_args, **_kwargs: transport,
    )
    factory = module._make_rh56_owner_factory(
        port="fake",
        baud=115200,
        hand_id=1,
        speed=module.RH56_SPEED_SET,
        force=module.RH56_FORCE_SET_G,
        open_targets=(1000,) * 6,
        minimum_angle_set_register_order=(0, 0, 0, 0, 0, 416),
        maximum_angle_set_register_order=(1000,) * 6,
    )
    factory(
        admission=SimpleNamespace(run_id="parameter-wiring"),
        action_ledger=object(),
        fault_callback=lambda _fault: None,
    )

    with pytest.raises(module.SupervisedV94RuntimeError):
        captured["session_factory"]()

    assert transport.speeds == module.RH56_SPEED_SET
    assert transport.forces == (module.RH56_FORCE_SET_G,) * 6
    actuator = captured["actuator_kwargs"]
    assert actuator["maximum_running_axis_current_ma"] == 1400
    assert actuator["stop_settle_max_axis_current_ma"] == 1000
    assert actuator["stop_max_axis_current_ma"] == 100
    assert actuator["maximum_feedback_age_s"] == pytest.approx(0.150)
    assert actuator["feedback_to_command_offset_units"] == (
        0,
        0,
        0,
        0,
        0,
        15,
    )
    assert actuator["stop_hold_command_minimum"] == (0, 0, 0, 0, 0, 416)
    assert actuator["stop_hold_command_maximum"] == (1000,) * 6
    owner = captured["owner_kwargs"]
    assert owner["feedback_period_s"] == pytest.approx(0.050)
    assert owner["feedback_hard_age_s"] == pytest.approx(0.150)
    assert owner["tracking_significant_gap_units"] == 50
    assert owner["tracking_min_progress_units"] == 3
    assert owner["tracking_timeout_s"] == pytest.approx(0.750)
