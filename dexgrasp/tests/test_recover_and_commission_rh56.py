from __future__ import annotations

import importlib.util
import builtins
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

from anydex_pipeline.rh56_commissioning import atomic_write_evidence, seal_evidence


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps/recover_and_commission_rh56.py"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tokens(app):
    return [
        "--confirm-installed",
        app.INSTALLED_TOKEN,
        "--confirm-24v-cutoff",
        app.POWER_TOKEN,
        "--confirm-franka-stop",
        app.STOP_TOKEN,
        "--confirm-workspace-clear",
        app.CLEAR_TOKEN,
        "--confirm-no-contact",
        app.NO_CONTACT_TOKEN,
        "--confirm-recovery",
        app.RECOVERY_TOKEN,
        "--confirm-wide-q6",
        app.WIDE_Q6_TOKEN,
        "--confirm-coupled-closure",
        app.COUPLED_TOKEN,
        "--confirm-exact-air-target",
        app.EXACT_AIR_TARGET_TOKEN,
    ]


class _FakePlan:
    def __init__(self, failed_path: Path, failed_sha256: str = "f" * 64):
        self.failed_evidence_path = failed_path
        self.failed_file_sha256 = failed_sha256
        self.failed_payload_sha256 = "e" * 64
        self.failed_run_id = "failed-run"
        self.target_q6 = 450
        self.step_units = 25
        self.coupled_targets = (798, 798, 798, 798, 978, 450)
        self.interrupted_targets = (1000, 1000, 1000, 1000, 1000, 648)
        self.q6_forward_waypoints = (975, 950, 925, 900, 875, 850, 825, 800)
        self.bend_return_waypoints = ()
        self.q6_return_waypoints = (698, 723, 748, 773, 798, 823, 848, 873, 898, 923, 948, 973, 998, 1000)
        self.original_speeds = (1000,) * 6
        self.original_forces = (500,) * 6
        self.recovery_mode = "interrupted_stage2_q6_return_v1"
        self.permitted_live_q6_range = (640, 656)
        self.allowed_recovery_routes = (
            "sealed_stage2_q6_return_v1",
            "profile_near_open_reset_v1",
        )
        self.profile_near_open_q6_range = (900, 1000)
        self.q6_open_min_angle = 975

    def as_binding(self):
        return {
            "kind": "interrupted_installed_rh56_commissioning_failure_v1",
            "path": str(self.failed_evidence_path),
            "file_sha256": self.failed_file_sha256,
            "payload_sha256": self.failed_payload_sha256,
            "run_id": self.failed_run_id,
            "target_q6": self.target_q6,
            "step_units": self.step_units,
            "coupled_targets": list(self.coupled_targets),
            "interrupted_targets": list(self.interrupted_targets),
            "bend_return_waypoints": [],
            "q6_return_waypoints": list(self.q6_return_waypoints),
            "original_speeds": list(self.original_speeds),
            "original_forces": list(self.original_forces),
            "recovery_mode": self.recovery_mode,
            "permitted_live_q6_range": list(self.permitted_live_q6_range),
            "allowed_recovery_routes": list(self.allowed_recovery_routes),
            "profile_near_open_q6_range": list(
                self.profile_near_open_q6_range
            ),
            "q6_open_min_angle": self.q6_open_min_angle,
        }


def _write(path: Path, data: bytes = b"test") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path.resolve()


def _context(app, tmp_path: Path):
    config_dir = tmp_path / "configs"
    base_config = {
        "inspire": {
            "open_targets": [1000] * 6,
            "arrival_tolerance_units": 25,
            "thumb_rotate_validated_realtime_range": [900, 1000],
        }
    }
    base = _write(
        config_dir / "base.json",
        (json.dumps(base_config, sort_keys=True) + "\n").encode("utf-8"),
    )
    snapshot = _write(tmp_path / "official.npz", b"official")
    stage1 = _write(tmp_path / "stage1.json", b"stage1")
    failed = _write(tmp_path / "failed-stage2.json", b"failed")
    output = (tmp_path / "workflow-output").resolve()
    plan = _FakePlan(failed, app.sha256_file(failed))
    immutable = (
        app._file_binding("failed_stage2", failed),
        app._file_binding("base_config", base),
        app._file_binding("official_snapshot", snapshot),
        app._file_binding("historical_stage1_evidence", stage1),
    )
    return app.WorkflowContext(
        workflow_id="12345678-1234-4234-8234-123456789abc",
        failed_stage2=failed,
        output_dir=output,
        recovery_output=output / "recovery.json",
        fresh_stage1_output=output / "fresh_stage1.json",
        stage2_output=output / "stage2.json",
        receipt_output=output / "workflow_receipt.json",
        base_config_path=base,
        base_config=base_config,
        snapshot_path=snapshot,
        candidate_index=0,
        candidate_score=1.5019598007202148,
        hand_targets=(798, 798, 798, 798, 978, 450),
        historical_stage1_evidence_path=stage1,
        q6_step=25,
        derived_profile_path=(config_dir / "derived.json").resolve(),
        recovery_plan=plan,
        immutable_inputs=immutable,
        historical_commissioning_sources=(
            {"name": "historical", "path": str(stage1), "sha256": app.sha256_file(stage1)},
        ),
        recovery_runtime_sources=(),
    )


def _args(app, context):
    return app.build_parser().parse_args(
        [
            "run",
            "--failed-stage2",
            str(context.failed_stage2),
            "--output-dir",
            str(context.output_dir),
            *_tokens(app),
        ]
    )


def test_parser_has_no_target_candidate_snapshot_or_config_override():
    app = _load("test_recover_commission_parser")
    subparsers = next(
        action
        for action in app.build_parser()._actions
        if isinstance(action, app.argparse._SubParsersAction)
    )
    run = subparsers.choices["run"]
    destinations = {action.dest for action in run._actions}
    assert {"failed_stage2", "output_dir"} <= destinations
    assert {
        "config",
        "snapshot",
        "candidate_index",
        "target_q6",
        "bend_targets",
        "hand_targets",
        "q6_step",
        "stage1_evidence",
    }.isdisjoint(destinations)


def test_import_does_not_load_device_transport_modules(monkeypatch):
    original = builtins.__import__
    forbidden = {"pylibfranka", "pyrealsense2", "serial"}

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            raise AssertionError("hardware transport imported: {}".format(name))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    app = _load("test_recover_commission_import_guard")
    assert app.RECEIPT_KIND.endswith("_v2")


def test_missing_token_stops_before_context_or_output(tmp_path):
    app = _load("test_recover_commission_tokens")
    context = _context(app, tmp_path)
    values = [
        "run",
        "--failed-stage2",
        str(context.failed_stage2),
        "--output-dir",
        str(context.output_dir),
        *_tokens(app),
    ]
    values = values[:-2]
    args = app.build_parser().parse_args(values)

    with pytest.raises(ValueError, match="before any child process"):
        app.run_workflow(
            args,
            context_loader=lambda *_args: (_ for _ in ()).throw(
                AssertionError("context was loaded before token validation")
            ),
        )
    assert not context.output_dir.exists()


def test_success_uses_original_candidate_targets_and_writes_read_only_receipt(
    tmp_path, capsys
):
    app = _load("test_recover_commission_success")
    context = _context(app, tmp_path)
    args = _args(app, context)
    calls = []
    validations = []

    def runner(stage, command):
        calls.append((stage, tuple(command)))
        if stage == "recovery":
            Path(command[command.index("--output") + 1]).write_bytes(b"recovery")
            os.chmod(context.recovery_output, 0o444)
        elif stage == "fresh_stage1":
            Path(command[command.index("--output") + 1]).write_bytes(b"fresh-stage1")
            os.chmod(context.fresh_stage1_output, 0o444)
        elif stage == "stage2":
            Path(command[command.index("--output") + 1]).write_bytes(b"stage2")
            os.chmod(context.stage2_output, 0o444)
        elif stage == "materialize_profile":
            staging = Path(command[command.index("--output-config") + 1])
            staging.write_bytes(b"derived-profile")
            os.chmod(staging, 0o444)
        return app.ChildResult(0)

    result = app.run_workflow(
        args,
        runner=runner,
        context_loader=lambda *_args: context,
        recovery_validator=lambda path, ctx: validations.append(
            ("recovery", path, ctx)
        )
        or {
            "request": {
                "allowed_recovery_routes": list(
                    context.recovery_plan.allowed_recovery_routes
                ),
                "selected_recovery_route": "profile_near_open_reset_v1",
                "route_dispatch_initial_q6": 968,
            },
            "result": {
                "recovery_route": "profile_near_open_reset_v1",
                "recovery_route_initial_q6": 968,
                "evidence_bound_path_executed": False,
            },
            "telemetry": [
                {
                    "phase": "reset_open_preflight",
                    "angles": [1000, 1000, 1000, 1000, 1000, 968],
                }
            ],
            "executed_q6_return_waypoints": [],
            "executed_reset_q6_waypoints": [918, 1000],
        },
        stage1_validator=lambda path, ctx: validations.append(
            ("fresh_stage1", path, ctx)
        )
        or {},
        stage2_validator=lambda path, ctx: validations.append(
            ("stage2", path, ctx)
        )
        or {},
        applied_validator=lambda evidence, profile, ctx: validations.append(
            ("applied", profile, ctx)
        )
        or {},
    )

    assert result == 0
    assert [name for name, _ in calls] == [
        "recovery",
        "fresh_stage1",
        "verify_fresh_stage1",
        "stage2",
        "verify_stage2",
        "materialize_profile",
        "verify_applied",
    ]
    stage2 = dict(calls)["stage2"]
    assert stage2[stage2.index("--candidate-index") + 1] == "0"
    assert stage2[stage2.index("--snapshot") + 1] == str(context.snapshot_path)
    assert stage2[stage2.index("--stage1-evidence") + 1] == str(
        context.fresh_stage1_output
    )
    assert stage2[stage2.index("--q6-step") + 1] == "25"
    assert "--target-q6" not in stage2
    assert "--bend-targets" not in stage2
    assert "--hand-targets" not in stage2
    assert [item[0] for item in validations] == [
        "recovery",
        "fresh_stage1",
        "stage2",
        "applied",
    ]

    receipt = json.loads(context.receipt_output.read_text(encoding="utf-8"))
    assert receipt["kind"] == app.RECEIPT_KIND
    assert receipt["schema_version"] == app.RECEIPT_SCHEMA_VERSION
    assert receipt["motion_authorized"] is False
    assert receipt["result"]["status"] == "pass"
    assert receipt["selection"]["candidate_index"] == 0
    assert receipt["selection"]["hand_targets"] == [798, 798, 798, 798, 978, 450]
    assert receipt["selection"]["manual_target_or_candidate_override_allowed"] is False
    assert receipt["recovery_execution"] == {
        "route": "profile_near_open_reset_v1",
        "evidence_bound_path_executed": False,
        "allowed_recovery_routes": [
            "sealed_stage2_q6_return_v1",
            "profile_near_open_reset_v1",
        ],
        "route_dispatch_initial_q6": 968,
        "reset_preflight_initial_q6": 968,
        "executed_q6_return_waypoints": [],
        "executed_reset_q6_waypoints": [918, 1000],
    }
    assert receipt["historical_commissioning_sources"] == [
        dict(context.historical_commissioning_sources[0])
    ]
    fresh_output = next(
        item
        for item in receipt["outputs"]
        if item["name"] == "stage1_evidence"
    )
    assert fresh_output["sha256"] == app.sha256_file(context.fresh_stage1_output)
    unsigned = {key: value for key, value in receipt.items() if key != "integrity"}
    assert receipt["integrity"]["payload_sha256"] == app.json_sha256(unsigned)
    assert context.receipt_output.stat().st_mode & 0o222 == 0
    output = capsys.readouterr().out
    assert "PROFILE={}".format(context.derived_profile_path) in output
    assert "RECEIPT={}".format(context.receipt_output) in output


def test_first_failed_child_stops_without_materialize(tmp_path):
    app = _load("test_recover_commission_failure")
    context = _context(app, tmp_path)
    args = _args(app, context)
    calls = []

    def runner(stage, command):
        calls.append(stage)
        if stage == "recovery":
            context.recovery_output.write_bytes(b"recovery")
            os.chmod(context.recovery_output, 0o444)
            return app.ChildResult(0)
        if stage == "fresh_stage1":
            context.fresh_stage1_output.write_bytes(b"fresh-stage1")
            os.chmod(context.fresh_stage1_output, 0o444)
            return app.ChildResult(0)
        if stage == "verify_fresh_stage1":
            return app.ChildResult(0)
        if stage == "stage2":
            return app.ChildResult(4)
        raise AssertionError("workflow continued after failed Stage-2")

    result = app.run_workflow(
        args,
        runner=runner,
        context_loader=lambda *_args: context,
        recovery_validator=lambda *_args: {},
        stage1_validator=lambda *_args: {},
        stage2_validator=lambda *_args: (_ for _ in ()).throw(
            AssertionError("failed Stage-2 was validated")
        ),
        applied_validator=lambda *_args: (_ for _ in ()).throw(
            AssertionError("failed Stage-2 was materialized")
        ),
    )

    assert result == 4
    assert calls == ["recovery", "fresh_stage1", "verify_fresh_stage1", "stage2"]
    assert not context.derived_profile_path.exists()
    receipt = json.loads(context.receipt_output.read_text(encoding="utf-8"))
    assert receipt["result"]["status"] == "fail"
    assert receipt["result"]["failed_stage"] == "stage2"
    assert [item["name"] for item in receipt["stages"]] == calls


def test_fresh_stage1_rejection_stops_before_exact_stage2(tmp_path):
    app = _load("test_recover_commission_fresh_stage1_locked")
    context = _context(app, tmp_path)
    args = _args(app, context)
    calls = []

    def runner(stage, _command):
        calls.append(stage)
        if stage == "recovery":
            context.recovery_output.write_bytes(b"recovery")
            os.chmod(context.recovery_output, 0o444)
        elif stage == "fresh_stage1":
            context.fresh_stage1_output.write_bytes(b"fresh-stage1")
            os.chmod(context.fresh_stage1_output, 0o444)
        return app.ChildResult(0)

    result = app.run_workflow(
        args,
        runner=runner,
        context_loader=lambda *_args: context,
        recovery_validator=lambda *_args: {},
        stage1_validator=lambda *_args: (_ for _ in ()).throw(
            ValueError("fresh current-source Stage1 is LOCKED")
        ),
        stage2_validator=lambda *_args: pytest.fail("Stage2 must not start"),
    )

    assert result == 1
    assert calls == ["recovery", "fresh_stage1", "verify_fresh_stage1"]
    receipt = json.loads(context.receipt_output.read_text(encoding="utf-8"))
    assert receipt["result"]["failed_stage"] == "verify_fresh_stage1"
    assert "current-source Stage1 is LOCKED" in receipt["result"]["error"]


def test_current_recovery_runtime_source_change_stops_after_recovery(tmp_path):
    app = _load("test_recover_commission_runtime_source_change")
    context = _context(app, tmp_path)
    runtime_path = _write(tmp_path / "runtime/recovery.py", b"version-one")
    runtime_binding = app._file_binding("recovery_cli", runtime_path)
    context = replace(
        context,
        immutable_inputs=context.immutable_inputs + (runtime_binding,),
        recovery_runtime_sources=(runtime_binding,),
    )
    args = _args(app, context)
    calls = []

    def runner(stage, _command):
        calls.append(stage)
        assert stage == "recovery"
        context.recovery_output.write_bytes(b"recovery")
        os.chmod(context.recovery_output, 0o444)
        runtime_path.write_bytes(b"version-two")
        return app.ChildResult(0)

    result = app.run_workflow(
        args,
        runner=runner,
        context_loader=lambda *_args: context,
        recovery_validator=lambda *_args: pytest.fail(
            "changed runtime source must stop before recovery validation"
        ),
    )

    assert result == 1
    assert calls == ["recovery"]
    receipt = json.loads(context.receipt_output.read_text(encoding="utf-8"))
    assert receipt["result"]["failed_stage"] == "recovery"
    assert "immutable input changed" in receipt["result"]["error"]


def test_interrupted_child_stops_workflow_and_records_130(tmp_path):
    app = _load("test_recover_commission_interrupted")
    context = _context(app, tmp_path)
    args = _args(app, context)
    calls = []

    def runner(stage, command):
        calls.append(stage)
        return app.ChildResult(130, interrupted=True)

    result = app.run_workflow(
        args,
        runner=runner,
        context_loader=lambda *_args: context,
        recovery_validator=lambda *_args: (_ for _ in ()).throw(
            AssertionError("interrupted recovery was validated")
        ),
    )

    assert result == 130
    assert calls == ["recovery"]
    receipt = json.loads(context.receipt_output.read_text(encoding="utf-8"))
    assert receipt["result"]["failed_stage"] == "recovery"
    assert receipt["stages"][0]["interrupted"] is True


def test_real_child_runner_forwards_only_sigint_and_waits(monkeypatch):
    app = _load("test_recover_commission_signal")
    calls = []

    class FakeProcess:
        pid = 43210

        def __init__(self):
            self.wait_count = 0

        def wait(self):
            self.wait_count += 1
            if self.wait_count == 1:
                raise KeyboardInterrupt()
            return 130

    process = FakeProcess()

    def popen(argv, **kwargs):
        calls.append(("popen", tuple(argv), kwargs))
        return process

    monkeypatch.setattr(app.subprocess, "Popen", popen)
    monkeypatch.setattr(
        app.os,
        "killpg",
        lambda pid, sig: calls.append(("killpg", pid, sig)),
    )

    result = app._run_child("recovery", ["fake-child"])

    assert result == app.ChildResult(130, interrupted=True)
    assert calls[0][2]["start_new_session"] is True
    assert calls[1] == ("killpg", process.pid, signal.SIGINT)
    assert process.wait_count == 2


def test_strict_recovery_validator_rejects_non_open_feedback(tmp_path):
    app = _load("test_recover_commission_strict_recovery")
    context = _context(app, tmp_path)
    source_bindings = []
    for name in sorted(app.RECOVERY_SOURCE_NAMES):
        path = _write(tmp_path / "sources" / name, name.encode("utf-8"))
        source_bindings.append(
            {"name": name, "path": str(path), "sha256": app.sha256_file(path)}
        )
    context = replace(
        context,
        recovery_runtime_sources=tuple(
            app._file_binding(
                item["name"],
                Path(item["path"]),
            )
            for item in source_bindings
        ),
    )
    snapshot = {
        "angle_targets": [-1] * 6,
        "angles": [1000] * 6,
        "currents": [0] * 6,
        "errors": [0] * 6,
        "statuses": [2] * 6,
        "speeds": [1000] * 6,
        "force_limits": [500] * 6,
    }
    payload = {
        "schema_version": 1,
        "kind": app.RECOVERY_EVIDENCE_KIND,
        "motion_authorized": False,
        "commissioning_unlock_claimed": False,
        "control_profile": {
            "path": str(context.base_config_path),
            "file_sha256": app.sha256_file(context.base_config_path),
            "parsed_sha256": app.json_sha256(context.base_config),
            "snapshot": context.base_config,
        },
        "failed_commissioning_binding": context.recovery_plan.as_binding(),
        "source_bindings": source_bindings,
        "operator_confirmations": {
            "installed_on_fr3": True,
            "24v_cutoff_ready": True,
            "franka_stop_ready": True,
            "workspace_clear": True,
            "no_contact_PLA_scope": True,
            "interrupted_open_recovery_confirmed": True,
        },
        "request": {
            "recovery_mode": context.recovery_plan.recovery_mode,
            "evidence_bound_q6_return_waypoints": list(
                context.recovery_plan.q6_return_waypoints
            ),
            "permitted_live_q6_range": list(
                context.recovery_plan.permitted_live_q6_range
            ),
            "allowed_recovery_routes": list(
                context.recovery_plan.allowed_recovery_routes
            ),
            "selected_recovery_route": "profile_near_open_reset_v1",
            "route_dispatch_initial_q6": 968,
            "profile_near_open_q6_range": [900, 1000],
            "q6_open_min_angle": 975,
        },
        "result": {
            "status": "pass",
            "adopted_disabled_verified": True,
            "recovered_open_verified": True,
            "disabled_verified": True,
            "operation_error": None,
            "stop_error": None,
            "cleanup_error": None,
            "recovery_route": "profile_near_open_reset_v1",
            "recovery_route_initial_q6": 968,
            "evidence_bound_path_executed": False,
        },
        "franka_read_only": {"verified": True},
        "route_dispatch": {
            "phase": "boundary_state_snapshot",
            "angle_targets": [-1] * 6,
            "angles": [1000, 1000, 1000, 1000, 1000, 968],
            "currents": [0] * 6,
            "errors": [0] * 6,
            "statuses": [2] * 6,
            "temperatures": [30] * 6,
        },
        "telemetry": [
            {
                "phase": "boundary_state_snapshot",
                "angle_targets": [-1] * 6,
                "angles": [1000, 1000, 1000, 1000, 1000, 968],
                "currents": [0] * 6,
                "errors": [0] * 6,
                "statuses": [2] * 6,
                "temperatures": [30] * 6,
            },
            {
                "phase": "reset_open_preflight",
                "angle_targets": [-1] * 6,
                "angles": [1000, 1000, 1000, 1000, 1000, 968],
                "currents": [0] * 6,
                "errors": [0] * 6,
                "statuses": [2] * 6,
                "temperatures": [30] * 6,
            },
        ],
        "executed_q6_return_waypoints": [],
        "executed_reset_q6_waypoints": [918, 1000],
        "final": {
            "angle_targets": [-1] * 6,
            "angles": [1000, 1000, 1000, 1000, 1000, 974],
            "currents": [0] * 6,
            "errors": [0] * 6,
            "statuses": [2] * 6,
            "original_settings_restored": True,
            "settings_restore_target": "reset_defaults",
            "snapshot_after_disable": snapshot,
        },
    }
    recovery = context.output_dir.parent / "bad-recovery.json"
    atomic_write_evidence(recovery, seal_evidence(payload))

    with pytest.raises(ValueError, match="not canonical open"):
        app._strict_recovery_evidence(recovery, context)

    payload["final"]["angles"][5] = 975
    accepted = context.output_dir.parent / "accepted-near-open-recovery.json"
    atomic_write_evidence(accepted, seal_evidence(payload))
    validated = app._strict_recovery_evidence(accepted, context)
    assert validated["result"]["recovery_route"] == "profile_near_open_reset_v1"
    assert validated["result"]["evidence_bound_path_executed"] is False

    payload["executed_reset_q6_waypoints"] = [100, 1000]
    escaped_reset = context.output_dir.parent / "escaped-reset-path.json"
    atomic_write_evidence(escaped_reset, seal_evidence(payload))
    with pytest.raises(ValueError, match="canonical profile path"):
        app._strict_recovery_evidence(escaped_reset, context)
    payload["executed_reset_q6_waypoints"] = [918, 1000]

    original_source = dict(payload["source_bindings"][0])
    alias_source = _write(
        tmp_path / "sources" / "same-bytes-alias",
        Path(original_source["path"]).read_bytes(),
    )
    payload["source_bindings"][0]["path"] = str(alias_source)
    payload["source_bindings"][0]["sha256"] = app.sha256_file(alias_source)
    rebound = context.output_dir.parent / "rebound-runtime-source.json"
    atomic_write_evidence(rebound, seal_evidence(payload))
    with pytest.raises(ValueError, match="differs from workflow runtime authority"):
        app._strict_recovery_evidence(rebound, context)
    payload["source_bindings"][0] = original_source

    payload["result"]["evidence_bound_path_executed"] = True
    false_claim = context.output_dir.parent / "false-sealed-path-claim.json"
    atomic_write_evidence(false_claim, seal_evidence(payload))
    with pytest.raises(ValueError, match="misstates whether the sealed path"):
        app._strict_recovery_evidence(false_claim, context)


def test_coupled_prefix_plan_does_not_offer_near_open_fallback(tmp_path):
    app = _load("test_recover_commission_prefix_route")
    context = _context(app, tmp_path)
    prefix_plan = _FakePlan(context.failed_stage2)
    prefix_plan.recovery_mode = "interrupted_coupled_close_prefix_v1"

    assert app._expected_recovery_routes(prefix_plan) == (
        "sealed_coupled_close_prefix_v1",
    )


def test_context_derivation_delegates_historical_profile_migration_to_plan_builder(
    tmp_path, monkeypatch
):
    app = _load("test_recover_commission_derivation")
    config = {"test": "current-reviewed-profile"}
    historical_config = {"test": "historical-profile"}
    base = _write(
        tmp_path / "configs/base.json",
        (json.dumps(config, sort_keys=True) + "\n").encode("utf-8"),
    )
    snapshot = _write(tmp_path / "official.npz", b"snapshot")
    stage1 = _write(tmp_path / "stage1.json", b"stage1")
    bound_source = _write(tmp_path / "bound-source.py", b"# bound\n")
    failed = tmp_path / "failed.json"
    payload = seal_evidence(
        {
            "control_profile": {
                "path": str(base),
                "file_sha256": "1" * 64,
                "parsed_sha256": app.json_sha256(historical_config),
                "snapshot": historical_config,
            },
            "snapshot_candidate": {
                "path": str(snapshot),
                "file_sha256": app.sha256_file(snapshot),
                "candidate": {
                    "index": 0,
                    "official_score": 1.5,
                    "hand_targets": [798, 798, 798, 798, 978, 450],
                },
            },
            "stage1_prerequisite": {
                "path": str(stage1),
                "file_sha256": app.sha256_file(stage1),
            },
            "source_bindings": [
                {
                    "name": "synthetic_source",
                    "path": str(bound_source),
                    "sha256": app.sha256_file(bound_source),
                }
            ],
            "request": {
                "target_source": "official_snapshot_candidate",
                "step_units": 25,
                "coupled_targets": [798, 798, 798, 798, 978, 450],
            },
        }
    )
    atomic_write_evidence(failed, payload)
    # Historical source hashes are sealed provenance, not current execution
    # authority. The exact recovery-plan builder is responsible for accepting
    # only its narrowly reviewed legacy artifact.
    bound_source.write_bytes(b"# current source changed\n")
    monkeypatch.setattr(app, "load_control_config", lambda path: (config, base))
    adapter_mesh = _write(tmp_path / "adapter.stl", b"adapter")
    adapter_provenance = _write(tmp_path / "adapter.json", b"{}")
    monkeypatch.setattr(
        app,
        "verify_adapter_assets",
        lambda *_args: SimpleNamespace(
            mesh_path=adapter_mesh,
            provenance_path=adapter_provenance,
        ),
    )

    def plan_builder(path, **kwargs):
        assert path == failed.resolve()
        assert kwargs["expected_config"] == config
        assert kwargs["expected_config_path"] == base
        return _FakePlan(failed.resolve(), app.sha256_file(failed))

    monkeypatch.setattr(app, "build_interrupted_recovery_plan", plan_builder)
    context = app._derive_context(failed, tmp_path / "new-output")

    assert context.candidate_index == 0
    assert context.base_config == config
    assert context.hand_targets == (798, 798, 798, 798, 978, 450)
    assert context.q6_step == 25
    assert context.snapshot_path == snapshot
    assert context.historical_stage1_evidence_path == stage1
    assert context.fresh_stage1_output == context.output_dir / "fresh_stage1.json"
    assert context.historical_commissioning_sources[0]["sha256"] != app.sha256_file(
        bound_source
    )
    assert {item.name for item in context.recovery_runtime_sources} == set(
        app.RECOVERY_SOURCE_NAMES
    )
    assert context.derived_profile_path.parent == base.parent
    assert not context.output_dir.exists()


def test_receipt_writer_is_o_excl_and_0444(tmp_path):
    app = _load("test_recover_commission_receipt_exclusive")
    path = tmp_path / "receipt.json"
    app._write_exclusive_readonly_json(path, {"status": "first"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "first"}
    assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        app._write_exclusive_readonly_json(path, {"status": "replacement"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "first"}


def test_receipt_writer_rejects_dangling_final_symlink_without_creating_target(
    tmp_path,
):
    app = _load("test_recover_commission_receipt_symlink")
    target = tmp_path / "must-not-be-created.json"
    alias = tmp_path / "receipt-alias.json"
    os.symlink(target, alias)

    with pytest.raises(FileExistsError):
        app._write_exclusive_readonly_json(alias, {"status": "must-not-write"})

    assert alias.is_symlink()
    assert not target.exists()


def test_receipt_creation_uses_exclusive_nofollow_and_exact_0444(
    tmp_path, monkeypatch
):
    app = _load("test_recover_commission_receipt_open_flags")
    real_open = app.os.open
    calls = []

    def tracking_open(path, flags, mode=0o777, *, dir_fd=None):
        calls.append((str(path), flags, mode, dir_fd))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(app.os, "open", tracking_open)
    receipt = tmp_path / "receipt.json"
    app._write_exclusive_readonly_json(receipt, {"status": "sealed"})

    create = next(item for item in calls if item[1] & os.O_CREAT)
    assert create[1] & os.O_EXCL
    assert create[1] & os.O_NOFOLLOW
    assert create[2] == 0o444
    assert receipt.stat().st_mode & 0o777 == 0o444


@pytest.mark.parametrize("leaf", ["output_dir", "derived_profile_path"])
def test_preflight_rejects_existing_or_dangling_output_leaf_before_child(
    tmp_path, leaf
):
    app = _load("test_recover_commission_output_leaf_" + leaf)
    context = _context(app, tmp_path)
    path = getattr(context, leaf)
    target = tmp_path / ("unexpected-target-" + leaf)
    os.symlink(target, path)
    args = _args(app, context)

    with pytest.raises(FileExistsError, match="including a symlink"):
        app.run_workflow(
            args,
            context_loader=lambda *_args: context,
            runner=lambda *_args: pytest.fail("child must not start"),
        )

    assert path.is_symlink()
    assert not target.exists()


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_preflight_rejects_existing_output_directory_leaf_before_child(
    tmp_path, kind
):
    app = _load("test_recover_commission_existing_output_" + kind)
    context = _context(app, tmp_path)
    if kind == "file":
        context.output_dir.write_bytes(b"occupied")
    else:
        context.output_dir.mkdir()
    args = _args(app, context)

    with pytest.raises(FileExistsError, match="including a symlink"):
        app.run_workflow(
            args,
            context_loader=lambda *_args: context,
            runner=lambda *_args: pytest.fail("child must not start"),
        )


def test_recovery_symlink_output_fails_closed_but_writes_sealed_receipt(tmp_path):
    app = _load("test_recover_commission_recovery_symlink")
    context = _context(app, tmp_path)
    args = _args(app, context)
    target = tmp_path / "must-not-be-created-recovery.json"
    calls = []

    def runner(stage, _command):
        calls.append(stage)
        assert stage == "recovery"
        os.symlink(target, context.recovery_output)
        return app.ChildResult(0)

    result = app.run_workflow(
        args,
        runner=runner,
        context_loader=lambda *_args: context,
        recovery_validator=lambda *_args: pytest.fail(
            "symlink output must not reach validation"
        ),
    )

    assert result == 1
    assert calls == ["recovery"]
    assert not target.exists()
    receipt = json.loads(context.receipt_output.read_text(encoding="utf-8"))
    assert receipt["result"]["status"] == "fail"
    assert receipt["result"]["failed_stage"] == "recovery"
    recovery = next(
        item for item in receipt["outputs"] if item["name"] == "recovery_evidence"
    )
    assert recovery["exists"] is True
    assert recovery["safe_regular_file"] is False
    assert recovery["sha256"] is None
    assert context.receipt_output.stat().st_mode & 0o777 == 0o444


def test_immutable_source_symlink_swap_stops_after_first_stage_and_receipts(
    tmp_path,
):
    app = _load("test_recover_commission_source_swap")
    context = _context(app, tmp_path)
    args = _args(app, context)
    original = context.snapshot_path
    replacement = _write(tmp_path / "same-snapshot-bytes.npz", original.read_bytes())
    calls = []

    def runner(stage, command):
        calls.append(stage)
        assert stage == "recovery"
        Path(command[command.index("--output") + 1]).write_bytes(b"recovery")
        os.chmod(context.recovery_output, 0o444)
        original.unlink()
        os.symlink(replacement, original)
        return app.ChildResult(0)

    result = app.run_workflow(
        args,
        runner=runner,
        context_loader=lambda *_args: context,
        recovery_validator=lambda *_args: pytest.fail(
            "source alias must stop before evidence validation"
        ),
    )

    assert result == 1
    assert calls == ["recovery"]
    receipt = json.loads(context.receipt_output.read_text(encoding="utf-8"))
    assert receipt["result"]["status"] == "fail"
    assert receipt["result"]["failed_stage"] == "recovery"
    assert "non-symlink" in receipt["result"]["error"]


def test_stage2_inode_swap_during_verify_stops_before_materialize(tmp_path):
    app = _load("test_recover_commission_stage2_swap")
    context = _context(app, tmp_path)
    args = _args(app, context)
    calls = []

    def runner(stage, command):
        calls.append(stage)
        if stage == "recovery":
            context.recovery_output.write_bytes(b"recovery")
            os.chmod(context.recovery_output, 0o444)
        elif stage == "fresh_stage1":
            context.fresh_stage1_output.write_bytes(b"fresh-stage1")
            os.chmod(context.fresh_stage1_output, 0o444)
        elif stage == "verify_fresh_stage1":
            pass
        elif stage == "stage2":
            context.stage2_output.write_bytes(b"stage2")
            os.chmod(context.stage2_output, 0o444)
        elif stage == "verify_stage2":
            replacement = context.output_dir / "stage2-replacement.json"
            replacement.write_bytes(b"stage2")
            os.chmod(replacement, 0o444)
            os.replace(replacement, context.stage2_output)
        else:
            pytest.fail("workflow continued after Stage-2 inode swap")
        return app.ChildResult(0)

    result = app.run_workflow(
        args,
        runner=runner,
        context_loader=lambda *_args: context,
        recovery_validator=lambda *_args: {},
        stage1_validator=lambda *_args: {},
        stage2_validator=lambda *_args: pytest.fail(
            "swapped Stage-2 evidence must not be validated"
        ),
    )

    assert result == 1
    assert calls == [
        "recovery",
        "fresh_stage1",
        "verify_fresh_stage1",
        "stage2",
        "verify_stage2",
    ]
    receipt = json.loads(context.receipt_output.read_text(encoding="utf-8"))
    assert receipt["result"]["status"] == "fail"
    assert receipt["result"]["failed_stage"] == "verify_stage2"
    assert "immutable input changed" in receipt["result"]["error"]


def test_derived_profile_symlink_race_is_not_followed(tmp_path):
    app = _load("test_recover_commission_derived_symlink_race")
    context = _context(app, tmp_path)
    args = _args(app, context)
    target = tmp_path / "must-not-be-created-derived.json"
    calls = []

    def runner(stage, command):
        calls.append(stage)
        if stage == "recovery":
            context.recovery_output.write_bytes(b"recovery")
            os.chmod(context.recovery_output, 0o444)
        elif stage == "fresh_stage1":
            context.fresh_stage1_output.write_bytes(b"fresh-stage1")
            os.chmod(context.fresh_stage1_output, 0o444)
        elif stage == "stage2":
            context.stage2_output.write_bytes(b"stage2")
            os.chmod(context.stage2_output, 0o444)
        elif stage == "materialize_profile":
            staging = Path(command[command.index("--output-config") + 1])
            staging.write_bytes(b"derived-profile")
            os.chmod(staging, 0o444)
            os.symlink(target, context.derived_profile_path)
        return app.ChildResult(0)

    result = app.run_workflow(
        args,
        runner=runner,
        context_loader=lambda *_args: context,
        recovery_validator=lambda *_args: {},
        stage1_validator=lambda *_args: {},
        stage2_validator=lambda *_args: {},
        applied_validator=lambda *_args: pytest.fail(
            "derived symlink race must stop before applied validation"
        ),
    )

    assert result == 1
    assert calls == [
        "recovery",
        "fresh_stage1",
        "verify_fresh_stage1",
        "stage2",
        "verify_stage2",
        "materialize_profile",
    ]
    assert context.derived_profile_path.is_symlink()
    assert not target.exists()
    receipt = json.loads(context.receipt_output.read_text(encoding="utf-8"))
    assert receipt["result"]["status"] == "fail"
    assert receipt["result"]["failed_stage"] == "materialize_profile"
