from __future__ import annotations

import importlib.util
import json
from types import ModuleType, SimpleNamespace
from pathlib import Path
import sys

import pytest

import anydex_pipeline.rh56_interrupted_recovery as recovery_module
from anydex_pipeline.control_config import load_control_config, verify_adapter_assets
from anydex_pipeline.rh56_commissioning import seal_evidence, sha256_file
from anydex_pipeline.inspire_sequence_driver import RH56SequenceDriverError
from anydex_pipeline.rh56_interrupted_recovery import (
    AUTHORIZED_HISTORICAL_CONTROL_PROFILE_MIGRATION_POLICY,
    AUTHORIZED_HISTORICAL_SOURCE_BINDINGS_SHA256,
    AUTHORIZED_HISTORICAL_SOURCE_POLICY,
    InterruptedRecoveryPlan,
    RECOVERY_EVIDENCE_KIND,
    RECOVERY_MODE_Q6_RETURN,
    RECOVERY_ROUTE_COUPLED_CLOSE_PREFIX,
    RECOVERY_ROUTE_NEAR_OPEN_RESET,
    RECOVERY_ROUTE_SEALED_Q6_RETURN,
    _canonical_return_waypoints,
    _derive_interrupted_waypoint,
    build_interrupted_recovery_plan,
)
from anydex_pipeline.rh56_reset_open import (
    DEFAULT_FORCES,
    DEFAULT_SPEEDS,
    RH56ResetSettingsRestoreError,
)


ROOT = Path(__file__).resolve().parents[1]
FAILED = ROOT / "runs/rh56_candidate51_exact_air_20260721_live.json"
STAGE2_INTERRUPTED = (
    ROOT / "runs/rh56_candidate0_exact_stage2_20260722_151314.json"
)
STAGE2_UNAUTHORIZED_RETRY = (
    ROOT / "runs/rh56_candidate0_exact_stage2_20260722_154512.json"
)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "apps/recover_installed_rh56.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tokens(app):
    return [
        "--confirm-installed", app.INSTALLED_TOKEN,
        "--confirm-24v-cutoff", app.POWER_TOKEN,
        "--confirm-franka-stop", app.STOP_TOKEN,
        "--confirm-workspace-clear", app.CLEAR_TOKEN,
        "--confirm-no-contact", app.NO_CONTACT_TOKEN,
        "--confirm-recovery", app.RECOVERY_TOKEN,
    ]


def _synthetic_recovery_plan():
    interrupted = (0, 1000, 1000, 1000, 1000, 646)
    bends, q6 = _canonical_return_waypoints(interrupted, 25)
    return InterruptedRecoveryPlan(
        failed_evidence_path=FAILED,
        failed_file_sha256="f" * 64,
        failed_payload_sha256="e" * 64,
        failed_run_id="synthetic-current-source-unit-test",
        target_q6=646,
        step_units=25,
        coupled_targets=(0, 358, 799, 911, 922, 646),
        interrupted_targets=interrupted,
        q6_forward_waypoints=tuple(range(975, 645, -25)) + (646,),
        bend_return_waypoints=bends,
        q6_return_waypoints=q6,
        original_speeds=(1000, 1000, 1000, 1000, 1000, 80),
        original_forces=(500, 500, 500, 500, 500, 80),
    )


def _stage2_plan():
    config, config_path = load_control_config(
        ROOT / "configs/fr3_rh56_v7_commissioning.json"
    )
    return build_interrupted_recovery_plan(
        STAGE2_INTERRUPTED,
        expected_config=config,
        expected_config_path=config_path,
    )


def _write_stage2_variant(tmp_path, name, mutate):
    value = json.loads(STAGE2_INTERRUPTED.read_text(encoding="utf-8"))
    value.pop("integrity")
    mutate(value)
    output = tmp_path / name
    output.write_text(
        json.dumps(seal_evidence(value), ensure_ascii=False), encoding="utf-8"
    )
    return output


def test_exact_151314_stage2_interrupt_derives_tight_live_q6_recovery():
    plan = _stage2_plan()

    assert plan.recovery_mode == RECOVERY_MODE_Q6_RETURN
    assert plan.failed_file_sha256 == (
        "335fb62ff3db22fd0cabec964aefa11fec2ae3c9f1b84ff229ec440366e72d44"
    )
    assert plan.failed_payload_sha256 == (
        "b5d80a26da4ec0f9a22d877da0f23ed7b29a5483f2ca90cc09fab3da8226422e"
    )
    assert plan.failed_run_id == "f2be3cc5-6045-43eb-bfab-450d13c52454"
    assert plan.coupled_targets == (798, 798, 798, 798, 978, 450)
    assert plan.interrupted_targets == (1000, 1000, 1000, 1000, 1000, 648)
    assert plan.permitted_live_q6_range == (640, 656)
    assert plan.bend_return_waypoints == ()
    assert plan.q6_return_waypoints == ()
    assert plan.allowed_recovery_routes == (
        RECOVERY_ROUTE_SEALED_Q6_RETURN,
        RECOVERY_ROUTE_NEAR_OPEN_RESET,
    )
    assert plan.profile_near_open_q6_range == (900, 1000)
    assert plan.q6_open_min_angle == 975
    assert plan.as_binding()["allowed_recovery_routes"] == [
        RECOVERY_ROUTE_SEALED_Q6_RETURN,
        RECOVERY_ROUTE_NEAR_OPEN_RESET,
    ]
    assert plan.historical_source_policy == AUTHORIZED_HISTORICAL_SOURCE_POLICY
    assert (
        plan.historical_source_bindings_sha256
        == AUTHORIZED_HISTORICAL_SOURCE_BINDINGS_SHA256
    )
    assert plan.as_binding()["historical_source_provenance"] == {
        "policy": AUTHORIZED_HISTORICAL_SOURCE_POLICY,
        "source_bindings_sha256": AUTHORIZED_HISTORICAL_SOURCE_BINDINGS_SHA256,
        "control_profile_migration": (
            AUTHORIZED_HISTORICAL_CONTROL_PROFILE_MIGRATION_POLICY
        ),
        "execution_authority": False,
            "live_migrations": [
                "commission_cli",
                "commission_evidence_module",
                "control_config_module",
                "franka_sequence_driver",
                "rh56_register_api",
                "rh56_sequence_driver",
        ],
    }


def test_exact_151314_historical_stage1_is_semantically_revalidated():
    evidence = json.loads(STAGE2_INTERRUPTED.read_text(encoding="utf-8"))
    config, config_path = load_control_config(
        ROOT / "configs/fr3_rh56_v7_commissioning.json"
    )

    binding, blockers = recovery_module._exact_historical_stage1_binding(
        evidence["stage1_prerequisite"],
        expected_config=config,
        expected_config_path=config_path,
    )

    assert blockers == []
    assert binding == evidence["stage1_prerequisite"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: config["franka"].__setitem__(
            "joint_limit_margin_rad", 0.049
        ),
        lambda config: config["franka"]["joint_limits_rad"][3].__setitem__(
            0, -3.0769
        ),
    ],
)
def test_exact_profile_migration_rejects_any_unreviewed_config_change(mutate):
    config, config_path = load_control_config(
        ROOT / "configs/fr3_rh56_v7_commissioning.json"
    )
    changed = json.loads(json.dumps(config))
    mutate(changed)

    with pytest.raises(ValueError, match="interrupted recovery is LOCKED"):
        build_interrupted_recovery_plan(
            STAGE2_INTERRUPTED,
            expected_config=changed,
            expected_config_path=config_path,
        )


def test_exact_profile_migration_rejects_a_different_profile_path(tmp_path):
    config, _ = load_control_config(
        ROOT / "configs/fr3_rh56_v7_commissioning.json"
    )
    copied_path = tmp_path / "copied-profile.json"
    copied_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different profile path"):
        build_interrupted_recovery_plan(
            STAGE2_INTERRUPTED,
            expected_config=config,
            expected_config_path=copied_path,
        )


def test_154512_retry_is_not_recovery_authority():
    config, config_path = load_control_config(
        ROOT / "configs/fr3_rh56_v7_commissioning.json"
    )

    with pytest.raises(ValueError, match="interrupted recovery is LOCKED"):
        build_interrupted_recovery_plan(
            STAGE2_UNAUTHORIZED_RETRY,
            expected_config=config,
            expected_config_path=config_path,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("run_id", "different-run-id"),
        lambda value: value["source_bindings"][0].__setitem__("sha256", "0" * 64),
    ],
)
def test_151314_identity_or_source_binding_change_is_rejected(
    tmp_path, mutate
):
    variant = _write_stage2_variant(
        tmp_path, "changed-stage2-identity.json", mutate
    )
    config, config_path = load_control_config(
        ROOT / "configs/fr3_rh56_v7_commissioning.json"
    )

    with pytest.raises(ValueError, match="interrupted recovery is LOCKED"):
        build_interrupted_recovery_plan(
            variant,
            expected_config=config,
            expected_config_path=config_path,
        )


def test_151314_file_identity_change_alone_is_rejected(tmp_path):
    value = json.loads(STAGE2_INTERRUPTED.read_text(encoding="utf-8"))
    variant = tmp_path / "same-payload-different-file-bytes.json"
    variant.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    assert value["integrity"] == json.loads(
        STAGE2_INTERRUPTED.read_text(encoding="utf-8")
    )["integrity"]
    config, config_path = load_control_config(
        ROOT / "configs/fr3_rh56_v7_commissioning.json"
    )

    with pytest.raises(ValueError, match="interrupted recovery is LOCKED"):
        build_interrupted_recovery_plan(
            variant,
            expected_config=config,
            expected_config_path=config_path,
        )


def test_exact_historical_policy_rejects_any_other_live_source_change(monkeypatch):
    evidence = json.loads(STAGE2_INTERRUPTED.read_text(encoding="utf-8"))
    original = recovery_module.sha256_file
    changed = str(
        ROOT / "apps/commission_installed_rh56.py"
    )

    def changed_hash(path):
        if str(Path(path).resolve()) == changed:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(recovery_module, "sha256_file", changed_hash)

    blockers = recovery_module._validate_exact_historical_source_bindings(
        evidence["source_bindings"]
    )
    assert blockers == [
        "authorized historical source live hash differs: " + changed
    ]


@pytest.mark.parametrize("command", ["inspect", "dry-run"])
def test_exact_151314_inspect_modes_never_enter_hardware(
    command, monkeypatch, capsys
):
    app = _load(f"test_stage2_{command.replace('-', '_')}")

    def forbidden_hardware(*_args, **_kwargs):
        raise AssertionError("offline recovery inspection entered hardware")

    monkeypatch.setattr(app, "_run_hardware_session", forbidden_hardware)
    result = app.main(
        [
            command,
            "--config", str(ROOT / "configs/fr3_rh56_v7_commissioning.json"),
            "--failed-evidence", str(STAGE2_INTERRUPTED),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert RECOVERY_MODE_Q6_RETURN in output
    assert '"permitted_live_q6_range": [' in output
    assert "640" in output and "656" in output
    assert AUTHORIZED_HISTORICAL_SOURCE_POLICY in output
    assert f"[{command} only] no hardware module was imported" in output


def test_recovery_runtime_sources_bind_reviewed_reset_open_module():
    app = _load("test_recovery_reset_source_binding")
    config, config_path = load_control_config(
        ROOT / "configs/fr3_rh56_v7_commissioning.json"
    )
    assets = verify_adapter_assets(config, config_path)

    bindings = app._source_bindings(assets)

    reset_binding = next(
        item for item in bindings if item["name"] == "rh56_reset_open"
    )
    expected = (
        ROOT / "src/anydex_pipeline/rh56_reset_open.py"
    ).resolve()
    assert reset_binding == {
        "name": "rh56_reset_open",
        "path": str(expected),
        "sha256": sha256_file(expected),
    }


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        (
            "not_keyboard_interrupt.json",
            lambda value: value["result"].__setitem__(
                "operation_error", "RuntimeError: synthetic"
            ),
            "interrupted recovery is LOCKED",
        ),
        (
            "already_returned.json",
            lambda value: value["result"].__setitem__("q6_return_pass", True),
            "interrupted recovery is LOCKED",
        ),
        (
            "bend_not_open.json",
            lambda value: value["final"]["angles"].__setitem__(0, 900),
            "interrupted recovery is LOCKED",
        ),
        (
            "mislabelled_close_prefix.json",
            lambda value: value["result"].__setitem__(
                "coupled_closure_pass", False
            ),
            "interrupted recovery is LOCKED",
        ),
        (
            "candidate_target_changed.json",
            lambda value: value["request"]["coupled_targets"].__setitem__(0, 799),
            "bound official candidate",
        ),
        (
            "manual_target_source.json",
            lambda value: value["request"].__setitem__(
                "target_source", "manual_cli"
            ),
            "manual target evidence must not contain a snapshot binding",
        ),
    ],
)
def test_other_stage2_failure_shapes_are_not_recovery_authority(
    tmp_path, name, mutate, message
):
    variant = _write_stage2_variant(tmp_path, name, mutate)
    config, config_path = load_control_config(
        ROOT / "configs/fr3_rh56_v7_commissioning.json"
    )

    with pytest.raises(ValueError, match=message):
        build_interrupted_recovery_plan(
            variant,
            expected_config=config,
            expected_config_path=config_path,
        )


def test_stage2_q6_return_telemetry_hole_is_rejected(tmp_path, monkeypatch):
    def add_hole(value):
        groups = value["observations"]["q6_return_steps"]
        sample = dict(groups[7]["feedback"][-1])
        sample["angle_targets"] = list(sample["angle_targets"])
        sample["angles"] = list(sample["angles"])
        target = groups[9]["target_q6"]
        sample["angle_targets"][5] = target
        sample["angles"][5] = target
        sample["statuses"] = [2] * 6
        sample["errors"] = [0] * 6
        sample["currents"] = [0] * 6
        groups[9]["feedback"] = [sample]

    variant = _write_stage2_variant(
        tmp_path, "q6_return_telemetry_hole.json", add_hole
    )
    resealed = json.loads(variant.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        recovery_module, "AUTHORIZED_STAGE2_FILE_SHA256", sha256_file(variant)
    )
    monkeypatch.setattr(
        recovery_module,
        "AUTHORIZED_STAGE2_PAYLOAD_SHA256",
        resealed["integrity"]["payload_sha256"],
    )
    config, config_path = load_control_config(
        ROOT / "configs/fr3_rh56_v7_commissioning.json"
    )

    with pytest.raises(ValueError, match="feedback differs from sealed all_feedback"):
        build_interrupted_recovery_plan(
            variant,
            expected_config=config,
            expected_config_path=config_path,
        )


def test_recovery_requires_every_exact_token_before_evidence_or_hardware(
    tmp_path, monkeypatch, capsys
):
    app = _load("test_recovery_tokens")
    calls = []
    monkeypatch.setattr(
        app,
        "build_interrupted_recovery_plan",
        lambda *_args, **_kwargs: calls.append("evidence"),
    )
    monkeypatch.setattr(
        app,
        "_run_hardware_session",
        lambda *_args, **_kwargs: calls.append("hardware"),
    )

    result = app.main(
        [
            "run",
            "--failed-evidence", str(FAILED),
            "--output", str(tmp_path / "recovery.json"),
        ]
    )

    assert result == 1
    assert calls == []
    assert "before hardware import" in capsys.readouterr().err


def test_historical_interrupted_record_locks_after_bound_driver_changes():
    config, config_path = load_control_config(
        ROOT / "configs/fr3_rh56_v7_commissioning.json"
    )
    try:
        build_interrupted_recovery_plan(
            FAILED, expected_config=config, expected_config_path=config_path
        )
    except ValueError as exc:
        assert "bound source file changed" in str(exc)
    else:
        raise AssertionError("stale live recovery evidence was accepted")


def test_recorded_interrupted_prefix_derives_only_pinky_then_q6_return():
    evidence = json.loads(FAILED.read_text(encoding="utf-8"))
    interrupted = _derive_interrupted_waypoint(
        evidence["observations"]["coupled_air_close_steps"],
        tuple(
            tuple(item)
            for item in evidence["request"]["coupled_command_waypoints"]
        ),
    )
    bends, q6 = _canonical_return_waypoints(
        interrupted, evidence["request"]["step_units"]
    )

    assert interrupted == (0, 1000, 1000, 1000, 1000, 646)
    assert bends[0] == (50, 1000, 1000, 1000, 1000, 646)
    assert bends[-1] == (1000, 1000, 1000, 1000, 1000, 646)
    assert all(item[1:5] == (1000, 1000, 1000, 1000) for item in bends)
    assert all(item[5] == 646 for item in bends)
    assert q6 == (
        696, 721, 746, 771, 796, 821, 846,
        871, 896, 921, 946, 971, 996, 1000,
    )


def test_tampered_interrupted_record_is_locked(tmp_path, monkeypatch):
    value = json.loads(FAILED.read_text(encoding="utf-8"))
    value.pop("integrity")
    value["request"]["coupled_command_waypoints"][0][0] = 974
    tampered = tmp_path / "tampered.json"
    tampered.write_text(
        json.dumps(seal_evidence(value), ensure_ascii=False), encoding="utf-8"
    )

    # Isolate canonical-path validation from the independently tested fact that
    # this historical live evidence is now stale after its bound driver and
    # control profile changed.
    monkeypatch.setattr(recovery_module, "_validate_bound_files", lambda _value: [])
    profile = value["control_profile"]
    config = profile["snapshot"]
    config_path = Path(profile["path"]).resolve()
    original_sha256_file = recovery_module.sha256_file

    def historical_profile_hash(path):
        if Path(path).expanduser().resolve() == config_path:
            return profile["file_sha256"]
        return original_sha256_file(path)

    monkeypatch.setattr(
        recovery_module, "sha256_file", historical_profile_hash
    )

    try:
        build_interrupted_recovery_plan(
            tampered, expected_config=config, expected_config_path=config_path
        )
    except ValueError as exc:
        assert "not canonical" in str(exc)
    else:
        raise AssertionError("tampered command path was accepted")


def test_coupled_telemetry_hole_is_not_a_recoverable_prefix():
    waypoints = [
        (975, 1000, 1000, 1000, 1000, 646),
        (950, 1000, 1000, 1000, 1000, 646),
        (925, 1000, 1000, 1000, 1000, 646),
    ]
    groups = [
        {
            "target": list(waypoints[0]),
            "feedback": [{
                "angle_targets": list(waypoints[0]),
                "errors": [0] * 6,
            }],
        },
        {"target": list(waypoints[1]), "feedback": []},
        {
            "target": list(waypoints[2]),
            "feedback": [{
                "angle_targets": list(waypoints[2]),
                "errors": [0] * 6,
            }],
        },
    ]

    try:
        _derive_interrupted_waypoint(groups, waypoints)
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("non-contiguous telemetry was accepted")


def test_confirmed_recovery_writes_non_unlocking_read_only_evidence(
    tmp_path, monkeypatch
):
    app = _load("test_recovery_confirmed")
    output = tmp_path / "recovery.json"
    calls = []

    def fake_runtime(args, config, plan):
        calls.append((args, config, plan))
        final = {
            "angle_targets": [-1] * 6,
            "angles": [1000] * 6,
            "currents": [0] * 6,
            "errors": [0] * 6,
            "statuses": [2] * 6,
            "temperatures": [30] * 6,
            "snapshot_after_disable": {},
        }
        return {
            "rh56_device": {"hand_id": 1},
            "franka_read_only": {"verified": True},
            "result": {
                "status": "pass",
                "adopted_disabled_verified": True,
                "recovered_open_verified": True,
                "disabled_verified": True,
                "recovery_route": RECOVERY_ROUTE_COUPLED_CLOSE_PREFIX,
                "recovery_route_initial_q6": 646,
                "evidence_bound_path_executed": True,
                "operation_error": None,
                "stop_error": None,
                "cleanup_error": None,
            },
            "route_dispatch": {
                "phase": "boundary_state_snapshot",
                "angle_targets": [-1] * 6,
                "angles": [1000, 1000, 1000, 1000, 1000, 646],
                "currents": [0] * 6,
                "errors": [0] * 6,
                "statuses": [2] * 6,
                "temperatures": [30] * 6,
            },
            "executed_q6_return_waypoints": list(plan.q6_return_waypoints),
            "executed_reset_q6_waypoints": [],
            "telemetry": [],
            "final": final,
        }

    monkeypatch.setattr(app, "_run_hardware_session", fake_runtime)
    monkeypatch.setattr(
        app, "build_interrupted_recovery_plan", lambda *_args, **_kwargs: _synthetic_recovery_plan()
    )
    result = app.main(
        [
            "run",
            "--failed-evidence", str(FAILED),
            "--output", str(output),
            *_tokens(app),
        ]
    )

    assert result == 0
    assert len(calls) == 1
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["kind"] == RECOVERY_EVIDENCE_KIND
    assert evidence["motion_authorized"] is False
    assert evidence["commissioning_unlock_claimed"] is False
    assert evidence["failed_commissioning_binding"]["interrupted_targets"] == [
        0, 1000, 1000, 1000, 1000, 646
    ]
    assert evidence["request"]["q6_return_waypoints"][-1] == 1000
    assert evidence["request"]["allowed_recovery_routes"] == [
        "sealed_coupled_close_prefix_v1"
    ]
    assert (
        evidence["request"]["selected_recovery_route"]
        == RECOVERY_ROUTE_COUPLED_CLOSE_PREFIX
    )
    assert evidence["request"]["route_dispatch_initial_q6"] == 646
    assert evidence["result"]["recovery_route_initial_q6"] == 646
    assert evidence["route_dispatch"]["angles"][5] == 646
    assert evidence["result"]["evidence_bound_path_executed"] is True
    assert evidence["executed_reset_q6_waypoints"] == []
    assert any(
        item["name"] == "rh56_reset_open"
        for item in evidence["source_bindings"]
    )
    assert output.stat().st_mode & 0o222 == 0


def test_near_open_settings_cleanup_failure_is_not_reported_as_stop_unconfirmed(
    monkeypatch,
):
    app = _load("test_recovery_near_cleanup_classification")
    config, _ = load_control_config(
        ROOT / "configs/fr3_rh56_v7_commissioning.json"
    )
    plan = _stage2_plan()

    class FakeHand:
        def snapshot(self):
            return {
                "hand_id": 1,
                "angle_targets": (-1,) * 6,
                "angles": (1000, 1000, 1000, 1000, 1000, 975),
                "currents": (0,) * 6,
                "errors": (0,) * 6,
                "statuses": (2,) * 6,
                "temperatures": (30,) * 6,
                "speeds": DEFAULT_SPEEDS,
                "force_limits": DEFAULT_FORCES,
            }

    class FakeDriver:
        def __init__(self):
            self.hand = FakeHand()
            self.telemetry = []
            self.last_recovery_route = None
            self.last_recovery_initial_q6 = None

        @classmethod
        def connect(cls, **_kwargs):
            return cls()

        def adopt_disabled_state_and_verify(self):
            return None

        def install_external_safety_check(self, callback):
            self.external_safety_check = callback

        def recover_interrupted_coupled_air_close_to_open(self, *_args, **_kwargs):
            self.external_safety_check()
            self.last_recovery_route = RECOVERY_ROUTE_NEAR_OPEN_RESET
            self.last_recovery_initial_q6 = 968
            return (918, 1000)

        def disable_and_verify(self):
            raise RH56ResetSettingsRestoreError(
                "all-six disable/idle is verified, but default restore failed"
            )

        def close(self):
            raise RH56SequenceDriverError(
                "physical stop is verified, but cleanup failed"
            )

    class FakeRobot:
        def read_once(self):
            return object()

    class FakeLimits:
        def __init__(self, **_kwargs):
            pass

    class FakeArm:
        def __init__(self, *_args):
            pass

        def _validate_state(self, *_args, **_kwargs):
            return None

    pylibfranka = ModuleType("pylibfranka")
    pylibfranka.Robot = lambda *_args, **_kwargs: FakeRobot()
    pylibfranka.RealtimeConfig = SimpleNamespace(kIgnore=object())
    franka_module = ModuleType("anydex_pipeline.franka_sequence_driver")
    franka_module.FrankaMotionLimits = FakeLimits
    franka_module.FrankaSequenceDriver = FakeArm
    monkeypatch.setitem(sys.modules, "pylibfranka", pylibfranka)
    monkeypatch.setitem(
        sys.modules, "anydex_pipeline.franka_sequence_driver", franka_module
    )
    monkeypatch.setattr(app, "InterruptedRecoveryDriver", FakeDriver)
    monkeypatch.setattr(app, "_franka_state_json", lambda _state: {"idle": True})
    args = app.build_parser().parse_args(
        [
            "run",
            "--failed-evidence",
            str(STAGE2_INTERRUPTED),
            "--output",
            "/tmp/not-written-near-cleanup-test.json",
            *_tokens(app),
        ]
    )

    runtime = app._run_hardware_session(args, config, plan)

    assert runtime["result"]["status"] == "fail"
    assert runtime["result"]["disabled_verified"] is True
    assert runtime["result"]["stop_error"] is None
    assert "RH56ResetSettingsRestoreError" in runtime["result"]["cleanup_error"]
    assert runtime["result"]["recovery_route"] == RECOVERY_ROUTE_NEAR_OPEN_RESET
