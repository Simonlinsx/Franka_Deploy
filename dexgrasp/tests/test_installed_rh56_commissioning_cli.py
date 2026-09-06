from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from anydex_pipeline.snapshot import (
    GraspCandidates,
    VisualizationSnapshot,
    save_snapshot_npz,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "apps/commission_installed_rh56.py"
    )
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
    ]


def _fake_runtime(args, config):
    target = int(args.target_q6)
    points = list(range(1000 - int(args.q6_step), target, -int(args.q6_step)))
    if not points or points[-1] != target:
        points.append(target)
    return {
        "franka_read_only": {"verified": True},
        "rh56_device": {"hand_id": 1},
        "result": {
            "status": "pass",
            "q6_sweep_pass": True,
            "coupled_closure_pass": bool(args.coupled_air_close),
            "reopened_and_verified": True,
            "disabled_verified": True,
            "operation_error": None,
            "stop_error": None,
        },
        "observations": {
            "all_feedback": [],
            "q6_steps": [],
            "coupled_air_close_feedback": [],
            "final_open_feedback": None,
            "disable_feedback": [],
            "actual_q6_range": None,
        },
        "final": {
            "angle_targets": [-1] * 6,
            "angles": [1000] * 6,
            "currents": [0] * 6,
            "errors": [0] * 6,
            "statuses": [2] * 6,
            "temperatures": [30] * 6,
            "reopened_and_verified": True,
            "disabled_verified": True,
            "snapshot_after_disable": {},
        },
        "q6_waypoints": points,
    }


def _official_snapshot(path: Path, *, calibration_id=None, targets=None):
    config = json.loads(
        (ROOT / "configs/fr3_rh56_v7_commissioning.json").read_text(
            encoding="utf-8"
        )
    )
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, ...], 3, axis=0)
    poses[:, 0, 3] = np.asarray([0.50, 0.51, 0.52])
    hand_poses = poses.copy()
    hand_poses[:, 2, 3] += 0.02
    hand_targets = np.asarray(
        targets
        or [
            [910, 911, 912, 913, 914, 1000],
            [900, 901, 902, 903, 904, 950],
            [920, 921, 922, 923, 924, 925],
        ],
        dtype=np.float32,
    )
    snapshot = VisualizationSnapshot(
        scene_points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        scene_colors=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        object_points=np.asarray([[0.5, 0.0, 0.1]], dtype=np.float32),
        object_colors=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        grasps=GraspCandidates(
            canonical_poses=poses,
            scores=np.asarray([0.8, 0.95, 0.95], dtype=np.float32),
            type_ids=np.asarray([1, 2, 3], dtype=np.int32),
            collision_free=np.asarray([False, False, False]),
            collision_checked=np.asarray([False, False, False]),
            selected_index=2,
            hand_poses=hand_poses,
            hand_angles=hand_targets,
            source_indices=np.asarray([10, 11, 12], dtype=np.int64),
        ),
        reference_frame=config["reference_frame"],
        T_reference_camera=np.eye(4, dtype=np.float64),
        frame_id=17,
        timestamp_s=1234.5,
        calibration_id=(
            config["calibration"]["id"]
            if calibration_id is None
            else calibration_id
        ),
        camera_serial=config["calibration"]["camera_serial"],
        model_name="AnyDexGrasp official representation + Inspire obj140 decision",
        representation_checkpoint_sha256="a" * 64,
        decision_checkpoint_sha256s=tuple(
            f"{index + 1:064x}" for index in range(8)
        ),
        official_source_commit="b" * 40,
    )
    save_snapshot_npz(path, snapshot)
    return snapshot


def test_missing_confirmations_cannot_reach_hardware_session(tmp_path, monkeypatch, capsys):
    app = _load("test_installed_rh56_missing_tokens")
    calls = []
    monkeypatch.setattr(app, "_run_hardware_session", lambda *args: calls.append(args))

    result = app.main(["run", "--output", str(tmp_path / "evidence.json")])

    assert result == 1
    assert calls == []
    assert "before hardware import" in capsys.readouterr().err
    assert not (tmp_path / "evidence.json").exists()


def test_wide_q6_and_coupled_each_require_an_extra_exact_token(
    tmp_path, monkeypatch, capsys
):
    app = _load("test_installed_rh56_wide_tokens")
    calls = []
    monkeypatch.setattr(app, "_run_hardware_session", lambda *args: calls.append(args))

    result = app.main(
        [
            "run",
            "--output",
            str(tmp_path / "wide.json"),
            "--target-q6",
            "646",
            *_tokens(app),
        ]
    )
    assert result == 1
    assert calls == []
    assert app.WIDE_Q6_TOKEN in capsys.readouterr().err

    result = app.main(
        [
            "run",
            "--output",
            str(tmp_path / "coupled.json"),
            "--coupled-air-close",
            *_tokens(app),
        ]
    )
    assert result == 1
    assert calls == []
    assert app.COUPLED_TOKEN in capsys.readouterr().err


def test_q6_below_900_requires_verified_stage1_before_hardware_import(
    tmp_path, monkeypatch, capsys
):
    app = _load("test_installed_rh56_stage1_dependency")
    calls = []
    monkeypatch.setattr(app, "_run_hardware_session", lambda *args: calls.append(args))

    common = [
        "run",
        "--output",
        str(tmp_path / "wide.json"),
        "--target-q6",
        "646",
        "--confirm-wide-q6",
        app.WIDE_Q6_TOKEN,
        *_tokens(app),
    ]
    assert app.main(common) == 1
    assert calls == []
    assert "--stage1-evidence" in capsys.readouterr().err

    stage1 = tmp_path / "failed-stage1.json"
    stage1.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        app,
        "build_stage1_prerequisite_binding",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Stage1 prerequisite is LOCKED: commissioning result is not pass")
        ),
    )
    assert app.main([*common, "--stage1-evidence", str(stage1)]) == 1
    assert calls == []
    assert "Stage1 prerequisite is LOCKED" in capsys.readouterr().err


def test_confirmed_run_writes_non_overwriting_evidence_and_never_changes_config(
    tmp_path, monkeypatch
):
    app = _load("test_installed_rh56_confirmed")
    output = tmp_path / "evidence.json"
    config_path = ROOT / "configs/fr3_rh56_v7_commissioning.json"
    config_before = config_path.read_bytes()
    calls = []

    def fake(args, config):
        calls.append((args, config))
        return _fake_runtime(args, config)

    monkeypatch.setattr(app, "_run_hardware_session", fake)
    monkeypatch.setattr(
        app,
        "verify_evidence",
        lambda *args, **kwargs: SimpleNamespace(passed=True, blockers=()),
    )
    result = app.main(
        ["run", "--output", str(output), *_tokens(app)]
    )

    assert result == 0
    assert len(calls) == 1
    assert config_path.read_bytes() == config_before
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["motion_authorized"] is False
    assert evidence["request"]["q6_waypoints"] == [975, 950, 925, 900]
    names = {item["name"] for item in evidence["source_bindings"]}
    assert {
        "rh56_hand_path",
        "rh56_reset_open",
        "actuator_to_joint_xlsx",
        "driver_to_angle_xls",
        "actuator_to_urdf_generator",
    } <= names
    assert output.stat().st_mode & 0o222 == 0

    # Existing evidence is rejected before the fake hardware session is called.
    assert app.main(["run", "--output", str(output), *_tokens(app)]) == 1
    assert len(calls) == 1


def test_snapshot_mode_auto_selects_best_score_with_stable_lowest_index_and_zero_is_explicit(
    tmp_path, monkeypatch
):
    app = _load("test_installed_rh56_snapshot_selection")
    snapshot_path = tmp_path / "official.npz"
    _official_snapshot(snapshot_path)
    calls = []

    def fake(args, config):
        calls.append((int(args.target_q6), tuple(args.bend_targets)))
        return _fake_runtime(args, config)

    monkeypatch.setattr(app, "_run_hardware_session", fake)
    monkeypatch.setattr(
        app,
        "verify_evidence",
        lambda *args, **kwargs: SimpleNamespace(passed=True, blockers=()),
    )
    common = [
        "--snapshot",
        str(snapshot_path),
        "--coupled-air-close",
        "--confirm-coupled-closure",
        app.COUPLED_TOKEN,
        *_tokens(app),
    ]

    auto_output = tmp_path / "auto.json"
    assert app.main(["run", "--output", str(auto_output), *common]) == 0
    auto = json.loads(auto_output.read_text(encoding="utf-8"))
    assert calls[0] == (950, (900, 901, 902, 903, 904))
    assert auto["snapshot_candidate"]["selection_method"] == (
        "highest_official_score_then_lowest_index"
    )
    assert auto["snapshot_candidate"]["candidate"]["index"] == 1
    assert auto["request"]["coupled_targets"] == [900, 901, 902, 903, 904, 950]
    assert auto["request"]["target_source"] == "official_snapshot_candidate"

    first_output = tmp_path / "first.json"
    assert (
        app.main(
            [
                "run",
                "--output",
                str(first_output),
                *common,
                "--candidate-index",
                "0",
            ]
        )
        == 0
    )
    first = json.loads(first_output.read_text(encoding="utf-8"))
    assert calls[1] == (1000, (910, 911, 912, 913, 914))
    assert first["snapshot_candidate"]["selection_method"] == (
        "explicit_candidate_index"
    )
    assert first["snapshot_candidate"]["candidate"]["index"] == 0


def test_snapshot_mode_rejects_manual_targets_and_requires_coupled_before_hardware(
    tmp_path, monkeypatch, capsys
):
    app = _load("test_installed_rh56_snapshot_exclusive")
    snapshot_path = tmp_path / "official.npz"
    _official_snapshot(snapshot_path)
    calls = []
    monkeypatch.setattr(app, "_run_hardware_session", lambda *args: calls.append(args))

    assert (
        app.main(
            [
                "run",
                "--output",
                str(tmp_path / "mixed.json"),
                "--snapshot",
                str(snapshot_path),
                "--target-q6",
                "950",
                "--coupled-air-close",
                *_tokens(app),
            ]
        )
        == 1
    )
    assert "cannot be combined" in capsys.readouterr().err
    assert (
        app.main(
            [
                "run",
                "--output",
                str(tmp_path / "uncoupled.json"),
                "--snapshot",
                str(snapshot_path),
                *_tokens(app),
            ]
        )
        == 1
    )
    assert "requires --coupled-air-close" in capsys.readouterr().err
    assert calls == []


def test_snapshot_calibration_and_targets_are_rejected_before_hardware_import(
    tmp_path, monkeypatch, capsys
):
    app = _load("test_installed_rh56_snapshot_preimport_validation")
    calls = []
    monkeypatch.setattr(app, "_run_hardware_session", lambda *args: calls.append(args))

    wrong_calibration = tmp_path / "wrong-calibration.npz"
    _official_snapshot(wrong_calibration, calibration_id="wrong")
    base = [
        "run",
        "--coupled-air-close",
        "--confirm-coupled-closure",
        app.COUPLED_TOKEN,
        *_tokens(app),
    ]
    assert (
        app.main(
            [
                *base,
                "--output",
                str(tmp_path / "wrong-calibration.json"),
                "--snapshot",
                str(wrong_calibration),
            ]
        )
        == 1
    )
    assert "calibration_id differs" in capsys.readouterr().err

    fractional = tmp_path / "fractional.npz"
    targets = np.full((3, 6), 900.0, dtype=np.float32)
    targets[1, 0] = 900.5
    _official_snapshot(fractional, targets=targets.tolist())
    assert (
        app.main(
            [
                *base,
                "--output",
                str(tmp_path / "fractional.json"),
                "--snapshot",
                str(fractional),
            ]
        )
        == 1
    )
    assert "integer registers" in capsys.readouterr().err
    assert calls == []


def test_snapshot_change_during_fake_run_is_detected_and_fails_evidence(
    tmp_path, monkeypatch
):
    app = _load("test_installed_rh56_snapshot_changed")
    snapshot_path = tmp_path / "official.npz"
    _official_snapshot(snapshot_path)
    output = tmp_path / "changed.json"

    def fake(args, config):
        runtime = _fake_runtime(args, config)
        snapshot_path.write_bytes(snapshot_path.read_bytes() + b"changed")
        return runtime

    monkeypatch.setattr(app, "_run_hardware_session", fake)
    assert (
        app.main(
            [
                "run",
                "--output",
                str(output),
                "--snapshot",
                str(snapshot_path),
                "--coupled-air-close",
                "--confirm-coupled-closure",
                app.COUPLED_TOKEN,
                *_tokens(app),
            ]
        )
        == 1
    )
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["result"]["status"] == "fail"
    assert "official snapshot changed" in evidence["result"]["operation_error"]


def test_target_zero_is_a_descending_small_step_path_not_a_direct_jump(
    tmp_path, monkeypatch
):
    app = _load("test_installed_rh56_target_zero")
    output = tmp_path / "zero.json"
    monkeypatch.setattr(app, "_run_hardware_session", _fake_runtime)
    monkeypatch.setattr(
        app,
        "verify_evidence",
        lambda *args, **kwargs: SimpleNamespace(passed=True, blockers=()),
    )
    stage1 = tmp_path / "stage1.json"
    stage1.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        app,
        "build_stage1_prerequisite_binding",
        lambda *args, **kwargs: {
            "kind": "installed_rh56_q6_stage1_pass_v1",
            "path": str(stage1),
            "file_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
            "run_id": "stage1",
            "completed_at_utc": "2026-07-18T00:00:00Z",
            "target_q6": 900,
            "control_profile_parsed_sha256": "c" * 64,
        },
    )

    result = app.main(
        [
            "run",
            "--output",
            str(output),
            "--target-q6",
            "0",
            "--q6-step",
            "25",
            "--stage1-evidence",
            str(stage1),
            "--confirm-wide-q6",
            app.WIDE_Q6_TOKEN,
            *_tokens(app),
        ]
    )

    assert result == 0
    evidence = json.loads(output.read_text(encoding="utf-8"))
    points = evidence["request"]["q6_waypoints"]
    assert points[0] == 975
    assert points[-1] == 0
    assert len(points) == 40
    assert all(a - b == 25 for a, b in zip(points, points[1:]))
    returned = evidence["request"]["q6_return_waypoints"]
    assert returned[0] == 50
    assert returned[-1] == 1000
    assert all(0 < b - a <= 50 for a, b in zip([0] + returned, returned))
    assert (
        evidence["request"]["q6_return_strategy"]
        == "canonical_reverse_bootstrap_v3"
    )
    assert evidence["stage1_prerequisite"]["target_q6"] == 900


def test_official_candidate_q6_1000_can_commission_exact_coupled_bends(
    tmp_path, monkeypatch
):
    app = _load("test_installed_rh56_target_1000")
    output = tmp_path / "q6_1000.json"
    monkeypatch.setattr(app, "_run_hardware_session", _fake_runtime)
    monkeypatch.setattr(
        app,
        "verify_evidence",
        lambda *args, **kwargs: SimpleNamespace(passed=True, blockers=()),
    )

    result = app.main(
        [
            "run",
            "--output",
            str(output),
            "--target-q6",
            "1000",
            "--coupled-air-close",
            "--bend-targets",
            "900",
            "900",
            "900",
            "900",
            "900",
            "--confirm-coupled-closure",
            app.COUPLED_TOKEN,
            *_tokens(app),
        ]
    )

    assert result == 0
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["request"]["target_q6"] == 1000
    assert evidence["request"]["q6_waypoints"] == [1000]
    assert evidence["request"]["q6_return_waypoints"] == [1000]
    assert evidence["request"]["coupled_targets"] == [900, 900, 900, 900, 900, 1000]


def test_candidate51_q6_return_uses_bootstrap_and_small_steps_not_direct_jump():
    app = _load("test_installed_rh56_candidate51_return")

    returned = app._requested_q6_return_waypoints(646, 25)

    assert returned == (
        696,
        721,
        746,
        771,
        796,
        821,
        846,
        871,
        896,
        921,
        946,
        971,
        996,
        1000,
    )
    assert all(0 < after - before <= 50 for before, after in zip((646,) + returned, returned))


def test_profile_aware_driver_keeps_bend_floor_separate_from_q6_endpoint():
    app = _load("test_installed_rh56_separate_open_floors")

    class FakeResetBase:
        def __init__(
            self,
            _hand,
            _api,
            *,
            q6_open_min_angle,
            open_min_angle,
            **_kwargs,
        ):
            self.q6_open_min_angle = q6_open_min_angle
            self.open_min_angle = open_min_angle
            self.feedback = None

        def _read_feedback(
            self,
            _phase,
            _started,
            *,
            allow_external_gate_bypass_after_disabled_readback=False,
        ):
            self.cleanup_bypass = (
                allow_external_gate_bypass_after_disabled_readback
            )
            return self.feedback

    Driver = app._build_installed_commissioning_driver_type(
        FakeResetBase,
        RuntimeError,
    )
    driver = Driver(object(), object(), q6_open_min_angle=975)
    assert driver.open_min_angle == 975
    assert driver.q6_open_min_angle == 975

    driver.feedback = SimpleNamespace(
        angles=(1000, 1000, 1000, 1000, 1000, 978)
    )
    assert driver._read_feedback("q6_sweep_preflight", 0.0) is driver.feedback

    driver.feedback = SimpleNamespace(
        angles=(1000, 1000, 979, 1000, 1000, 978)
    )
    with pytest.raises(RuntimeError, match="all five bend axes >= 980"):
        driver._read_feedback("coupled_air_close_preflight", 0.0)

    driver.feedback = SimpleNamespace(
        angles=(900, 900, 900, 900, 900, 450)
    )
    assert (
        driver._read_feedback("q6_return_adopt_verify", 0.0)
        is driver.feedback
    )
    driver.feedback = SimpleNamespace(
        angles=(1000, 1000, 1000, 1000, 1000, 978)
    )
    driver._read_feedback(
        "disable_verify",
        0.0,
        allow_external_gate_bypass_after_disabled_readback=True,
    )
    assert driver.cleanup_bypass is True


def test_fresh_commissioning_accepts_only_motionless_profile_open_q6():
    app = _load("test_installed_rh56_motionless_q6_gate")
    snapshot = {"angles": [1000, 998, 1000, 1000, 995, 978]}

    assert (
        app._require_fresh_commissioning_q6_open(
            snapshot, q6_open_min_angle=975
        )
        == 978
    )
    snapshot["angles"][5] = 974
    with pytest.raises(RuntimeError, match="evidence-bound recovery"):
        app._require_fresh_commissioning_q6_open(
            snapshot, q6_open_min_angle=975
        )


def test_hardware_session_disables_hand_before_constructing_franka(
    monkeypatch,
):
    """A stale RH56 target cannot remain active during Franka construction."""

    app = _load("test_installed_rh56_hardware_order")
    events = []

    class FakeRobot:
        def __init__(self, *_args):
            events.append("franka.connect")

        def read_once(self):
            events.append("franka.read")
            return object()

    pylibfranka = ModuleType("pylibfranka")
    pylibfranka.Robot = FakeRobot
    pylibfranka.RealtimeConfig = SimpleNamespace(kIgnore=object())
    monkeypatch.setitem(sys.modules, "pylibfranka", pylibfranka)

    franka_module = ModuleType("anydex_pipeline.franka_sequence_driver")

    class FakeLimits:
        def __init__(self, **_kwargs):
            pass

    class FakeArm:
        def __init__(self, robot, _backend, _limits):
            self.robot = robot

        def _validate_state(self, _state, **_kwargs):
            events.append("franka.validate")

    franka_module.FrankaMotionLimits = FakeLimits
    franka_module.FrankaSequenceDriver = FakeArm
    monkeypatch.setitem(
        sys.modules, "anydex_pipeline.franka_sequence_driver", franka_module
    )

    inspire_module = ModuleType("anydex_pipeline.inspire_sequence_driver")

    class FakeHand:
        def snapshot(self):
            return {
                "hand_id": 1,
                "angles": [1000, 998, 1000, 1000, 995, 978],
            }

    class FakeDriver:
        def __init__(self, hand=None, _api=None, **_kwargs):
            self.hand = FakeHand() if hand is None else hand
            self.telemetry = []
            self.disabled = False
            self.safety_check = None

        @classmethod
        def connect(cls, **kwargs):
            events.append("rh56.connect")
            return cls(
                FakeHand(),
                object(),
                q6_open_min_angle=kwargs["q6_open_min_angle"],
            )

        @property
        def feedback_envelope_policy(self):
            return app.rh56_feedback_envelope_policy(25)

        def adopt_disabled_state_and_verify(self):
            events.append("rh56.adopt_disable")
            self.disabled = True

        def install_external_safety_check(self, callback):
            assert self.disabled
            events.append("rh56.install_franka_gate")
            self.safety_check = callback

        def reset_to_open(self, **_kwargs):
            assert self.safety_check is not None
            events.append("rh56.reset_open")
            self.safety_check()
            return ()

        def commission_thumb_sweep(self, target, *, step_units, **_kwargs):
            events.append("rh56.q6_sweep")
            self.safety_check()
            return app._requested_q6_waypoints(target, step_units)

        def return_commissioned_thumb_to_open(self, **_kwargs):
            events.append("rh56.return_open")
            self.safety_check()
            return app._requested_q6_return_waypoints(900, 25)

        def disable_and_verify(self):
            events.append("rh56.final_disable")

        def close(self):
            events.append("rh56.close")

    inspire_module.RH56SequenceDriverError = RuntimeError
    monkeypatch.setitem(
        sys.modules, "anydex_pipeline.inspire_sequence_driver", inspire_module
    )
    reset_module = ModuleType("anydex_pipeline.rh56_reset_open")
    reset_module.RH56ResetOpenDriver = FakeDriver
    monkeypatch.setitem(
        sys.modules, "anydex_pipeline.rh56_reset_open", reset_module
    )
    monkeypatch.setattr(app, "_franka_state_json", lambda _state: {"ok": True})
    monkeypatch.setattr(
        app,
        "group_telemetry",
        lambda *_args, **_kwargs: {
            "all_feedback": [],
            "adopt_disable_feedback": [],
            "q6_steps": [],
            "q6_return_steps": [],
            "coupled_air_close_feedback": [],
            "coupled_air_close_steps": [],
            "coupled_air_return_steps": [],
            "final_open_feedback": None,
            "disable_feedback": [],
            "actual_q6_range": None,
            "actual_q6_return_range": None,
        },
    )

    config, _ = app.load_control_config(
        ROOT / "configs/fr3_rh56_v7_commissioning.json"
    )
    args = app.build_parser().parse_args(
        ["run", "--output", "/tmp/not-written-order-test.json", *_tokens(app)]
    )
    result = app._run_hardware_session(args, config)

    assert result["result"]["status"] == "pass"
    assert events.index("rh56.adopt_disable") < events.index("franka.connect")
    assert events.index("franka.validate") < events.index(
        "rh56.install_franka_gate"
    )
    assert events.index("rh56.install_franka_gate") < events.index(
        "rh56.reset_open"
    )


def test_failed_motion_distinguishes_verified_stop_from_unconfirmed_stop(
    tmp_path, monkeypatch, capsys
):
    app = _load("test_installed_rh56_stop_wording")

    safe = _fake_runtime(
        app.build_parser().parse_args(
            ["run", "--output", str(tmp_path / "unused.json"), *_tokens(app)]
        ),
        {},
    )
    safe["result"].update(
        status="fail",
        q6_return_pass=False,
        reopened_and_verified=False,
        operation_error="simulated acceptance timeout",
        stop_error=None,
        disabled_verified=True,
    )
    safe["final"]["disabled_verified"] = True
    monkeypatch.setattr(app, "_run_hardware_session", lambda *_args: safe)

    assert (
        app.main(
            [
                "run",
                "--output",
                str(tmp_path / "safe-failure.json"),
                *_tokens(app),
            ]
        )
        == 1
    )
    stderr = capsys.readouterr().err
    assert "[safe stop verified]" in stderr
    assert "STOP UNCONFIRMED" not in stderr

    unsafe = json.loads(json.dumps(safe))
    unsafe["result"]["disabled_verified"] = False
    unsafe["result"]["stop_error"] = "simulated serial loss"
    unsafe["final"]["disabled_verified"] = False
    monkeypatch.setattr(app, "_run_hardware_session", lambda *_args: unsafe)
    assert (
        app.main(
            [
                "run",
                "--output",
                str(tmp_path / "unsafe-failure.json"),
                *_tokens(app),
            ]
        )
        == 1
    )
    assert "[STOP UNCONFIRMED]" in capsys.readouterr().err


def test_materialize_config_update_creates_exact_read_only_sibling_without_overwrite(
    tmp_path, monkeypatch
):
    app = _load("test_installed_rh56_materialize_config")
    source = tmp_path / "profile.json"
    output = tmp_path / "profile_candidate51.json"
    original = {
        "schema_version": 1,
        "inspire": {
            "thumb_rotate_validated_realtime_range": [900, 1000],
            "six_axis_coupled_closure_commissioned": False,
            "commissioned_air_closure_targets": [],
            "open_speed": 40,
        },
        "tool": {"adapter_asset": "../assets/adapter/example.stl"},
    }
    source.write_text(json.dumps(original), encoding="utf-8")
    evidence = {
        "control_profile": {"snapshot": original},
        "integrity": {"payload_sha256": "a" * 64},
    }
    proposal = {
        "inspire.thumb_rotate_validated_realtime_range": [646, 1000],
        "inspire.six_axis_coupled_closure_commissioned": True,
        "inspire.commissioned_air_closure_targets": [
            [0, 358, 799, 911, 922, 646]
        ],
        "scope": "low_speed_unloaded_no_contact_PLA",
        "evidence_payload_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        app,
        "verify_evidence",
        lambda *_args, **_kwargs: SimpleNamespace(passed=True, blockers=()),
    )
    monkeypatch.setattr(app, "load_evidence", lambda _path: (evidence, Path(_path)))
    monkeypatch.setattr(app, "derived_config_proposal", lambda _evidence: proposal)

    def verify_materialized(_evidence, config, **_kwargs):
        value = json.loads(Path(config).read_text(encoding="utf-8"))
        assert value["inspire"]["thumb_rotate_validated_realtime_range"] == [646, 1000]
        assert value["inspire"]["six_axis_coupled_closure_commissioned"] is True
        assert value["inspire"]["commissioned_air_closure_targets"] == [
            [0, 358, 799, 911, 922, 646]
        ]
        assert value["inspire"]["open_speed"] == 40
        assert value["tool"] == original["tool"]
        return SimpleNamespace(passed=True, blockers=())

    monkeypatch.setattr(app, "verify_applied_config", verify_materialized)

    assert (
        app.main(
            [
                "materialize-config-update",
                "--evidence",
                str(tmp_path / "stage2.json"),
                "--config",
                str(source),
                "--output-config",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(source.read_text(encoding="utf-8")) == original
    assert output.is_file()
    assert output.stat().st_mode & 0o222 == 0

    # No-replace semantics protect an already reviewed derived profile.
    assert (
        app.main(
            [
                "materialize-config-update",
                "--evidence",
                str(tmp_path / "stage2.json"),
                "--config",
                str(source),
                "--output-config",
                str(output),
            ]
        )
        == 1
    )


def test_materialize_config_update_requires_passing_coupled_evidence_and_sibling_path(
    tmp_path, monkeypatch, capsys
):
    app = _load("test_installed_rh56_materialize_locked")
    source = tmp_path / "profile.json"
    source.write_text('{"inspire": {}}', encoding="utf-8")
    output = tmp_path / "derived.json"
    calls = []

    def locked(*_args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(passed=False, blockers=("coupled evidence missing",))

    monkeypatch.setattr(app, "verify_evidence", locked)
    assert (
        app.main(
            [
                "materialize-config-update",
                "--evidence",
                str(tmp_path / "failed.json"),
                "--config",
                str(source),
                "--output-config",
                str(output),
            ]
        )
        == 2
    )
    assert calls == [{"config_path": source.resolve(), "require_coupled": True}]
    assert not output.exists()
    assert "coupled evidence missing" in capsys.readouterr().err

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert (
        app.main(
            [
                "materialize-config-update",
                "--evidence",
                str(tmp_path / "failed.json"),
                "--config",
                str(source),
                "--output-config",
                str(elsewhere / "derived.json"),
            ]
        )
        == 1
    )
    assert not (elsewhere / "derived.json").exists()
