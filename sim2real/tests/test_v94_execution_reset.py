from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from dexgrasp.apps.reset_installed_rh56_open import RH56ResetOpenProof
from sim2real.deployment import execution_reset as reset


def _rh56_proof() -> RH56ResetOpenProof:
    return RH56ResetOpenProof(
        q6_waypoints=(),
        final_angles=(1000, 998, 1000, 1000, 995, 979),
        final_targets=(-1,) * 6,
        final_currents_ma=(0,) * 6,
        final_statuses=(2,) * 6,
        final_speeds=(1000,) * 6,
        final_forces_g=(500,) * 6,
    )


def _franka_proof() -> reset.FrankaV94HomeResetProof:
    target = (0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741)
    digest = hashlib.sha256(
        np.ascontiguousarray(np.asarray(target, dtype="<f8")).tobytes()
    ).hexdigest()
    return reset.FrankaV94HomeResetProof(
        target_q_rad=target,
        target_sha256_f64_le=digest,
        initial_linf_delta_rad=0.2,
        maximum_start_delta_rad=1.21,
        final_linf_error_rad=0.0001,
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += duration


def test_reset_order_cap_and_minimum_dwell(monkeypatch, tmp_path) -> None:
    events = []
    clock = _Clock()
    monkeypatch.setattr(
        reset,
        "_verify_v94_reset_profile_and_assets",
        lambda *_args: events.append("offline"),
    )

    def rh56(profile, **kwargs):
        events.append("rh56")
        assert profile == {"profile": "v94"}
        assert kwargs["max_axis_current_ma"] == 1000
        return _rh56_proof()

    def franka(profile, **kwargs):
        events.append("franka")
        assert profile == {"profile": "v94"}
        assert kwargs["maximum_start_delta_rad"] == pytest.approx(1.21)
        np.testing.assert_array_equal(
            kwargs["contract_q_home_rad"],
            np.asarray(_franka_proof().target_q_rad),
        )
        return _franka_proof()

    def sleep(duration):
        events.append("dwell")
        clock.sleep(duration)

    proof = reset.run_v94_execution_reset(
        profile={"profile": "v94"},
        profile_path=tmp_path / "profile.json",
        bundle_path=tmp_path / "deploy.zip",
        contract_q_home_rad=_franka_proof().target_q_rad,
        rh56_reset_runner=rh56,
        franka_reset_runner=franka,
        sleep=sleep,
        monotonic=clock.monotonic,
    )

    assert events == ["offline", "rh56", "franka", "dwell"]
    assert proof.dwell_requested_s == pytest.approx(0.5)
    assert proof.dwell_elapsed_s >= 0.5


def test_thrown_force500_uses_reset_local_80g_without_mutating_runtime_profile(
    monkeypatch, tmp_path
) -> None:
    events = []
    clock = _Clock()
    profile = {
        "profile_id": "fr3_rh56_v60_palmcatch_single_tick_v1",
        "inspire": {"force_limit_g": 500},
    }
    monkeypatch.setattr(
        reset, "_verify_v94_reset_profile_and_assets", lambda *_args: None
    )
    monkeypatch.setattr(
        reset, "load_commissioned_rh56_force_set_g", lambda _profile: 500
    )

    def rh56(reset_profile, **kwargs):
        events.append("rh56")
        assert reset_profile is not profile
        assert reset_profile["inspire"]["force_limit_g"] == 80
        assert kwargs["max_axis_current_ma"] == 1000
        return _rh56_proof()

    def franka(runtime_profile, **_kwargs):
        events.append("franka")
        assert runtime_profile is profile
        assert runtime_profile["inspire"]["force_limit_g"] == 500
        return _franka_proof()

    reset.run_v94_execution_reset(
        profile=profile,
        profile_path=tmp_path / "profile.json",
        bundle_path=tmp_path / "deploy.zip",
        contract_q_home_rad=_franka_proof().target_q_rad,
        rh56_reset_runner=rh56,
        franka_reset_runner=franka,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert profile["inspire"]["force_limit_g"] == 500
    assert events == ["rh56", "franka"]


@pytest.mark.parametrize(
    ("profile_tolerance", "expected_tolerance"),
    (
        (0.02, reset.RESET_FRANKA_ARRIVAL_TOLERANCE_RAD),
        (0.001, 0.001),
    ),
)
def test_default_franka_reset_uses_tight_v94_runtime_only_tolerance(
    monkeypatch,
    tmp_path,
    profile_tolerance: float,
    expected_tolerance: float,
) -> None:
    target = np.asarray(_franka_proof().target_q_rad, dtype=np.float64)
    digest = hashlib.sha256(
        np.ascontiguousarray(target.astype("<f8")).tobytes()
    ).hexdigest()
    source = {
        "franka": {
            "default_q_rad": target.tolist(),
            "default_arrival_tolerance_rad": profile_tolerance,
        }
    }
    runtime_config = {
        "franka": {
            "default_q_rad": target.tolist(),
            "default_arrival_tolerance_rad": profile_tolerance,
        }
    }
    observed = {}

    monkeypatch.setattr(
        reset.franka_training_reset,
        "_training_target",
        lambda _path: (target.copy(), {"q_home_sha256_f64_le": digest}),
    )
    monkeypatch.setattr(
        reset.franka_training_reset,
        "_reset_config",
        lambda config, wanted: runtime_config,
    )

    def run_reset(config, *, maximum_start_delta_rad):
        observed["config"] = config
        observed["maximum_start_delta_rad"] = maximum_start_delta_rad
        return SimpleNamespace(
            initial_linf_delta_rad=0.2,
            final_linf_error_rad=0.0001,
        )

    monkeypatch.setattr(
        reset.franka_training_reset.DEFAULT_RESET,
        "run_reset",
        run_reset,
    )

    proof = reset._default_franka_reset(
        source,
        bundle_path=tmp_path / "deploy.zip",
        contract_q_home_rad=target,
        maximum_start_delta_rad=1.21,
    )

    assert source["franka"]["default_arrival_tolerance_rad"] == profile_tolerance
    assert (
        observed["config"]["franka"]["default_arrival_tolerance_rad"]
        == pytest.approx(expected_tolerance)
    )
    assert observed["maximum_start_delta_rad"] == pytest.approx(1.21)
    assert observed["config"]["franka"][
        "default_max_joint_velocity_rad_s"
    ] == pytest.approx(reset.RESET_FRANKA_MAX_JOINT_SPEED_RAD_S)
    assert observed["config"]["franka"][
        "default_min_duration_s"
    ] == pytest.approx(reset.RESET_FRANKA_MIN_SEGMENT_DURATION_S)
    assert proof.final_linf_error_rad == pytest.approx(0.0001)


def test_default_franka_reset_uses_sha_bound_task_target_not_bundle_home(
    monkeypatch, tmp_path
) -> None:
    task_target = np.asarray(
        [0.0, -1.2, 0.0, -2.2, 0.0, 1.4, np.pi / 4.0],
        dtype=np.float64,
    )
    bundle_target = np.asarray(_franka_proof().target_q_rad, dtype=np.float64)
    observed = {}
    runtime_config = {
        "franka": {
            "default_q_rad": task_target.tolist(),
            "default_arrival_tolerance_rad": 0.02,
        }
    }

    monkeypatch.setattr(
        reset.franka_training_reset,
        "_training_target",
        lambda _path: (bundle_target.copy(), {"source": "verified bundle"}),
    )

    def reset_config(config, wanted):
        np.testing.assert_array_equal(wanted, task_target)
        observed["input"] = config
        return runtime_config

    monkeypatch.setattr(reset.franka_training_reset, "_reset_config", reset_config)
    monkeypatch.setattr(
        reset.franka_training_reset.DEFAULT_RESET,
        "run_reset",
        lambda config, *, maximum_start_delta_rad: SimpleNamespace(
            initial_linf_delta_rad=0.1,
            final_linf_error_rad=0.001,
        ),
    )

    proof = reset._default_franka_reset(
        {"franka": {"default_q_rad": task_target.tolist()}},
        bundle_path=tmp_path / "deploy.zip",
        contract_q_home_rad=task_target,
        maximum_start_delta_rad=1.21,
    )

    assert observed["input"]["franka"]["default_q_rad"] == task_target.tolist()
    np.testing.assert_array_equal(proof.target_q_rad, task_target)


def test_reset_profile_accepts_exact_float32_roundtrip_of_decimal_home(
    monkeypatch, tmp_path
) -> None:
    profile_target = np.asarray(
        [0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741], dtype=np.float64
    )
    bundle_target = profile_target.astype(np.float32).astype(np.float64)
    profile = {
        "franka": {
            "default_q_rad": profile_target.tolist(),
            "default_q_provenance": {"evidence_artifact": "legacy.json"},
        }
    }
    observed = {}
    monkeypatch.setattr(
        reset,
        "verify_adapter_assets",
        lambda value, source: observed.update(value=value, source=source),
    )

    reset._verify_v94_reset_profile_and_assets(
        profile, tmp_path / "profile.json", bundle_target
    )

    assert observed["source"] == tmp_path / "profile.json"
    assert observed["value"]["franka"]["default_q_provenance"][
        "evidence_artifact"
    ] is None


def test_reset_profile_rejects_representable_q_home_change(
    monkeypatch, tmp_path
) -> None:
    profile_target = np.asarray(
        [0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741], dtype=np.float64
    )
    changed = profile_target.astype(np.float32).astype(np.float64)
    changed[1] += 1.0e-4
    profile = {
        "franka": {
            "default_q_rad": profile_target.tolist(),
            "default_q_provenance": {"evidence_artifact": "legacy.json"},
        }
    }
    monkeypatch.setattr(reset, "verify_adapter_assets", lambda *_args: None)

    with pytest.raises(
        reset.V94ExecutionResetError,
        match="default_q_rad differs from the prepared task q_home",
    ):
        reset._verify_v94_reset_profile_and_assets(
            profile, tmp_path / "profile.json", changed
        )


@pytest.mark.parametrize("failure_phase", ("rh56", "franka"))
def test_reset_failure_prevents_later_phases(
    monkeypatch, tmp_path, failure_phase: str
) -> None:
    events = []
    monkeypatch.setattr(
        reset, "_verify_v94_reset_profile_and_assets", lambda *_args: None
    )

    def rh56(*_args, **_kwargs):
        events.append("rh56")
        if failure_phase == "rh56":
            raise RuntimeError("thumb_rotate current 106mA exceeded strict cap 100mA")
        return _rh56_proof()

    def franka(*_args, **_kwargs):
        events.append("franka")
        if failure_phase == "franka":
            raise RuntimeError("Franka STOP UNCONFIRMED")
        return _franka_proof()

    with pytest.raises(RuntimeError):
        reset.run_v94_execution_reset(
            profile={},
            profile_path=tmp_path / "profile.json",
            bundle_path=tmp_path / "deploy.zip",
            contract_q_home_rad=_franka_proof().target_q_rad,
            rh56_reset_runner=rh56,
            franka_reset_runner=franka,
            sleep=lambda _duration: events.append("dwell"),
        )

    expected = ["rh56"] if failure_phase == "rh56" else ["rh56", "franka"]
    assert events == expected


def test_short_dwell_is_rejected_after_both_verified_resets(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        reset, "_verify_v94_reset_profile_and_assets", lambda *_args: None
    )
    clock = _Clock()

    with pytest.raises(reset.V94ExecutionResetError, match="shorter than 0.5s"):
        reset.run_v94_execution_reset(
            profile={},
            profile_path=tmp_path / "profile.json",
            bundle_path=tmp_path / "deploy.zip",
            contract_q_home_rad=_franka_proof().target_q_rad,
            rh56_reset_runner=lambda *_args, **_kwargs: _rh56_proof(),
            franka_reset_runner=lambda *_args, **_kwargs: _franka_proof(),
            sleep=lambda _duration: clock.sleep(0.49),
            monotonic=clock.monotonic,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("nan_requested", "stationary dwell"),
        ("nan_elapsed", "stationary dwell"),
        ("wrong_target", "target differs"),
        ("wrong_hash", "SHA-256"),
        ("over_start_envelope", "start envelope"),
        ("over_reset_arrival_envelope", "reset arrival envelope"),
    ),
)
def test_reset_proof_rejects_nonfinite_or_inconsistent_contract(
    mutation: str, message: str
) -> None:
    rh56 = _rh56_proof()
    franka = _franka_proof()
    requested = 0.5
    elapsed = 0.5
    if mutation == "nan_requested":
        requested = float("nan")
    elif mutation == "nan_elapsed":
        elapsed = float("nan")
    elif mutation == "wrong_target":
        franka = reset.FrankaV94HomeResetProof(
            **{
                **franka.__dict__,
                "target_q_rad": (0.1,) + franka.target_q_rad[1:],
            }
        )
    elif mutation == "wrong_hash":
        franka = reset.FrankaV94HomeResetProof(
            **{**franka.__dict__, "target_sha256_f64_le": "0" * 64}
        )
    elif mutation == "over_start_envelope":
        franka = reset.FrankaV94HomeResetProof(
            **{**franka.__dict__, "initial_linf_delta_rad": 1.210001}
        )
    elif mutation == "over_reset_arrival_envelope":
        franka = reset.FrankaV94HomeResetProof(
            **{
                **franka.__dict__,
                "final_linf_error_rad": 0.0102506,
            }
        )
    proof = reset.V94ExecutionResetProof(
        rh56=rh56,
        franka=franka,
        dwell_requested_s=requested,
        dwell_elapsed_s=elapsed,
    )

    with pytest.raises(reset.V94ExecutionResetError, match=message):
        reset.validate_v94_execution_reset_proof(
            proof,
            contract_q_home_rad=_franka_proof().target_q_rad,
            maximum_start_delta_rad=1.21,
        )


def test_reset_proof_accepts_exact_v94_arrival_boundary() -> None:
    franka = reset.FrankaV94HomeResetProof(
        **{
            **_franka_proof().__dict__,
            "final_linf_error_rad": (
                reset.RESET_FRANKA_ARRIVAL_TOLERANCE_RAD
            ),
        }
    )
    proof = reset.V94ExecutionResetProof(
        rh56=_rh56_proof(),
        franka=franka,
        dwell_requested_s=0.5,
        dwell_elapsed_s=0.5,
    )

    assert (
        reset.validate_v94_execution_reset_proof(
            proof,
            contract_q_home_rad=_franka_proof().target_q_rad,
            maximum_start_delta_rad=1.21,
        )
        is proof
    )
