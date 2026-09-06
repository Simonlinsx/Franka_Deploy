from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from robot_control.rh56.linux_transport import (
    RH56_COMPACT_READ_ATTEMPTS,
    RH56_COMPACT_READ_TOTAL_TIMEOUT_S,
)
from robot_control.rh56.actuator import (
    RH56_SUPERVISED_EXCHANGE_TIMEOUT_S,
    RH56_SUPERVISED_WRITE_RESPONSE_GRACE_S,
)
from sim2real.deployment.runner import (
    RH56_COMMAND_MAX_AGE_S,
    RH56_EXACT_READBACK_REQUEST_COUNT,
    RH56_EXACT_READBACK_TIMEOUT_S,
    RH56_WRITE_RESPONSE_GRACE_S,
    SupervisedRequest,
    main,
)
from sim2real.runtime.supervised_v94_runtime import RH56_WATCHDOG_S


def _fake_irq_host(tmp_path: Path, *, requested: str = "8", effective: str = "8"):
    route_table = tmp_path / "proc/net/route"
    route_table.parent.mkdir(parents=True)
    route_table.write_text(
        "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n"
        "eno1 00000000 00000000 0001 0 0 500 00000000 0 0 0\n"
        "franka0 000010AC 00000000 0001 0 0 10 00FFFFFF 0 0 0\n"
    )
    sys_class_net = tmp_path / "sys/class/net"
    (sys_class_net / "franka0/device/msi_irqs").mkdir(parents=True)
    (sys_class_net / "franka0/device/msi_irqs/127").touch()
    proc_root = tmp_path / "proc"
    proc_irq = proc_root / "irq"
    proc_interrupts = proc_root / "interrupts"
    proc_interrupts.write_text(
        " ".join(f"CPU{cpu}" for cpu in range(12)) + "\n"
        + "127: "
        + " ".join("100" if cpu == 8 else "0" for cpu in range(12))
        + " PCI-MSI franka0\n"
    )
    (proc_irq / "127").mkdir(parents=True)
    (proc_irq / "127/effective_affinity_list").write_text(effective)
    (proc_irq / "127/smp_affinity_list").write_text(requested)
    cpu_topology = tmp_path / "sys/devices/system/cpu"
    for cpu in range(12):
        root = cpu_topology / f"cpu{cpu}/topology"
        root.mkdir(parents=True)
        (root / "physical_package_id").write_text("0")
        (root / "core_id").write_text(str(cpu // 2))
    return (
        route_table,
        sys_class_net,
        proc_root,
        proc_irq,
        proc_interrupts,
        cpu_topology,
    )


def _fake_irq_guard(tmp_path: Path) -> tuple[Path, Path]:
    guard_directory = tmp_path / "run/franka-nic-irq-guard"
    guard_directory.mkdir(parents=True, mode=0o755)
    guard_directory.chmod(0o755)
    state = guard_directory / "state"
    state.write_text(
        "schema 2\n"
        "interface franka0\n"
        "pinned_cpu 8\n"
        "irqbalance_runtime_masked 0\n"
        "irqbalance_was_active 1\n"
        "irq 127 0-11 0-11\n"
    )
    state.chmod(0o644)
    runtime_mask = tmp_path / "run/systemd/system/irqbalance.service"
    runtime_mask.parent.mkdir(parents=True)
    runtime_mask.symlink_to("/dev/null")
    return state, runtime_mask


def test_frozen_nic_guard_binds_root_helper_state_and_immediate_irq_token(
    tmp_path,
):
    import sim2real.deployment.runner as cli

    route, sys_net, proc_root, proc_irq, _interrupts, _topology = (
        _fake_irq_host(tmp_path)
    )
    state, runtime_mask = _fake_irq_guard(tmp_path)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"franka": {"ip": "172.16.0.2"}}))
    stability_payload = {
        "interface": "franka0",
        "irq_numbers": [127],
        "effective_cpus": [8],
        "requested_cpus": [8],
    }
    import hashlib

    request = SupervisedRequest(
        run_id="frozen-irq-guard",
        steps=1,
        bundle=tmp_path / "deploy.zip",
        profile=profile,
        pcd_config=tmp_path / "pcd.yaml",
        execute=True,
        franka_nic_interface="franka0",
        franka_nic_irq_numbers=(127,),
        franka_nic_irq_cpus=(8,),
        franka_nic_irq_requested_cpus=(8,),
        franka_nic_irq_stability_token=hashlib.sha256(
            json.dumps(
                stability_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    )
    cli._verify_frozen_franka_nic_admission(
        request,
        route_table=route,
        sys_class_net=sys_net,
        proc_irq=proc_irq,
        proc_root=proc_root,
        irqbalance_pid_file=tmp_path / "run/irqbalance.pid",
        guard_state_path=state,
        runtime_mask_path=runtime_mask,
        guard_required_owner_uid=os.getuid(),
        guard_required_owner_gid=os.getgid(),
    )

    (proc_irq / "127/effective_affinity_list").write_text("10")
    with pytest.raises(
        cli.SupervisedV94RunError,
        match="not exclusively/stably pinned|changed",
    ):
        cli._verify_frozen_franka_nic_admission(
            request,
            route_table=route,
            sys_class_net=sys_net,
            proc_irq=proc_irq,
            proc_root=proc_root,
            irqbalance_pid_file=tmp_path / "run/irqbalance.pid",
            guard_state_path=state,
            runtime_mask_path=runtime_mask,
            guard_required_owner_uid=os.getuid(),
            guard_required_owner_gid=os.getgid(),
        )


def test_cpu_partition_reserves_stably_pinned_nic_and_idle_servo_sibling(
    monkeypatch, tmp_path
):
    import sim2real.deployment.runner as cli

    route, sys_net, proc_root, proc_irq, interrupts, topology = _fake_irq_host(
        tmp_path
    )
    # Make CPU0 busier than CPU1 and the last physical core busier than the
    # other candidates. The exact servo logical CPU must be the quieter
    # sibling of the least-loaded non-NIC physical core.
    for irq, cpu in (
        (200, 0),
        (201, 10),
        (202, 11),
        (203, 2),
        (204, 4),
        (205, 6),
    ):
        root = proc_irq / str(irq)
        root.mkdir()
        (root / "effective_affinity_list").write_text(str(cpu))
    interrupts.write_text(
        interrupts.read_text()
        + "200: "
        + " ".join("99999999" if cpu == 0 else "0" for cpu in range(12))
        + " PCI-MSI nvme0q0\n"
        + "201: "
        + " ".join("1000" if cpu == 10 else "0" for cpu in range(12))
        + " PCI-MSI misc-a\n"
        + "202: "
        + " ".join("1000" if cpu == 11 else "0" for cpu in range(12))
        + " PCI-MSI misc-b\n"
        + "203: "
        + " ".join("10" if cpu == 2 else "0" for cpu in range(12))
        + " PCI-MSI misc-c\n"
        + "204: "
        + " ".join("20" if cpu == 4 else "0" for cpu in range(12))
        + " PCI-MSI misc-d\n"
        + "205: "
        + " ".join("30" if cpu == 6 else "0" for cpu in range(12))
        + " PCI-MSI misc-e\n"
    )
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(12)))

    partition = cli._deployment_cpu_partition(
        robot_ip="172.16.0.2",
        reserve_visualization=True,
        route_table=route,
        sys_class_net=sys_net,
        proc_irq=proc_irq,
        proc_interrupts=interrupts,
        proc_root=proc_root,
        irqbalance_pid_file=tmp_path / "run/irqbalance.pid",
        cpu_topology=topology,
    )

    assert partition.nic_interface == "franka0"
    assert partition.nic_irq_numbers == (127,)
    assert partition.nic_irq_cpus == (8,)
    assert partition.nic_irq_requested_cpus == (8,)
    assert partition.nic_reserved_cpus == (8, 9)
    assert partition.servo_sibling_cpus == (2, 3)
    assert partition.servo_cpu == 3
    assert partition.servo_idle_sibling_cpus == (2,)
    assert partition.visualization_cpus == (4, 5)
    assert set(partition.parent_cpus) == {0, 1, 6, 7, 10, 11}
    assert len(partition.nic_irq_stability_token) == 64


def test_cpu_partition_refuses_active_irqbalance_before_any_robot_access(
    monkeypatch, tmp_path
):
    import sim2real.deployment.runner as cli

    route, sys_net, proc_root, proc_irq, interrupts, topology = _fake_irq_host(
        tmp_path
    )
    (proc_root / "42").mkdir()
    (proc_root / "42/comm").write_text("irqbalance\n")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(12)))

    with pytest.raises(cli.SupervisedV94RunError, match="irqbalance is active"):
        cli._deployment_cpu_partition(
            robot_ip="172.16.0.2",
            route_table=route,
            sys_class_net=sys_net,
            proc_irq=proc_irq,
            proc_interrupts=interrupts,
            proc_root=proc_root,
            irqbalance_pid_file=tmp_path / "run/irqbalance.pid",
            cpu_topology=topology,
        )


def test_cpu_partition_never_places_servo_on_current_xhci_irq_core(
    monkeypatch, tmp_path
):
    import sim2real.deployment.runner as cli

    route, sys_net, proc_root, proc_irq, interrupts, topology = _fake_irq_host(
        tmp_path
    )
    (proc_irq / "200").mkdir()
    (proc_irq / "200/effective_affinity_list").write_text("2")
    interrupts.write_text(
        interrupts.read_text()
        + "200: "
        + " ".join("1" if cpu == 2 else "0" for cpu in range(12))
        + " PCI-MSI xhci_hcd\n"
    )
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(12)))

    partition = cli._deployment_cpu_partition(
        robot_ip="172.16.0.2",
        route_table=route,
        sys_class_net=sys_net,
        proc_irq=proc_irq,
        proc_interrupts=interrupts,
        proc_root=proc_root,
        irqbalance_pid_file=tmp_path / "run/irqbalance.pid",
        cpu_topology=topology,
    )
    assert set(partition.servo_sibling_cpus).isdisjoint({2, 3})


def test_cpu_partition_refuses_requested_effective_irq_drift(
    monkeypatch, tmp_path
):
    import sim2real.deployment.runner as cli

    route, sys_net, proc_root, proc_irq, interrupts, topology = _fake_irq_host(
        tmp_path, requested="8", effective="10"
    )
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(12)))

    with pytest.raises(cli.SupervisedV94RunError, match="not exclusively/stably pinned"):
        cli._deployment_cpu_partition(
            robot_ip="172.16.0.2",
            route_table=route,
            sys_class_net=sys_net,
            proc_irq=proc_irq,
            proc_interrupts=interrupts,
            proc_root=proc_root,
            irqbalance_pid_file=tmp_path / "run/irqbalance.pid",
            cpu_topology=topology,
        )


def test_irq_guard_script_has_syntax_and_explicit_restore_path() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "dexgrasp/scripts/configure_franka_nic_irq.sh"
    )
    lifecycle_test = script.with_name(
        "test_configure_franka_nic_irq_no_hardware.sh"
    )
    assert script.is_file() and os.access(script, os.X_OK)
    assert lifecycle_test.is_file() and os.access(lifecycle_test, os.X_OK)
    subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(["bash", "-n", str(lifecycle_test)], check=True)
    subprocess.run([str(lifecycle_test)], check=True)
    source = script.read_text()
    assert "systemctl mask --runtime --now irqbalance.service" in source
    assert "systemctl unmask --runtime irqbalance.service" in source
    assert "systemctl mask --runtime --now irqbalance.service" in source
    assert "systemctl unmask --runtime irqbalance.service" in source
    assert "schema $STATE_SCHEMA" in source
    assert "chmod 0644" in source
    assert 'STATE_SCHEMA="2"' in source
    assert 'SYSTEMD_RUNTIME_UNIT="/run/systemd/system/irqbalance.service"' in source
    assert "restore_internal" in source
    assert "trap " in source
    assert "effective_affinity_list" in source


def test_compute_isolation_rechecks_pre_reset_irq_token_before_runtime(
    monkeypatch, tmp_path
):
    import sim2real.deployment.runner as cli

    request = SupervisedRequest(
        run_id="irq-toctou",
        steps=1,
        bundle=tmp_path / "deploy.zip",
        profile=tmp_path / "profile.json",
        pcd_config=tmp_path / "pcd.yaml",
        execute=True,
        franka_servo_cpu=2,
        franka_servo_sibling_cpus=(2, 3),
        franka_nic_irq_stability_token="a" * 64,
    )
    monkeypatch.setattr(cli, "_profile_franka_ip", lambda _path: "172.16.0.2")
    monkeypatch.setattr(
        cli, "_verify_frozen_franka_nic_admission", lambda _request: None
    )
    monkeypatch.setattr(
        cli,
        "_deployment_cpu_partition",
        lambda **_kwargs: SimpleNamespace(
            nic_irq_stability_token="b" * 64,
            servo_cpu=2,
            servo_sibling_cpus=(2, 3),
        ),
    )
    with pytest.raises(
        cli.SupervisedV94RunError,
        match="changed after preflight",
    ):
        with cli._deployment_compute_isolation(request):
            pytest.fail("runtime must not start after IRQ affinity drift")


def test_help_preserves_argparse_success_exit(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--help"])

    assert raised.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert "FAILED" not in captured.err


def test_cli_rh56_timeout_summary_matches_transport_and_actuator_contracts():
    assert (
        RH56_WRITE_RESPONSE_GRACE_S
        == RH56_SUPERVISED_WRITE_RESPONSE_GRACE_S
        == 0.020
    )
    assert RH56_EXACT_READBACK_REQUEST_COUNT == RH56_COMPACT_READ_ATTEMPTS == 1
    assert (
        RH56_EXACT_READBACK_TIMEOUT_S
        == RH56_COMPACT_READ_TOTAL_TIMEOUT_S
        == RH56_SUPERVISED_EXCHANGE_TIMEOUT_S
        == RH56_COMMAND_MAX_AGE_S
        == RH56_WATCHDOG_S
        == 0.050
    )


def test_default_is_hardware_inert_dry_run(capsys):
    called = []
    assert main(["--steps", "5"], real_runner=lambda request: called.append(request)) == 0
    assert called == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["hardware_access"] is False
    assert payload["steps"] == 5
    assert payload["policy_mode"] == {
        "checkpoint_control_dt_must_match": True,
        "control_dt_s": 1.0 / 60.0,
        "name": "60hz",
        "policy_rate_hz": 60.0,
    }
    assert payload["hard_step_cap"] == 720
    assert payload["hard_guards"]["franka_maximum_session_duration_s"] == 15.0
    assert payload["hard_guards"]["franka_command_speed_rad_s"] == 0.5
    assert payload["hard_guards"]["franka_command_acceleration_rad_s2"] == 5.0
    assert payload["hard_guards"]["franka_command_jerk_rad_s3"] == 250.0
    assert (
        payload["hard_guards"]["franka_measured_velocity_fault_rad_s"]
        == 0.7
    )
    assert payload["hard_guards"]["franka_episode_delta_rad"] == 1.21
    assert (
        payload["hard_guards"]["franka_fci_max_recoverable_control_period_s"]
        == 0.021
    )
    assert payload["hard_guards"]["franka_parent_heartbeat_receipt_timeout_s"] == 0.1
    assert payload["hard_guards"]["franka_policy_target_max_age_s"] == 0.05
    assert payload["hard_guards"]["franka_policy_inter_target_timeout_s"] == 0.5
    assert (
        payload["hard_guards"]["franka_observation_action_max_age_s"]
        == 0.040
    )
    assert payload["hard_guards"]["franka_observation_hard_max_age_s"] == 0.05
    assert payload["hard_guards"]["rh56_command_max_age_s"] == 0.05
    assert payload["hard_guards"]["rh56_inter_command_watchdog_s"] == 0.5
    assert payload["hard_guards"]["rh56_stop_timeout_s"] == 5.0
    assert payload["hard_guards"]["camera_runtime_frame_timeout_s"] == 0.1
    assert payload["hard_guards"]["camera_formal_publication_stall_s"] == 0.4
    assert (
        payload["hard_guards"]["object_roi_preflight_max_valid_mask_gap_s"]
        == 0.4
    )
    assert payload["hard_guards"]["object_pointcloud_max_age_s"] == 0.2
    assert (
        payload["hard_guards"][
            "maximum_consecutive_no_stage_observation_hold_s"
        ]
        == 0.4
    )
    assert "guarded_recoverable_fail_closed" in (
        payload["hard_guards"]["stale_palm_contract"]
    )
    assert "explicit_object_mask_mode_legacy" in (
        payload["hard_guards"]["stale_palm_contract"]
    )
    assert "honor_enabled_config" in (
        payload["hard_guards"]["formal_online_sam2"]
    )
    assert payload["hard_guards"]["rh56_speed_set"] == [
        600,
        600,
        600,
        600,
        600,
        600,
    ]
    assert payload["sim_control_alignment"]["contract_version"] == (
        "inspire_v202_20hz_adapter_v7_20260727"
    )
    assert payload["sim_control_alignment"]["rh56"]["speed_set"] == [
        600,
        600,
        600,
        600,
        600,
        600,
    ]
    assert payload["sim_control_alignment"]["rh56"]["response_profile"] == (
        "uniform_speed_600_identified_free_space"
    )
    assert payload["hard_guards"]["rh56_force_set_g"] == 80
    assert payload["hard_guards"]["rh56_running_current_ma"] == 1400
    assert (
        payload["hard_guards"]["rh56_stop_settle_transient_current_ma"]
        == 1000
    )
    assert payload["hard_guards"]["rh56_post_disable_idle_current_ma"] == 100
    assert payload["hard_guards"]["rh56_target_schedule_rate_hz"] == 20.0
    assert payload["hard_guards"]["rh56_target_schedule_semantics"] == (
        "20hz_latest_legal_absolute_target;"
        "v94_three_policy_tick_transparent_envelope"
    )
    assert payload["hard_guards"][
        "rh56_target_contract_envelope_units_per_update"
    ] == [
        137,
        143,
        158,
        158,
        251,
        120,
    ]
    assert payload["hard_guards"]["rh56_nominal_feedback_rate_hz"] == 20.0
    assert (
        payload["hard_guards"]["rh56_feedback_sample_hold_max_age_s"]
        == 0.150
    )
    assert payload["hard_guards"]["rh56_target_slew_units_per_update"] == [
        137,
        143,
        158,
        158,
        251,
        120,
    ]
    assert payload["hard_guards"]["rh56_feedback_to_command_offset_units"] == [
        0,
        0,
        0,
        0,
        0,
        15,
    ]
    assert payload["hard_guards"]["rh56_disabled_open_tolerance_units"] == [
        25,
        25,
        25,
        25,
        25,
        26,
    ]
    assert payload["hard_guards"]["rh56_tracking_significant_gap_units"] == 50
    assert payload["hard_guards"]["rh56_write_response_grace_s"] == 0.020
    assert payload["hard_guards"]["rh56_exact_readback_request_count"] == 1
    assert payload["hard_guards"]["rh56_exact_readback_timeout_s"] == 0.050
    assert payload["hard_guards"]["rh56_tracking_min_progress_units"] == 3
    assert payload["hard_guards"]["rh56_tracking_timeout_s"] == 0.750
    assert payload["hard_guards"]["camera_max_policy_actions_per_frame"] == 6
    assert payload["hard_guards"]["policy_schedule"] == "legacy_minimum_period"
    assert (
        payload["hard_guards"]["qd_g015_startup_non_actuated_policy_steps"]
        == 0
    )
    assert (
        payload["automatic_reset"]["franka_arrival_linf_tolerance_rad"]
        == 0.005
    )
    assert (
        payload["automatic_reset"]["franka_arrival_linf_tolerance_rad"]
        < payload["hard_guards"]["franka_start_linf_from_v94_home_rad"]
    )
    assert payload["live_visualization"]["requested"] is False
    assert payload["object_roi"]["mode"] == "pinned_fixed_roi"
    assert payload["object_roi"]["xywh"] is None
    assert payload["object_mask"] == {
        "mode": "guarded",
        "provider_publication_mode": "adaptive_fusion",
        "recovery_publication_mode": "unified_three_evidence",
        "legacy_comparison_available": True,
        "requested_object_mask_mode": "guarded",
        "effective_object_mask_mode": "guarded_v2",
        "effective_provider_mask_publication_mode": "guarded_sam2_primary",
        "effective_provider_recovery_publication_mode": (
            "unified_three_evidence"
        ),
        "stale_palm_policy": (
            "recoverable_fail_closed_no_policy_history_mapper_stage"
        ),
        "stale_palm_legacy_compatibility_switch": (
            "explicit_--object-mask-mode=legacy_only"
        ),
    }
    assert "confirmation_phrase" not in payload


def test_closed_loop_arrival_gate_is_explicit_in_dry_run_summary(capsys):
    called = []
    assert main(
        [
            "--steps",
            "5",
            "--arrival-gated",
            "--arrival-tolerance-rad",
            "0.004",
            "--arrival-timeout-s",
            "0.30",
        ],
        real_runner=lambda request: called.append(request),
    ) == 0
    assert called == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["closed_loop_arrival_gate"] == {
        "enabled": True,
        "scope": "franka_only_before_next_checkpoint_inference",
        "arrival_criterion": "per_axis_tolerance_or_crossing_latched",
        "tolerance_rad": 0.004,
        "per_target_timeout_s": 0.30,
        "policy_timing": "arrival_driven_diagnostic_not_fixed_rate",
        "native_franka_v225_1khz_interpolator_retained": True,
        "libfranka_official_rate_limiter_retained": True,
    }
    assert payload["hard_guards"]["franka_arrival_gate_enabled"] is True


@pytest.mark.parametrize(
    "args, message",
    [
        (["--arrival-tolerance-rad", "0.0009"], "0.001..0.030"),
        (["--arrival-timeout-s", "0.41"], "0.10..0.40"),
    ],
)
def test_closed_loop_arrival_gate_bounds_are_rejected_before_runner(
    capsys, args, message
):
    called = []
    assert main(args, real_runner=lambda request: called.append(request)) == 2
    assert called == []
    assert message in capsys.readouterr().err


def test_20hz_mode_is_hardware_inert_and_changes_all_rate_bound_guards(capsys):
    called = []
    assert (
        main(
            ["--steps", "240", "--policy-rate-hz", "20"],
            real_runner=lambda request: called.append(request),
        )
        == 0
    )
    assert called == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["policy_mode"]["name"] == "20hz"
    assert payload["policy_mode"]["control_dt_s"] == 0.05
    assert payload["hard_step_cap"] == 240
    assert payload["hard_guards"]["policy_rate_hz"] == 20.0
    assert payload["hard_guards"]["camera_max_policy_actions_per_frame"] == 3
    assert payload["hard_guards"]["policy_schedule"] == (
        "phase_locked_no_catch_up"
    )
    assert payload["hard_guards"]["rh56_target_schedule_rate_hz"] == 20.0
    assert payload["hard_guards"]["rh56_target_schedule_semantics"] == (
        "20hz_latest_legal_absolute_target;"
        "v94_one_policy_tick_transparent_envelope"
    )
    assert payload["hard_guards"][
        "rh56_target_contract_envelope_units_per_update"
    ] == [46, 48, 53, 53, 84, 40]


def test_20hz_mode_refuses_more_than_twelve_seconds_before_hardware(capsys):
    called = []
    assert (
        main(
            ["--steps", "241", "--policy-rate-hz", "20"],
            real_runner=lambda request: called.append(request),
        )
        == 2
    )
    assert called == []
    assert "steps must be an integer in 1..240" in capsys.readouterr().err


def test_external_checkpoint_cli_is_explicit_and_hardware_inert(capsys):
    checkpoint = Path(__file__).resolve().parents[2] / (
        "data/test_fixtures/student_pretrain_epoch_0001.pt"
    )
    called = []
    assert main(
        [
            "--steps",
            "5",
            "--checkpoint",
            str(checkpoint),
            "--select-object-roi",
        ],
        real_runner=lambda request: called.append(request),
    ) == 0
    assert called == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["hardware_access"] is False
    assert payload["checkpoint"]["source"] == "external_path_override"
    assert payload["checkpoint"]["path"] == str(checkpoint.resolve())
    assert "golden_action_equality_does_not_apply" in (
        payload["checkpoint"]["validation"]
    )


def test_external_checkpoint_execution_requires_current_roi(capsys):
    checkpoint = Path(__file__).resolve().parents[2] / (
        "data/test_fixtures/student_pretrain_epoch_0001.pt"
    )
    called = []
    assert main(
        [
            "--execute",
            "--yes-i-am-supervising",
            "--steps",
            "1",
            "--run-id",
            "checkpoint-with-stale-roi",
            "--checkpoint",
            str(checkpoint),
        ],
        real_runner=lambda request: called.append(request),
    ) == 2
    assert called == []
    assert "requires --object-text, --select-object-roi, or --object-roi" in (
        capsys.readouterr().err
    )


def test_step_cap_rejected_before_runner(capsys):
    called = []
    assert main(["--steps", "721"], real_runner=lambda request: called.append(request)) == 2
    assert called == []
    assert "1..720" in capsys.readouterr().err


def test_execute_needs_run_id_and_supervision_flag(capsys):
    called = []
    assert main(["--execute", "--steps", "1"], real_runner=lambda r: called.append(r)) == 2
    assert called == []
    assert main(
        [
            "--execute",
            "--steps",
            "1",
            "--run-id",
            "trial-001",
        ],
        real_runner=lambda r: called.append(r),
    ) == 2
    assert called == []


def test_explicit_execute_and_supervision_flags_call_real_runner_once(capsys):
    calls = []

    def runner(request):
        calls.append(request)
        return {"result": "PASS", "completed_policy_steps": request.steps}

    code = main(
        [
            "--execute",
            "--yes-i-am-supervising",
            "--steps",
            "5",
            "--run-id",
            "trial-005",
        ],
        real_runner=runner,
    )
    assert code == 0
    assert len(calls) == 1
    assert '"confirmation_phrase"' not in capsys.readouterr().out


def test_execute_console_is_compact_by_default_and_verbose_is_opt_in(capsys):
    def runner(request):
        print("[RealSense] verbose transport detail")
        print("[Rollout START] test milestone")
        return {
            "result": "PASS",
            "completed_policy_steps": request.steps,
            "franka_stop_verified": True,
            "rh56_disabled_verified": True,
        }

    base = [
        "--execute",
        "--yes-i-am-supervising",
        "--steps",
        "2",
        "--run-id",
        "compact-console",
    ]
    assert main(base, real_runner=runner) == 0
    compact = capsys.readouterr().out
    assert "[Deployment START]" in compact
    assert "[Rollout START]" in compact
    assert "[Rollout PASS] completed=2/2" in compact
    assert "RealSense" not in compact
    assert '"hard_guards"' not in compact

    verbose = base[:-1] + ["verbose-console", "--verbose-console"]
    assert main(verbose, real_runner=runner) == 0
    full = capsys.readouterr().out
    assert '"hard_guards"' in full
    assert "[RealSense] verbose transport detail" in full


def test_live_visualization_is_explicit_and_crosses_only_authorized_runner(capsys):
    called = []
    assert main(
        [
            "--steps",
            "5",
            "--live-visualization",
            "--live-visualization-rate-hz",
            "12",
        ],
        real_runner=lambda request: called.append(request),
    ) == 0
    assert called == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["live_visualization"]["requested"] is True
    assert payload["live_visualization"]["update_rate_hz"] == 12.0

    assert main(
        [
            "--execute",
            "--steps",
            "5",
            "--run-id",
            "trial-viz",
            "--live-visualization",
        ],
        real_runner=lambda request: called.append(request),
    ) == 2
    assert called == []


def test_authorized_live_visualization_flag_reaches_runner(capsys):
    called = []

    def runner(request):
        called.append(request)
        return {"result": "PASS", "completed_policy_steps": request.steps}

    assert main(
        [
            "--execute",
            "--yes-i-am-supervising",
            "--steps",
            "5",
            "--run-id",
            "trial-viz-ok",
            "--live-visualization",
        ],
        real_runner=runner,
    ) == 0
    assert len(called) == 1
    assert called[0].live_visualization is True
    assert called[0].live_visualization_rate_hz == 10.0


def test_record_video_uses_background_pipeline_without_live_windows(
    capsys, tmp_path
):
    called = []
    output = tmp_path / "test-run.mp4"

    assert main(
        [
            "--steps",
            "5",
            "--policy-rate-hz",
            "20",
            "--record-video",
            str(output),
        ],
        real_runner=lambda request: called.append(request),
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["live_visualization"]["requested"] is False
    assert (
        payload["live_visualization"]["background_output_pipeline_enabled"]
        is True
    )
    assert payload["live_visualization"]["update_rate_hz"] == 10.0
    assert payload["live_visualization"]["record_video_rate_hz"] == 20.0
    assert payload["live_visualization"]["record_video_overlay"] == "none"
    assert payload["live_visualization"]["record_mask_video_path"] == str(
        (tmp_path / "test-run_mask.mp4").resolve()
    )
    assert called == []

    def runner(request):
        called.append(request)
        return {"result": "PASS", "completed_policy_steps": request.steps}

    assert main(
        [
            "--execute",
            "--yes-i-am-supervising",
            "--steps",
            "5",
            "--run-id",
            "trial-video-ok",
            "--record-video",
            str(output),
        ],
        real_runner=runner,
    ) == 0
    assert len(called) == 1
    assert called[0].live_visualization is False
    assert called[0].record_video_path == output.resolve()


def test_record_video_refuses_non_mp4_and_existing_output(capsys, tmp_path):
    called = []
    bad = tmp_path / "run.avi"
    assert main(
        ["--record-video", str(bad)],
        real_runner=lambda request: called.append(request),
    ) == 2
    assert "must end in .mp4" in capsys.readouterr().err

    existing = tmp_path / "run.mp4"
    existing.write_bytes(b"keep")
    assert main(
        ["--record-video", str(existing)],
        real_runner=lambda request: called.append(request),
    ) == 2
    assert "refuses to overwrite" in capsys.readouterr().err
    assert existing.read_bytes() == b"keep"
    assert called == []

    sidecar = tmp_path / "fresh_mask.mp4"
    sidecar.write_bytes(b"keep-mask")
    assert main(
        ["--record-video", str(tmp_path / "fresh.mp4")],
        real_runner=lambda request: called.append(request),
    ) == 2
    assert "mask sidecar" in capsys.readouterr().err
    assert sidecar.read_bytes() == b"keep-mask"
    assert called == []


def test_record_policy_io_is_opt_in_and_reaches_authorized_runner(
    monkeypatch, capsys, tmp_path
):
    import sim2real.deployment.runner as cli

    monkeypatch.setattr(
        cli,
        "default_policy_io_path",
        lambda run_id: tmp_path / f"{run_id}_policy_io.npz",
    )
    called = []
    assert main(
        ["--steps", "5", "--run-id", "policy-io-dry", "--record-policy-io"],
        real_runner=lambda request: called.append(request),
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["policy_io_recording"] == {
        "requested": True,
        "path": str(tmp_path / "policy-io-dry_policy_io.npz"),
        "capture": "accepted_non_actuated_startup_and_dual_ack_ticks_only",
        "control_path": "in_memory_copy_only_then_post_stop_atomic_npz",
        "normalization": "checkpoint_constants_applied_after_stop",
    }
    assert called == []

    def runner(request):
        called.append(request)
        return {"result": "PASS", "completed_policy_steps": request.steps}

    assert main(
        [
            "--execute",
            "--yes-i-am-supervising",
            "--steps",
            "5",
            "--run-id",
            "policy-io-real",
            "--record-policy-io",
        ],
        real_runner=runner,
    ) == 0
    assert len(called) == 1
    assert called[0].record_policy_io is True


def test_record_policy_io_refuses_existing_output(monkeypatch, capsys, tmp_path):
    import sim2real.deployment.runner as cli

    existing = tmp_path / "existing.npz"
    existing.write_bytes(b"keep")
    monkeypatch.setattr(cli, "default_policy_io_path", lambda _run_id: existing)
    assert main(
        ["--run-id", "policy-io-existing", "--record-policy-io"],
    ) == 2
    assert "refuses to overwrite" in capsys.readouterr().err
    assert existing.read_bytes() == b"keep"


def test_current_object_roi_modes_are_explicit_and_mutually_exclusive(capsys):
    called = []
    assert main(
        ["--steps", "1", "--object-roi", "420", "220", "120", "110"],
        real_runner=lambda request: called.append(request),
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["object_roi"]["mode"] == "camera_only_numeric_preflight"
    assert payload["object_roi"]["xywh"] == [420, 220, 120, 110]
    assert called == []

    with pytest.raises(SystemExit) as raised:
        main(
            [
                "--steps",
                "1",
                "--object-roi",
                "420",
                "220",
                "120",
                "110",
                "--select-object-roi",
            ],
            real_runner=lambda request: called.append(request),
        )
    assert raised.value.code == 2
    assert called == []

    assert main(
        ["--steps", "1", "--object-text", "  green   ball  "],
        real_runner=lambda request: called.append(request),
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["object_roi"]["mode"] == "camera_only_text_grounding_preflight"
    assert payload["object_roi"]["text"] == "green ball"
    assert payload["object_roi"]["xywh"] is None
    assert called == []

    with pytest.raises(SystemExit) as raised:
        main(
            [
                "--steps",
                "1",
                "--object-text",
                "green ball",
                "--select-object-roi",
            ],
            real_runner=lambda request: called.append(request),
        )
    assert raised.value.code == 2
    assert called == []


def test_object_mask_mode_preserves_guarded_and_legacy_ab_paths(capsys):
    called = []
    assert main(
        ["--steps", "1", "--object-mask-mode", "legacy"],
        real_runner=lambda request: called.append(request),
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["object_mask"] == {
        "mode": "legacy",
        "provider_publication_mode": "semantic_sam2",
        "recovery_publication_mode": "semantic_sam2_direct",
        "legacy_comparison_available": True,
        "requested_object_mask_mode": "legacy",
        "effective_object_mask_mode": "legacy",
        "effective_provider_mask_publication_mode": "semantic_sam2",
        "effective_provider_recovery_publication_mode": "semantic_sam2_direct",
        "stale_palm_policy": "legacy_explicit_compatibility",
        "stale_palm_legacy_compatibility_switch": (
            "explicit_--object-mask-mode=legacy_only"
        ),
    }
    assert called == []


@pytest.mark.parametrize(
    ("mode", "recovery"),
    [
        ("guarded", "unified_three_evidence"),
        ("guarded_v2", "unified_three_evidence"),
        ("guarded_v1", "legacy_double_confirm"),
    ],
)
def test_object_mask_mode_exposes_versioned_guarded_recovery(
    capsys, mode, recovery
):
    assert main(["--steps", "1", "--object-mask-mode", mode]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["object_mask"]["mode"] == mode
    assert payload["object_mask"]["provider_publication_mode"] == (
        "adaptive_fusion"
    )
    assert payload["object_mask"]["recovery_publication_mode"] == recovery


def test_policy_rgbd_resolution_exposes_native_and_simulator_aligned_ab(capsys):
    called = []
    assert main(
        ["--steps", "1", "--policy-rgbd-resolution", "424x240"],
        real_runner=lambda request: called.append(request),
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["policy_rgbd"] == {
        "resolution": [424, 240],
        "native_d435_and_sam2_resolution": [848, 480],
        "adapter": "aligned_stride2_no_interpolation",
        "camera_intrinsics_scaled_with_resolution": True,
    }
    assert called == []


@pytest.mark.parametrize(
    "roi",
    [
        ("-1", "2", "3", "4"),
        ("1", "2", "0", "4"),
        ("1", "2", "3", "0"),
    ],
)
def test_invalid_current_object_roi_is_rejected_before_runner(roi, capsys):
    called = []
    assert main(
        ["--steps", "1", "--object-roi", *roi],
        real_runner=lambda request: called.append(request),
    ) == 2
    assert called == []


def test_isolated_roi_selector_environment_removes_ros_isaac_and_qt_paths():
    import sim2real.deployment.runner as cli

    environment = cli._isolated_roi_selector_environment(
        {
            "PATH": "/usr/bin",
            "DISPLAY": ":1",
            "PYTHONPATH": (
                "/opt/ros/humble/lib/python3.10/site-packages:"
                "/tmp/isaacgym/python:/wanted"
            ),
            "LD_LIBRARY_PATH": (
                "/opt/ros/humble/lib:/tmp/isaacgym/lib:"
                "/usr/local/cuda/lib64"
            ),
            "QT_PLUGIN_PATH": "/opt/ros/humble/qt",
            "ROS_DISTRO": "humble",
            "AMENT_PREFIX_PATH": "/opt/ros/humble",
            "ISAAC_PATH": "/tmp/isaacgym",
        }
    )

    assert environment["DISPLAY"] == ":1"
    assert environment["LD_LIBRARY_PATH"] == "/usr/local/cuda/lib64"
    assert "/opt/ros" not in environment["PYTHONPATH"]
    assert "isaac" not in environment["PYTHONPATH"].lower()
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "QT_PLUGIN_PATH" not in environment
    assert "ROS_DISTRO" not in environment
    assert "AMENT_PREFIX_PATH" not in environment
    assert "ISAAC_PATH" not in environment


def test_isolated_roi_selector_private_pipe_success(monkeypatch, tmp_path):
    import sim2real.deployment.runner as cli

    nonce = "1" * 32
    roi = [401, 211, 131, 121]
    observed = {}

    class Process:
        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            raise AssertionError("successful selector must not be terminated")

        def kill(self):
            raise AssertionError("successful selector must not be killed")

    def popen(command, **kwargs):
        observed["command"] = tuple(command)
        observed["kwargs"] = kwargs
        descriptor = kwargs["pass_fds"][0]
        os.write(
            descriptor,
            (
                json.dumps(
                    {
                        "protocol": cli.OBJECT_ROI_SELECTOR_PROTOCOL,
                        "nonce": nonce,
                        "status": "ok",
                        "roi_xywh": roi,
                        "camera_serial": "camera-1",
                        "calibration_id": "calibration-1",
                    },
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8"),
        )
        return Process()

    monkeypatch.setattr(cli.secrets, "token_hex", lambda _size: nonce)
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(
        cli,
        "_isolated_roi_selector_environment",
        lambda: {"PYTHONPATH": "clean"},
    )
    request = SupervisedRequest(
        run_id="selector-success",
        steps=1,
        bundle=tmp_path / "deploy.zip",
        profile=tmp_path / "profile.json",
        pcd_config=tmp_path / "pcd.yaml",
        execute=True,
        select_object_roi=True,
    )

    selected = cli._select_object_roi_isolated(
        request,
        expected_camera_serial="camera-1",
        expected_calibration_id="calibration-1",
    )

    assert selected == tuple(roi)
    assert observed["command"][:3] == (
        cli.sys.executable,
        "-m",
        "sim2real.observation.roi_selector",
    )
    assert observed["kwargs"]["env"] == {"PYTHONPATH": "clean"}
    assert observed["kwargs"]["stdin"] is cli.subprocess.DEVNULL
    assert observed["kwargs"]["stderr"] is cli.subprocess.DEVNULL
    assert observed["kwargs"]["close_fds"] is True
    assert len(observed["kwargs"]["pass_fds"]) == 1


def test_text_roi_selector_enables_child_compact_console(monkeypatch, tmp_path):
    import sim2real.deployment.runner as cli

    nonce = "2" * 32
    observed = {}

    class Process:
        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            raise AssertionError("successful selector must not be terminated")

        def kill(self):
            raise AssertionError("successful selector must not be killed")

    def popen(command, **kwargs):
        observed["command"] = tuple(command)
        descriptor = kwargs["pass_fds"][0]
        os.write(
            descriptor,
            (
                json.dumps(
                    {
                        "protocol": cli.OBJECT_ROI_SELECTOR_PROTOCOL,
                        "nonce": nonce,
                        "status": "ok",
                        "roi_xywh": [100, 80, 20, 20],
                        "camera_serial": "camera-1",
                        "calibration_id": "calibration-1",
                    },
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8"),
        )
        return Process()

    monkeypatch.setattr(cli.secrets, "token_hex", lambda _size: nonce)
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(
        cli,
        "_isolated_roi_selector_environment",
        lambda: {"PYTHONPATH": "clean"},
    )
    request = SupervisedRequest(
        run_id="selector-text-compact",
        steps=1,
        bundle=tmp_path / "deploy.zip",
        profile=tmp_path / "profile.json",
        pcd_config=tmp_path / "pcd.yaml",
        execute=False,
        object_text="red triangular beanbag toy",
    )

    assert cli._select_object_roi_isolated(
        request,
        expected_camera_serial="camera-1",
        expected_calibration_id="calibration-1",
    ) == (100, 80, 20, 20)
    assert "--object-text" in observed["command"]
    assert "red triangular beanbag toy" in observed["command"]
    assert "--compact-console" in observed["command"]


def test_isolated_roi_selector_gui_crash_opens_no_robot_interface(
    monkeypatch, tmp_path
):
    import sim2real.deployment.runner as cli

    class Process:
        def poll(self):
            return -signal.SIGABRT

        def wait(self, timeout=None):
            return -signal.SIGABRT

        def terminate(self):
            raise AssertionError("crashed selector is already stopped")

        def kill(self):
            raise AssertionError("crashed selector is already stopped")

    monkeypatch.setattr(cli.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    request = SupervisedRequest(
        run_id="selector-crash",
        steps=1,
        bundle=tmp_path / "deploy.zip",
        profile=tmp_path / "profile.json",
        pcd_config=tmp_path / "pcd.yaml",
        execute=True,
        select_object_roi=True,
    )

    with pytest.raises(
        cli.SupervisedV94RunError,
        match="crashed with SIGABRT; the selector opened no robot interfaces",
    ):
        cli._select_object_roi_isolated(
            request,
            expected_camera_serial="camera-1",
            expected_calibration_id="calibration-1",
        )


def test_keyboard_interrupt_reports_130_after_runner_cleanup(capsys):
    def interrupt(_request):
        raise KeyboardInterrupt

    code = main(
        [
            "--execute",
            "--yes-i-am-supervising",
            "--steps",
            "1",
            "--run-id",
            "trial-int",
        ],
        real_runner=interrupt,
    )
    assert code == 130
    assert "interrupted" in capsys.readouterr().err


@pytest.mark.parametrize("retain_live_provider", [False, True])
def test_interactive_roi_preflight_proves_mask_depth_and_policy_points(
    monkeypatch,
    tmp_path,
    retain_live_provider,
):
    import sim2real.observation.capture as capture
    import sim2real.observation.live_preview as preview
    import sim2real.deployment.runner as cli
    from sim2real.deployment.bundle import DeployBundle
    from sim2real.contracts.v94 import V94Contract

    bundle = Path(__file__).resolve().parents[2] / "data/test_fixtures/sim2real/deploy.zip"
    contract = V94Contract.from_bundle(DeployBundle(bundle))
    height, width = contract.camera_height, contract.camera_width
    roi = (400, 210, 130, 120)
    mask = np.zeros((height, width), dtype=bool)
    mask[235:305, 430:500] = True
    depth = np.zeros((height, width), dtype=np.uint16)
    depth[mask] = 930
    payload = {
        "rgb": np.zeros((3, height, width, 3), dtype=np.uint8),
        "depth_raw": np.stack([depth, depth, depth]),
        "object_mask": np.stack([mask, mask, mask]),
        "frame_camera_K": np.repeat(
            contract.camera_K[None], 3, axis=0
        ),
        "frame_camera_distortion": np.zeros((3, 5), dtype=np.float64),
        "frame_depth_scale_m_per_unit": np.full(
            3, contract.depth_scale_m_per_unit, dtype=np.float64
        ),
        # Startup inference may skip three native 30 Hz frames.  This remains
        # fresh under the formal 400 ms publication-liveness contract.
        "camera_timestamp_s": np.asarray([1.0, 1.033, 1.166]),
        # GPU/tracker processing may skip native camera sequence numbers even
        # though these are three adjacent valid provider publications.
        "camera_frame_id": np.asarray([10, 12, 15], dtype=np.int64),
    }
    stopped = []
    provider = SimpleNamespace(
        last_bbox_initialization_evidence=SimpleNamespace(
            source="grabcut",
            prompt_bbox_xyxy=np.asarray([400, 210, 530, 330], dtype=np.int32),
            mask_bbox_xyxy=np.asarray([430, 235, 500, 305], dtype=np.int32),
            mask_area_px=int(mask.sum()),
        ),
        extrinsics=SimpleNamespace(
            calibration_id=contract.calibration_id,
            camera_serial=contract.camera_serial,
            T_base_camera=contract.T_base_camera_optical.copy(),
        ),
        stop=lambda: stopped.append(True),
    )

    class Guard:
        def start(self):
            return self

        def close(self):
            pass

    monkeypatch.setattr(capture, "_ComputeThreadGuard", lambda _threads: Guard())
    selector_calls = []
    monkeypatch.setattr(
        cli,
        "_select_object_roi_isolated",
        lambda request, **kwargs: selector_calls.append(
            (request, kwargs)
        )
        or roi,
    )

    def initialize_provider(_config, requested_roi, **kwargs):
        assert tuple(requested_roi) == roi
        assert kwargs["disable_online_sam2"] is False
        assert kwargs["object_mask_mode"] == "guarded"
        return provider, object()

    monkeypatch.setattr(
        preview,
        "_initialize_provider",
        initialize_provider,
    )
    monkeypatch.setattr(
        capture,
        "capture_valid_frames",
        lambda *_args, **_kwargs: (
            payload,
            {
                "provider_steps": 7,
                "valid_publications": 3,
                "invalid_publications": 4,
                "stored_frames": 3,
                "invalid_reasons": {},
            },
        ),
    )
    handoff = object()
    adopted = []
    if retain_live_provider:
        import sim2real.runtime.v94_live_observation_owner as observation_owner

        def adopt_provider(**kwargs):
            adopted.append(kwargs)
            return handoff

        monkeypatch.setattr(
            observation_owner,
            "open_prevalidated_d435_handoff",
            adopt_provider,
        )
    pcd_config = tmp_path / "pcd.yaml"
    pcd_config.write_text("camera: test\n", encoding="utf-8")
    request = SupervisedRequest(
        run_id="roi-preflight",
        steps=1,
        bundle=bundle,
        profile=tmp_path / "profile.json",
        pcd_config=pcd_config,
        execute=True,
        select_object_roi=True,
        policy_rgbd_resolution="424x240",
    )

    result = cli._preflight_current_object_roi(
        request,
        retain_live_provider=retain_live_provider,
    )

    assert result.object_roi_xywh == roi
    assert result.interactive_roi_gui_used is True
    assert result.object_roi_preflight_valid_frames == 3
    assert result.object_roi_preflight_invalid_frames == 4
    assert result.object_roi_preflight_min_policy_points == int(mask[::2, ::2].sum())
    assert result.object_roi_preflight_depth_p50_m == pytest.approx(0.93)
    assert len(result.object_roi_preflight_sha256) == 64
    assert len(selector_calls) == 1
    assert selector_calls[0][1] == {
        "expected_camera_serial": contract.camera_serial,
        "expected_calibration_id": contract.calibration_id,
    }
    if retain_live_provider:
        assert stopped == []
        assert result.prewarmed_camera_handoff is handoff
        assert len(adopted) == 1
        assert adopted[0]["provider"] is provider
        assert adopted[0]["roi_xywh"] == roi
        assert adopted[0]["object_mask_mode"] == "guarded"
        assert adopted[0]["last_preflight_frame_id"] == 15
        assert adopted[0]["last_preflight_timestamp_s"] == pytest.approx(1.166)
        assert adopted[0]["preflight_sha256"] == result.object_roi_preflight_sha256
    else:
        assert stopped == [True]
        assert result.prewarmed_camera_handoff is None


def test_object_roi_preflight_progress_uses_formal_camera_liveness_bound():
    import sim2real.deployment.runner as cli

    gaps = cli._validate_object_roi_preflight_progress(
        [2, 3, 4],
        [10.0, 10.033, 10.166],
    )
    np.testing.assert_allclose(gaps, [0.033, 0.133], atol=1.0e-12)

    with pytest.raises(
        cli.SupervisedV94RunError,
        match="publication gap exceeded 400ms",
    ):
        cli._validate_object_roi_preflight_progress(
            [2, 3, 4],
            [10.0, 10.033, 10.434],
        )


def test_real_runner_resets_before_object_preflight_and_compute_isolation(
    monkeypatch, tmp_path
):
    import sim2real.deployment.runner as cli
    import sim2real.deployment.lease as deployment_lease
    import sim2real.runtime.supervised_v94_runtime as runtime
    import sim2real.deployment.execution_reset as reset

    events = []
    preparation = SimpleNamespace(
        output=tmp_path / "audit.json",
        profile={"profile": "v94"},
        contract=SimpleNamespace(q_home_rad=(0.0,) * 7),
        bundle_sha256="a" * 64,
        profile_file_sha256="b" * 64,
        pcd_config_sha256="c" * 64,
        checkpoint_source="bundle_primary",
        checkpoint_path=None,
        checkpoint_sha256="e" * 64,
        native_build=SimpleNamespace(binary_sha256="d" * 64),
        pinned_checkpoint_bytes=b"checkpoint",
    )
    reset_proof = object()

    class Handoff:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    handoff = Handoff()

    def preflight(request, **kwargs):
        assert kwargs == {"retain_live_provider": True}
        events.append("roi_preflight")
        return replace(request, prewarmed_camera_handoff=handoff)

    monkeypatch.setattr(
        cli,
        "_preflight_current_object_roi",
        preflight,
    )
    monkeypatch.setattr(
        runtime,
        "prepare_supervised_v94",
        lambda _request: events.append("prepare") or preparation,
    )
    monkeypatch.setattr(
        runtime,
        "prepare_supervised_v94_artifacts",
        lambda _request: events.append("artifact_prepare") or preparation,
    )
    monkeypatch.setattr(
        cli,
        "_admit_deployment_compute_isolation",
        lambda request: events.append("host_irq_admission") or request,
    )
    monkeypatch.setattr(
        reset,
        "run_v94_execution_reset",
        lambda **_kwargs: events.append("reset") or reset_proof,
    )

    class Lease:
        def __enter__(self):
            events.append("lease_enter")
            return self

        def mark_phase(self, phase):
            events.append(f"phase:{phase}")

        def finalize(self, *, result):
            events.append(f"finalize:{result}")

        def __exit__(self, *_args):
            events.append("lease_exit")

    monkeypatch.setattr(
        deployment_lease,
        "acquire_v94_deployment_lease",
        lambda **_kwargs: Lease(),
    )

    @contextmanager
    def isolation(request):
        events.append("isolation_enter")
        yield request
        events.append("isolation_exit")

    monkeypatch.setattr(cli, "_deployment_compute_isolation", isolation)

    @contextmanager
    def persistent_sam2(_pcd_config):
        events.append("sam2_prewarm_enter")
        yield {"service": "dynamic-pcd-online-sam2"}
        events.append("sam2_prewarm_exit")

    monkeypatch.setattr(
        cli, "_persistent_online_sam2_service", persistent_sam2
    )
    expected_preparation = preparation

    def run(
        request,
        *,
        preparation,
        execution_reset,
        prewarmed_camera_handoff,
    ):
        events.append("runtime")
        assert preparation is expected_preparation
        assert execution_reset is reset_proof
        assert prewarmed_camera_handoff is handoff
        return {"result": "PASS"}

    monkeypatch.setattr(runtime, "run_supervised_v94", run)
    request = SupervisedRequest(
        run_id="reset-order",
        steps=1,
        bundle=tmp_path / "deploy.zip",
        profile=tmp_path / "profile.json",
        pcd_config=tmp_path / "pcd.yaml",
        execute=True,
        select_object_roi=True,
    )

    assert cli._run_real(request) == {"result": "PASS"}
    assert handoff.close_calls == 1
    assert events == [
        "artifact_prepare",
        "host_irq_admission",
        "lease_enter",
        "phase:perception_model_prewarm",
        "sam2_prewarm_enter",
        "phase:automatic_reset",
        "reset",
        "phase:object_preflight",
        "roi_preflight",
        "prepare",
        "phase:policy_runtime",
        "isolation_enter",
        "runtime",
        "isolation_exit",
        "finalize:PASS",
        "sam2_prewarm_exit",
        "lease_exit",
    ]


def test_persistent_sam2_service_starts_once_and_closes_after_all_phases(
    monkeypatch, tmp_path
):
    import sim2real.deployment.runner as cli

    calls = []

    class Manager:
        def start(self):
            calls.append("start")
            return {"service": "dynamic-pcd-online-sam2"}

        def reset(self):
            calls.append("reset")

        def close(self):
            calls.append("close")

    manager = Manager()
    monkeypatch.setattr(
        cli,
        "_build_persistent_online_sam2_manager",
        lambda _path: manager,
    )

    with cli._persistent_online_sam2_service(tmp_path / "pcd.yaml") as health:
        calls.append("grounding_preflight_runtime")
        assert health["service"] == "dynamic-pcd-online-sam2"

    assert calls == [
        "start",
        "grounding_preflight_runtime",
        "reset",
        "close",
    ]


def test_persistent_sam2_service_closes_if_startup_fails(monkeypatch, tmp_path):
    import sim2real.deployment.runner as cli

    calls = []

    class Manager:
        def start(self):
            calls.append("start")
            raise RuntimeError("compile failed")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(
        cli,
        "_build_persistent_online_sam2_manager",
        lambda _path: Manager(),
    )

    with pytest.raises(RuntimeError, match="compile failed"):
        with cli._persistent_online_sam2_service(tmp_path / "pcd.yaml"):
            pytest.fail("startup failure must not enter the camera phases")

    assert calls == ["start", "close"]


def test_persistent_sam2_service_is_noop_when_online_sam2_is_disabled(
    monkeypatch, tmp_path
):
    import sim2real.deployment.runner as cli

    monkeypatch.setattr(
        cli,
        "_build_persistent_online_sam2_manager",
        lambda _path: None,
    )
    with cli._persistent_online_sam2_service(tmp_path / "pcd.yaml") as health:
        assert health is None


def test_real_runner_rejects_native_artifact_before_camera_roi(
    monkeypatch, tmp_path
):
    import sim2real.deployment.runner as cli
    import sim2real.runtime.supervised_v94_runtime as runtime

    camera_calls: list[bool] = []

    def reject_stale_native(_request):
        raise runtime.SupervisedV94RuntimeError("stale native protocol")

    monkeypatch.setattr(
        runtime,
        "prepare_supervised_v94_artifacts",
        reject_stale_native,
    )
    monkeypatch.setattr(
        cli,
        "_preflight_current_object_roi",
        lambda request: camera_calls.append(True) or request,
    )
    request = SupervisedRequest(
        run_id="artifact-before-camera",
        steps=1,
        bundle=tmp_path / "deploy.zip",
        profile=tmp_path / "profile.json",
        pcd_config=tmp_path / "pcd.yaml",
        execute=True,
        select_object_roi=True,
    )
    with pytest.raises(
        runtime.SupervisedV94RuntimeError,
        match="stale native protocol",
    ):
        cli._run_real(request)
    assert camera_calls == []
