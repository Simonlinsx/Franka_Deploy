#!/usr/bin/env python3
"""One foreground command for reset -> perception -> selection -> execution.

The user-facing workflow intentionally hides snapshots, candidate indices,
audit directories, and the internal one-shot session.  Those artifacts remain
on disk for reproducibility, but they are implementation details rather than
shell variables the operator must copy.

The currently implemented execution backend is the reviewed no-contact air
round trip.  Loaded lift is exposed as a fail-before-motion mode until a
payload/load-rated profile is supplied; camera geometry cannot determine mass.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_sim2real_supervised.json"
DEFAULT_CAMERA_CONFIG = WORKSPACE / "perception/configs/d435_default.yaml"
DEFAULT_DYNAMIC_PYTHON = Path(
    "/home/qiaoguanren/anaconda3/envs/dynamic/bin/python"
)
RUN_CONFIRMATION = "RUN_FR3_RH56_ONE_SHOT"


class StageError(RuntimeError):
    def __init__(self, name: str, returncode: int) -> None:
        super().__init__("stage {!r} failed with exit {}".format(name, returncode))
        self.name = name
        self.returncode = int(returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One command: reset FR3/RH56, interactive bbox+SAM2, official "
            "AnyDex, automatic executable-candidate selection, live preview, "
            "and foreground execution"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--q7-rad", type=float, default=1.3525)
    parser.add_argument("--lift-height-m", type=float, default=0.05)
    parser.add_argument(
        "--execution-mode",
        choices=("air", "lift"),
        default="air",
        help="lift fails before motion until a loaded payload profile is configured",
    )
    parser.add_argument("--dynamic-python", type=Path, default=DEFAULT_DYNAMIC_PYTHON)
    parser.add_argument(
        "--confirm-run",
        default=None,
        help="noninteractive exact confirmation token; omit for one terminal prompt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the one-command orchestration without opening hardware/camera",
    )
    return parser


def _default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "runs" / "one_shot_{}".format(stamp)


def _run_stage(name: str, command: Sequence[str], *, env=None) -> None:
    values = [str(item) for item in command]
    print("\n[one-shot:{}] {}".format(name, shlex.join(values)), flush=True)
    completed = subprocess.run(values, cwd=str(ROOT), env=env, check=False)
    if int(completed.returncode) != 0:
        raise StageError(name, int(completed.returncode))


def _confirm(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    token = args.confirm_run
    if token is None:
        print(
            "\n即将执行真机一体化流程。请确认：RH56/适配器牢固；线缆有余量；"
            "全扫掠空间无人无物；Franka 可立即停止；RH56 24V 可立即切断。",
            flush=True,
        )
        token = input("输入 {} 继续：".format(RUN_CONFIRMATION)).strip()
    if token != RUN_CONFIRMATION:
        raise ValueError("run confirmation must exactly equal {}".format(RUN_CONFIRMATION))


def _validate_loaded_mode(config: dict, lift_height_m: float) -> None:
    if not 0.0 < float(lift_height_m) <= 0.20:
        raise ValueError("--lift-height-m must be in (0,0.20]")
    loaded = config.get("loaded_lift")
    if not isinstance(loaded, dict):
        raise ValueError(
            "loaded lift is not configured: object mass/inertia and a load-rated "
            "adapter approval are required; D435 geometry cannot infer mass"
        )
    required = (
        "payload_mass_kg",
        "payload_com_hand_m",
        "payload_inertia_hand_kg_m2",
        "max_lift_height_m",
    )
    missing = [name for name in required if loaded.get(name) is None]
    if missing:
        raise ValueError("loaded_lift profile is incomplete: " + ", ".join(missing))
    if float(lift_height_m) > float(loaded["max_lift_height_m"]):
        raise ValueError("requested lift height exceeds the loaded profile")


def _host_preflight(
    config: dict,
    args: argparse.Namespace,
    *,
    runner=subprocess.run,
) -> None:
    """Reject missing serial/CUDA infrastructure before any motion stage."""

    failures: list[str] = []
    inspire = config.get("inspire")
    port_value = inspire.get("port") if isinstance(inspire, dict) else None
    if not isinstance(port_value, str) or not port_value.strip():
        failures.append("control profile has no inspire.port")
    else:
        port = Path(port_value).expanduser()
        if not port.exists():
            failures.append(
                "RH56 serial port is missing: {} (after reboot, check whether "
                "brltty reclaimed the CH341 interface)".format(port)
            )

    device = str(args.device).strip().lower()
    if device.startswith("cuda"):
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            failures.append("nvidia-smi is not installed")
        else:
            try:
                result = runner(
                    [nvidia_smi, "-L"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                failures.append("NVIDIA driver health check failed: {}".format(exc))
            else:
                if int(result.returncode) != 0:
                    detail = (result.stderr or result.stdout or "").strip()
                    failures.append(
                        "NVIDIA driver is unavailable{}".format(
                            ": " + detail if detail else ""
                        )
                    )

        if not any(item.startswith("NVIDIA driver") for item in failures):
            cuda_probe = (
                "import sys, torch; "
                "d=torch.device(sys.argv[1]); "
                "assert torch.cuda.is_available(), 'torch.cuda.is_available() is false'; "
                "x=torch.empty(1, device=d); "
                "print(torch.cuda.get_device_name(d))"
            )
            try:
                result = runner(
                    [str(args.dynamic_python.expanduser().resolve()), "-c", cuda_probe, device],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                failures.append("PyTorch CUDA probe failed: {}".format(exc))
            else:
                if int(result.returncode) != 0:
                    detail = (result.stderr or result.stdout or "").strip()
                    failures.append(
                        "PyTorch cannot allocate on {}{}".format(
                            device, ": " + detail if detail else ""
                        )
                    )
                else:
                    print(
                        "[preflight] CUDA ready: {}".format(result.stdout.strip()),
                        flush=True,
                    )

    if failures:
        raise RuntimeError(
            "host preflight failed before confirmation/motion:\n- "
            + "\n- ".join(failures)
        )
    print("[preflight] RH56 serial ready: {}".format(port_value), flush=True)


def _commands(
    args: argparse.Namespace,
    *,
    output: Path,
    snapshot: Path,
    raw: Path,
    selection_plan: Path,
    selected_index: Optional[int],
) -> dict[str, list[str]]:
    config = str(args.config.expanduser().resolve())
    camera = str(args.camera_config.expanduser().resolve())
    commands = {
        "reset_rh56": [
            str(ROOT / "scripts/reset_installed_rh56_open.sh"), "run",
            "--config", config,
            "--confirm-installed", "RH56_INSTALLED_ON_FR3",
            "--confirm-24v-cutoff", "RH56_24V_CUTOFF_READY",
            "--confirm-franka-stop", "FR3_STOP_READY",
            "--confirm-workspace-clear", "INSTALLED_AIR_WORKSPACE_CLEAR",
            "--confirm-no-contact", "PLA_LOW_SPEED_NO_CONTACT",
            "--confirm-reset-open", "RH56_RESET_OPEN",
        ],
        "reset_franka": [
            str(ROOT / "scripts/reset_franka_default.sh"),
            "--config", config,
            "--confirm-installed", "RH56_INSTALLED_ON_FR3",
            "--confirm-hand-open", "RH56_OPEN_DISABLED",
            "--confirm-workspace-clear", "FR3_RH56_CURRENT_TO_DEFAULT_SWEEP_CLEAR",
            "--confirm-stop-ready", "FR3_STOP_READY",
            "--confirm-pla-low-speed", "PLA_LOW_SPEED_UNLOADED_ONLY",
        ],
        "perception": [
            str(ROOT / "scripts/run_grasp_generation_live_preview.sh"),
            str(snapshot),
            "--raw-output", str(raw),
            "--control-config", config,
            "--camera-config", camera,
            "--sam2",
            "--device", str(args.device),
            "--top-k", str(int(args.top_k)),
            "--trust-official-checkpoints",
            "--no-preview",
        ],
        "select": [
            str(ROOT / "scripts/select_executable_candidate.sh"),
            "--snapshot", str(snapshot),
            "--config", config,
            "--q7-rad", "{:.17g}".format(float(args.q7_rad)),
            "--output", str(selection_plan),
        ],
    }
    if selected_index is not None:
        prepare_dir = output / "execution"
        commands["prepare"] = [
            str(ROOT / "scripts/run_installed_air_grasp.sh"), "prepare",
            "--config", config,
            "--camera-config", camera,
            "--snapshot", str(snapshot),
            "--selected-index", str(int(selected_index)),
            "--output-dir", str(prepare_dir),
            "--confirm-installed", "RH56_INSTALLED_ON_FR3",
            "--confirm-24v-cutoff", "RH56_24V_CUTOFF_READY",
            "--confirm-franka-stop", "FR3_STOP_READY",
            "--confirm-rh56-workspace-clear", "INSTALLED_AIR_WORKSPACE_CLEAR",
            "--confirm-no-contact", "PLA_LOW_SPEED_NO_CONTACT",
            "--confirm-rh56-reset-open", "RH56_RESET_OPEN",
            "--confirm-hand-open", "RH56_OPEN_DISABLED",
            "--confirm-default-sweep-clear", "FR3_RH56_CURRENT_TO_DEFAULT_SWEEP_CLEAR",
            "--confirm-pla-low-speed", "PLA_LOW_SPEED_UNLOADED_ONLY",
            "--confirm-stationary-q", "CURRENT_Q_READ_ONLY_AND_STATIONARY",
        ]
        session = prepare_dir / "installed_air_grasp_session.json"
        commands["execute"] = [
            str(ROOT / "scripts/run_installed_air_grasp.sh"), "run",
            "--session", str(session),
            "--confirm-workspace-clear", "FR3_RH56_WORKSPACE_CLEAR",
            "--confirm-immediate-stop", "IMMEDIATE_STOP_AND_24V_CUT_READY",
            "--confirm-air-grasp", "FR3_RH56_AIR_GRASP_NO_CONTACT_NO_LIFT",
            "--confirm-q6-preshape", "RH56_Q6_PRESHAPE_VERIFIED",
            "--confirm-installed-collision-model", "INSTALLED_TOOL_COLLISIONS_VERIFIED",
            "--interactive-execution-confirmation",
        ]
    return commands


def run_once(args: argparse.Namespace) -> Path:
    config_path = args.config.expanduser().resolve()
    camera_path = args.camera_config.expanduser().resolve()
    dynamic_python = args.dynamic_python.expanduser().resolve()
    if not config_path.is_file() or not camera_path.is_file():
        raise FileNotFoundError("control or camera config does not exist")
    if not dynamic_python.is_file() or not os.access(dynamic_python, os.X_OK):
        raise FileNotFoundError("dynamic Python is not executable")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.execution_mode == "lift":
        _validate_loaded_mode(config, float(args.lift_height_m))
        raise ValueError(
            "loaded profile preflight passed, but the one-shot loaded executor "
            "is not yet connected; no motion was started"
        )
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    output = (
        _default_output()
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    if output.exists():
        raise FileExistsError("output directory already exists: {}".format(output))
    snapshot = (
        output / "official_snapshot.npz"
        if args.snapshot is None
        else args.snapshot.expanduser().resolve()
    )
    raw = output / "raw_snapshot.npz"
    selection_plan = output / "auto_selected_joint_plan.json"
    commands = _commands(
        args,
        output=output,
        snapshot=snapshot,
        raw=raw,
        selection_plan=selection_plan,
        selected_index=None,
    )
    if args.dry_run:
        print("[one-shot] DRY RUN: no hardware, camera, or GUI is opened")
        for name, command in commands.items():
            if name == "perception" and args.snapshot is not None:
                continue
            print("[dry-run:{}] {}".format(name, shlex.join(command)))
        print(
            "[dry-run] after automatic selection, prepare+viewer+execution run "
            "inside this same foreground command"
        )
        return output

    _host_preflight(config, args)
    _confirm(args)
    output.mkdir(parents=True, exist_ok=False)
    _run_stage("reset_rh56", commands["reset_rh56"])
    _run_stage("reset_franka", commands["reset_franka"])
    if args.snapshot is None:
        env = os.environ.copy()
        env["DEXGRASP_DYNAMIC_PYTHON"] = str(dynamic_python)
        _run_stage("bbox_sam_anydex", commands["perception"], env=env)
    elif not snapshot.is_file():
        raise FileNotFoundError("supplied snapshot does not exist: {}".format(snapshot))
    _run_stage("auto_select", commands["select"])
    selected_artifact = json.loads(selection_plan.read_text(encoding="utf-8"))
    selected = int(selected_artifact["candidate"]["index"])
    commands = _commands(
        args,
        output=output,
        snapshot=snapshot,
        raw=raw,
        selection_plan=selection_plan,
        selected_index=selected,
    )
    _run_stage("fresh_audit", commands["prepare"])
    _run_stage("viewer_and_execution", commands["execute"])
    print("[one-shot] completed output={} selected={}".format(output, selected))
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_once(args)
        return 0
    except KeyboardInterrupt:
        print(
            "[one-shot] Ctrl+C received; wait for the active child to confirm "
            "Franka stop and RH56 disable",
            file=sys.stderr,
        )
        return 130
    except StageError as exc:
        print("[one-shot] {}".format(exc), file=sys.stderr)
        return exc.returncode if exc.returncode else 2
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print("[one-shot] rejected before next stage: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
