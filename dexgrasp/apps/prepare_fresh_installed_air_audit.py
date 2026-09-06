#!/usr/bin/python3
"""Prepare one fresh installed-tool air-audit evidence bundle.

The only device opened by this workflow is the explicitly selected D435.  It
never imports a Franka or Inspire driver and never issues a motion command.
The caller supplies a stationary, read-only Franka joint observation.  A
future executor must still read live q again and satisfy every runtime gate.

The workflow keeps two point-cloud roles separate:

* the official snapshot's object cloud binds the AnyDex candidate and pose;
* the newly captured full scene supplies observed environment obstacles.

Before object returns are removed from the latter, the filter requires the
saved object cloud to align with the new observation.  It never ICP-adjusts an
old grasp pose to make a moved object appear valid.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from anydex_pipeline.air_audit_readiness import (  # noqa: E402
    validate_air_candidate_commissioning,
)
from anydex_pipeline.control_config import load_control_config  # noqa: E402
from anydex_pipeline.control_plan import is_official_snapshot  # noqa: E402
from anydex_pipeline.installed_tool_audit import (  # noqa: E402
    load_installed_tool_audit,
)
from anydex_pipeline.rh56_commissioning import (  # noqa: E402
    atomic_write_evidence,
    seal_evidence,
)
from anydex_pipeline.snapshot import load_snapshot_npz  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
DEFAULT_CAMERA_CONFIG = WORKSPACE / "perception/configs/d435_default.yaml"
DEFAULT_ANYDEX_ROOT = ROOT / "third_party/AnyDexGrasp"
STATIONARY_Q_TOKEN = "CURRENT_Q_READ_ONLY_AND_STATIONARY"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _q(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError("--capture-q-rad must contain seven finite radians")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Camera-only/offline plan -> fresh D435 capture -> installed-return "
            "filter -> schema-v2 conditional air-audit preparation. No Franka "
            "or RH56 transport is opened."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, default=51)
    parser.add_argument(
        "--capture-q-rad",
        type=float,
        nargs=7,
        required=True,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
        help="stationary read-only Franka q that remains unchanged during capture",
    )
    parser.add_argument(
        "--confirm-stationary-q",
        metavar=STATIONARY_Q_TOKEN,
        required=True,
        help=(
            "exact acknowledgement that capture q came from a fresh read-only "
            "state and Franka will remain stationary"
        ),
    )
    parser.add_argument("--q7-rad", type=float, default=1.3525)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anydex-root", type=Path, default=DEFAULT_ANYDEX_ROOT)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--capture-frames", type=int, default=5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print child commands without opening D435 or writing files",
    )
    return parser


def _preflight(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.confirm_stationary_q != STATIONARY_Q_TOKEN:
        raise ValueError(
            "--confirm-stationary-q must exactly equal {}".format(
                STATIONARY_Q_TOKEN
            )
        )
    if args.warmup_frames < 0 or args.capture_frames < 1:
        raise ValueError("capture frame counts are invalid")
    q_capture = _q(args.capture_q_rad)
    config, config_path = load_control_config(args.config)
    snapshot_path = args.snapshot.expanduser().resolve()
    camera_config = args.camera_config.expanduser().resolve()
    anydex_root = args.anydex_root.expanduser().resolve()
    for path, name in (
        (snapshot_path, "snapshot"),
        (camera_config, "camera config"),
        (anydex_root, "AnyDex root"),
    ):
        if not path.exists():
            raise FileNotFoundError("{} does not exist: {}".format(name, path))
    snapshot = load_snapshot_npz(snapshot_path)
    if not is_official_snapshot(snapshot):
        raise ValueError("snapshot is not provenance-complete official AnyDexGrasp output")
    if snapshot.reference_frame != "robot_base":
        raise ValueError("snapshot reference_frame must be robot_base")
    if snapshot.calibration_id != str(config["calibration"]["id"]):
        raise ValueError("snapshot/control-profile calibration_id mismatch")
    if snapshot.camera_serial != str(config["calibration"]["camera_serial"]):
        raise ValueError("snapshot/control-profile camera serial mismatch")
    index = int(args.candidate_index)
    if index < 0 or index >= snapshot.grasps.count:
        raise ValueError("--candidate-index is outside the snapshot")
    if snapshot.grasps.hand_angles is None or snapshot.grasps.hand_poses is None:
        raise ValueError("selected candidate has no RH56 pose/register targets")
    target = np.asarray(snapshot.grasps.hand_angles[index], dtype=np.float64)
    limits = np.asarray(config["franka"]["joint_limits_rad"], dtype=np.float64)
    margin = float(config["franka"]["joint_limit_margin_rad"])
    if np.any(q_capture < limits[:, 0] + margin) or np.any(
        q_capture > limits[:, 1] - margin
    ):
        raise ValueError("--capture-q-rad violates configured Franka joint margins")
    commissioning_blocker = ""
    try:
        validate_air_candidate_commissioning(config, target)
    except ValueError as exc:
        # Capture/filter remain useful diagnostic evidence, but no schema-v2
        # PASS can be claimed until this separate hardware evidence exists.
        commissioning_blocker = str(exc)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            "output directory already exists; choose a new run directory: {}".format(
                output_dir
            )
        )
    return {
        "config": config,
        "config_path": config_path,
        "snapshot": snapshot,
        "snapshot_path": snapshot_path,
        "camera_config": camera_config,
        "anydex_root": anydex_root,
        "candidate_index": index,
        "selected_targets": tuple(int(value) for value in np.rint(target)),
        "q_capture": q_capture,
        "output_dir": output_dir,
        "commissioning_blocker": commissioning_blocker,
    }


def _commands(context: Mapping[str, Any], args: argparse.Namespace) -> Mapping[str, list[str]]:
    output = Path(context["output_dir"])
    q = ["{:.17g}".format(float(value)) for value in context["q_capture"]]
    config = str(context["config_path"])
    snapshot = str(context["snapshot_path"])
    anydex = str(context["anydex_root"])
    index = str(context["candidate_index"])
    return {
        "plan": [
            str(ROOT / "scripts/plan_installed_air_candidate.sh"),
            "--snapshot", snapshot,
            "--config", config,
            "--candidate-index", index,
            "--q7-rad", "{:.17g}".format(float(args.q7_rad)),
            "--start-q-rad", *q,
            "--output", str(output / "joint_plan.json"),
        ],
        "capture": [
            str(ROOT / "scripts/capture_live_scene.sh"),
            "--config", str(context["camera_config"]),
            "--output", str(output / "live_scene.npz"),
            "--capture-q-rad", *q,
            "--expected-camera-serial", str(context["snapshot"].camera_serial),
            "--expected-calibration-id", str(context["snapshot"].calibration_id),
            "--warmup-frames", str(int(args.warmup_frames)),
            "--capture-frames", str(int(args.capture_frames)),
        ],
        "filter": [
            str(ROOT / "scripts/filter_installed_scene.sh"),
            "--live-scene", str(output / "live_scene.npz"),
            "--snapshot", snapshot,
            "--output", str(output / "filtered_scene.npz"),
            "--evidence", str(output / "filtered_scene.evidence.json"),
            "--anydex-root", anydex,
        ],
        "bootstrap_capture": [
            str(ROOT / "scripts/capture_live_scene.sh"),
            "--config", str(context["camera_config"]),
            "--output", str(output / "bootstrap_live_scene.npz"),
            "--capture-q-rad", *q,
            "--expected-camera-serial", str(context["snapshot"].camera_serial),
            "--expected-calibration-id", str(context["snapshot"].calibration_id),
            "--warmup-frames", str(int(args.warmup_frames)),
            "--capture-frames", str(int(args.capture_frames)),
        ],
        "bootstrap_filter": [
            str(ROOT / "scripts/filter_installed_scene.sh"),
            "--live-scene", str(output / "bootstrap_live_scene.npz"),
            "--snapshot", snapshot,
            "--output", str(output / "bootstrap_filtered_scene.npz"),
            "--evidence", str(output / "bootstrap_filtered_scene.evidence.json"),
            "--anydex-root", anydex,
        ],
        "static_cache": [
            str(ROOT / "scripts/generate_installed_air_audit.sh"),
            "--config", config,
            "--snapshot", snapshot,
            "--filtered-scene", str(output / "bootstrap_filtered_scene.npz"),
            "--candidate-index", index,
            "--joint-plan", str(output / "joint_plan.json"),
            "--anydex-root", anydex,
            "--build-static-cache",
            "--output", str(output / "static_collision_cache.json"),
        ],
        "audit": [
            str(ROOT / "scripts/generate_installed_air_audit.sh"),
            "--config", config,
            "--snapshot", snapshot,
            "--filtered-scene", str(output / "filtered_scene.npz"),
            "--candidate-index", index,
            "--joint-plan", str(output / "joint_plan.json"),
            "--anydex-root", anydex,
            "--static-cache", str(output / "static_collision_cache.json"),
            "--output", str(output / "installed_air_audit_v2.json"),
        ],
    }


def _output_bindings(output_dir: Path) -> Mapping[str, Any]:
    result = {}
    for name in (
        "joint_plan.json",
        "bootstrap_live_scene.npz",
        "bootstrap_filtered_scene.npz",
        "bootstrap_filtered_scene.evidence.json",
        "static_collision_cache.json",
        "live_scene.npz",
        "filtered_scene.npz",
        "filtered_scene.evidence.json",
        "installed_air_audit_v2.json",
    ):
        path = output_dir / name
        result[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": _sha256_file(path) if path.is_file() else "",
        }
    return result


def _write_receipt(
    context: Mapping[str, Any],
    *,
    status: str,
    completed_steps: Sequence[str],
    blockers: Sequence[str],
    audit: Optional[Mapping[str, Any]] = None,
) -> Path:
    output_dir = Path(context["output_dir"])
    payload = seal_evidence(
        {
            "schema_version": 1,
            "artifact_type": "fresh_installed_air_audit_preparation_receipt",
            "created_at_unix_s": time.time(),
            "status": status,
            "completed_steps": list(completed_steps),
            "blockers": list(blockers),
            "inputs": {
                "config": {
                    "path": str(context["config_path"]),
                    "sha256": _sha256_file(Path(context["config_path"])),
                },
                "snapshot": {
                    "path": str(context["snapshot_path"]),
                    "sha256": _sha256_file(Path(context["snapshot_path"])),
                },
                "candidate_index": int(context["candidate_index"]),
                "selected_targets": list(context["selected_targets"]),
                "capture_q_rad": np.asarray(
                    context["q_capture"], dtype=np.float64
                ).tolist(),
                "capture_q_source": "operator-supplied fresh read-only state",
                "stationary_q_confirmation": STATIONARY_Q_TOKEN,
            },
            "outputs": _output_bindings(output_dir),
            "audit_decision": (
                dict(audit["decision"]) if audit is not None else None
            ),
            "point_cloud_roles": {
                "anydex_candidate_and_object_geometry": "snapshot:object_points",
                "observed_environment_geometry": "filtered_scene.npz:filtered_scene_points",
                "object_alignment_required_before_filtering": True,
                "old_grasp_pose_icp_adjustment_allowed": False,
                "unseen_camera_space_authoritative": False,
            },
            "robot_or_hand_transport_opened": False,
            "motion_authorized": False,
        }
    )
    path = output_dir / "preparation_receipt.json"
    atomic_write_evidence(path, payload)
    return path


def prepare(
    args: argparse.Namespace,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    context = _preflight(args)
    output_dir = Path(context["output_dir"])
    commands = _commands(context, args)
    if bool(args.dry_run):
        print(
            "[dry-run] no D435/Franka/RH56 connection; no directory or artifact is created"
        )
        dry_steps = (
            ("plan", "capture", "filter")
            if str(context["commissioning_blocker"])
            else (
                "plan",
                "bootstrap_capture",
                "bootstrap_filter",
                "static_cache",
                "capture",
                "filter",
                "audit",
            )
        )
        for step in dry_steps:
            print("[dry-run][{}] {}".format(step, shlex.join(commands[step])))
        blocker = str(context["commissioning_blocker"])
        if blocker:
            print("[dry-run][AUDIT LOCKED] six-axis commissioning gate: {}".format(blocker))
            return 3
        print("[dry-run] commissioning precondition is ready; geometry is not evaluated")
        return 0
    output_dir.mkdir(parents=True, exist_ok=False)
    completed = []
    print(
        "[safety] D435 perception + offline geometry only; no Franka/RH56 "
        "driver is imported or opened"
    )
    print(
        "[authority] snapshot object_points bind candidate/pose; fresh filtered "
        "scene supplies observed environment only; unseen space remains conditional"
    )
    commissioning_blocker = str(context["commissioning_blocker"])
    preparation_steps = (
        ("plan", "capture", "filter")
        if commissioning_blocker
        else (
            "plan",
            "bootstrap_capture",
            "bootstrap_filter",
            "static_cache",
            # This second capture starts the unchanged 120 s freshness clock.
            "capture",
            "filter",
        )
    )
    for step in preparation_steps:
        if step == "static_cache":
            print(
                "[fresh-air-audit] step=static_cache "
                "(offline exact-mesh audit; typically several minutes; "
                "progress follows)",
                flush=True,
            )
        else:
            print("[fresh-air-audit] step={}".format(step), flush=True)
        result = runner(commands[step], cwd=str(ROOT), check=False)
        if int(result.returncode) != 0:
            blocker = "{} command failed with exit {}".format(step, result.returncode)
            receipt = _write_receipt(
                context,
                status="failed",
                completed_steps=completed,
                blockers=(blocker,),
            )
            print("[fresh-air-audit][FAILED] {}".format(blocker))
            print("[fresh-air-audit] receipt={}".format(receipt))
            return int(result.returncode) or 2
        completed.append(step)

    if commissioning_blocker:
        blocker = "six-axis commissioning gate: {}".format(
            commissioning_blocker
        )
        receipt = _write_receipt(
            context,
            status="audit_locked",
            completed_steps=completed,
            blockers=(blocker,),
        )
        print("[fresh-air-audit][AUDIT LOCKED] {}".format(blocker))
        print(
            "[fresh-air-audit] capture/filter were retained for diagnostics, but "
            "their 120 s freshness cannot be reused after commissioning"
        )
        print("[fresh-air-audit] receipt={}".format(receipt))
        return 3

    print("[fresh-air-audit] step=audit")
    result = runner(commands["audit"], cwd=str(ROOT), check=False)
    audit_path = output_dir / "installed_air_audit_v2.json"
    audit = None
    blockers = []
    if audit_path.is_file():
        audit = load_installed_tool_audit(
            audit_path, verify_files=True, require_pass=False
        )
        blockers.extend(str(value) for value in audit["decision"]["reasons"])
    if int(result.returncode) != 0 or audit is None or not audit["decision"]["passed"]:
        if audit is None:
            blockers.append(
                "audit command failed with exit {} and wrote no valid artifact".format(
                    result.returncode
                )
            )
        receipt = _write_receipt(
            context,
            status="audit_locked",
            completed_steps=completed + (["audit"] if audit is not None else []),
            blockers=blockers,
            audit=audit,
        )
        print("[fresh-air-audit][AUDIT LOCKED]")
        for blocker in blockers:
            print("[fresh-air-audit][blocker] {}".format(blocker))
        print("[fresh-air-audit] receipt={}".format(receipt))
        return 3

    completed.append("audit")
    captured_at = float(audit["bindings"]["scene"]["captured_at_s"])
    max_age = float(audit["policies"]["max_scene_age_s"])
    expires_at = captured_at + max_age
    remaining = expires_at - time.time()
    if remaining <= 0.0:
        blocker = (
            "fresh scene expired during offline audit generation; recapture into "
            "a new output directory"
        )
        receipt = _write_receipt(
            context,
            status="audit_locked",
            completed_steps=completed + ["audit"],
            blockers=(blocker,),
            audit=audit,
        )
        print("[fresh-air-audit][AUDIT LOCKED] {}".format(blocker))
        print("[fresh-air-audit] receipt={}".format(receipt))
        return 3
    receipt = _write_receipt(
        context,
        status="evidence_pass",
        completed_steps=completed,
        blockers=(),
        audit=audit,
    )
    print(
        "[fresh-air-audit] EVIDENCE-PASS expires_at_unix_s={:.6f} "
        "remaining_now_s={:.2f}".format(expires_at, remaining)
    )
    print("[fresh-air-audit] audit={}".format(audit_path))
    print("[fresh-air-audit] receipt={}".format(receipt))
    print(
        "[fresh-air-audit] motion_authorized=false; executor live-q, scene-age, "
        "workspace-clear, immediate-stop, cable, load, and hardware gates remain"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return prepare(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print("[fatal] {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
