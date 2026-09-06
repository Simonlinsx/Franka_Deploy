#!/usr/bin/env python3
"""Orchestrate Dynamic capture -> Official GPU inference -> Dynamic preview.

The process is perception-only.  It never imports or launches a Franka/RH56
driver.  The two Python environments remain separate subprocesses and exchange
only strict, pickle-free snapshot NPZ files.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Callable, Mapping, Optional, Sequence
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anydex_pipeline.snapshot import (  # noqa: E402
    VisualizationSnapshot,
    has_complete_official_provenance,
    load_snapshot_npz,
)
from anydex_pipeline.candidate_summary import candidate_summary_lines  # noqa: E402


DEFAULT_DYNAMIC_PYTHON = Path(
    os.environ.get(
        "DEXGRASP_WORKFLOW_DYNAMIC_PYTHON",
        os.environ.get("DEXGRASP_SHELL_PYTHON", sys.executable),
    )
)
DEFAULT_CAMERA_CONFIG = WORKSPACE / "perception/configs/d435_default.yaml"
DEFAULT_CONTROL_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
DEFAULT_CHECKPOINT = ROOT / "weights/logs/model/checkpoint.tar.18"
DEFAULT_MODEL_DIR = ROOT / "weights/logs/model/inspire_model/obj140"
DEFAULT_UPSTREAM = ROOT / "third_party/AnyDexGrasp"
DEFAULT_MANIFEST = ROOT / "weights/MANIFEST.sha256"
DEFAULT_ACTIVATION = ROOT / "scripts/activate_official_runtime.sh"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Perception-only two-environment workflow: Dynamic D435 capture/SAM2, "
            "Official AnyDex GPU inference, Dynamic live point-cloud/grasp preview"
        )
    )
    parser.add_argument("output", type=Path, help="final official snapshot .npz")
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="raw scene/object snapshot (default: OUTPUT stem + .raw.npz)",
    )
    parser.add_argument("--dynamic-python", type=Path, default=DEFAULT_DYNAMIC_PYTHON)
    parser.add_argument("--official-activation", type=Path, default=DEFAULT_ACTIVATION)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--control-config", type=Path, default=DEFAULT_CONTROL_CONFIG)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="pixel ROI for capture; omit for interactive selection",
    )
    parser.add_argument("--sam2", action="store_true")
    parser.add_argument("--scene-stride", type=int, default=4)
    parser.add_argument("--settle-valid-frames", type=int, default=5)
    parser.add_argument("--capture-timeout-s", type=float, default=20.0)

    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--inspire-model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--weights-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--decision-score-threshold", type=float, default=None)
    parser.add_argument(
        "--trust-official-checkpoints",
        action="store_true",
        help="required after manifest verification because official .pth files use pickle",
    )

    parser.add_argument("--selected-index", type=int, default=None)
    parser.add_argument(
        "--preview-source", choices=("realsense", "snapshot"), default="realsense"
    )
    parser.add_argument(
        "--preview-execution-mode",
        choices=("air", "contact"),
        default="air",
        help="target contract shown by the final viewer (default: audited air pose)",
    )
    parser.add_argument("--lift-preview-m", type=float, default=0.05)
    parser.add_argument(
        "--hand-mesh-resolution", choices=("full", "simplified"), default="simplified"
    )
    parser.add_argument("--no-target-hand-mesh", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--preview-validate-only", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate paths/assets and print all subprocess commands without camera/GPU",
    )
    return parser


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _raw_output_path(args: argparse.Namespace, official: Path) -> Path:
    if args.raw_output is not None:
        return _resolved(args.raw_output)
    return official.with_name(official.stem + ".raw.npz")


def _decision_paths(model_dir: Path) -> tuple[Path, ...]:
    root = model_dir / "480" if (model_dir / "480").is_dir() else model_dir
    output = []
    for grasp_type in range(1, 9):
        candidates = sorted((root / str(grasp_type)).glob("*.pth"))
        if len(candidates) != 1:
            raise ValueError(
                f"Inspire type {grasp_type} requires exactly one checkpoint; "
                f"found {len(candidates)} under {root / str(grasp_type)}"
            )
        output.append(candidates[0].resolve())
    return tuple(output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_weight_manifest(
    manifest_path: Path,
    checkpoint: Path,
    decisions: Sequence[Path],
) -> None:
    root = manifest_path.parent.resolve()
    entries: dict[Path, str] = {}
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read weights manifest {manifest_path}: {exc}") from exc
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"malformed weights manifest line {line_number}")
        digest, relative_text = parts
        if any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"malformed SHA-256 on manifest line {line_number}")
        relative = Path(relative_text.lstrip("*"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path on line {line_number}: {relative}")
        resolved = (root / relative).resolve()
        if resolved in entries:
            raise ValueError(f"duplicate manifest entry: {relative}")
        entries[resolved] = digest

    required = (checkpoint.resolve(),) + tuple(
        Path(item).resolve() for item in decisions
    )
    missing = [str(path) for path in required if path not in entries]
    if missing:
        raise ValueError(
            "requested official weights are not bound by manifest: "
            + "; ".join(missing)
        )
    for path in required:
        if not path.is_file():
            raise ValueError(f"manifest-bound weight is missing: {path}")
        actual = _sha256(path)
        if actual != entries[path]:
            raise ValueError(
                f"official weight SHA-256 mismatch: {path}; "
                f"expected={entries[path]} actual={actual}"
            )
    print(f"[weights] manifest PASS: {manifest_path} ({len(required)} files)")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")


def validate_request(args: argparse.Namespace) -> Mapping[str, Path]:
    official = _resolved(args.output)
    raw = _raw_output_path(args, official)
    dynamic_python = _resolved(args.dynamic_python)
    activation = _resolved(args.official_activation)
    camera_config = _resolved(args.camera_config)
    control_config = _resolved(args.control_config)
    checkpoint = _resolved(args.checkpoint)
    model_dir = _resolved(args.inspire_model_dir)
    upstream = _resolved(args.upstream_root)
    manifest = _resolved(args.weights_manifest)

    if official.suffix.lower() != ".npz" or raw.suffix.lower() != ".npz":
        raise ValueError("official output and raw output must both end in .npz")
    if official == raw:
        raise ValueError("official output and raw output must be different files")
    non_files = [
        str(path)
        for path in (raw, official)
        if path.exists() and not path.is_file()
    ]
    if non_files:
        raise ValueError(
            "output path exists but is not a regular file: " + "; ".join(non_files)
        )
    # Dry-run never creates, replaces, or publishes either output.  Existing
    # regular files are therefore valid command-preview destinations, while
    # directories and aliasing remain request errors above.
    if not args.dry_run and not args.overwrite:
        existing = [str(path) for path in (raw, official) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing output: " + "; ".join(existing)
            )
    if not dynamic_python.is_file() or not os.access(dynamic_python, os.X_OK):
        raise ValueError(f"dynamic Python is not executable: {dynamic_python}")
    for path, label in (
        (activation, "official activation script"),
        (camera_config, "camera config"),
        (control_config, "control config"),
        (checkpoint, "representation checkpoint"),
        (manifest, "weights manifest"),
        (ROOT / "apps/capture_raw_object_snapshot.py", "raw capture app"),
        (ROOT / "apps/check_official_inference_runtime.py", "official preflight app"),
        (ROOT / "apps/infer_snapshot.py", "official inference app"),
        (ROOT / "apps/live_pipeline_preview.py", "live preview app"),
    ):
        _require_file(path, label)
    if not upstream.is_dir():
        raise ValueError(f"official upstream root does not exist: {upstream}")

    if args.roi is not None:
        x1, y1, x2, y2 = (int(value) for value in args.roi)
        if min(x1, y1, x2, y2) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("--roi requires non-negative X1 Y1 X2 Y2 with X2>X1,Y2>Y1")
    if args.scene_stride < 1 or args.settle_valid_frames < 1:
        raise ValueError("--scene-stride and --settle-valid-frames must be >= 1")
    if not math.isfinite(args.capture_timeout_s) or args.capture_timeout_s <= 0:
        raise ValueError("--capture-timeout-s must be finite and positive")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.selected_index is not None:
        if args.selected_index < 0:
            raise ValueError("--selected-index must be non-negative")
        if args.selected_index >= args.top_k:
            raise ValueError("--selected-index must be smaller than --top-k")
    if args.decision_score_threshold is not None and not math.isfinite(
        args.decision_score_threshold
    ):
        raise ValueError("--decision-score-threshold must be finite")
    if not str(args.device).strip().startswith("cuda"):
        raise ValueError("--device must select CUDA for official inference")
    if not args.trust_official_checkpoints:
        raise ValueError(
            "official Inspire .pth files are pickle-backed; pass "
            "--trust-official-checkpoints after verifying their source"
        )
    if not math.isfinite(args.lift_preview_m) or args.lift_preview_m <= 0:
        raise ValueError("--lift-preview-m must be finite and positive")
    if not math.isfinite(args.duration) or args.duration < 0:
        raise ValueError("--duration must be finite and non-negative")

    decisions = _decision_paths(model_dir)
    _verify_weight_manifest(manifest, checkpoint, decisions)
    return {
        "official": official,
        "raw": raw,
        "dynamic_python": dynamic_python,
        "activation": activation,
        "camera_config": camera_config,
        "control_config": control_config,
        "checkpoint": checkpoint,
        "model_dir": model_dir,
        "upstream": upstream,
    }


def _official_python_command(activation: Path, arguments: Sequence[str]) -> list[str]:
    code = 'set -euo pipefail; source "$1"; shift; exec "$DEXGRASP_OFFICIAL_PYTHON" "$@"'
    return [
        "bash",
        "-c",
        code,
        "dexgrasp-official-stage",
        str(activation),
        *[str(value) for value in arguments],
    ]


def build_stage_commands(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    raw_temp: Path,
    official_temp: Path,
) -> Mapping[str, list[str]]:
    dynamic = str(paths["dynamic_python"])
    capture = [
        dynamic,
        str(ROOT / "apps/capture_raw_object_snapshot.py"),
        "--output",
        str(raw_temp),
        "--config",
        str(paths["camera_config"]),
        "--scene-stride",
        str(args.scene_stride),
        "--settle-valid-frames",
        str(args.settle_valid_frames),
        "--capture-timeout-s",
        str(args.capture_timeout_s),
    ]
    if args.roi is not None:
        capture.extend(("--roi", *[str(value) for value in args.roi]))
    if args.sam2:
        capture.append("--sam2")

    preflight = _official_python_command(
        paths["activation"],
        [
            str(ROOT / "apps/check_official_inference_runtime.py"),
            "--checkpoint",
            str(paths["checkpoint"]),
            "--inspire-model-dir",
            str(paths["model_dir"]),
            "--upstream-root",
            str(paths["upstream"]),
            "--device",
            str(args.device),
        ],
    )
    inference_args = [
        str(ROOT / "apps/infer_snapshot.py"),
        str(raw_temp),
        "--output",
        str(official_temp),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--inspire-model-dir",
        str(paths["model_dir"]),
        "--upstream-root",
        str(paths["upstream"]),
        "--device",
        str(args.device),
        "--top-k",
        str(args.top_k),
        "--trust-official-checkpoints",
    ]
    if args.decision_score_threshold is not None:
        inference_args.extend(
            ("--decision-score-threshold", str(args.decision_score_threshold))
        )
    inference = _official_python_command(paths["activation"], inference_args)

    selected = 0 if args.selected_index is None else int(args.selected_index)
    preview = [
        dynamic,
        str(ROOT / "apps/live_pipeline_preview.py"),
        str(paths["official"]),
        "--control-config",
        str(paths["control_config"]),
        "--camera-config",
        str(paths["camera_config"]),
        "--selected-index",
        str(selected),
        "--source",
        str(args.preview_source),
        "--execution-mode",
        str(args.preview_execution_mode),
        "--lift-preview-m",
        str(args.lift_preview_m),
        "--scene-stride",
        str(args.scene_stride),
        "--max-candidates",
        str(args.top_k),
        "--hand-mesh-resolution",
        str(args.hand_mesh_resolution),
        "--duration",
        str(args.duration),
    ]
    if args.no_target_hand_mesh:
        preview.append("--no-target-hand-mesh")
    if args.preview_validate_only:
        preview.append("--validate-only")
    return {
        "official_preflight": preflight,
        "dynamic_capture": capture,
        "official_inference": inference,
        "dynamic_preview": preview,
    }


def _validate_raw_snapshot(snapshot: VisualizationSnapshot) -> None:
    if snapshot.grasps.count != 0 or snapshot.grasps.selected_index != -1:
        raise ValueError("raw snapshot must contain no grasp candidates")
    if len(snapshot.object_points) < 20 or len(snapshot.scene_points) < 1:
        raise ValueError("raw snapshot has insufficient scene/object points")
    if snapshot.reference_frame != "robot_base":
        raise ValueError("raw snapshot reference frame must be robot_base")
    if not snapshot.camera_serial or not snapshot.calibration_id:
        raise ValueError("raw snapshot lacks camera/calibration provenance")
    if snapshot.representation_checkpoint_sha256 or snapshot.decision_checkpoint_sha256s:
        raise ValueError("raw snapshot unexpectedly contains model provenance")


def _validate_official_snapshot(
    raw: VisualizationSnapshot,
    official: VisualizationSnapshot,
) -> None:
    if official.grasps.count < 1:
        raise ValueError("official inference returned no candidates")
    if official.grasps.hand_poses is None or official.grasps.hand_angles is None:
        raise ValueError("official snapshot lacks Inspire hand poses/angles")
    if np.asarray(official.grasps.hand_angles).shape != (official.grasps.count, 6):
        raise ValueError("official Inspire hand angles must have shape [K,6]")
    if not has_complete_official_provenance(official):
        raise ValueError("official snapshot model/checkpoint/source provenance is incomplete")
    if "AnyDexGrasp official" not in official.model_name:
        raise ValueError(f"unexpected official model identity: {official.model_name!r}")
    scalar_fields = (
        "reference_frame",
        "frame_id",
        "timestamp_s",
        "calibration_id",
        "camera_serial",
        "scene_excludes_object",
    )
    for name in scalar_fields:
        if getattr(official, name) != getattr(raw, name):
            raise ValueError(f"official inference changed capture field {name}")
    array_fields = (
        "T_reference_camera",
        "scene_points",
        "scene_colors",
        "object_points",
        "object_colors",
    )
    for name in array_fields:
        if not np.array_equal(np.asarray(getattr(official, name)), np.asarray(getattr(raw, name))):
            raise ValueError(f"official inference changed capture array {name}")


def _validate_selected_index(snapshot: VisualizationSnapshot, selected_index: int) -> None:
    if not 0 <= int(selected_index) < snapshot.grasps.count:
        raise ValueError(
            f"selected index {selected_index} is outside official candidates "
            f"0..{snapshot.grasps.count - 1}"
        )


def _print_candidate_summary(
    snapshot: VisualizationSnapshot,
    *,
    selected_index: int,
    requested_top_k: int,
) -> None:
    for line in candidate_summary_lines(
        snapshot,
        selected_index=selected_index,
        requested_top_k=requested_top_k,
    ):
        print(line, flush=True)


def _unlink_if_same_file(path: Path, source: Path) -> None:
    """Remove ``path`` only when it is still the hard-link we published."""

    try:
        if path.exists() and source.exists() and os.path.samefile(path, source):
            path.unlink()
    except OSError:
        # The original publication error remains the useful exception.  Never
        # unlink a path whose identity can no longer be proven.
        pass


def _publish_without_overwrite(
    pairs: Sequence[tuple[Path, Path]],
) -> None:
    """Publish with atomic no-clobber hard links; official must be last.

    A pre-publication ``exists`` check has a TOCTOU race.  ``os.link`` performs
    creation and the EEXIST check in one kernel operation.  The temporary and
    final names are deliberately in the same directory/filesystem.
    """

    created: list[tuple[Path, Path]] = []
    try:
        for temporary, destination in pairs:
            os.link(temporary, destination)
            created.append((temporary, destination))
    except BaseException:
        for temporary, destination in reversed(created):
            _unlink_if_same_file(destination, temporary)
        raise


def _publish_with_rollback(
    pairs: Sequence[tuple[Path, Path]],
    *,
    token: str,
) -> None:
    """Replace a validated pair and restore the old pair on ordinary failure."""

    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        non_files = [
            str(destination)
            for _, destination in pairs
            if destination.exists() and not destination.is_file()
        ]
        if non_files:
            raise ValueError(
                "output path became a non-file before publication: "
                + "; ".join(non_files)
            )
        for _, destination in pairs:
            if destination.exists():
                backup = destination.with_name(
                    f".{destination.name}.{token}.workflow-backup"
                )
                os.replace(destination, backup)
                backups.append((destination, backup))
        for temporary, destination in pairs:
            os.replace(temporary, destination)
            published.append(destination)
    except BaseException as publish_error:
        rollback_errors: list[str] = []
        for destination in reversed(published):
            try:
                destination.unlink(missing_ok=True)
            except OSError as exc:
                rollback_errors.append(f"remove {destination}: {exc}")
        for destination, backup in reversed(backups):
            try:
                os.replace(backup, destination)
            except OSError as exc:
                rollback_errors.append(f"restore {destination}: {exc}")
        if rollback_errors:
            raise RuntimeError(
                "output publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from publish_error
        raise
    else:
        for _, backup in backups:
            try:
                backup.unlink(missing_ok=True)
            except OSError as exc:
                print(f"[warning] stale workflow backup remains: {backup}: {exc}")


def _publish_outputs(
    *,
    raw_temp: Path,
    raw_output: Path,
    official_temp: Path,
    official_output: Path,
    overwrite: bool,
    token: str,
) -> None:
    # Raw is published first and the self-contained official snapshot last.
    # Thus the official path is the workflow's commit marker: observing a new
    # official file implies its matching raw capture was already published.
    pairs = ((raw_temp, raw_output), (official_temp, official_output))
    if overwrite:
        _publish_with_rollback(pairs, token=token)
    else:
        _publish_without_overwrite(pairs)


StageRunner = Callable[[str, Sequence[str]], None]


def _run_stage(name: str, command: Sequence[str]) -> None:
    print(f"\n[stage:{name}] {shlex.join([str(value) for value in command])}", flush=True)
    subprocess.run([str(value) for value in command], check=True)


def run_workflow(
    args: argparse.Namespace,
    *,
    stage_runner: StageRunner = _run_stage,
) -> tuple[Path, Path]:
    paths = validate_request(args)
    token = uuid.uuid4().hex
    raw_temp = paths["raw"].with_name(f".{paths['raw'].stem}.{token}.tmp.npz")
    official_temp = paths["official"].with_name(
        f".{paths['official'].stem}.{token}.tmp.npz"
    )
    commands = build_stage_commands(args, paths, raw_temp, official_temp)
    print("[safety] perception-only: D435/file/GPU stages; no Franka or Inspire process")
    print(f"[workflow] raw={paths['raw']}")
    print(f"[workflow] official={paths['official']}")
    if args.dry_run:
        for name, command in commands.items():
            if name == "dynamic_preview" and args.no_preview:
                continue
            print(f"[dry-run:{name}] {shlex.join(command)}")
        return paths["raw"], paths["official"]

    paths["raw"].parent.mkdir(parents=True, exist_ok=True)
    paths["official"].parent.mkdir(parents=True, exist_ok=True)
    selected = 0 if args.selected_index is None else int(args.selected_index)
    try:
        # Check GPU/native extensions before opening the camera.
        stage_runner("official_preflight", commands["official_preflight"])
        stage_runner("dynamic_capture", commands["dynamic_capture"])
        raw_snapshot = load_snapshot_npz(raw_temp)
        _validate_raw_snapshot(raw_snapshot)
        stage_runner("official_inference", commands["official_inference"])
        official_snapshot = load_snapshot_npz(official_temp)
        _validate_official_snapshot(raw_snapshot, official_snapshot)
        _print_candidate_summary(
            official_snapshot,
            selected_index=selected,
            requested_top_k=args.top_k,
        )
        _validate_selected_index(official_snapshot, selected)

        _publish_outputs(
            raw_temp=raw_temp,
            raw_output=paths["raw"],
            official_temp=official_temp,
            official_output=paths["official"],
            overwrite=bool(args.overwrite),
            token=token,
        )
        print(
            "[workflow] official snapshot PASS candidates={} selected={} raw={} official={}".format(
                official_snapshot.grasps.count,
                selected,
                paths["raw"],
                paths["official"],
            )
        )
        if not args.no_preview:
            stage_runner("dynamic_preview", commands["dynamic_preview"])
        return paths["raw"], paths["official"]
    finally:
        for temporary in (raw_temp, official_temp):
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_workflow(args)
        return 0
    except KeyboardInterrupt:
        print(
            "[interrupted] perception workflow stopped; no robot/hand was connected",
            file=sys.stderr,
        )
        return 130
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
