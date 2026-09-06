#!/usr/bin/env python3
"""Live D435 + saved AnyDex target + optional current-EEF preview.

This process is deliberately perception-only.  It never imports libfranka or
the Inspire serial driver.  Current EEF telemetry, when requested, is consumed
either from a legacy atomically replaced waypoint JSON file or from the
reviewed read-only native mapping, so the viewer can never compete for device
ownership or issue motion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import importlib
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SRC = ROOT / "src"
DYNAMIC_PCD_ROOT = WORKSPACE / "perception"
for path in (SRC, DYNAMIC_PCD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anydex_pipeline.control_config import load_control_config  # noqa: E402
from anydex_pipeline.continuous_telemetry import (  # noqa: E402
    NATIVE_TELEMETRY_ABI_SCHEMA_SHA256,
    NativeContinuousTelemetryAdapter,
    NativeTelemetryVisibility,
    native_viewer_feedback,
    telemetry_identity_from_native_header,
)
from anydex_pipeline.air_target_poses import derive_air_target_poses  # noqa: E402
from anydex_pipeline.control_plan import (  # noqa: E402
    ExecutionConfig,
    build_control_plan,
)
from anydex_pipeline.pipeline_preview import (  # noqa: E402
    PipelinePreviewContract,
    PoseState,
    build_pipeline_preview_contract,
    load_pose_state_json,
    pose_error,
)
from anydex_pipeline.snapshot import (  # noqa: E402
    GraspCandidates,
    VisualizationSnapshot,
    load_snapshot_npz,
)
from anydex_pipeline.telemetry_session_manifest import (  # noqa: E402
    LoadedTelemetrySessionManifest,
    load_telemetry_session_manifest,
    sha256_file,
)
from anydex_pipeline.viewer_ready import (  # noqa: E402
    normalize_unfollowed_leaf,
    publish_viewer_ready,
)


DEFAULT_CONTROL_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
DEFAULT_CAMERA_CONFIG = DYNAMIC_PCD_ROOT / "configs/d435_default.yaml"


@dataclass(frozen=True)
class PreviewInputs:
    snapshot: VisualizationSnapshot
    control_config: dict[str, Any]
    contract: PipelinePreviewContract
    execution_mode: str
    telemetry_manifest: Optional[LoadedTelemetrySessionManifest] = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Perception-only live preview: calibrated D435 scene, saved AnyDex "
            "object/target/hand mesh, optional current EEF/hand pose, and pose error"
        )
    )
    parser.add_argument("snapshot", type=Path, help="saved AnyDex snapshot NPZ")
    parser.add_argument(
        "--control-config", type=Path, default=DEFAULT_CONTROL_CONFIG
    )
    parser.add_argument(
        "--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG
    )
    parser.add_argument("--selected-index", type=int, default=None)
    parser.add_argument(
        "--source",
        choices=("realsense", "snapshot"),
        default="realsense",
        help="realsense refreshes the calibrated scene; snapshot is offline",
    )
    feedback_source = parser.add_mutually_exclusive_group()
    feedback_source.add_argument(
        "--pose-state",
        type=Path,
        default=None,
        help=(
            "optional atomically replaced JSON containing current T_reference_EE; "
            "the viewer never opens a Franka connection"
        ),
    )
    feedback_source.add_argument(
        "--continuous-telemetry",
        type=Path,
        default=None,
        help=(
            "read-only native telemetry mapping; requires an exact immutable "
            "--telemetry-session-manifest and never opens a device"
        ),
    )
    parser.add_argument(
        "--telemetry-session-manifest",
        type=Path,
        default=None,
        help="immutable run/artifact identity for --continuous-telemetry",
    )
    parser.add_argument(
        "--continuous-telemetry-python-dir",
        type=Path,
        default=(
            None
            if not os.environ.get("ANYDEX_TELEMETRY_PYTHON_DIR")
            else Path(os.environ["ANYDEX_TELEMETRY_PYTHON_DIR"])
        ),
        help=(
            "directory containing the reviewed _anydex_telemetry reader module; "
            "defaults to ANYDEX_TELEMETRY_PYTHON_DIR then normal import paths"
        ),
    )
    parser.add_argument(
        "--telemetry-read-attempts",
        type=int,
        default=8,
        help="bounded native double-slot read attempts per stream",
    )
    parser.add_argument(
        "--telemetry-wait-seconds",
        type=float,
        default=60.0,
        help=(
            "continuous mode only: wait this long for the producer to create "
            "and commit the read-only mapping; 0 performs one immediate open"
        ),
    )
    parser.add_argument(
        "--pose-max-age-s", type=float, default=0.50,
        help="reject current-pose samples older than this many seconds",
    )
    parser.add_argument(
        "--arm-max-age-s",
        type=float,
        default=0.25,
        help="continuous mode: independently hide older measured arm feedback",
    )
    parser.add_argument(
        "--hand-max-age-s",
        type=float,
        default=0.75,
        help="continuous mode: independently hide older RH56 ANGLE_ACT readback",
    )
    parser.add_argument(
        "--error-target",
        choices=("auto", "pregrasp", "grasp", "lift"),
        default="auto",
        help=(
            "pose-error target; auto follows executor telemetry and falls back "
            "to the matching pregrasp/grasp/lift stage"
        ),
    )
    parser.add_argument(
        "--execution-mode",
        choices=("air", "contact"),
        default="air",
        help=(
            "target contract to display: air exactly reuses the installed PLA "
            "air-retreat/pregrasp formula; contact shows the nominal AnyDex pose"
        ),
    )
    parser.add_argument(
        "--lift-preview-m",
        type=float,
        default=0.05,
        help="vertical +robot_base-Z lift shown only as a locked preview",
    )
    parser.add_argument("--scene-stride", type=int, default=4)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--frame-timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--max-consecutive-frame-errors",
        type=int,
        default=10,
        help="close the preview after this many consecutive D435 frame failures",
    )
    parser.add_argument("--status-hz", type=float, default=5.0)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds before closing; 0 means until window close/Ctrl-C",
    )
    parser.add_argument(
        "--hand-mesh-resolution",
        choices=("full", "simplified"),
        default="simplified",
    )
    parser.add_argument("--no-target-hand-mesh", action="store_true")
    parser.add_argument(
        "--show-current-hand-mesh",
        action="store_true",
        help=(
            "render a green RH56 mesh reconstructed from telemetry hand.angles "
            "through the checksum-pinned official XLS mapping and URDF; this is "
            "a six-register model reconstruction, not 12 independent joint sensing"
        ),
    )
    parser.add_argument(
        "--window-name",
        default="AnyDex live preview | target + current EE/hand",
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        default=None,
        help=(
            "optional one-shot O_EXCL/0444 marker published only after the "
            "Open3D window, calibrated D435, and native read-only telemetry "
            "reader are ready, immediately before waiting for the mapping"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and print the preview contract without camera/Open3D",
    )
    return parser


def _selected_snapshot(
    snapshot: VisualizationSnapshot, selected_index: Optional[int]
) -> VisualizationSnapshot:
    selected = (
        int(snapshot.grasps.selected_index)
        if selected_index is None
        else int(selected_index)
    )
    if not 0 <= selected < snapshot.grasps.count:
        raise ValueError(
            "selected index {} is outside candidates 0..{}".format(
                selected, snapshot.grasps.count - 1
            )
        )
    grasps: GraspCandidates = replace(snapshot.grasps, selected_index=selected)
    return replace(snapshot, grasps=grasps)


def _snapshot_for_execution_mode(
    snapshot: VisualizationSnapshot,
    control_config: dict[str, Any],
    execution_mode: str,
) -> VisualizationSnapshot:
    """Return an in-memory target view; never rewrite the official NPZ."""

    if execution_mode == "contact":
        return snapshot
    if execution_mode != "air":
        raise ValueError("unknown execution mode {!r}".format(execution_mode))
    selected = int(snapshot.grasps.selected_index)
    hand_poses = snapshot.grasps.hand_poses
    if hand_poses is None:
        raise ValueError("air preview requires official hand poses")
    grasp = control_config["grasp"]
    tool = control_config["tool"]
    air_targets = derive_air_target_poses(
        np.asarray(snapshot.grasps.canonical_poses[selected], dtype=np.float64),
        np.asarray(hand_poses[selected], dtype=np.float64),
        np.asarray(snapshot.grasps.approach_axis_local, dtype=np.float64),
        np.asarray(tool["T_EE_hand"], dtype=np.float64),
        retreat_distance_m=float(grasp["air_retreat_distance_m"]),
        pregrasp_extra_distance_m=float(grasp["air_pregrasp_distance_m"]),
    )
    adjusted_hand_poses = np.asarray(hand_poses, dtype=np.float64).copy()
    adjusted_hand_poses[selected] = air_targets.T_reference_hand_final_air
    return replace(
        snapshot,
        grasps=replace(snapshot.grasps, hand_poses=adjusted_hand_poses),
    )


def load_preview_inputs(args: argparse.Namespace) -> PreviewInputs:
    config, _config_path = load_control_config(args.control_config)
    snapshot = _selected_snapshot(
        load_snapshot_npz(args.snapshot), args.selected_index
    )
    calibration = config["calibration"]
    if snapshot.reference_frame != config["reference_frame"]:
        raise ValueError(
            "snapshot/control reference-frame mismatch: {!r} != {!r}".format(
                snapshot.reference_frame, config["reference_frame"]
            )
        )
    if str(snapshot.camera_serial) != str(calibration["camera_serial"]):
        raise ValueError(
            "snapshot camera serial {} != control config {}".format(
                snapshot.camera_serial, calibration["camera_serial"]
            )
        )
    if str(snapshot.calibration_id) != str(calibration["id"]):
        raise ValueError(
            "snapshot calibration {} != control config {}".format(
                snapshot.calibration_id, calibration["id"]
            )
        )

    snapshot = _snapshot_for_execution_mode(
        snapshot, config, str(args.execution_mode)
    )
    grasp = config["grasp"]
    tool = config["tool"]
    air_mode = str(args.execution_mode) == "air"
    execution = ExecutionConfig(
        default_q=np.asarray(config["franka"]["default_q_rad"], dtype=np.float64),
        inspire_open_angles=np.asarray(
            config["inspire"]["open_targets"], dtype=np.float64
        ),
        pregrasp_distance_m=float(
            grasp[
                "air_pregrasp_distance_m"
                if air_mode
                else "pregrasp_distance_m"
            ]
        ),
        final_insertion_m=(
            0.0 if air_mode else float(grasp["extra_final_insertion_m"])
        ),
        enable_thumb_preshape=bool(config["inspire"]["thumb_preshape_required"]),
    )
    plan = build_control_plan(
        snapshot,
        T_EE_hand=np.asarray(tool["T_EE_hand"], dtype=np.float64),
        mount_transform_commissioned=bool(tool["mount_transform_commissioned"]),
        config=execution,
        allow_diagnostic=True,
        selected_index=int(snapshot.grasps.selected_index),
    )
    contract = build_pipeline_preview_contract(
        plan,
        lift_distance_m=float(args.lift_preview_m),
        adapter_material=str(tool["material"]),
        low_speed_unloaded_only=bool(tool["low_speed_unloaded_commissioning_only"]),
    )
    inputs = PreviewInputs(
        snapshot=snapshot,
        control_config=config,
        contract=contract,
        execution_mode=str(args.execution_mode),
    )
    if args.continuous_telemetry is not None:
        manifest = load_telemetry_session_manifest(
            args.telemetry_session_manifest, verify_files=True
        )
        _validate_telemetry_manifest_binding(args, inputs, manifest)
        inputs = replace(inputs, telemetry_manifest=manifest)
    return inputs


def _validate_telemetry_manifest_binding(
    args: argparse.Namespace,
    inputs: PreviewInputs,
    manifest: LoadedTelemetrySessionManifest,
) -> None:
    """Bind the viewer inputs to one immutable session without native imports."""

    execution = manifest.payload["execution"]
    selected = int(inputs.snapshot.grasps.selected_index)
    if int(execution["selected_index"]) != selected:
        raise ValueError(
            "telemetry manifest selected_index={} != viewer selected_index={}".format(
                execution["selected_index"], selected
            )
        )
    if sha256_file(args.snapshot) != manifest.identity.source_snapshot_sha256:
        raise ValueError("telemetry manifest does not bind the viewer snapshot bytes")
    if (
        sha256_file(args.control_config)
        != manifest.identity.control_config_sha256
    ):
        raise ValueError(
            "telemetry manifest does not bind the viewer control-config bytes"
        )

    def require_pose_match(actual: np.ndarray, expected: Any, label: str) -> None:
        if not np.allclose(
            np.asarray(actual, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("telemetry manifest {} differs from viewer".format(label))

    air = execution["air_target_contract"]
    if inputs.execution_mode == "air":
        require_pose_match(
            inputs.contract.T_reference_EE_grasp,
            air["T_reference_EE_final_air"],
            "final-air target",
        )
        require_pose_match(
            inputs.contract.stages[1].T_reference_EE,
            air["T_reference_EE_pregrasp"],
            "air pregrasp",
        )
        require_pose_match(
            inputs.contract.target_hand_pose,
            air["T_reference_hand_final_air"],
            "final-air hand target",
        )
    else:
        require_pose_match(
            inputs.contract.T_reference_EE_grasp,
            air["T_reference_EE_nominal"],
            "nominal target",
        )


def print_preview_contract(inputs: PreviewInputs) -> None:
    contract = inputs.contract
    print(
        "[preview] selected={} execution_mode={} frame={} camera={} calibration={}".format(
            contract.selected_index,
            inputs.execution_mode,
            inputs.snapshot.reference_frame,
            inputs.snapshot.camera_serial,
            inputs.snapshot.calibration_id,
        )
    )
    if inputs.execution_mode == "air":
        grasp_config = inputs.control_config["grasp"]
        print(
            "[air-contract] nominal_retreat={:.3f}m extra_pregrasp={:.3f}m "
            "shared_with=planner,audit,executor".format(
                float(grasp_config["air_retreat_distance_m"]),
                float(grasp_config["air_pregrasp_distance_m"]),
            )
        )
    else:
        print(
            "[contact-reference] nominal AnyDex pose only; current PLA profile "
            "does not authorize contact execution"
        )
    if inputs.telemetry_manifest is not None:
        manifest = inputs.telemetry_manifest
        print(
            "[telemetry-manifest] OFFLINE VERIFIED run_uuid={} file_sha256={} "
            "motion_authorized=false".format(
                manifest.identity.run_uuid,
                manifest.manifest_file_sha256,
            )
        )
    for index, stage in enumerate(contract.stages, start=1):
        if stage.target_q is not None:
            target = "q={}".format(np.array2string(stage.target_q, precision=5))
        else:
            target = "xyz={}".format(
                np.array2string(stage.T_reference_EE[:3, 3], precision=5)
            )
        if stage.inspire_target6 is not None:
            target += " inspire={}".format(list(stage.inspire_target6))
        print("[preview-stage {}/5] {} {}".format(index, stage.name, target))
    print(
        "[lift-preview] +robot_base-Z={:.3f}m hardware_execution_allowed={}".format(
            contract.lift_distance_m, contract.hardware_execution_allowed
        )
    )
    for blocker in contract.hardware_blockers:
        print("  - {}".format(blocker))


def _target_pose(contract: PipelinePreviewContract, name: str) -> np.ndarray:
    if name == "pregrasp":
        return contract.stages[1].T_reference_EE.copy()
    if name == "grasp":
        return contract.T_reference_EE_grasp
    if name == "lift":
        return contract.T_reference_EE_lift
    raise ValueError("unknown error target {!r}".format(name))


def _automatic_target(
    contract: PipelinePreviewContract, state: PoseState
) -> tuple[str, np.ndarray]:
    """Choose the honest active target for waypoint-rate executor telemetry."""

    if state.target_T_reference_EE is not None:
        return "telemetry", state.target_T_reference_EE.copy()
    stage = state.stage.strip().lower()
    if "setdown" in stage:
        return "grasp", contract.T_reference_EE_grasp
    if "pregrasp" in stage:
        return "pregrasp", contract.stages[1].T_reference_EE.copy()
    if "lift" in stage or stage == "load_applied":
        return "lift", contract.T_reference_EE_lift
    if "grasp" in stage or "thumb_preshape" in stage or "close" in stage:
        return "grasp", contract.T_reference_EE_grasp
    return "pregrasp", contract.stages[1].T_reference_EE.copy()


def _active_target(
    contract: PipelinePreviewContract,
    requested: str,
    state: Optional[PoseState],
) -> tuple[str, np.ndarray]:
    if requested != "auto":
        return requested, _target_pose(contract, requested)
    if state is None:
        return "grasp", contract.T_reference_EE_grasp
    return _automatic_target(contract, state)


class _NativeTelemetrySource:
    """Read-only native mapping source; it owns no robot, hand, or camera API."""

    def __init__(
        self,
        mapping_path: Path,
        manifest: LoadedTelemetrySessionManifest,
        *,
        python_dir: Optional[Path],
        arm_max_age_s: float,
        hand_max_age_s: float,
        read_attempts: int,
        wait_seconds: float = 60.0,
        before_mapping_wait: Optional[Callable[[], None]] = None,
    ) -> None:
        if python_dir is not None:
            resolved_python_dir = Path(python_dir).expanduser().resolve()
            if not resolved_python_dir.is_dir():
                raise RuntimeError(
                    "continuous telemetry Python directory does not exist: {}".format(
                        resolved_python_dir
                    )
                )
            if str(resolved_python_dir) not in sys.path:
                sys.path.insert(0, str(resolved_python_dir))
        try:
            native = importlib.import_module("_anydex_telemetry")
        except ImportError as exc:
            raise RuntimeError(
                "cannot import the read-only _anydex_telemetry module; set "
                "--continuous-telemetry-python-dir or "
                "ANYDEX_TELEMETRY_PYTHON_DIR"
            ) from exc
        if getattr(native, "ABI_MAJOR", None) != 1:
            raise RuntimeError("native telemetry ABI major is not 1")
        if (
            getattr(native, "ABI_SCHEMA_SHA256", None)
            != NATIVE_TELEMETRY_ABI_SCHEMA_SHA256
        ):
            raise RuntimeError(
                "native telemetry ABI schema digest is not the reviewed schema"
            )
        lock_free = getattr(native, "platform_is_supported_lock_free", None)
        if not callable(lock_free) or not bool(lock_free()):
            raise RuntimeError(
                "native telemetry requires lock-free uint64 atomics on this platform"
            )
        reader_type = getattr(native, "TelemetryReader", None)
        open_read_only = getattr(reader_type, "open_read_only", None)
        if not callable(open_read_only):
            raise RuntimeError(
                "native telemetry module has no reviewed read-only reader API"
            )
        # Preserve the final path component.  The reviewed native reader owns
        # the actual O_NOFOLLOW/open policy; resolving a dangling leaf here
        # would silently redirect it to a different mapping name.
        mapping = normalize_unfollowed_leaf(mapping_path)
        wait = float(wait_seconds)
        if not np.isfinite(wait) or wait < 0.0:
            raise ValueError("telemetry wait seconds must be finite and non-negative")
        if before_mapping_wait is not None:
            before_mapping_wait()
        self.reader = self._open_read_only_with_wait(
            open_read_only,
            mapping,
            wait_seconds=wait,
        )
        self.header = self.reader.header()
        header_identity, _created_wall, _created_mono = (
            telemetry_identity_from_native_header(self.header)
        )
        if header_identity != manifest.identity:
            raise RuntimeError(
                "native mapping run UUID/artifact hashes do not match the session manifest"
            )
        self.adapter = NativeContinuousTelemetryAdapter(
            manifest.identity,
            arm_max_age_s=float(arm_max_age_s),
            hand_max_age_s=float(hand_max_age_s),
        )
        self.read_attempts = int(read_attempts)
        producer = str(self.header.get("producer_name", "unknown"))
        robot_id = str(self.header.get("robot_id", "unknown"))
        print(
            "[telemetry] READ ONLY mapping={} run_uuid={} producer={} robot_id={}".format(
                mapping, manifest.identity.run_uuid, producer, robot_id
            )
        )
        print(
            "[telemetry] arm=measured Franka O_T_EE feedback; hand=measured "
            "ANGLE_ACT -> official model reconstruction, not a measured surface"
        )

    @staticmethod
    def _is_retryable_open_error(exc: Exception) -> bool:
        """Return true only for an absent or not-yet-committed mapping.

        The reviewed binding currently raises ``RuntimeError`` with its native
        ``InitCode`` embedded in this exact text.  Keep matching deliberately
        narrow: permission, ABI/layout incompatibility, mmap failure, and every
        other error must fail immediately rather than being hidden by a wait.
        """

        if isinstance(exc, FileNotFoundError):
            return True
        message = str(exc)
        return message in (
            "open_read_only failed (code=3): open failed: errno=2",
            "open_read_only failed (code=7): telemetry mapping initialization is not committed",
        )

    @staticmethod
    def _is_zero_size_initialization_race(exc: Exception, mapping: Path) -> bool:
        """Recognize only the O_EXCL-before-ftruncate producer window.

        A zero-byte regular file can be visible for a few instructions after
        the producer wins O_EXCL and before it installs the fixed 2,112-byte
        layout.  Every nonzero incompatible file remains a fatal ABI error.
        """

        if str(exc) != (
            "open_read_only failed (code=8): telemetry ABI header or "
            "provenance is incompatible"
        ):
            return False
        try:
            metadata = mapping.lstat()
        except OSError:
            return False
        return stat.S_ISREG(metadata.st_mode) and metadata.st_size == 0

    @classmethod
    def _open_read_only_with_wait(
        cls,
        open_read_only: Any,
        mapping: Path,
        *,
        wait_seconds: float,
    ) -> Any:
        deadline = time.monotonic() + wait_seconds
        announced = False
        attempts = 0
        zero_size_deadline: Optional[float] = None
        while True:
            attempts += 1
            try:
                # This is the only mapping operation used by the viewer.  It
                # cannot create, truncate, initialize, or modify the path.
                return open_read_only(str(mapping))
            except Exception as exc:
                # KeyboardInterrupt is a BaseException and deliberately passes
                # straight through to main's normal Ctrl-C cleanup path.
                now = time.monotonic()
                zero_size_race = cls._is_zero_size_initialization_race(
                    exc, mapping
                )
                if zero_size_race and zero_size_deadline is None:
                    zero_size_deadline = min(deadline, now + 0.50)
                retryable = cls._is_retryable_open_error(exc) or (
                    zero_size_race
                    and zero_size_deadline is not None
                    and now < zero_size_deadline
                )
                if not retryable:
                    raise
                if now >= deadline:
                    raise RuntimeError(
                        "read-only telemetry mapping did not become ready within "
                        "{:.3f}s after {} attempt(s): {}".format(
                            wait_seconds, attempts, exc
                        )
                    ) from exc
                if not announced:
                    print(
                        "[telemetry] waiting up to {:.3f}s for producer-created "
                        "read-only mapping={} ({})".format(
                            wait_seconds, mapping, exc
                        )
                    )
                    announced = True
                active_deadline = (
                    deadline
                    if not zero_size_race or zero_size_deadline is None
                    else zero_size_deadline
                )
                time.sleep(min(0.10, max(0.0, active_deadline - now)))

    def next(self) -> NativeTelemetryVisibility:
        # Re-read the immutable header so in-place mutation or mapping reuse is
        # detected by the stateful adapter instead of trusting a cached copy.
        # This is viewer-side metadata I/O, never part of a producer hot loop.
        current_header = self.reader.header()
        arm = self.reader.read_arm(self.read_attempts)
        hand = self.reader.read_hand(self.read_attempts)
        return self.adapter.accept(current_header, arm, hand)

    def close(self) -> None:
        # pybind owns only one read-only mmap/fd; releasing it never writes to the
        # producer or either device.
        self.reader = None


class _RealSenseSceneSource:
    """Lazily imported calibrated scene source; no robot transports."""

    def __init__(
        self,
        camera_config_path: Path,
        *,
        expected_serial: str,
        expected_calibration_id: str,
        scene_stride: int,
        warmup_frames: int,
        frame_timeout_ms: int,
    ) -> None:
        from dynamic_pcd.calibration.io import resolve_extrinsics
        from dynamic_pcd.camera.realsense_camera import RealSenseCamera
        from dynamic_pcd.config import load_config
        from dynamic_pcd.pointcloud.extractor import ObjectPointCloudExtractor

        self.frame_timeout_ms = int(frame_timeout_ms)
        self.scene_stride = int(scene_stride)
        cfg = load_config(str(Path(camera_config_path).expanduser().resolve()))
        extrinsics = resolve_extrinsics(cfg.get("extrinsics"))
        configured_serial = str(cfg.get("camera", {}).get("serial") or "")
        if configured_serial != expected_serial:
            raise RuntimeError(
                "camera config serial {} != snapshot {}".format(
                    configured_serial or "missing", expected_serial
                )
            )
        if not bool(extrinsics.calibrated):
            raise RuntimeError("live preview requires calibrated camera extrinsics")
        if str(extrinsics.reference_frame) != "robot_base":
            raise RuntimeError("live preview extrinsics must use robot_base")
        if str(extrinsics.camera_serial or "") != expected_serial:
            raise RuntimeError("extrinsics camera serial does not match snapshot")
        if str(extrinsics.calibration_id or "") != expected_calibration_id:
            raise RuntimeError("extrinsics calibration id does not match snapshot")

        pointcloud_cfg = dict(cfg["pointcloud"])
        pointcloud_cfg["z_min"] = float(cfg["camera"].get("z_min", 0.25))
        pointcloud_cfg["z_max"] = float(cfg["camera"].get("z_max", 1.20))
        self.extractor = ObjectPointCloudExtractor(
            pointcloud_cfg, T_base_camera=extrinsics.T_base_camera
        )
        self.camera = RealSenseCamera(cfg["camera"])
        self.camera.start()
        if str(self.camera.device_serial or "") != expected_serial:
            self.stop()
            raise RuntimeError("opened RealSense serial does not match snapshot")
        for _ in range(int(warmup_frames)):
            self.camera.get_frame(timeout_ms=self.frame_timeout_ms)
        print(
            "[camera] {} serial={} live scene in robot_base".format(
                self.camera.device_name, self.camera.device_serial
            )
        )

    def next(self) -> tuple[np.ndarray, np.ndarray, int]:
        frame = self.camera.get_frame(timeout_ms=self.frame_timeout_ms)
        cloud = self.extractor.extract_scene(
            frame, exclude_mask=None, stride=self.scene_stride
        )
        points = np.asarray(cloud.points, dtype=np.float64)
        colors = np.asarray(cloud.colors, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or colors.shape != points.shape
        ):
            raise RuntimeError(
                "live scene arrays are malformed: points={} colors={}".format(
                    points.shape, colors.shape
                )
            )
        valid = (
            np.all(np.isfinite(points), axis=1)
            & np.all(np.isfinite(colors), axis=1)
        )
        return points[valid], np.clip(colors[valid], 0.0, 1.0), int(frame.frame_id)

    def stop(self) -> None:
        camera = getattr(self, "camera", None)
        if camera is not None:
            camera.stop()


class _CurrentPoseOverlay:
    def __init__(
        self,
        o3d: Any,
        visualizer: Any,
        *,
        T_EE_hand: np.ndarray,
        target_ee: np.ndarray,
        current_hand_model: Optional[Any],
        current_hand_mapper: Optional[Any],
        show_current_hand_mesh: bool,
    ) -> None:
        self.o3d = o3d
        self.visualizer = visualizer
        self.T_EE_hand = np.asarray(T_EE_hand, dtype=np.float64)
        self.target_ee = np.asarray(target_ee, dtype=np.float64)
        self.show_current_hand_mesh = bool(show_current_hand_mesh)
        self.current_hand_model = current_hand_model
        self.current_hand_mapper = current_hand_mapper
        if self.show_current_hand_mesh and (
            self.current_hand_model is None or self.current_hand_mapper is None
        ):
            raise ValueError(
                "current hand mesh requires the official RH56 mapper and URDF model"
            )
        self._create_frame_geometries()
        self.current_meshes: list[Any] = []
        self._current_link_transforms: dict[str, np.ndarray] = {}
        self._frames_added = False
        self._meshes_added = False
        self._last_ee = np.eye(4, dtype=np.float64)

    def _create_frame_geometries(self) -> None:
        """Create pristine current-feedback geometry at the identity pose."""

        self.current_ee_frame = (
            self.o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.045)
        )
        self.current_hand_frame = (
            self.o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.035)
        )
        self.error_line = self.o3d.geometry.LineSet()
        self.error_line.lines = self.o3d.utility.Vector2iVector([[0, 1]])
        self.error_line.colors = self.o3d.utility.Vector3dVector(
            [[1.0, 0.15, 0.15]]
        )

    def _hide_current_hand_mesh(self) -> None:
        if self._meshes_added:
            for mesh in self.current_meshes:
                self.visualizer.remove_geometry(mesh, reset_bounding_box=False)
        self.current_meshes = []
        self._current_link_transforms = {}
        self._meshes_added = False

    def hide_current_feedback(self) -> None:
        """Remove every current-state overlay when telemetry is unavailable.

        The coordinate frames and error line are recreated at identity so a later
        valid sample cannot accidentally compound a new transform onto stale
        geometry that had merely been removed from the visualizer.
        """

        if self._frames_added:
            for geometry in (
                self.current_ee_frame,
                self.current_hand_frame,
                self.error_line,
            ):
                self.visualizer.remove_geometry(
                    geometry, reset_bounding_box=False
                )
        self._hide_current_hand_mesh()
        self._frames_added = False
        self._last_ee = np.eye(4, dtype=np.float64)
        self._create_frame_geometries()

    def _update_current_hand_mesh(
        self,
        current_hand: np.ndarray,
        hand_angles: Optional[tuple[int, ...]],
    ) -> None:
        if not self.show_current_hand_mesh:
            return
        if hand_angles is None:
            # Never retain a prior finger shape after feedback disappears.
            self._hide_current_hand_mesh()
            return

        converter = getattr(
            self.current_hand_mapper, "feedback_to_joint_positions_rad", None
        )
        if not callable(converter):
            raise RuntimeError(
                "official RH56 mapper exposes no reviewed ANGLE_ACT conversion"
            )
        joints = converter(hand_angles)
        transforms = self.current_hand_model.link_mesh_transforms(
            current_hand, joints
        )
        link_names = [link.name for link in self.current_hand_model.links]
        if set(transforms) != set(link_names):
            raise RuntimeError("official RH56 FK did not return every visual link")

        if not self._meshes_added:
            meshes = self.current_hand_model.build_link_meshes(current_hand, joints)
            if len(meshes) != len(link_names):
                raise RuntimeError("official RH56 model returned an invalid mesh count")
            for mesh in meshes:
                mesh.paint_uniform_color((0.10, 0.95, 0.35))
                mesh.compute_vertex_normals()
                self.visualizer.add_geometry(mesh, reset_bounding_box=False)
            self.current_meshes = list(meshes)
            self._current_link_transforms = {
                name: np.asarray(transforms[name], dtype=np.float64).copy()
                for name in link_names
            }
            self._meshes_added = True
            return

        for name, mesh in zip(link_names, self.current_meshes):
            previous = self._current_link_transforms[name]
            current = np.asarray(transforms[name], dtype=np.float64)
            mesh.transform(current @ np.linalg.inv(previous))
            self._current_link_transforms[name] = current.copy()

    def update(self, state: PoseState, *, target_ee: np.ndarray) -> None:
        current_ee = np.asarray(state.T_reference_EE, dtype=np.float64)
        self.target_ee = np.asarray(target_ee, dtype=np.float64)
        current_hand = current_ee @ self.T_EE_hand
        if not self._frames_added:
            self.current_ee_frame.transform(current_ee)
            self.current_hand_frame.transform(current_hand)
            self._last_ee = current_ee.copy()
            for geometry in (
                self.current_ee_frame,
                self.current_hand_frame,
                self.error_line,
            ):
                self.visualizer.add_geometry(geometry, reset_bounding_box=False)
            self._frames_added = True
        else:
            ee_delta = current_ee @ np.linalg.inv(self._last_ee)
            self.current_ee_frame.transform(ee_delta)
            self.current_hand_frame.transform(ee_delta)
            self._last_ee = current_ee.copy()

        self._update_current_hand_mesh(current_hand, state.hand_angles)

        self.error_line.points = self.o3d.utility.Vector3dVector(
            [current_ee[:3, 3], self.target_ee[:3, 3]]
        )
        for geometry in (
            self.current_ee_frame,
            self.current_hand_frame,
            self.error_line,
        ):
            self.visualizer.update_geometry(geometry)
        for mesh in self.current_meshes:
            self.visualizer.update_geometry(mesh)


def run_viewer(args: argparse.Namespace, inputs: PreviewInputs) -> None:
    from anydex_pipeline.visualization import (
        VisualizationStyle,
        build_open3d_geometries,
    )
    import open3d as o3d

    hand_builder = None
    if not args.no_target_hand_mesh or args.show_current_hand_mesh:
        from anydex_pipeline.inspire_hand_model import (
            SelectedInspireHandLinkMeshBuilder,
        )

        hand_builder = SelectedInspireHandLinkMeshBuilder(
            ROOT / "third_party/AnyDexGrasp",
            mesh_resolution=args.hand_mesh_resolution,
        )
    current_hand_mapper = None
    if args.show_current_hand_mesh:
        from anydex_pipeline.rh56_actuator_mapping import (
            OfficialRH56ActuatorMapper,
        )

        current_hand_mapper = OfficialRH56ActuatorMapper.from_anydex_root(
            ROOT / "third_party/AnyDexGrasp"
        )
        print(
            "[hand-mesh] green=current ANGLE_ACT -> checksum-pinned official "
            "XLS -> URDF reconstruction; workbook_sha256={}".format(
                current_hand_mapper.sha256
            )
        )
        print(
            "[hand-mesh] model-derived from six actuator registers; not 12 "
            "independent joint sensing and not a physical ground-truth surface"
        )
    style = VisualizationStyle(
        max_candidates=int(args.max_candidates), show_selected_hand_frame=True
    )
    bundle = build_open3d_geometries(
        inputs.snapshot,
        style=style,
        selected_hand_link_mesh_builder=(
            None if args.no_target_hand_mesh else hand_builder
        ),
    )
    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window(window_name=str(args.window_name)):
        raise RuntimeError("Open3D could not create a preview window")
    try:
        for geometry in bundle.geometry_list():
            visualizer.add_geometry(geometry)
        render = visualizer.get_render_option()
        if render is not None:
            render.point_size = float(args.point_size)

        active_target_name, target_ee = _active_target(
            inputs.contract, args.error_target, None
        )
        target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.065)
        target_frame.transform(target_ee)
        target_frame_pose = target_ee.copy()
        visualizer.add_geometry(target_frame, reset_bounding_box=False)
        lift_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.055)
        lift_frame.transform(inputs.contract.T_reference_EE_lift)
        visualizer.add_geometry(lift_frame, reset_bounding_box=False)

        overlay = _CurrentPoseOverlay(
            o3d,
            visualizer,
            T_EE_hand=inputs.contract.T_EE_hand,
            target_ee=target_ee,
            current_hand_model=(None if hand_builder is None else hand_builder.model),
            current_hand_mapper=current_hand_mapper,
            show_current_hand_mesh=bool(args.show_current_hand_mesh),
        )
        source = None
        if args.source == "realsense":
            source = _RealSenseSceneSource(
                args.camera_config,
                expected_serial=str(inputs.snapshot.camera_serial),
                expected_calibration_id=str(inputs.snapshot.calibration_id),
                scene_stride=int(args.scene_stride),
                warmup_frames=int(args.warmup_frames),
                frame_timeout_ms=int(args.frame_timeout_ms),
            )

        started = time.monotonic()
        last_status = 0.0
        last_pose_key = None
        last_pose: Optional[PoseState] = None
        last_native_visibility: Optional[NativeTelemetryVisibility] = None
        last_native_feedback = None
        last_pose_error = ""
        frame_id = int(inputs.snapshot.frame_id)
        consecutive_frame_errors = 0
        telemetry_source = None
        try:
            if args.continuous_telemetry is not None:
                if inputs.telemetry_manifest is None:
                    raise RuntimeError(
                        "continuous telemetry manifest was not validated offline"
                    )
                ready_callback = None
                if args.ready_file is not None:
                    if source is None:
                        raise RuntimeError(
                            "viewer readiness requires a started calibrated D435"
                        )
                    if not visualizer.poll_events():
                        raise RuntimeError(
                            "Open3D window closed before viewer readiness"
                        )
                    visualizer.update_renderer()
                    ready_callback = lambda: publish_viewer_ready(args.ready_file)
                telemetry_source = _NativeTelemetrySource(
                    args.continuous_telemetry,
                    inputs.telemetry_manifest,
                    python_dir=args.continuous_telemetry_python_dir,
                    arm_max_age_s=float(args.arm_max_age_s),
                    hand_max_age_s=float(args.hand_max_age_s),
                    read_attempts=int(args.telemetry_read_attempts),
                    wait_seconds=float(args.telemetry_wait_seconds),
                    before_mapping_wait=ready_callback,
                )
            while True:
                if args.duration > 0.0 and time.monotonic() - started >= args.duration:
                    break
                if not visualizer.poll_events():
                    break
                if source is not None:
                    try:
                        points, colors, frame_id = source.next()
                        consecutive_frame_errors = 0
                        bundle.scene_point_cloud.points = o3d.utility.Vector3dVector(points)
                        displayed = np.clip(
                            colors * style.scene_brightness + style.scene_color_floor,
                            0.0,
                            1.0,
                        )
                        bundle.scene_point_cloud.colors = o3d.utility.Vector3dVector(displayed)
                        visualizer.update_geometry(bundle.scene_point_cloud)
                    except RuntimeError as exc:
                        consecutive_frame_errors += 1
                        print(
                            "[camera][retry {}/{}] {}".format(
                                consecutive_frame_errors,
                                args.max_consecutive_frame_errors,
                                exc,
                            )
                        )
                        if consecutive_frame_errors >= args.max_consecutive_frame_errors:
                            raise RuntimeError(
                                "live D435 exceeded consecutive frame-error limit"
                            ) from exc

                candidate_state: Optional[PoseState] = None
                candidate_key = None
                feedback_requested = False
                feedback_error = ""
                if args.pose_state is not None:
                    feedback_requested = True
                    try:
                        candidate_state = load_pose_state_json(
                            args.pose_state, max_age_s=float(args.pose_max_age_s)
                        )
                        candidate_key = (
                            candidate_state.sequence,
                            candidate_state.timestamp_unix_s,
                        )
                    except (OSError, TypeError, ValueError) as exc:
                        feedback_error = str(exc)
                elif telemetry_source is not None:
                    feedback_requested = True
                    try:
                        last_native_visibility = telemetry_source.next()
                        feedback = native_viewer_feedback(last_native_visibility)
                        last_native_feedback = feedback
                        if feedback is None:
                            feedback_error = "{}; {}".format(
                                last_native_visibility.arm_status,
                                last_native_visibility.hand_status,
                            )
                        else:
                            candidate_state = PoseState(
                                reference_frame="robot_base",
                                timestamp_unix_s=(
                                    feedback.arm_timestamp_unix_ns / 1.0e9
                                ),
                                T_reference_EE=feedback.T_reference_EE,
                                stage=feedback.stage,
                                source=feedback.ee_semantics,
                                sequence=feedback.arm_sequence,
                                hand_angles=feedback.hand_angles,
                            )
                            candidate_key = feedback.update_key
                    except (OSError, RuntimeError, TypeError, ValueError) as exc:
                        feedback_error = str(exc)
                        last_native_feedback = None

                if feedback_requested:
                    if candidate_state is not None and not feedback_error:
                        if candidate_key != last_pose_key:
                            next_target_name, next_target = _active_target(
                                inputs.contract, args.error_target, candidate_state
                            )
                            if not np.allclose(
                                next_target,
                                target_frame_pose,
                                atol=1.0e-12,
                                rtol=0.0,
                            ):
                                target_frame.transform(
                                    next_target @ np.linalg.inv(target_frame_pose)
                                )
                                target_frame_pose = next_target.copy()
                                visualizer.update_geometry(target_frame)
                            active_target_name = next_target_name
                            target_ee = next_target
                            overlay.update(candidate_state, target_ee=target_ee)
                            last_pose = candidate_state
                            last_pose_key = candidate_key
                        last_pose_error = ""
                    else:
                        last_pose_error = feedback_error or "feedback unavailable"
                        # A stale/missing atomic sample invalidates every current
                        # overlay, not only the optional green hand mesh.  Leaving
                        # an old EE frame or error line visible would mislabel a
                        # historical pose as current.
                        overlay.hide_current_feedback()
                        # Force a recovered file to be rendered even if its
                        # sequence/timestamp key is identical to the last valid
                        # sample seen before the transient read failure.
                        last_pose = None
                        last_pose_key = None

                now = time.monotonic()
                if now - last_status >= 1.0 / float(args.status_hz):
                    if last_pose is None or last_pose_error:
                        detail = "measured_feedback=HIDDEN"
                        if last_pose_error:
                            detail += " ({})".format(last_pose_error)
                    else:
                        error = pose_error(last_pose.T_reference_EE, target_ee)
                        if telemetry_source is not None:
                            stage_epoch = (
                                "n/a"
                                if last_native_feedback is None
                                else str(last_native_feedback.stage_epoch)
                            )
                            detail = (
                                "stage={} stage_epoch={} measured_EE_age={:.3f}s "
                                "position_error={:.1f}mm rotation_error={:.2f}deg"
                            ).format(
                                last_pose.stage,
                                stage_epoch,
                                last_pose.age_s(),
                                error.position_mm,
                                error.rotation_deg,
                            )
                            if last_native_visibility is not None:
                                detail += " arm_status={!r} hand_status={!r}".format(
                                    last_native_visibility.arm_status,
                                    last_native_visibility.hand_status,
                                )
                        else:
                            detail = (
                                "stage={} waypoint_feedback_age={:.3f}s "
                                "position_error={:.1f}mm rotation_error={:.2f}deg"
                            ).format(
                                last_pose.stage,
                                last_pose.age_s(),
                                error.position_mm,
                                error.rotation_deg,
                            )
                        if args.show_current_hand_mesh:
                            if last_pose.hand_angles is None:
                                detail += " ANGLE_ACT=UNAVAILABLE hand_mesh=HIDDEN"
                            else:
                                detail += " ANGLE_ACT={} hand_mesh=official-model-reconstruction".format(
                                    list(last_pose.hand_angles)
                                )
                    print(
                        "[live] frame={} target={} {}".format(
                            frame_id, active_target_name, detail
                        )
                    )
                    last_status = now
                visualizer.update_renderer()
                if source is None:
                    time.sleep(0.01)
        finally:
            if telemetry_source is not None:
                telemetry_source.close()
            if source is not None:
                source.stop()
    finally:
        visualizer.destroy_window()


def _validate_numeric_args(args: argparse.Namespace) -> None:
    positive = {
        "--pose-max-age-s": args.pose_max_age_s,
        "--arm-max-age-s": args.arm_max_age_s,
        "--hand-max-age-s": args.hand_max_age_s,
        "--lift-preview-m": args.lift_preview_m,
        "--status-hz": args.status_hz,
        "--point-size": args.point_size,
    }
    for name, value in positive.items():
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("{} must be finite and positive".format(name))
    if args.scene_stride < 1 or args.max_candidates < 1:
        raise ValueError("--scene-stride and --max-candidates must be >= 1")
    if args.telemetry_read_attempts < 1:
        raise ValueError("--telemetry-read-attempts must be >= 1")
    if args.warmup_frames < 0 or args.frame_timeout_ms < 1:
        raise ValueError("--warmup-frames must be >= 0 and timeout must be positive")
    if args.max_consecutive_frame_errors < 1:
        raise ValueError("--max-consecutive-frame-errors must be >= 1")
    if not np.isfinite(float(args.duration)) or float(args.duration) < 0.0:
        raise ValueError("--duration must be finite and non-negative")
    if (
        not np.isfinite(float(args.telemetry_wait_seconds))
        or float(args.telemetry_wait_seconds) < 0.0
    ):
        raise ValueError(
            "--telemetry-wait-seconds must be finite and non-negative"
        )
    if (args.continuous_telemetry is None) != (
        args.telemetry_session_manifest is None
    ):
        raise ValueError(
            "--continuous-telemetry and --telemetry-session-manifest are required together"
        )
    if (
        args.show_current_hand_mesh
        and args.pose_state is None
        and args.continuous_telemetry is None
    ):
        raise ValueError(
            "--show-current-hand-mesh requires --pose-state or --continuous-telemetry"
        )
    if args.ready_file is not None:
        if args.source != "realsense":
            raise ValueError("--ready-file requires --source realsense")
        if args.continuous_telemetry is None:
            raise ValueError("--ready-file requires --continuous-telemetry")
        if args.validate_only:
            raise ValueError("--ready-file cannot be used with --validate-only")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        "[safety] PREVIEW ONLY: no Franka/Inspire transport is imported or opened; "
        "the lift frame is not motion authorization"
    )
    try:
        _validate_numeric_args(args)
        inputs = load_preview_inputs(args)
        print_preview_contract(inputs)
        if args.validate_only:
            print("[preview] VALIDATED ONLY; camera and Open3D were not imported")
            return 0
        run_viewer(args, inputs)
        return 0
    except KeyboardInterrupt:
        print("[preview] interrupted; viewer/camera cleanup complete")
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print("[preview] failed: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
