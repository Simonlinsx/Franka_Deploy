#!/usr/bin/env python3
"""Thrown-object trigger to policy inference with read-only robot state.

This diagnostic deliberately has no action mapper, actuator, reset, or robot
command owner.  It opens Franka and RH56 only through the repository's
read-only state adapters, transforms the real RGB-D object cloud into the
measured palm frame, and starts the selected checkpoint only after the sealed
throw trigger fires.  Every policy input and output is saved for audit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Optional, Sequence

import numpy as np

from sim2real import perception
from sim2real.console_output import compact_deployment_console, emit_operator_line
from sim2real.deployment.bundle import load_checkpoint_safely
from sim2real.io import FrankaStateReader, InspireStateReader
from sim2real.observation.live_preview import _Latest, _hand_worker
from sim2real.policy import RollingStudentPolicy
from sim2real.contracts.v94 import V94Contract
from sim2real.v94_kinematics import (
    KinematicVelocityTracker,
    RH56FeedbackMapper,
    RH56FingertipKinematics,
    T_base_policy_palm_from_franka,
)
from sim2real.observation.model import PolicyHistory, Proprio67Builder


MAX_FRANKA_SPEED_RAD_S = 0.05
MAX_FRANKA_QHOME_ERROR_RAD = 0.005
MAX_HAND_SAMPLE_AGE_S = 0.50
RH56_READ_RATE_HZ = 60.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_only_controller_state29(franka: Any) -> np.ndarray:
    """Build the stationary pre-command V258 controller-state observation.

    ``FrankaStateReader`` exposes measured and desired q/dq but, correctly,
    never creates the native command shaper used by execution.  In this
    no-command shadow the held command equals the measured desired q and the
    unavailable desired acceleration is exactly zero.  This is recorded as a
    read-only approximation and is never presented as an executed-controller
    sample.
    """

    q = np.asarray(franka.q, dtype=np.float32)
    q_d = np.asarray(franka.q_desired, dtype=np.float32)
    dq_d = np.asarray(franka.dq_desired, dtype=np.float32)
    if (
        q.shape != (7,)
        or q_d.shape != (7,)
        or dq_d.shape != (7,)
        or not np.all(np.isfinite(np.concatenate((q, q_d, dq_d))))
    ):
        raise RuntimeError("read-only Franka controller inputs are invalid")
    result = np.zeros(29, dtype=np.float32)
    # held_q_cmd == q_d in a shadow with no command shaper.
    result[0:7] = 0.0
    result[7:14] = np.clip((q_d - q) / np.float32(0.05), -4.0, 4.0)
    result[14:21] = np.clip(dq_d / np.float32(0.5), -1.0, 1.0)
    result[21:28] = 0.0
    result[28] = 1.0
    return result


class _ShadowPolicyRollout:
    """Pure policy-rate history/inference owner with no command mapping."""

    def __init__(
        self,
        policy: RollingStudentPolicy,
        *,
        control_dt_s: float,
        monotonic=time.monotonic,
        perf_counter=time.perf_counter,
    ) -> None:
        dt = float(control_dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("shadow control_dt_s must be positive")
        self.policy = policy
        self.control_dt_s = dt
        self.monotonic = monotonic
        self.perf_counter = perf_counter
        self.history = PolicyHistory(
            length=policy.history_length,
            point_feature_dim=policy.point_feature_dim,
            proprio_dim=policy.proprio_dim,
        )
        self.next_due_monotonic_s: Optional[float] = None
        self.last_camera_frame_id: Optional[int] = None
        self.pretrigger_inference_count = 0
        self.records: list[dict[str, Any]] = []
        self.points_history: list[np.ndarray] = []
        self.valid_history: list[np.ndarray] = []
        self.proprio_history: list[np.ndarray] = []
        self.normalized_points_history: list[np.ndarray] = []
        self.normalized_proprio_history: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []

    def maybe_infer(
        self,
        *,
        point_frame: Any,
        proprio67: np.ndarray,
        controller_state29: Optional[np.ndarray],
        camera_frame_id: int,
        camera_timestamp_s: float,
        trigger_result: Optional[dict[str, object]],
        trigger_detected_monotonic_s: Optional[float],
    ) -> Optional[dict[str, Any]]:
        if trigger_result is None:
            return None
        if trigger_detected_monotonic_s is None:
            raise RuntimeError("detected trigger has no monotonic timestamp")
        frame_id = int(camera_frame_id)
        if self.last_camera_frame_id == frame_id:
            return None
        now = float(self.monotonic())
        if self.next_due_monotonic_s is not None and now + 1.0e-9 < self.next_due_monotonic_s:
            return None
        if self.next_due_monotonic_s is None:
            self.next_due_monotonic_s = now
        while self.next_due_monotonic_s <= now + 1.0e-9:
            self.next_due_monotonic_s += self.control_dt_s

        base = np.asarray(proprio67, dtype=np.float32)
        if base.shape != (67,) or not np.all(np.isfinite(base)):
            raise RuntimeError("shadow proprio67 is invalid")
        if self.policy.proprio_dim == 96:
            if controller_state29 is None:
                raise RuntimeError("96D shadow policy requires controller_state29")
            controller = np.asarray(controller_state29, dtype=np.float32)
            if controller.shape != (29,) or not np.all(np.isfinite(controller)):
                raise RuntimeError("shadow controller_state29 is invalid")
            proprio = np.concatenate((base, controller)).astype(np.float32)
        elif self.policy.proprio_dim == 67:
            proprio = base
        else:
            raise RuntimeError("unsupported shadow policy proprio dimension")

        point_history, valid_history, proprio_history = self.history.append(
            point_frame, proprio
        )
        started = float(self.perf_counter())
        output = self.policy.act(point_history, valid_history, proprio_history)
        inference_s = float(self.perf_counter()) - started
        action = np.asarray(output.action13, dtype=np.float32)
        if (
            action.shape != (13,)
            or not np.all(np.isfinite(action))
            or np.any(action < -1.0)
            or np.any(action > 1.0)
        ):
            raise RuntimeError("shadow policy output is invalid")

        normalized_points = (
            point_history[None] - np.asarray(self.policy.point_mean, dtype=np.float32)
        ) / np.asarray(self.policy.point_std, dtype=np.float32)
        normalized_proprio = (
            proprio_history[None]
            - np.asarray(self.policy.proprio_mean, dtype=np.float32)
        ) / np.asarray(self.policy.proprio_std, dtype=np.float32)
        record = {
            "logical_index": len(self.records),
            "camera_frame_id": frame_id,
            "camera_timestamp_s": float(camera_timestamp_s),
            "pointcloud_status": str(point_frame.status),
            "pointcloud_source_frame_id": int(point_frame.frame_id),
            "source_valid_points": int(point_frame.source_valid_points),
            "policy_inference_ms": 1000.0 * inference_s,
            "trigger_to_action_ms": 1000.0
            * (float(self.monotonic()) - float(trigger_detected_monotonic_s)),
            "action13": action.tolist(),
            "action_max_abs": float(np.max(np.abs(action))),
            "predicted_hold_logit": float(output.predicted_hold_logit),
            "robot_command_staged": False,
            "robot_hardware_writes": False,
        }
        self.records.append(record)
        self.points_history.append(point_history.copy())
        self.valid_history.append(valid_history.copy())
        self.proprio_history.append(proprio_history.copy())
        self.normalized_points_history.append(normalized_points[0].copy())
        self.normalized_proprio_history.append(normalized_proprio[0].copy())
        self.actions.append(action.copy())
        self.last_camera_frame_id = frame_id
        return record


class ReadOnlyThrownPolicyShadow:
    """Hardware read-only state adapter used by ``perception.run`` hooks."""

    def __init__(self) -> None:
        self.contract: Optional[V94Contract] = None
        self.request: Optional[Any] = None
        self.policy: Optional[RollingStudentPolicy] = None
        self.rollout: Optional[_ShadowPolicyRollout] = None
        self.proprio_builder: Optional[Proprio67Builder] = None
        self.velocity_tracker: Optional[KinematicVelocityTracker] = None
        self.fingertip_model: Optional[RH56FingertipKinematics] = None
        self.franka_reader: Optional[FrankaStateReader] = None
        self.hand_reader: Optional[InspireStateReader] = None
        self.hand_latest = _Latest()
        self.hand_stop = threading.Event()
        self.hand_thread: Optional[threading.Thread] = None
        self.latest_franka: Optional[Any] = None
        self.latest_T_base_palm: Optional[np.ndarray] = None
        self.checkpoint_path: Optional[Path] = None
        self.checkpoint_sha256 = ""
        self.started = False
        self.closed = False
        self.franka_interface_opened = False
        self.rh56_interface_opened = False

    def start(
        self,
        *,
        contract: V94Contract,
        request: Any,
        point_feature_dim: int,
    ) -> None:
        if self.started:
            raise RuntimeError("policy shadow was started twice")
        if request.checkpoint is None:
            raise RuntimeError("policy shadow requires --checkpoint")
        if float(request.policy_rate_hz) != 20.0:
            raise RuntimeError("thrown policy shadow requires the V57 20 Hz contract")
        self.contract = contract
        self.request = request
        self.checkpoint_path = Path(request.checkpoint).resolve()
        payload = self.checkpoint_path.read_bytes()
        self.checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
        checkpoint = load_checkpoint_safely(payload)
        self.policy = RollingStudentPolicy(checkpoint)
        if self.policy.point_feature_dim != int(point_feature_dim):
            raise RuntimeError("checkpoint and projector point-feature modes differ")
        self.rollout = _ShadowPolicyRollout(
            self.policy, control_dt_s=float(contract.control_dt_s)
        )

        profile_path = Path(request.profile).resolve()
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        franka = profile.get("franka")
        inspire = profile.get("inspire")
        if not isinstance(franka, dict) or not isinstance(inspire, dict):
            raise RuntimeError("shadow profile lacks Franka/RH56 sections")
        self.proprio_builder = Proprio67Builder(
            q_home_rad=contract.q_home_rad,
            q_hand_close_rad=contract.q_hand_close_rad,
        )
        self.velocity_tracker = KinematicVelocityTracker(maximum_dt_s=0.25)
        self.fingertip_model = RH56FingertipKinematics(contract)
        mapper = RH56FeedbackMapper(q_hand_close_rad=contract.q_hand_close_rad)
        self.hand_reader = InspireStateReader(
            port=str(inspire.get("port")),
            baud=int(inspire.get("baud", 115200)),
            hand_id=int(inspire.get("hand_id", 1)),
            timeout_s=0.5,
            debug=False,
            snapshot_mode="compact_policy",
        )
        self.hand_thread = threading.Thread(
            target=_hand_worker,
            args=(
                self.hand_reader,
                mapper,
                self.hand_stop,
                self.hand_latest,
                RH56_READ_RATE_HZ,
            ),
            name="thrown-policy-shadow-rh56-readonly",
            daemon=True,
        )
        self.hand_thread.start()
        self.rh56_interface_opened = True
        self.franka_reader = FrankaStateReader(
            str(franka.get("ip")), enforce_realtime=False
        )
        self.franka_reader.start()
        self.franka_interface_opened = True
        self.sample_T_base_palm()
        assert self.latest_franka is not None
        measured_q = np.asarray(self.latest_franka.q, dtype=np.float64)
        expected_q = np.asarray(contract.q_home_rad, dtype=np.float64)
        qhome_error = float(np.max(np.abs(measured_q - expected_q)))
        if qhome_error > MAX_FRANKA_QHOME_ERROR_RAD:
            raise RuntimeError(
                "Franka is not at the selected thrown task q_home: "
                f"max_error={qhome_error:.6f}rad > "
                f"{MAX_FRANKA_QHOME_ERROR_RAD:.6f}rad; shadow inference was not started"
            )
        deadline = time.monotonic() + 5.0
        while self.hand_latest.get() is None:
            self.hand_latest.raise_if_failed()
            if time.monotonic() >= deadline:
                raise RuntimeError("RH56 read-only shadow sample timed out")
            time.sleep(0.01)
        self._validate_hand(self.hand_latest.get())
        self.started = True

    @staticmethod
    def _validate_franka(franka: Any) -> None:
        if franka.current_errors:
            raise RuntimeError(
                f"Franka read-only shadow reports errors: {franka.current_errors}"
            )
        if float(np.max(np.abs(np.asarray(franka.dq)))) > MAX_FRANKA_SPEED_RAD_S:
            raise RuntimeError("Franka must remain stationary in policy shadow mode")

    @staticmethod
    def _validate_hand(hand: Any) -> None:
        if hand is None:
            raise RuntimeError("RH56 read-only shadow has no sample")
        if not np.array_equal(np.asarray(hand.angle_targets), np.full(6, -1)):
            raise RuntimeError("RH56 must be disabled (ANGLE_SET=-1) in shadow mode")
        if np.any(np.asarray(hand.errors) != 0):
            raise RuntimeError("RH56 reports an error in shadow mode")

    def sample_T_base_palm(self) -> np.ndarray:
        if self.franka_reader is None or self.contract is None:
            raise RuntimeError("policy shadow is not initialized")
        franka = self.franka_reader.read()
        self._validate_franka(franka)
        palm = T_base_policy_palm_from_franka(
            T_base_ee=franka.T_base_ee,
            F_T_EE=franka.F_T_EE,
            T_flange_policy_palm=self.contract.T_flange_policy_palm,
        )
        self.latest_franka = franka
        self.latest_T_base_palm = palm.copy()
        return palm

    def maybe_infer(
        self,
        *,
        point_frame: Any,
        camera_frame_id: int,
        camera_timestamp_s: float,
        trigger_result: Optional[dict[str, object]],
        trigger_detected_monotonic_s: Optional[float],
    ) -> None:
        if trigger_result is None:
            return
        if any(
            item is None
            for item in (
                self.rollout,
                self.policy,
                self.proprio_builder,
                self.velocity_tracker,
                self.fingertip_model,
                self.latest_franka,
                self.latest_T_base_palm,
            )
        ):
            raise RuntimeError("policy shadow inference state is incomplete")
        self.hand_latest.raise_if_failed()
        hand = self.hand_latest.get()
        self._validate_hand(hand)
        hand_age_s = time.time() - float(hand.timestamp_s)
        if not math.isfinite(hand_age_s) or not 0.0 <= hand_age_s <= MAX_HAND_SAMPLE_AGE_S:
            raise RuntimeError(f"RH56 shadow sample is stale: age={hand_age_s:.3f}s")
        franka = self.latest_franka
        palm = self.latest_T_base_palm
        velocity = self.velocity_tracker.update(
            captured_at_s=float(franka.captured_at_s),
            T_base_palm=palm,
            hand_q_policy_order_rad=hand.q_policy_rad,
        )
        tips = self.fingertip_model.positions_base(
            angle_act_register_order=hand.angle_act,
            T_base_palm=palm,
        )
        proprio67 = self.proprio_builder.build(
            franka_q_rad=franka.q,
            franka_dq_rad_s=franka.dq,
            rh56_virtual_q_policy_order_rad=hand.q_policy_rad,
            rh56_virtual_dq_policy_order_rad_s=hand.dq_policy_rad_s,
            T_base_palm=palm,
            palm_linear_velocity_base_m_s=velocity.linear_base_m_s,
            palm_angular_velocity_base_rad_s=velocity.angular_base_rad_s,
            fingertip_positions_base_m=tips,
            # Nothing is executed, so hypothetical outputs must never feed
            # back into the next observation.
            previous_executed_action13=self.policy.initial_previous_action13,
        )
        controller = (
            _read_only_controller_state29(franka)
            if self.policy.proprio_dim == 96
            else None
        )
        record = self.rollout.maybe_infer(
            point_frame=point_frame,
            proprio67=proprio67,
            controller_state29=controller,
            camera_frame_id=camera_frame_id,
            camera_timestamp_s=camera_timestamp_s,
            trigger_result=trigger_result,
            trigger_detected_monotonic_s=trigger_detected_monotonic_s,
        )
        if record is not None:
            label = (
                "[Policy shadow FIRST ACTION]"
                if int(record["logical_index"]) == 0
                else "[Policy shadow]"
            )
            emit_operator_line(
                f"{label} tick={record['logical_index']} "
                f"frame={record['camera_frame_id']} "
                f"trigger_to_action={record['trigger_to_action_ms']:.1f}ms "
                f"inference={record['policy_inference_ms']:.2f}ms "
                f"max|a|={record['action_max_abs']:.3f}; robot writes=0"
            )

    def close(self) -> None:
        if self.closed:
            return
        errors: list[str] = []
        self.hand_stop.set()
        if self.hand_thread is not None:
            self.hand_thread.join(timeout=5.0)
            if self.hand_thread.is_alive():
                errors.append("RH56 read-only thread did not stop")
        if self.hand_reader is not None and (
            self.hand_thread is None or not self.hand_thread.is_alive()
        ):
            try:
                self.hand_reader.close()
            except BaseException as exc:
                errors.append(f"RH56 close failed: {exc}")
        if self.franka_reader is not None:
            try:
                self.franka_reader.close()
            except BaseException as exc:
                errors.append(f"Franka close failed: {exc}")
        self.closed = not errors
        if errors:
            raise RuntimeError("policy shadow cleanup failed: " + "; ".join(errors))

    def finalize(self, *, save_directory: Path) -> dict[str, object]:
        if self.rollout is None or self.checkpoint_path is None:
            raise RuntimeError("policy shadow never initialized")
        if not self.closed:
            raise RuntimeError("policy shadow interfaces are not closed")
        if _sha256_file(self.checkpoint_path) != self.checkpoint_sha256:
            raise RuntimeError("policy shadow checkpoint changed during the run")
        records = self.rollout.records
        count = len(records)
        output = Path(save_directory) / "policy_shadow_io.npz"
        if output.exists():
            raise FileExistsError(f"policy shadow refuses to overwrite: {output}")

        policy = self.policy
        assert policy is not None
        history = int(policy.history_length)
        feature_dim = int(policy.point_feature_dim)
        proprio_dim = int(policy.proprio_dim)
        points = (
            np.stack(self.rollout.points_history).astype(np.float32)
            if count
            else np.empty((0, history, 128, feature_dim), dtype=np.float32)
        )
        valid = (
            np.stack(self.rollout.valid_history).astype(np.float32)
            if count
            else np.empty((0, history, 128), dtype=np.float32)
        )
        proprio = (
            np.stack(self.rollout.proprio_history).astype(np.float32)
            if count
            else np.empty((0, history, proprio_dim), dtype=np.float32)
        )
        normalized_points = (
            np.stack(self.rollout.normalized_points_history).astype(np.float32)
            if count
            else np.empty((0, history, 128, feature_dim), dtype=np.float32)
        )
        normalized_proprio = (
            np.stack(self.rollout.normalized_proprio_history).astype(np.float32)
            if count
            else np.empty((0, history, proprio_dim), dtype=np.float32)
        )
        actions = (
            np.stack(self.rollout.actions).astype(np.float32)
            if count
            else np.empty((0, 13), dtype=np.float32)
        )
        with output.open("xb") as stream:
            np.savez_compressed(
                stream,
                pointcloud_history_metric=points,
                pointcloud_valid_history=valid,
                proprio_history_raw=proprio,
                pointcloud_history_normalized=normalized_points,
                proprio_history_normalized=normalized_proprio,
                action13=actions,
                camera_frame_id=np.asarray(
                    [item["camera_frame_id"] for item in records], dtype=np.int64
                ),
                camera_timestamp_s=np.asarray(
                    [item["camera_timestamp_s"] for item in records],
                    dtype=np.float64,
                ),
                policy_inference_ms=np.asarray(
                    [item["policy_inference_ms"] for item in records],
                    dtype=np.float64,
                ),
                trigger_to_action_ms=np.asarray(
                    [item["trigger_to_action_ms"] for item in records],
                    dtype=np.float64,
                ),
                checkpoint_sha256=np.asarray(self.checkpoint_sha256),
                robot_hardware_writes=np.asarray(False),
            )
        inference_ms = np.asarray(
            [item["policy_inference_ms"] for item in records], dtype=np.float64
        )
        trigger_ms = np.asarray(
            [item["trigger_to_action_ms"] for item in records], dtype=np.float64
        )
        return {
            "schema": "thrown_read_only_policy_shadow_v1",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "history_length": history,
            "point_feature_dim": feature_dim,
            "proprio_dim": proprio_dim,
            "control_rate_hz": 1.0 / self.rollout.control_dt_s,
            "inference_count": count,
            "pretrigger_inference_count": self.rollout.pretrigger_inference_count,
            "first_trigger_to_action_ms": (
                None if not count else float(trigger_ms[0])
            ),
            "policy_inference_p50_ms": (
                None if not count else float(np.percentile(inference_ms, 50))
            ),
            "policy_inference_p95_ms": (
                None if not count else float(np.percentile(inference_ms, 95))
            ),
            "policy_inference_max_ms": (
                None if not count else float(np.max(inference_ms))
            ),
            "action_max_abs": (
                None if not count else float(np.max(np.abs(actions)))
            ),
            "all_outputs_finite": bool(count and np.all(np.isfinite(actions))),
            "policy_io_archive": str(output),
            "proprio_controller_state_source": (
                "read_only_franka_q_qd_dq_d_plus_zero_unavailable_ddq_d"
            ),
            "franka_interface_opened": self.franka_interface_opened,
            "rh56_interface_opened": self.rh56_interface_opened,
            "interfaces_closed": self.closed,
            "robot_command_staged": False,
            "robot_hardware_writes": False,
        }


def build_parser():
    parser = perception.build_parser()
    parser.description = (
        "Read-only thrown-object trigger to real policy inference; opens robot "
        "state interfaces but has no robot command path."
    )
    for action in parser._actions:
        if getattr(action, "dest", "") == "test_rollout_trigger":
            action.help = (
                "thrown-task test: arm the production object-motion/entry "
                "trigger, then start read-only policy inference; robot state "
                "interfaces are read-only and robot writes remain zero"
            )
    return parser


def _validate_args(args: Any) -> None:
    if not bool(args.test_rollout_trigger):
        raise ValueError("policy shadow requires --test-rollout-trigger")
    if args.checkpoint is None:
        raise ValueError("policy shadow requires --checkpoint")
    if float(args.post_trigger_capture_s) <= 0.0:
        raise ValueError("policy shadow requires positive post-trigger capture")


def main(argv: Optional[Sequence[str]] = None) -> int:
    shadow = ReadOnlyThrownPolicyShadow()
    try:
        args = build_parser().parse_args(argv)
        _validate_args(args)
        compact = not bool(args.verbose_console)
        with compact_deployment_console(enabled=compact):
            result = perception.run(args, read_only_policy_shadow=shadow)
        report_path = perception._save_compact_report(result)
        policy_summary = result.get("policy_shadow")
        inference_count = (
            policy_summary.get("inference_count")
            if isinstance(policy_summary, dict)
            else None
        )
        first_latency = (
            policy_summary.get("first_trigger_to_action_ms")
            if isinstance(policy_summary, dict)
            else None
        )
        emit_operator_line(
            (
                "[Policy shadow PASS] "
                if result.get("result") == "PASS"
                else "[Policy shadow FAILED] "
            )
            + f"inferences={inference_count} "
            + f"first_trigger_to_action_ms={first_latency} "
            + "robot_writes=0 "
            + f"video={result.get('record_video_path')} report={report_path}",
            error=result.get("result") != "PASS",
        )
        return 0 if result.get("result") == "PASS" else 1
    except KeyboardInterrupt:
        try:
            shadow.close()
        except BaseException:
            pass
        print(
            "Thrown policy shadow: interrupted; read-only interfaces closed; "
            "no robot command was issued",
            file=sys.stderr,
        )
        return 130
    except SystemExit:
        raise
    except BaseException as exc:
        try:
            shadow.close()
        except BaseException as cleanup_exc:
            print(
                f"Thrown policy shadow cleanup also failed: {cleanup_exc}",
                file=sys.stderr,
            )
        print(
            f"Thrown policy shadow: FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
