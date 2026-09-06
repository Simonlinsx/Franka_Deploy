from __future__ import annotations

"""Non-blocking process coordinator for keyboard-triggered hover jobs.

The perception/Open3D process must keep publishing while pylibfranka owns its
control loop.  This coordinator therefore launches the existing fail-closed
``hover_over_object`` CLI in a separate process instead of controlling the
robot from a GUI callback or perception thread.
"""

import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence


class InteractiveHoverController:
    """Coordinate plan/execute/cancel requests without blocking perception."""

    def __init__(
        self,
        *,
        config_path: str,
        zmq_addr: str,
        robot_ip: str,
        clearance_m: float,
        loaded_calibration_id: Optional[str],
        enable_robot_motion: bool = False,
        confirm_calibration_id: Optional[str] = None,
        confirm_workspace_clear: bool = False,
        confirm_eef_clear: bool = False,
        confirm_descent_clear: bool = False,
        observe_timeout_s: float = 30.0,
        arm_timeout_s: float = 5.0,
        python_executable: Optional[str] = None,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.10 <= float(clearance_m) <= 0.30:
            raise ValueError("interactive hover clearance must be in [0.10, 0.30]m")
        if float(observe_timeout_s) <= 0.0:
            raise ValueError("observe timeout must be positive")
        if not 2.0 <= float(arm_timeout_s) <= 30.0:
            raise ValueError("motion arm timeout must be in [2, 30]s")

        self.config_path = str(Path(config_path))
        self.zmq_addr = str(zmq_addr)
        self.robot_ip = str(robot_ip)
        self.clearance_m = float(clearance_m)
        self.loaded_calibration_id = (
            None if loaded_calibration_id is None else str(loaded_calibration_id)
        )
        self.enable_robot_motion = bool(enable_robot_motion)
        self.confirm_calibration_id = (
            None if confirm_calibration_id is None else str(confirm_calibration_id)
        )
        self.confirm_workspace_clear = bool(confirm_workspace_clear)
        self.confirm_eef_clear = bool(confirm_eef_clear)
        self.confirm_descent_clear = bool(confirm_descent_clear)
        self.observe_timeout_s = float(observe_timeout_s)
        self.arm_timeout_s = float(arm_timeout_s)
        self.python_executable = python_executable or sys.executable
        self._popen = popen_factory
        self._monotonic = monotonic

        self.process: Optional[subprocess.Popen] = None
        self.job_kind: Optional[str] = None
        self.armed_until: Optional[float] = None
        self.status = "IDLE"

    @property
    def active(self) -> bool:
        self.poll()
        return self.process is not None

    @property
    def armed(self) -> bool:
        self._expire_arm()
        return self.armed_until is not None

    def _motion_gate_error(self) -> Optional[str]:
        if not self.enable_robot_motion:
            return "robot motion is disabled; restart with --enable_robot_motion"
        if not self.loaded_calibration_id:
            return "loaded calibration has no calibration_id"
        if self.confirm_calibration_id != self.loaded_calibration_id:
            return "--confirm-calibration-id does not match the loaded calibration"
        if not self.confirm_workspace_clear:
            return "--confirm-workspace-clear is required"
        if not self.confirm_eef_clear:
            return "--confirm-eef-clear is required"
        if not self.confirm_descent_clear:
            return "--confirm-descent-clear is required"
        return None

    def _base_command(self) -> list[str]:
        return [
            self.python_executable,
            "-m",
            "dynamic_pcd.apps.hover_over_object",
            "--config",
            self.config_path,
            "--addr",
            self.zmq_addr,
            "--robot-ip",
            self.robot_ip,
            "--observe-timeout",
            f"{self.observe_timeout_s:g}",
            "--allow-descent",
            "--clearance-m",
            f"{self.clearance_m:g}",
        ]

    def build_command(self, *, execute: bool) -> list[str]:
        command = self._base_command()
        if execute:
            error = self._motion_gate_error()
            if error is not None:
                raise RuntimeError(error)
            command.extend(
                [
                    "--execute",
                    "--confirm-calibration-id",
                    str(self.confirm_calibration_id),
                    "--confirm-workspace-clear",
                    "--confirm-eef-clear",
                    "--confirm-descent-clear",
                ]
            )
        return command

    def _start(self, *, execute: bool) -> bool:
        self.poll()
        if self.process is not None:
            print(f"[Interactive hover][WARN] {self.job_kind} job is already active")
            return False
        command = self.build_command(execute=execute)
        kind = "MOTION" if execute else "PLAN"
        print(f"[Interactive hover] starting {kind}: {' '.join(command)}")
        # Inherit stdout/stderr so pylibfranka progress cannot fill a PIPE and
        # block.  A new session prevents a terminal Ctrl-C intended for the GUI
        # from being delivered to the child accidentally; X sends SIGINT only
        # to this child.
        self.process = self._popen(command, start_new_session=True)
        self.job_kind = kind
        self.armed_until = None
        self.status = f"{kind} RUNNING"
        return True

    def start_plan(self, *, packet_valid: bool) -> bool:
        if not packet_valid:
            print("[Interactive hover][WARN] cannot plan: current target packet is invalid")
            self.status = "TARGET INVALID"
            return False
        return self._start(execute=False)

    def arm_motion(self, *, packet_valid: bool, roi_locked: bool) -> bool:
        self.poll()
        if self.process is not None:
            print(f"[Interactive hover][WARN] {self.job_kind} job is already active")
            return False
        error = self._motion_gate_error()
        if error is not None:
            print(f"[Interactive hover][DENY] {error}")
            self.status = "MOTION DISABLED"
            return False
        if not packet_valid:
            print("[Interactive hover][DENY] current target packet is invalid")
            self.status = "TARGET INVALID"
            return False
        if not roi_locked:
            print("[Interactive hover][DENY] press L to lock the settled target before motion")
            self.status = "LOCK TARGET FIRST"
            return False
        self.armed_until = self._monotonic() + self.arm_timeout_s
        self.status = f"ARMED: PRESS Y ({self.arm_timeout_s:g}s)"
        print(
            f"[Interactive hover][ARMED] press Y within {self.arm_timeout_s:g}s "
            "to execute one frozen hover; X cancels"
        )
        return True

    def confirm_motion(self, *, packet_valid: bool, roi_locked: bool) -> bool:
        if not self.armed:
            print("[Interactive hover][DENY] motion is not armed; press M first")
            return False
        if not packet_valid:
            self.disarm("target became invalid")
            return False
        if not roi_locked:
            self.disarm("target ROI is no longer locked")
            return False
        return self._start(execute=True)

    def disarm(self, reason: str = "operator request") -> None:
        if self.armed_until is not None:
            print(f"[Interactive hover] disarmed: {reason}")
        self.armed_until = None
        if self.process is None:
            self.status = "IDLE"

    def request_cancel(self) -> bool:
        self.disarm("cancel key")
        self.poll()
        if self.process is None:
            print("[Interactive hover] no active plan/motion job")
            return False
        print(f"[Interactive hover] sending SIGINT to active {self.job_kind} job")
        self.process.send_signal(signal.SIGINT)
        self.status = f"{self.job_kind} STOPPING"
        return True

    def poll(self) -> Optional[int]:
        self._expire_arm()
        if self.process is None:
            return None
        return_code = self.process.poll()
        if return_code is None:
            return None
        kind = self.job_kind or "JOB"
        self.status = f"{kind} DONE" if return_code == 0 else f"{kind} ERROR rc={return_code}"
        print(f"[Interactive hover] {kind} exited with code {return_code}")
        self.process = None
        self.job_kind = None
        return int(return_code)

    def _expire_arm(self) -> None:
        if self.armed_until is None:
            return
        if self._monotonic() > self.armed_until:
            self.armed_until = None
            if self.process is None:
                self.status = "ARM EXPIRED"
            print("[Interactive hover] motion arm expired; press M again")

    def shutdown(self, timeout_s: float = 8.0) -> None:
        """Stop a child cooperatively before the perception publisher closes."""

        self.disarm("application shutdown")
        self.poll()
        if self.process is None:
            return
        process = self.process
        kind = self.job_kind or "JOB"
        print(f"[Interactive hover] stopping active {kind} before shutdown")
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=float(timeout_s))
        except subprocess.TimeoutExpired:
            # Never SIGTERM/SIGKILL a process that may own a libfranka control
            # loop.  Send a second cooperative interrupt and keep the publisher
            # alive long enough for its perception watchdog to fail closed.
            print(
                "[Interactive hover][WARN] child did not stop after SIGINT; "
                "sending SIGINT again (do not close the perception publisher yet)"
            )
            process.send_signal(signal.SIGINT)
            process.wait(timeout=float(timeout_s))
        finally:
            self.poll()

