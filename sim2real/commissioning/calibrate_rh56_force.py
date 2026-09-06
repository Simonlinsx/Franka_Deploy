"""Calibrate the six RH56 actuator force sensors in a stable unloaded pose.

This utility never opens or commands Franka and never writes a positive RH56
motion target.  It implements the force-calibration transaction from the
Inspire Robots RH56 V1.09 manual, register 1009 (``GESTURE_FORCE_CLB``).  The
device-side calibration routine may internally open the hand; the utility
always disables all six targets and restores SPEED_SET/FORCE_SET afterwards.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import select
import statistics
import time
from typing import Any, Dict, List, Sequence, Tuple

from examples.inspire_rh56_test import (
    JOINTS,
    JOINT_LABELS,
    READ_COMMAND,
    REG_ANGLE_SET,
    REG_FORCE_ACT,
    REG_FORCE_SET,
    REG_SPEED_SET,
    RESPONSE_HEADER,
    RH56Error,
    RH56Hand,
    WRITE_COMMAND,
    LinuxSerial,
    build_write_frame,
    checksum,
    find_serial_port,
    hex_bytes,
    parse_response,
)


FORCE_CALIBRATION_REGISTER = 1009
FORCE_CALIBRATION_COMPLETION_ADDRESS = 2
FORCE_CALIBRATION_CONFIRMATION = "RH56_UNLOADED_FORCE_CALIBRATION"
DEFAULT_SAMPLES = 40
DEFAULT_SAMPLE_PERIOD_S = 0.05
DEFAULT_COMPLETION_TIMEOUT_S = 8.0
SINGLE_REPLY_MAX_UNLOADED_MEDIAN_ABS_GF = 50.0


class RH56ForceCalibrationError(RuntimeError):
    """The force calibration transaction or its pre/post checks failed."""


def _pop_response_frame(buffer: bytearray) -> bytes | None:
    """Pop one complete checksum-valid RH56 response from ``buffer``."""

    while True:
        header_at = buffer.find(RESPONSE_HEADER)
        if header_at < 0:
            if len(buffer) > 1:
                del buffer[:-1]
            return None
        if header_at:
            del buffer[:header_at]
        if len(buffer) < 4:
            return None
        total_length = int(buffer[3]) + 5
        if len(buffer) < total_length:
            return None
        frame = bytes(buffer[:total_length])
        del buffer[:total_length]
        if checksum(frame[:-1]) != frame[-1]:
            raise RH56ForceCalibrationError(
                "force calibration received a bad-checksum response: "
                + hex_bytes(frame)
            )
        return frame


def _write_request(serial_port: LinuxSerial, request: bytes, deadline: float) -> None:
    if serial_port.fd is None:
        raise RH56ForceCalibrationError("serial port is not open")
    view = memoryview(request)
    while view:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RH56ForceCalibrationError(
                "force calibration request write timed out"
            )
        try:
            written = os.write(serial_port.fd, view)
        except BlockingIOError:
            _, writable, _ = select.select([], [serial_port.fd], [], remaining)
            if not writable:
                raise RH56ForceCalibrationError(
                    "force calibration serial write was not ready"
                )
            continue
        if written <= 0:
            raise RH56ForceCalibrationError(
                "force calibration serial write made no progress"
            )
        view = view[written:]


def execute_official_force_calibration(
    serial_port: LinuxSerial,
    hand_id: int,
    completion_timeout_s: float = DEFAULT_COMPLETION_TIMEOUT_S,
) -> Tuple[Tuple[bytes, ...], bool]:
    """Execute the V1.09 register-1009 command and inspect its replies.

    The command is deliberately never retried: after the complete request has
    entered the serial driver, a missing response leaves the device-side apply
    state unknown.
    """

    if not math.isfinite(completion_timeout_s) or completion_timeout_s <= 0:
        raise ValueError("completion_timeout_s must be positive and finite")
    if serial_port.fd is None:
        raise RH56ForceCalibrationError("serial port is not open")

    request = build_write_frame(
        int(hand_id), FORCE_CALIBRATION_REGISTER, b"\x01"
    )
    serial_port.discard_input()
    if serial_port.debug:
        print("TX calibration: " + hex_bytes(request))
    deadline = time.monotonic() + float(completion_timeout_s)
    _write_request(serial_port, request, deadline)

    frames: List[bytes] = []
    buffer = bytearray()
    while len(frames) < 2:
        frame = _pop_response_frame(buffer)
        if frame is not None:
            frames.append(frame)
            if serial_port.debug:
                print("RX calibration: " + hex_bytes(frame))
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if frames:
                # The installed RH56 firmware has been observed to apply the
                # calibration but emit only one of the V1.09 manual's two
                # replies.  Do not retry the side-effecting write.  The caller
                # proves completion from the post-calibration unloaded zero
                # residual before declaring PASS.
                break
            received = hex_bytes(bytes(buffer)) if buffer else "<nothing>"
            raise RH56ForceCalibrationError(
                "force calibration received no reply; "
                f"trailing={received}; APPLY_UNKNOWN=true; do not retry blindly"
            )
        readable, _, _ = select.select([serial_port.fd], [], [], remaining)
        if not readable:
            continue
        try:
            chunk = os.read(serial_port.fd, 4096)
        except BlockingIOError:
            continue
        if chunk:
            buffer.extend(chunk)

    accepted_seen = False
    completed_seen = False
    for frame in frames:
        try:
            accepted = parse_response(
                frame,
                int(hand_id),
                WRITE_COMMAND,
                FORCE_CALIBRATION_REGISTER,
                1,
            )
            if accepted == b"\x01":
                accepted_seen = True
                continue
        except RH56Error:
            pass
        try:
            completed = parse_response(
                frame,
                int(hand_id),
                READ_COMMAND,
                FORCE_CALIBRATION_COMPLETION_ADDRESS,
                1,
            )
            if completed == b"\x00":
                completed_seen = True
                continue
        except RH56Error:
            pass
        raise RH56ForceCalibrationError(
            "force calibration returned an unexpected response: "
            + hex_bytes(frame)
        )
    if not (accepted_seen or completed_seen):
        raise RH56ForceCalibrationError(
            "force calibration returned no recognized response"
        )
    if len(frames) >= 2 and not (accepted_seen and completed_seen):
        raise RH56ForceCalibrationError(
            "force calibration returned two replies but did not prove both "
            "acceptance and completion"
        )
    return tuple(frames), bool(accepted_seen and completed_seen)


def _validate_unloaded_idle(snapshot: Dict[str, object], phase: str) -> None:
    targets = tuple(int(value) for value in snapshot["angle_targets"])
    errors = tuple(int(value) for value in snapshot["errors"])
    statuses = tuple(int(value) for value in snapshot["statuses"])
    currents = tuple(int(value) for value in snapshot["currents"])
    temperatures = tuple(int(value) for value in snapshot["temperatures"])
    if targets != (-1,) * 6:
        raise RH56ForceCalibrationError(
            f"{phase}: all six ANGLE_SET values must be -1; actual={targets}"
        )
    if any(errors):
        raise RH56ForceCalibrationError(
            f"{phase}: RH56 reports actuator errors={errors}"
        )
    if not all(value in (2, 0xFF) for value in statuses):
        raise RH56ForceCalibrationError(
            f"{phase}: RH56 must be stationary/idle; statuses={statuses}"
        )
    if any(abs(value) > 100 for value in currents):
        raise RH56ForceCalibrationError(
            f"{phase}: unloaded idle current must be <=100mA per axis; "
            f"currents={currents}"
        )
    if sum(abs(value) for value in currents) > 200:
        raise RH56ForceCalibrationError(
            f"{phase}: unloaded total idle current must be <=200mA; "
            f"currents={currents}"
        )
    if max(temperatures) >= 60:
        raise RH56ForceCalibrationError(
            f"{phase}: actuator temperature is too high; "
            f"temperatures={temperatures}"
        )


def _write_six_once_or_verify(
    hand: RH56Hand, address: int, values: Sequence[int], label: str
) -> None:
    expected = tuple(int(value) for value in values)
    try:
        hand.write_six_shorts(address, expected, retries=0)
    except RH56Error as write_error:
        try:
            actual = tuple(
                int(value)
                for value in hand.read_six_shorts(address, retries=0)
            )
        except RH56Error as read_error:
            raise RH56ForceCalibrationError(
                f"{label} write response and exact readback both failed: "
                f"write_error={write_error}; read_error={read_error}"
            ) from write_error
        if actual != expected:
            raise RH56ForceCalibrationError(
                f"{label} write failed and exact readback differs: "
                f"expected={expected}, actual={actual}; write_error={write_error}"
            ) from write_error
    actual = tuple(int(value) for value in hand.read_six_shorts(address))
    if actual != expected:
        raise RH56ForceCalibrationError(
            f"{label} exact readback mismatch: expected={expected}, actual={actual}"
        )


def _force_samples(
    hand: RH56Hand, count: int, sample_period_s: float
) -> List[Tuple[int, ...]]:
    result: List[Tuple[int, ...]] = []
    for sample_index in range(count):
        result.append(
            tuple(int(value) for value in hand.read_six_shorts(REG_FORCE_ACT))
        )
        if sample_index + 1 < count:
            time.sleep(sample_period_s)
    return result


def _force_statistics(samples: Sequence[Sequence[int]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("force sample sequence must not be empty")
    axes: List[Dict[str, Any]] = []
    for axis, (name, label) in enumerate(zip(JOINTS, JOINT_LABELS)):
        values = [int(sample[axis]) for sample in samples]
        axes.append(
            {
                "axis": axis,
                "name": name,
                "label": label,
                "median_gf": float(statistics.median(values)),
                "mean_gf": float(statistics.fmean(values)),
                "stdev_gf": (
                    float(statistics.stdev(values)) if len(values) > 1 else 0.0
                ),
                "min_gf": min(values),
                "max_gf": max(values),
            }
        )
    return {"sample_count": len(samples), "axes": axes}


def _print_statistics(title: str, stats: Dict[str, Any]) -> None:
    print(title)
    print("axis          median(gf)   mean(gf)  stdev(gf)       min..max")
    for axis in stats["axes"]:
        print(
            f"{axis['name']:<13} {axis['median_gf']:>10.1f} "
            f"{axis['mean_gf']:>10.1f} {axis['stdev_gf']:>10.2f} "
            f"{axis['min_gf']:>6}..{axis['max_gf']:<6}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Official RH56 unloaded force-sensor calibration; RH56 only, "
            "Franka is never opened or commanded"
        )
    )
    parser.add_argument("--port", help="serial device; auto-detected if unique")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--id", type=int, default=1)
    parser.add_argument("--serial-timeout", type=float, default=0.5)
    parser.add_argument(
        "--completion-timeout", type=float, default=DEFAULT_COMPLETION_TIMEOUT_S
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--sample-period", type=float, default=DEFAULT_SAMPLE_PERIOD_S
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--confirm-unloaded",
        metavar="TOKEN",
        required=True,
        help=(
            "requires exact token " + FORCE_CALIBRATION_CONFIRMATION
        ),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        help="optional JSON output; defaults to dexgrasp/runs/<timestamp>.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.confirm_unloaded != FORCE_CALIBRATION_CONFIRMATION:
        raise SystemExit(
            "REFUSED: --confirm-unloaded must be "
            + FORCE_CALIBRATION_CONFIRMATION
        )
    if not 1 <= int(args.id) <= 254:
        raise SystemExit("REFUSED: --id must be in 1..254")
    if not 10 <= int(args.samples) <= 200:
        raise SystemExit("REFUSED: --samples must be in 10..200")
    if not math.isfinite(args.sample_period) or not 0.02 <= args.sample_period <= 1:
        raise SystemExit("REFUSED: --sample-period must be in 0.02..1s")

    port = args.port or find_serial_port()
    evidence_path = args.evidence
    if evidence_path is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        evidence_path = Path("dexgrasp/runs") / (
            f"rh56_force_calibration_{timestamp}.json"
        )
    evidence_path = evidence_path.resolve()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "[RH56 ONLY] Franka interface will not be opened; the host sends no "
        "positive motion target."
    )
    print(
        "[CALIBRATION] Keep the open hand completely unloaded and untouched; "
        "the device-side routine may internally open the hand."
    )
    with LinuxSerial(
        port, int(args.baud), float(args.serial_timeout), bool(args.debug)
    ) as serial_port:
        hand = RH56Hand(serial_port, int(args.id))
        first = hand.snapshot()
        _validate_unloaded_idle(first, "preflight sample 1")
        time.sleep(0.25)
        second = hand.snapshot()
        _validate_unloaded_idle(second, "preflight sample 2")
        angle_drift = max(
            abs(int(after) - int(before))
            for before, after in zip(first["angles"], second["angles"])
        )
        position_drift = max(
            abs(int(after) - int(before))
            for before, after in zip(first["positions"], second["positions"])
        )
        if angle_drift > 2 or position_drift > 3:
            raise RH56ForceCalibrationError(
                "unloaded hand is not stationary: "
                f"angle_drift={angle_drift}, position_drift={position_drift}"
            )

        before_samples = _force_samples(
            hand, int(args.samples), float(args.sample_period)
        )
        before_stats = _force_statistics(before_samples)
        _print_statistics("[BEFORE] unloaded FORCE_ACT", before_stats)

        calibration_frames: Tuple[bytes, ...] = ()
        dual_reply_verified = False
        calibration_error: BaseException | None = None
        try:
            (
                calibration_frames,
                dual_reply_verified,
            ) = execute_official_force_calibration(
                serial_port,
                int(args.id),
                completion_timeout_s=float(args.completion_timeout),
            )
        except BaseException as exc:
            calibration_error = exc
        restore_errors: List[str] = []
        try:
            _write_six_once_or_verify(
                hand, REG_ANGLE_SET, (-1,) * 6, "post-calibration disable"
            )
        except BaseException as exc:
            restore_errors.append(str(exc))
        if not restore_errors:
            try:
                _write_six_once_or_verify(
                    hand,
                    REG_SPEED_SET,
                    tuple(int(value) for value in second["speeds"]),
                    "SPEED_SET restore",
                )
                _write_six_once_or_verify(
                    hand,
                    REG_FORCE_SET,
                    tuple(int(value) for value in second["force_limits"]),
                    "FORCE_SET restore",
                )
                _write_six_once_or_verify(
                    hand, REG_ANGLE_SET, (-1,) * 6, "final disable"
                )
            except BaseException as exc:
                restore_errors.append(str(exc))
        if restore_errors:
            raise RH56ForceCalibrationError(
                "post-calibration stop/settings restore failed: "
                + "; ".join(restore_errors)
            ) from calibration_error
        if calibration_error is not None:
            raise calibration_error
        if dual_reply_verified:
            print(
                "[DEVICE] official calibration acceptance and completion "
                "replies verified"
            )
        else:
            print(
                "[DEVICE] firmware emitted one recognized calibration reply; "
                "completion will be proved from unloaded zero residual"
            )
        time.sleep(0.5)

        post_snapshot = hand.snapshot()
        _validate_unloaded_idle(post_snapshot, "post-calibration")
        after_samples = _force_samples(
            hand, int(args.samples), float(args.sample_period)
        )
        after_stats = _force_statistics(after_samples)
        _print_statistics("[AFTER] unloaded FORCE_ACT", after_stats)

        max_after_median_abs = max(
            abs(float(axis["median_gf"])) for axis in after_stats["axes"]
        )
        if (
            not dual_reply_verified
            and max_after_median_abs
            > SINGLE_REPLY_MAX_UNLOADED_MEDIAN_ABS_GF
        ):
            raise RH56ForceCalibrationError(
                "firmware omitted the completion reply and final unloaded "
                f"median residual {max_after_median_abs:.1f}gf exceeds "
                f"{SINGLE_REPLY_MAX_UNLOADED_MEDIAN_ABS_GF:.1f}gf; "
                "calibration completion is not proven"
            )

    before_l1 = sum(abs(axis["median_gf"]) for axis in before_stats["axes"])
    after_l1 = sum(abs(axis["median_gf"]) for axis in after_stats["axes"])
    result = {
        "schema": "rh56_unloaded_force_calibration_v1",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "port": port,
        "hand_id": int(args.id),
        "franka_opened_or_commanded": False,
        "rh56_positive_motion_target_written_by_host": False,
        "rh56_stop_target_minus_one_written": True,
        "device_calibration_may_move_hand": True,
        "calibration_register": FORCE_CALIBRATION_REGISTER,
        "calibration_reply_frames_hex": [
            frame.hex(" ").upper() for frame in calibration_frames
        ],
        "manual_two_reply_completion_verified": dual_reply_verified,
        "single_reply_zero_residual_limit_gf": (
            SINGLE_REPLY_MAX_UNLOADED_MEDIAN_ABS_GF
        ),
        "before": before_stats,
        "after": after_stats,
        "median_abs_sum_before_gf": before_l1,
        "median_abs_sum_after_gf": after_l1,
        "preflight_angles": [int(value) for value in second["angles"]],
        "post_angles": [int(value) for value in post_snapshot["angles"]],
        "post_errors": [int(value) for value in post_snapshot["errors"]],
        "post_statuses": [int(value) for value in post_snapshot["statuses"]],
    }
    fd = os.open(
        evidence_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except BaseException:
        try:
            evidence_path.unlink()
        except OSError:
            pass
        raise

    print(
        "[PASS] device force calibration completed; unloaded median |force| "
        f"sum {before_l1:.1f} -> {after_l1:.1f} gf"
    )
    print(f"[evidence] {evidence_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
