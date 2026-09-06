#!/usr/bin/env python3
"""Feed the production Python ARM packet into the real C++ contract gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import uuid
import zlib
from pathlib import Path

import numpy as np

from sim2real.deployment.bundle import DeployBundle
from robot_control.franka.native_session import (
    NativeHelloExpectation,
    V94NativeMessageKind,
    V94NativePODCodec,
    V94_NATIVE_PROTOCOL_MAGIC,
    V94_NATIVE_PROTOCOL_VERSION,
)
from robot_control.franka.session import (
    MotionAuthorization,
    _issue_supervised_franka_preflight_token,
    load_experimental_supervised_franka_envelope,
)
from sim2real.tasks.thrown_contract import load_v57_thrown_task_contract


HEADER = struct.Struct("<IHHIIQQ16sII")
HELLO = struct.Struct("<4I32s20s32s8d")
NONCE = bytes(range(0xA0, 0xB0))
LIBFRANKA_SHA256 = bytes.fromhex(
    "956d2f7e85e3c4e127899734a170dff7c91f17f560a4fa2147739631ad721a3d"
)
LIBFRANKA_COMMIT = bytes.fromhex("9f9304ec0ac897eff3219a67f612b959948535e2")
TEST_BUILD_SHA256 = bytes.fromhex("42" * 32)


def child_packet(kind: V94NativeMessageKind, payload: bytes) -> bytes:
    header = HEADER.pack(
        V94_NATIVE_PROTOCOL_MAGIC,
        V94_NATIVE_PROTOCOL_VERSION,
        int(kind),
        len(payload),
        0,
        1,
        1,
        NONCE,
        0,
        0,
    )
    packet = bytearray(header + payload)
    struct.pack_into("<I", packet, 48, zlib.crc32(packet) & 0xFFFFFFFF)
    return bytes(packet)


def production_arm_packet(workspace: Path) -> bytes:
    safety_profile = os.environ.get("ANYDEX_V94_SAFETY_PROFILE", "v94")
    if safety_profile == "v94_tabletop":
        profile_path = (
            workspace
            / "dexgrasp/configs/fr3_rh56_v94_seq286_20hz_commissioned.json"
        )
        profile_json = json.loads(profile_path.read_text(encoding="utf-8"))
        reference_q_rad = np.asarray(
            profile_json["franka"]["default_q_rad"], dtype=np.float32
        )
    elif safety_profile == "v60_palmcatch":
        profile_path = (
            workspace
            / "dexgrasp/configs/fr3_rh56_v60_palmcatch_first_motion.json"
        )
        profile_json = json.loads(profile_path.read_text(encoding="utf-8"))
        reference_q_rad = np.asarray(
            profile_json["franka"]["default_q_rad"], dtype=np.float32
        )
    elif safety_profile == "v61_sixexpert":
        profile_path = (
            workspace
            / "dexgrasp/configs/fr3_rh56_v61_sixexpert_40tick.json"
        )
        profile_json = json.loads(profile_path.read_text(encoding="utf-8"))
        reference_q_rad = np.asarray(
            profile_json["franka"]["default_q_rad"], dtype=np.float32
        )
    elif safety_profile == "v94":
        profile_path = (
            workspace
            / "dexgrasp/configs/fr3_rh56_v57_thrown_alpha0p5_20hz_commissioned.json"
        )
        task = load_v57_thrown_task_contract(
            workspace / "perception/v57_real_test_reset_and_throw_ranges.yaml",
            selected_curriculum="alpha_0_5",
            expected_sha256=(
                "7d0646b1a1592895aeb0a7f591d65d771eb6b67cf14154d4aab1b54b0609f4c4"
            ),
        )
        reference_q_rad = task.franka_q_home_rad
    else:
        raise RuntimeError("unknown ANYDEX_V94_SAFETY_PROFILE")
    bundle = DeployBundle(workspace / "data/test_fixtures/sim2real/deploy.zip")
    bundle.verify()
    envelope = load_experimental_supervised_franka_envelope(profile_path)
    codec = V94NativePODCodec(
        hello_expectation=NativeHelloExpectation(
            state_decimation=16,
            safety_limits_schema=3,
            libfranka_sha256=LIBFRANKA_SHA256,
            libfranka_source_commit=LIBFRANKA_COMMIT,
            producer_build_sha256=TEST_BUILD_SHA256,
        ),
        # This is the production admission value.  In particular, deploy.zip
        # stores it as float32; reading the profile JSON as float64 would miss
        # the mismatch that caused the first native supervised run to fail.
        reference_q_rad=reference_q_rad,
        maximum_target_count=720,
        monotonic_ns=lambda: 10_000_000_000,
    )
    hello_payload = HELLO.pack(
        1234,
        16,
        V94_NATIVE_PROTOCOL_VERSION,
        3,
        LIBFRANKA_SHA256,
        LIBFRANKA_COMMIT,
        TEST_BUILD_SHA256,
        0.50,
        5.0,
        250.0,
        0.01,
        0.020,
        1.21,
        0.01,
        0.0008,
    )
    codec.decode_critical_packet(
        child_packet(V94NativeMessageKind.HELLO, hello_payload)
    )
    run_id = "python-cpp-arm-contract"
    authorization = MotionAuthorization(
        run_id=run_id,
        authorization_id=str(uuid.UUID("12345678-1234-5678-1234-567812345678")),
        issued_monotonic_s=5.0,
        expires_monotonic_s=20.0,
    )
    token = _issue_supervised_franka_preflight_token(
        envelope=envelope,
        run_id=run_id,
        confirmed_permit_sha256=hashlib.sha256(b"cross-language permit").hexdigest(),
        issued_monotonic_s=5.0,
        expires_monotonic_s=20.0,
    )
    return codec.encode_arm(
        run_id=run_id,
        authorization=authorization,
        preflight_token=token,
        envelope=envelope,
        maximum_cycles=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cpp_test", type=Path)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[4]
    packet = production_arm_packet(workspace)
    completed = subprocess.run(
        [str(args.cpp_test)],
        input=packet,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        sys.stderr.buffer.write(completed.stderr)
        return completed.returncode
    sys.stdout.buffer.write(completed.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
