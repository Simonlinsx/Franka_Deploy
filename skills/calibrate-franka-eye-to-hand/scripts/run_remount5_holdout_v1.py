#!/usr/bin/env python3
"""Run the already-reviewed remount5 H01--H06 holdout plans once."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml


WORKSPACE = Path("/home/qiaoguanren/code/franka")
ROOT = WORKSPACE / "beta/dynamic_object_pcd"
RUNS = ROOT / "calibration_runs"
PYTHON = WORKSPACE / ".venv/bin/python"
MOTION = WORKSPACE / "skills/calibrate-franka-eye-to-hand/scripts/move_calibration_pose.py"
INSPECT = WORKSPACE / "skills/calibrate-franka-eye-to-hand/scripts/inspect_aruco_frame.py"
TELEMETRY = WORKSPACE / "skills/calibrate-franka-eye-to-hand/scripts/capture_aruco_frame_telemetry.py"
CONFIG = ROOT / "configs/d435_default.yaml"
MANIFEST = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-holdout-v2-plans/manifest.json"
MANIFEST_SHA = "ce9e497da6d0c6824c6d961ebd21ea9fe7e2e057c1ef9b491672e927a4933fdb"
HOLDOUT = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-holdout-v1.yaml"
ARTIFACTS = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-holdout-v2"
ARTIFACTS = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-holdout-v2-resume-from-H02"
CLAIMS = RUNS / ".holdout-edge-claims"
SERIAL = "342222071785"
INITIAL_HOLDOUT_SHA = "9be983990eece85650d1e8d4dec3078e38ceeef0b2df53737dffd327bde586db"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def publish_json(path: Path, document) -> str:
    payload = json.dumps(document, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def run_command(label: str, argv, log_path: Path) -> str:
    print(f"[holdout:{label}] $ {' '.join(map(str, argv))}", flush=True)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.Popen(
        [str(x) for x in argv], cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        print(f"[holdout:{label}] {line}", end="", flush=True)
    code = process.wait()
    output = "".join(lines)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    if code != 0:
        raise RuntimeError(f"{label} failed with exit code {code}")
    return output


def sentinel(output: str, prefix: str):
    matches = [line[len(prefix):] for line in output.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {prefix} sentinel")
    return json.loads(matches[0])


def transform_error(left, right):
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    translation = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    c = float(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return translation, math.degrees(math.acos(c))


def load_dataset(path: Path):
    with path.open("rb") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("samples"), list):
        raise RuntimeError("invalid holdout dataset")
    return document


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def check_dataset(before, after, expected_count: int, target_T) -> None:
    if len(after["samples"]) != expected_count:
        raise RuntimeError("holdout sample count did not increment exactly once")
    if before is not None:
        if len(before["samples"]) + 1 != len(after["samples"]):
            raise RuntimeError("holdout append count mismatch")
        for index, sample in enumerate(before["samples"]):
            if canonical(sample) != canonical(after["samples"][index]):
                raise RuntimeError(f"holdout append changed sample {index + 1}")
        for field in ("camera", "target", "robot", "transform_convention", "units"):
            if canonical(before.get(field)) != canonical(after.get(field)):
                raise RuntimeError(f"holdout append changed metadata {field}")
    if after.get("camera", {}).get("serial") != SERIAL:
        raise RuntimeError("holdout camera serial mismatch")
    target = after.get("target", {})
    if (target.get("dictionary"), target.get("marker_id"), target.get("marker_length_m")) != ("DICT_6X6_50", 42, 0.19):
        raise RuntimeError("holdout target metadata mismatch")
    sample = after["samples"][-1]
    translation, rotation = transform_error(sample["T_base_ee"], target_T)
    if translation > 0.003 or rotation > 1.0:
        raise RuntimeError(f"holdout EEF target mismatch {translation*1000:.3f}mm/{rotation:.3f}deg")
    if before is not None:
        dt, dr = transform_error(before["samples"][-1]["T_base_ee"], sample["T_base_ee"])
        if dt < 0.010 and dr < 3.0:
            raise RuntimeError(f"holdout samples too similar {dt*1000:.3f}mm/{dr:.3f}deg")


def inspect_target(stem: str):
    raw = ARTIFACTS / f"{stem}.inspection-raw.png"
    annotated = ARTIFACTS / f"{stem}.inspection-annotated.png"
    output = run_command(
        f"{stem}:inspect",
        [PYTHON, INSPECT, "--config", CONFIG, "--camera-serial", SERIAL,
         "--raw-output", raw, "--annotated-output", annotated,
         "--frames", "30", "--dictionary", "DICT_6X6_50"],
        ARTIFACTS / f"{stem}.inspect.log",
    )
    data = sentinel(output, "ARUCO_FRAME_INSPECTION_JSON=")
    if data.get("camera_serial") != SERIAL or data.get("image_width") != 848 or data.get("image_height") != 480:
        raise RuntimeError("inspection camera identity/profile mismatch")
    detections = [d for d in data.get("detections", []) if d.get("dictionary") == "DICT_6X6_50" and d.get("marker_id") == 42]
    if len(detections) != 1:
        raise RuntimeError("inspection did not find exactly one requested marker")
    detection = detections[0]
    if not detection.get("complete_in_image") or detection.get("minimum_image_margin_px", 0) < 20 or detection.get("edge_min_px", 0) < 80:
        raise RuntimeError("inspection FOV gate failed")
    return data, raw, annotated


def capture_holdout(stem: str, target_T, expected_count: int, before):
    telemetry_path = ARTIFACTS / f"{stem}.frame-telemetry.json"
    output = run_command(
        f"{stem}:telemetry",
        [PYTHON, TELEMETRY, "--config", CONFIG, "--camera-serial", SERIAL,
         "--dictionary", "DICT_6X6_50", "--marker-id", "42",
         "--marker-length-m", "0.19", "--frames", "120",
         "--output", telemetry_path, "--max-reprojection-error-px", "0.5",
         "--max-translation-deviation-m", "0.001",
         "--max-rotation-deviation-deg", "0.3"],
        ARTIFACTS / f"{stem}.telemetry.log",
    )
    summary = sentinel(output, "ARUCO_FRAME_TELEMETRY_JSON=")
    if summary.get("captured") != 120 or summary.get("valid", 0) < 118 or summary.get("reprojection_gate_pass", 0) < 118 or summary.get("coherent", 0) < 114:
        raise RuntimeError("telemetry count gate failed")
    detailed = json.loads(telemetry_path.read_text(encoding="utf-8"))
    coherence = detailed["coherence"]
    if coherence["all_reprojection_pass_translation_jitter_p95_m"] > 0.001 or coherence["all_reprojection_pass_rotation_jitter_p95_deg"] > 0.3 or coherence["all_reprojection_pass_reprojection_error_p95_px"] > 0.5:
        raise RuntimeError("telemetry untrimmed p95 gate failed")

    debug = ARTIFACTS / f"{stem}.capture.png"
    output = run_command(
        f"{stem}:capture",
        [PYTHON, "-m", "dynamic_pcd.apps.calibrate_eye_to_hand", "capture",
         "--config", CONFIG, "--robot-ip", "172.16.0.2", "--camera-serial", SERIAL,
         "--target-type", "aruco", "--dictionary", "DICT_6X6_50", "--marker-id", "42",
         "--marker-length", "0.19", "--frames", "120",
         "--min-valid-frame-fraction", "0.95", "--min-valid-frames", "114",
         "--max-reprojection-error", "0.5", "--max-target-translation-jitter", "0.001",
         "--max-target-rotation-jitter", "0.3", "--min-all-reprojection-pass-frames", "118",
         "--max-all-reprojection-pass-translation-jitter", "0.001",
         "--max-all-reprojection-pass-rotation-jitter", "0.3",
         "--max-all-reprojection-pass-reprojection-error", "0.5",
         "--max-stationary-translation", "0.0005", "--max-stationary-rotation", "0.1",
         "--min-pose-translation", "0.01", "--min-pose-rotation", "3.0",
         "--debug-image", debug, "--output", HOLDOUT],
        ARTIFACTS / f"{stem}.capture.log",
    )
    formal = sentinel(output, "CAPTURE_UNTRIMMED_P95_JSON=")
    if formal.get("requested_frame_count") != 120 or formal.get("all_reprojection_pass_count", 0) < 118:
        raise RuntimeError("formal capture frame gate failed")
    if formal["all_reprojection_pass_translation_jitter_p95_m"] > 0.001 or formal["all_reprojection_pass_rotation_jitter_p95_deg"] > 0.3 or formal["all_reprojection_pass_reprojection_error_p95_px"] > 0.5:
        raise RuntimeError("formal capture untrimmed p95 gate failed")
    if not debug.is_file() or debug.stat().st_size == 0:
        raise RuntimeError("formal debug image missing")
    after = load_dataset(HOLDOUT)
    check_dataset(before, after, expected_count, target_T)
    return after, formal, detailed


def main() -> int:
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("holdout manifest SHA mismatch")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for key in ("source_training_master", "source_training_completion_receipt", "source_training_dataset"):
        path = Path(manifest[key])
        if sha256(path) != manifest[key + "_sha256"]:
            raise RuntimeError(f"source evidence drift: {key}")
    if not HOLDOUT.is_file() or sha256(HOLDOUT) != INITIAL_HOLDOUT_SHA:
        raise RuntimeError("two-sample recovered H01/H02 holdout prefix SHA mismatch")
    initial_document = load_dataset(HOLDOUT)
    if len(initial_document["samples"]) != 2:
        raise RuntimeError("resume requires exactly two recovered holdout samples")
    if ARTIFACTS.exists():
        raise FileExistsError("resume artifacts must be absent at first run")
    if shutil.disk_usage(ROOT).free < 1024 ** 3:
        raise RuntimeError("less than 1 GiB free storage")
    remaining_edges = manifest["edges"][4:]
    if remaining_edges[0]["start_pose_id"] != "H02" or remaining_edges[-1]["target_pose_id"] != "T01":
        raise RuntimeError("resume edge suffix mismatch")
    for edge in remaining_edges:
        if sha256(Path(edge["plan"])) != edge["plan_sha256"]:
            raise RuntimeError("edge plan SHA mismatch")
        claim = CLAIMS / f"{MANIFEST_SHA}-{edge['ordinal']:02d}-{edge['start_pose_id']}-to-{edge['target_pose_id']}.claim.json"
        if claim.exists():
            raise RuntimeError(f"edge already claimed; retry forbidden: {claim}")

    ARTIFACTS.mkdir(parents=True, exist_ok=False)
    CLAIMS.mkdir(parents=True, exist_ok=True)
    dataset = initial_document
    capture_count = 2
    completed = []
    for edge in remaining_edges:
        ordinal = edge["ordinal"]
        start = edge["start_pose_id"]
        target = edge["target_pose_id"]
        stem = f"{ordinal:02d}-{start}-to-{target}"
        plan = Path(edge["plan"])
        digest = edge["plan_sha256"]
        claim = CLAIMS / f"{MANIFEST_SHA}-{ordinal:02d}-{start}-to-{target}.claim.json"
        try:
            preview_output = run_command(
                f"{stem}:preview",
                [PYTHON, MOTION, "preview", "--plan", plan, "--pose-id", target,
                 "--expect-plan-sha256", digest, "--json"],
                ARTIFACTS / f"{stem}.preview.log",
            )
            preview = json.loads(preview_output)
            if not preview.get("run_authorized") or preview.get("plan_sha256") != digest:
                raise RuntimeError("motion preview authorization failed")
            publish_json(claim, {"state": "claimed_before_motion", "manifest_sha256": MANIFEST_SHA,
                                 "ordinal": ordinal, "start_pose_id": start, "target_pose_id": target,
                                 "plan": str(plan), "plan_sha256": digest,
                                 "holdout_sample_count_before": capture_count})
            motion_output = run_command(
                f"{stem}:motion",
                [PYTHON, MOTION, "run", "--plan", plan, "--pose-id", target,
                 "--robot-ip", "172.16.0.2", "--confirm-plan-sha256", digest,
                 "--confirm-pose-id", target, "--confirm-e-stop", "FR3_ESTOP_REACHABLE",
                 "--confirm-swept-volume", "FR3_CALIBRATION_SWEPT_VOLUME_CLEAR",
                 "--confirm-target-rigid", "FR3_CALIBRATION_TARGET_RIGID",
                 "--confirm-camera-fixed", "EYE_TO_HAND_CAMERA_FIXED"],
                ARTIFACTS / f"{stem}.motion.log",
            )
            if f"CAPTURE_READY pose={target} plan_sha256={digest}" not in motion_output or '"success_qualified":true' not in motion_output:
                raise RuntimeError("motion did not produce qualified CAPTURE_READY")
            inspection, raw, annotated = inspect_target(stem)
            record = {"ordinal": ordinal, "edge": f"{start}_to_{target}", "plan_sha256": digest,
                      "inspection_raw_sha256": sha256(raw), "inspection_annotated_sha256": sha256(annotated),
                      "capture_holdout_id": edge.get("capture_holdout_id")}
            capture_id = edge.get("capture_holdout_id")
            if capture_id is not None:
                capture_count += 1
                dataset, formal, telemetry = capture_holdout(
                    stem, preview["target_T_base_ee"], capture_count, dataset
                )
                record.update({"holdout_sample_index": capture_count, "holdout_id": capture_id,
                               "dataset_sha256_after": sha256(HOLDOUT), "formal_untrimmed": formal,
                               "telemetry_sha256": sha256(ARTIFACTS / f"{stem}.frame-telemetry.json")})
            publish_json(ARTIFACTS / f"{stem}.complete.json", record)
            completed.append(record)
            print(f"[holdout] EDGE_COMPLETE {start}_to_{target} captures={capture_count}", flush=True)
        except BaseException as exc:
            publish_json(ARTIFACTS / f"{stem}.failed.json", {
                "state": "failed_no_automatic_retry", "ordinal": ordinal,
                "edge": f"{start}_to_{target}", "plan_sha256": digest,
                "error": f"{type(exc).__name__}: {exc}",
                "holdout_dataset_state": "present" if HOLDOUT.exists() else "absent",
                "holdout_dataset_sha256": sha256(HOLDOUT) if HOLDOUT.exists() else None,
            })
            raise

    if capture_count != 6 or dataset is None or len(dataset["samples"]) != 6:
        raise RuntimeError("holdout sequence did not produce exactly six samples")
    completion = {
        "schema_version": 1, "kind": "franka_eye_to_hand_holdout_sequence_receipt",
        "state": "independent_holdout_capture_verified_complete",
        "manifest": str(MANIFEST), "manifest_sha256": MANIFEST_SHA,
        "holdout_dataset": str(HOLDOUT), "holdout_dataset_sha256": sha256(HOLDOUT),
        "final_holdout_sample_count": 6, "final_pose_id": "T01", "completed_edges": completed,
        "holdout_dataset_never_used_for_solve": True,
    }
    receipt = ARTIFACTS / "holdout-sequence.complete.json"
    publish_json(receipt, completion)
    print(f"[holdout] COMPLETE samples=6 dataset_sha256={sha256(HOLDOUT)} receipt={receipt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
