"""Lossless playback of an ``object_pcd_rgbd_case_v1`` recording."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterator, Optional, Sequence

import cv2
import numpy as np

from dynamic_pcd.types import CameraIntrinsics, RGBDFrame


EXPECTED_SCHEMA = "dynamic_object_pcd_rgbd_case_v1"


class EndOfRecording(EOFError):
    """The requested recording has no remaining RGB-D frame."""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> tuple[Dict[str, Any], ...]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path}:{line_number} must contain one JSON object"
                    )
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSONL {path}: {exc}") from exc
    return tuple(records)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_integrity(root: Path, relative_manifest: str) -> None:
    manifest_path = Path(str(relative_manifest))
    if (
        manifest_path.is_absolute()
        or ".." in manifest_path.parts
        or manifest_path.name != "MANIFEST.sha256"
    ):
        raise ValueError("recording integrity_manifest path is unsafe")
    checksum_path = root / manifest_path
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read integrity manifest: {exc}") from exc
    expected: Dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
            raise ValueError(
                f"{checksum_path}:{line_number} is not a SHA256 entry"
            )
        relative = Path(parts[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"{checksum_path}:{line_number} contains an unsafe path"
            )
        key = relative.as_posix()
        if key in expected:
            raise ValueError(f"duplicate integrity entry for {key}")
        expected[key] = parts[0]
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if set(expected) != actual_files:
        missing = sorted(actual_files - set(expected))
        extra = sorted(set(expected) - actual_files)
        raise ValueError(
            "recording integrity file set differs: "
            f"unlisted={missing[:3]} missing={extra[:3]}"
        )
    for relative, expected_digest in expected.items():
        actual_digest = _sha256(root / relative)
        if actual_digest != expected_digest:
            raise ValueError(f"recording checksum mismatch: {relative}")


class RecordedRGBDCase:
    """Validated random/sequential access to one lossless case directory."""

    def __init__(
        self, path: Path | str, *, verify_integrity: bool = True
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.manifest = _read_json(self.path / "manifest.json")
        if self.manifest.get("schema") != EXPECTED_SCHEMA:
            raise ValueError(
                f"unsupported recording schema {self.manifest.get('schema')!r}"
            )
        if not bool(self.manifest.get("complete", False)):
            raise ValueError("recording manifest is not marked complete")
        if verify_integrity:
            integrity_manifest = self.manifest.get("integrity_manifest")
            if not isinstance(integrity_manifest, str):
                raise ValueError("recording has no integrity_manifest")
            _verify_integrity(self.path, integrity_manifest)
        self.records = _read_jsonl(self.path / "frames.jsonl")
        initialization_record = self.manifest.get("initialization_frame")
        if not isinstance(initialization_record, dict):
            raise ValueError("recording has no initialization_frame object")
        self.initialization_record: Dict[str, Any] = initialization_record
        expected_count = int(self.manifest.get("frame_count", -1))
        if expected_count <= 0 or len(self.records) != expected_count:
            raise ValueError(
                f"recording frame count is {len(self.records)}, expected {expected_count}"
            )
        self.width = int(self.manifest["image_width"])
        self.height = int(self.manifest["image_height"])
        self.depth_scale = float(self.manifest["depth_scale_m_per_unit"])
        if (
            self.width <= 0
            or self.height <= 0
            or not math.isfinite(self.depth_scale)
            or self.depth_scale <= 0.0
        ):
            raise ValueError("recording image/depth metadata is invalid")
        intrinsics = self.manifest.get("camera_intrinsics")
        if not isinstance(intrinsics, dict):
            raise ValueError("recording has no camera_intrinsics object")
        distortion = intrinsics.get("distortion", ())
        if not isinstance(distortion, Sequence):
            raise ValueError("recording camera distortion must be a sequence")
        self.intrinsics = CameraIntrinsics(
            width=int(intrinsics["width"]),
            height=int(intrinsics["height"]),
            fx=float(intrinsics["fx"]),
            fy=float(intrinsics["fy"]),
            ppx=float(intrinsics["ppx"]),
            ppy=float(intrinsics["ppy"]),
            model=str(intrinsics.get("model", "")),
            distortion=tuple(float(value) for value in distortion),
        )
        if (
            self.intrinsics.width != self.width
            or self.intrinsics.height != self.height
        ):
            raise ValueError("recording intrinsics/image dimensions differ")
        previous_frame_id: Optional[int] = None
        previous_sensor_id: Optional[int] = None
        previous_depth_sensor_id: Optional[int] = None
        previous_timestamp_s: Optional[float] = None
        image_paths: set[str] = set()
        for index, record in enumerate(self.records):
            if int(record.get("index", -1)) != index:
                raise ValueError(f"record {index} has a non-contiguous index")
            frame_id = int(record["frame_id"])
            sensor_id = int(record["sensor_frame_number"])
            depth_sensor_id = int(record["depth_sensor_frame_number"])
            timestamp_s = float(record["camera_timestamp_s"])
            if previous_frame_id is not None and frame_id <= previous_frame_id:
                raise ValueError("recorded frame_id must strictly increase")
            if previous_sensor_id is not None and sensor_id <= previous_sensor_id:
                raise ValueError(
                    "recorded color sensor frame number must strictly increase"
                )
            if (
                previous_depth_sensor_id is not None
                and depth_sensor_id <= previous_depth_sensor_id
            ):
                raise ValueError(
                    "recorded depth sensor frame number must strictly increase"
                )
            if (
                previous_timestamp_s is not None
                and timestamp_s <= previous_timestamp_s
            ):
                raise ValueError(
                    "recorded camera timestamps must strictly increase"
                )
            previous_frame_id = frame_id
            previous_sensor_id = sensor_id
            previous_depth_sensor_id = depth_sensor_id
            previous_timestamp_s = timestamp_s
            self._validate_image_paths(record, label=f"record {index}")
            for name in ("color_path", "depth_path"):
                relative = Path(str(record[name])).as_posix()
                if relative in image_paths:
                    raise ValueError(
                        f"record {index} reuses image path {relative}"
                    )
                image_paths.add(relative)
        self._validate_image_paths(
            self.initialization_record, label="initialization frame"
        )
        for name in ("color_path", "depth_path"):
            relative = Path(str(self.initialization_record[name])).as_posix()
            if relative in image_paths:
                raise ValueError(
                    f"initialization frame reuses image path {relative}"
                )
        first_record = self.records[0]
        for name in (
            "frame_id",
            "sensor_frame_number",
            "depth_sensor_frame_number",
            "camera_timestamp_s",
        ):
            if float(self.initialization_record[name]) >= float(
                first_record[name]
            ):
                raise ValueError(
                    f"initialization frame {name} must precede video frame 0"
                )

    def _validate_image_paths(
        self, record: Dict[str, Any], *, label: str
    ) -> None:
        for name in ("color_path", "depth_path"):
            relative = Path(str(record[name]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{label} has an unsafe {name}")
            if not (self.path / relative).is_file():
                raise FileNotFoundError(self.path / relative)

    def __len__(self) -> int:
        return len(self.records)

    def frame(self, index: int) -> RGBDFrame:
        position = int(index)
        if not 0 <= position < len(self.records):
            raise IndexError(position)
        return self._frame_from_record(
            self.records[position], label=f"record {position}"
        )

    def initialization_frame(self) -> RGBDFrame:
        return self._frame_from_record(
            self.initialization_record, label="initialization frame"
        )

    def _frame_from_record(
        self, record: Dict[str, Any], *, label: str
    ) -> RGBDFrame:
        color = cv2.imread(
            str(self.path / str(record["color_path"])), cv2.IMREAD_COLOR
        )
        depth = cv2.imread(
            str(self.path / str(record["depth_path"])), cv2.IMREAD_UNCHANGED
        )
        if (
            color is None
            or color.dtype != np.uint8
            or color.shape != (self.height, self.width, 3)
        ):
            raise ValueError(f"{label} color PNG is invalid")
        if (
            depth is None
            or depth.dtype != np.uint16
            or depth.shape != (self.height, self.width)
        ):
            raise ValueError(f"{label} depth PNG is invalid")
        return RGBDFrame(
            color_bgr=np.ascontiguousarray(color),
            depth_raw=np.ascontiguousarray(depth),
            depth_scale=self.depth_scale,
            intrinsics=self.intrinsics,
            timestamp=float(record["camera_timestamp_s"]),
            frame_id=int(record["frame_id"]),
            retrieved_at_s=(
                None
                if record.get("retrieved_at_s") is None
                else float(record["retrieved_at_s"])
            ),
            timestamp_domain=str(record.get("timestamp_domain", "")),
            depth_timestamp_s=(
                None
                if record.get("depth_timestamp_s") is None
                else float(record["depth_timestamp_s"])
            ),
            color_depth_timestamp_skew_s=(
                None
                if record.get("color_depth_timestamp_skew_s") is None
                else float(record["color_depth_timestamp_skew_s"])
            ),
            color_depth_epoch_timestamp_skew_s=(
                None
                if record.get("color_depth_epoch_timestamp_skew_s") is None
                else float(record["color_depth_epoch_timestamp_skew_s"])
            ),
            rejected_timestamp_skew_frames=int(
                record.get("rejected_timestamp_skew_frames_total", 0)
            ),
            dropped_queued_framesets=int(
                record.get("dropped_queued_framesets_total", 0)
            ),
            sensor_frame_number=int(record["sensor_frame_number"]),
            depth_sensor_frame_number=int(record["depth_sensor_frame_number"]),
            rejected_transport_stale_frames=int(
                record.get("rejected_transport_stale_frames_total", 0)
            ),
            retrieved_monotonic_s=(
                None
                if record.get("retrieved_monotonic_s") is None
                else float(record["retrieved_monotonic_s"])
            ),
            host_clock_pair_span_s=(
                None
                if record.get("host_clock_pair_span_s") is None
                else float(record["host_clock_pair_span_s"])
            ),
            capture_diagnostic=dict(record.get("capture_diagnostic") or {}),
        )

    def __iter__(self) -> Iterator[RGBDFrame]:
        for index in range(len(self)):
            yield self.frame(index)


class RecordedRGBDCamera:
    """``RealSenseCamera``-compatible source backed by a recorded case."""

    def __init__(
        self,
        case: RecordedRGBDCase | Path | str,
        *,
        realtime: bool = True,
        rate: float = 1.0,
    ) -> None:
        self.case = (
            case if isinstance(case, RecordedRGBDCase) else RecordedRGBDCase(case)
        )
        self.realtime = bool(realtime)
        self.rate = float(rate)
        if not math.isfinite(self.rate) or self.rate <= 0.0:
            raise ValueError("playback rate must be finite and positive")
        self.device_serial = str(self.case.manifest.get("camera_serial", ""))
        self.device_name = str(self.case.manifest.get("camera_name", "recorded D435"))
        self.device_firmware_version = str(
            self.case.manifest.get("camera_firmware", "")
        )
        self.device_usb_type_descriptor = str(
            self.case.manifest.get("camera_usb_type", "")
        )
        self.sdk_version = str(self.case.manifest.get("camera_sdk_version", ""))
        self.depth_scale = self.case.depth_scale
        self.started = False
        self._index = 0
        self._wall_start_s = 0.0
        self._capture_start_s = 0.0

    def start(self) -> None:
        self._index = 0
        self._wall_start_s = time.monotonic()
        self._capture_start_s = float(self.case.records[0]["camera_timestamp_s"])
        self.started = True

    def get_frame(self, timeout_ms: int = 1000) -> RGBDFrame:
        del timeout_ms
        if not self.started:
            self.start()
        if self._index >= len(self.case):
            raise EndOfRecording(str(self.case.path))
        record = self.case.records[self._index]
        if self.realtime:
            capture_elapsed = (
                float(record["camera_timestamp_s"]) - self._capture_start_s
            ) / self.rate
            wait = self._wall_start_s + capture_elapsed - time.monotonic()
            if wait > 0.0:
                time.sleep(wait)
        frame = self.case.frame(self._index)
        self._index += 1
        return frame

    def stop(self) -> None:
        self.started = False

    def seek(self, index: int) -> None:
        """Position the next read at one validated recording index.

        This is intentionally an offline-only camera operation.  It lets a
        text detector initialize a replay from the exact saved RGB-D frame on
        which its bbox was produced, without pretending that the bbox belongs
        to a newer frame.
        """

        position = int(index)
        if not 0 <= position <= len(self.case):
            raise IndexError(position)
        self._index = position
        if self.started:
            self.rebase_timing()

    def rebase_timing(self) -> None:
        """Start a fresh wall-clock schedule at the next recorded frame.

        Tracker/SAM initialization may take much longer than one camera
        period.  Rebasing prevents offline playback from bursting through
        several sequential frames merely to catch up with time spent
        initializing frame memory.
        """

        if self._index >= len(self.case):
            return
        self._wall_start_s = time.monotonic()
        self._capture_start_s = float(
            self.case.records[self._index]["camera_timestamp_s"]
        )

    def __enter__(self) -> "RecordedRGBDCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
