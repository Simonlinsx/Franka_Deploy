#!/usr/bin/env python3
"""Camera-only OpenCV ROI selector used by the supervised V94 parent.

This module is intentionally a separate process boundary.  It never imports,
opens, or writes Franka/RH56 interfaces.  The parent supplies a private result
file descriptor and independently reopens the D435 with the returned numeric
ROI for its three-frame mask/depth/point-cloud preflight.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import signal
import sys
from typing import Mapping, Optional, Sequence


ROI_SELECTOR_PROTOCOL = "v94_isolated_object_roi_selection_v1"
MAX_RESULT_BYTES = 4096


def _grounding_candidate_depth_admission(
    frame: object,
    bbox_xyxy: object,
    *,
    z_min_m: float,
    z_max_m: float,
    minimum_valid_depth_ratio: float = 0.25,
    require_interior: bool = True,
    T_base_camera: object = None,
    workspace_center_base_m: object = None,
    workspace_max_horizontal_distance_m: object = None,
) -> tuple[bool, str]:
    """Reject semantic boxes that cannot initialize robot-facing RGB-D.

    This gate is deliberately weaker than the provider's formal mask/point
    cloud admission.  It only prevents an impossible background proposal from
    terminating automatic search before SAM2 gets a usable target.
    """

    import numpy as np

    bbox = np.asarray(bbox_xyxy, dtype=np.int64).reshape(4)
    depth_raw = np.asarray(getattr(frame, "depth_raw"))
    if depth_raw.ndim != 2:
        return False, "candidate depth image is not HxW"
    height, width = depth_raw.shape
    x1 = int(np.clip(bbox[0], 0, width))
    y1 = int(np.clip(bbox[1], 0, height))
    x2 = int(np.clip(bbox[2], 0, width))
    y2 = int(np.clip(bbox[3], 0, height))
    if x2 <= x1 or y2 <= y1:
        return False, "candidate bbox is empty after clipping"
    if require_interior and (x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height):
        return (
            False,
            f"candidate bbox touches image boundary: bbox={[x1, y1, x2, y2]} "
            f"image={width}x{height}",
        )
    scale = float(getattr(frame, "depth_scale"))
    if not np.isfinite(scale) or scale <= 0.0:
        return False, "camera depth scale is invalid"
    z_min = float(z_min_m)
    z_max = float(z_max_m)
    if not np.isfinite(z_min) or not np.isfinite(z_max) or not 0.0 < z_min < z_max:
        raise ValueError("grounding candidate depth range must be finite z_min<z_max")
    crop_m = depth_raw[y1:y2, x1:x2].astype(np.float32) * scale
    in_range = np.isfinite(crop_m) & (crop_m >= z_min) & (crop_m <= z_max)
    bbox_pixels = int(crop_m.size)
    valid_pixels = int(np.count_nonzero(in_range))
    minimum_ratio = float(minimum_valid_depth_ratio)
    if not np.isfinite(minimum_ratio) or not 0.0 < minimum_ratio <= 1.0:
        raise ValueError("minimum_valid_depth_ratio must be in (0,1]")
    required_pixels = max(4, int(np.ceil(minimum_ratio * float(bbox_pixels))))
    measured = crop_m[np.isfinite(crop_m) & (crop_m > 0.0)]
    measured_median = None if measured.size == 0 else float(np.median(measured))
    if valid_pixels < required_pixels:
        median_text = "none" if measured_median is None else f"{measured_median:.3f}m"
        return (
            False,
            f"depth outside task range [{z_min:.2f},{z_max:.2f}]m: "
            f"in_range={valid_pixels}/{bbox_pixels} required={required_pixels} "
            f"measured_p50={median_text}",
        )
    accepted_median = float(np.median(crop_m[in_range]))
    workspace_reason = ""
    workspace_values = (
        T_base_camera,
        workspace_center_base_m,
        workspace_max_horizontal_distance_m,
    )
    if any(value is not None for value in workspace_values):
        if not all(value is not None for value in workspace_values):
            raise ValueError(
                "grounding workspace admission requires transform, center and radius"
            )
        transform = np.asarray(T_base_camera, dtype=np.float64)
        center = np.asarray(workspace_center_base_m, dtype=np.float64).reshape(3)
        radius = float(workspace_max_horizontal_distance_m)
        intrinsics = getattr(frame, "intrinsics", None)
        if (
            transform.shape != (4, 4)
            or not np.all(np.isfinite(transform))
            or not np.all(np.isfinite(center))
            or not np.isfinite(radius)
            or radius <= 0.0
            or intrinsics is None
        ):
            raise ValueError("grounding workspace configuration is invalid")
        fx = float(getattr(intrinsics, "fx"))
        fy = float(getattr(intrinsics, "fy"))
        ppx = float(getattr(intrinsics, "ppx"))
        ppy = float(getattr(intrinsics, "ppy"))
        if not np.all(np.isfinite([fx, fy, ppx, ppy])) or fx <= 0.0 or fy <= 0.0:
            raise ValueError("grounding workspace camera intrinsics are invalid")
        rows, columns = np.nonzero(in_range)
        depths = crop_m[in_range].astype(np.float64)
        pixels_u = columns.astype(np.float64) + float(x1)
        pixels_v = rows.astype(np.float64) + float(y1)
        points_camera = np.column_stack(
            (
                (pixels_u - ppx) / fx * depths,
                (pixels_v - ppy) / fy * depths,
                depths,
                np.ones_like(depths),
            )
        )
        points_base = (transform @ points_camera.T).T[:, :3]
        base_center = np.median(points_base, axis=0)
        horizontal_distance = float(np.linalg.norm(base_center[:2] - center[:2]))
        if horizontal_distance > radius:
            return (
                False,
                "candidate is outside the task acquisition corridor: "
                f"base_center={base_center.tolist()} "
                f"horizontal_distance={horizontal_distance:.3f}m>"
                f"{radius:.3f}m",
            )
        workspace_reason = (
            f" base_center={[round(float(value), 3) for value in base_center]} "
            f"workspace_distance={horizontal_distance:.3f}/{radius:.3f}m"
        )
    return (
        True,
        f"depth PASS in_range={valid_pixels}/{bbox_pixels} "
        f"p50={accepted_median:.3f}m range=[{z_min:.2f},{z_max:.2f}]m"
        f"{workspace_reason}",
    )


def _grounding_preview_process(
    payload_queue: object,
    stop_event: object,
    cancelled_event: object,
    ready_connection: object,
    window_name: str,
) -> None:
    """Tk/Pillow preview isolated from the parent's OpenCV/Qt runtime."""

    root = None
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        import tkinter as tk

        import numpy as np
        from PIL import Image, ImageDraw, ImageFont, ImageTk

        root = tk.Tk()
        root.title(window_name)
        root.geometry("848x600+20+40")
        root.resizable(True, True)
        label = tk.Label(root, background="black")
        label.pack(fill=tk.BOTH, expand=True)

        def cancel(_event: object = None) -> None:
            cancelled_event.set()

        root.protocol("WM_DELETE_WINDOW", cancel)
        root.bind("q", cancel)
        root.bind("Q", cancel)
        root.bind("<Escape>", cancel)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18
            )
        except BaseException:
            font = ImageFont.load_default()
        root.update_idletasks()
        root.update()
        ready_connection.send({"status": "READY"})
        ready_connection.close()
        photo = None
        latest = None
        while not stop_event.is_set() and not cancelled_event.is_set():
            while True:
                try:
                    latest = payload_queue.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                color_bgr, state, bbox_xyxy, detail, prompt = latest
                color = {
                    "searching": (255, 215, 0),
                    "candidate": (0, 255, 255),
                    "rejected": (255, 80, 0),
                    "tracked": (0, 255, 0),
                }.get(str(state), (255, 255, 255))
                rgb = np.ascontiguousarray(
                    np.asarray(color_bgr, dtype=np.uint8)[..., ::-1]
                )
                image = Image.fromarray(rgb, mode="RGB")
                scale = max(1, int(np.ceil(848.0 / max(1, image.width))))
                image = image.resize(
                    (image.width * scale, image.height * scale),
                    resample=Image.Resampling.NEAREST,
                )
                draw = ImageDraw.Draw(image)
                if bbox_xyxy is not None:
                    bbox = np.asarray(bbox_xyxy, dtype=np.int32).reshape(4)
                    draw.rectangle(
                        tuple(int(value) * scale for value in bbox.tolist()),
                        outline=color,
                        width=max(2, 2 * scale),
                    )
                lines = (
                    f"{str(state).upper()}: {prompt}",
                    str(detail),
                    (
                        "KEEP HOLDING - wait for green ARMED before throwing"
                        if state != "tracked"
                        else "TRACKED - switching to final mask/cloud view"
                    ),
                    "Esc/Q cancels",
                )
                overlay_height = 30 * len(lines) + 14
                draw.rectangle(
                    (0, 0, image.width, overlay_height), fill=(0, 0, 0)
                )
                for index, line in enumerate(lines):
                    fill = color if index == 0 else (245, 245, 245)
                    draw.text((13, 10 + 29 * index), line, font=font, fill=fill)
                photo = ImageTk.PhotoImage(image=image)
                label.configure(image=photo)
                latest = None
            try:
                root.update_idletasks()
                root.update()
            except BaseException:
                cancelled_event.set()
                break
            stop_event.wait(0.015)
    except BaseException as exc:
        try:
            ready_connection.send(
                {"status": "ERROR", "detail": f"{type(exc).__name__}: {exc}"}
            )
        except BaseException:
            pass
    finally:
        try:
            ready_connection.close()
        except BaseException:
            pass
        if root is not None:
            try:
                root.destroy()
            except BaseException:
                pass


class _GroundingSearchPreview:
    """Latest-only isolated preview; never loads Qt in the camera owner."""

    def __init__(self, prompt: str) -> None:
        self.prompt = str(prompt)
        self.window_name = "Thrown object grounding"
        self.enabled = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        self._warned = False
        self._context = mp.get_context("spawn")
        self._queue = None
        self._stop = None
        self._cancelled = None
        self._process = None

    def _open(self) -> None:
        if not self.enabled or self._process is not None:
            return
        self._queue = self._context.Queue(maxsize=1)
        self._stop = self._context.Event()
        self._cancelled = self._context.Event()
        parent_ready, child_ready = self._context.Pipe(duplex=False)
        self._process = self._context.Process(
            target=_grounding_preview_process,
            args=(
                self._queue,
                self._stop,
                self._cancelled,
                child_ready,
                self.window_name,
            ),
            name="thrown-object-grounding-preview",
            daemon=True,
        )
        self._process.start()
        child_ready.close()
        try:
            if not parent_ready.poll(5.0):
                raise RuntimeError("grounding preview did not become ready in 5s")
            response = parent_ready.recv()
            if response.get("status") != "READY":
                raise RuntimeError(
                    "grounding preview failed: "
                    + str(response.get("detail", "unknown error"))
                )
        except BaseException:
            self.close()
            raise
        finally:
            parent_ready.close()

    def show(
        self,
        frame: object,
        state: str,
        bbox_xyxy: Optional[object],
        detail: str,
    ) -> None:
        if not self.enabled:
            if not self._warned:
                print(
                    "[Object grounding][WARN] live preview unavailable because "
                    "DISPLAY/WAYLAND_DISPLAY is absent",
                    flush=True,
                )
                self._warned = True
            return
        import numpy as np
        self._open()
        assert self._queue is not None
        assert self._cancelled is not None
        if self._cancelled.is_set():
            raise KeyboardInterrupt("operator cancelled object grounding preview")
        payload = (
            np.ascontiguousarray(
                np.asarray(getattr(frame, "color_bgr"), dtype=np.uint8)
            ).copy(),
            str(state),
            (
                None
                if bbox_xyxy is None
                else np.asarray(bbox_xyxy, dtype=np.int32).reshape(4).copy()
            ),
            str(detail),
            self.prompt,
        )
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                pass
        if self._process is not None and not self._process.is_alive():
            raise RuntimeError("grounding preview process stopped unexpectedly")

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        assert self._stop is not None
        self._stop.set()
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        for ipc in (self._queue,):
            if ipc is not None:
                try:
                    ipc.close()
                    ipc.join_thread()
                except BaseException:
                    pass
        self._process = None
        self._queue = None
        self._stop = None
        self._cancelled = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select one V94 object ROI using only the configured D435."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--pcd-config", type=Path, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--result-fd", type=int, required=True)
    parser.add_argument(
        "--object-text",
        default=None,
        help=(
            "automatically wait for this category using camera-only "
            "YOLO-World -> SAM2 instead of opening the ROI GUI"
        ),
    )
    parser.add_argument(
        "--compact-console",
        action="store_true",
        help="show only operator-facing grounding stages and failures",
    )
    return parser


def _write_record(descriptor: int, record: Mapping[str, object]) -> None:
    if int(descriptor) < 3:
        raise ValueError("result fd must be a dedicated inherited descriptor")
    encoded = (
        json.dumps(
            dict(record),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if len(encoded) > MAX_RESULT_BYTES:
        raise RuntimeError("ROI selector result exceeded its fixed bound")
    view = memoryview(encoded)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("ROI selector result pipe stopped accepting data")
        view = view[written:]


def _select(
    *,
    bundle_path: Path,
    pcd_config_path: Path,
    object_text: Optional[str] = None,
) -> dict[str, object]:
    """Open only the D435 and resolve an interactive or text prompt."""

    import numpy as np

    from .capture import _initialized_roi_evidence
    from sim2real.deployment.bundle import DeployBundle
    from .live_preview import _initialize_provider
    from .camera_profile import resolve_runtime_camera_contract
    from sim2real.contracts.v94 import V94Contract

    bundle = DeployBundle(bundle_path.expanduser().resolve())
    bundle.verify()
    contract = resolve_runtime_camera_contract(
        V94Contract.from_bundle(bundle), pcd_config_path
    )
    provider = None
    prompt_manager = None
    try:
        normalized_text = (
            None if object_text is None else str(object_text).strip()
        )
        if normalized_text:
            # Import through the configured point-cloud package only inside
            # this camera-only child.  No Franka/RH56 module is imported.
            from dynamic_pcd.apps.realtime_masked_pcd import (
                prompt_service_launcher_args,
                prompt_service_launcher_path,
                validate_prompt_service_health,
                validate_prompting_config,
                wait_for_prompt_target,
            )
            from dynamic_pcd.config import load_config
            from dynamic_pcd.provider.object_pcd_provider import (
                ObjectPCDProvider,
            )
            from dynamic_pcd.segmentation.prompt_protocol import validate_prompt
            from dynamic_pcd.segmentation.prompt_replay import (
                RecentRGBDFrameBuffer,
            )
            from dynamic_pcd.segmentation.prompt_runtime import (
                PromptServiceManager,
            )

            normalized_text = validate_prompt(normalized_text)
            config = load_config(str(pcd_config_path.expanduser().resolve()))
            prompt_cfg = config.get("prompting", {})
            validate_prompting_config(prompt_cfg)
            prompt_manager = PromptServiceManager(
                addr=str(prompt_cfg["service_addr"]),
                autostart=bool(prompt_cfg["service_autostart"]),
                startup_timeout_s=float(prompt_cfg["startup_timeout_s"]),
                request_timeout_ms=max(
                    1, int(1000.0 * float(prompt_cfg["request_timeout_s"]))
                ),
                launcher_path=str(prompt_service_launcher_path(prompt_cfg)),
                launcher_args=prompt_service_launcher_args(
                    prompt_cfg, prompt=normalized_text
                ),
            )
            # Load every model before opening the D435 so the camera readiness
            # gate describes the stream immediately preceding SEARCHING.
            prompt_health = prompt_manager.start()
            validate_prompt_service_health(prompt_cfg, prompt_health)
            provider = ObjectPCDProvider(config)
            provider.start()
            fps = max(1.0, float(config["camera"].get("fps", 30)))
            frame_buffer = RecentRGBDFrameBuffer(
                retention_s=float(prompt_cfg.get("replay_buffer_s", 2.0)),
                capacity_frames=max(
                    2,
                    int(
                        np.ceil(
                            float(prompt_cfg.get("replay_buffer_s", 2.0))
                            * fps
                        )
                    )
                    + 2,
                ),
            )
            detector_backend = str(
                prompt_cfg.get("detector_backend", "grounding_dino")
            ).strip().lower()
            preview = _GroundingSearchPreview(normalized_text)
            print(
                "[Object grounding SEARCHING] live camera preview opened; "
                "hold the target naturally in view; small hand jitter is allowed; "
                "do not throw yet",
                flush=True,
            )
            try:
                while True:
                    detected_frame, result = wait_for_prompt_target(
                        provider=provider,
                        prompt_manager=prompt_manager,
                        prompt=normalized_text,
                        frame_buffer=frame_buffer,
                        box_threshold=float(prompt_cfg["box_threshold"]),
                        text_threshold=float(prompt_cfg["text_threshold"]),
                        mask_threshold=float(prompt_cfg["mask_threshold"]),
                        top_k=int(prompt_cfg["top_k"]),
                        search_interval_s=float(
                            prompt_cfg.get("search_interval_s", 0.5)
                        ),
                        detector_backend=detector_backend,
                        reference_bbox_xyxy=prompt_cfg.get(
                            "search_reference_roi_xyxy"
                        ),
                        yolo_world_entry_filter_config=prompt_cfg,
                        preview_callback=preview.show,
                    )
                    bbox = np.asarray(
                        result.bbox_xyxy, dtype=np.int32
                    ).reshape(4)
                    depth_ok, depth_reason = _grounding_candidate_depth_admission(
                        detected_frame,
                        bbox,
                        z_min_m=float(config["camera"]["z_min"]),
                        z_max_m=float(config["camera"]["z_max"]),
                        minimum_valid_depth_ratio=float(
                            config["tracker"].get(
                                "component_min_valid_depth_ratio", 0.25
                            )
                        ),
                        require_interior=True,
                        T_base_camera=provider.extrinsics.T_base_camera,
                        workspace_center_base_m=prompt_cfg.get(
                            "grounding_workspace_center_base_m"
                        ),
                        workspace_max_horizontal_distance_m=prompt_cfg.get(
                            "grounding_workspace_max_horizontal_distance_m"
                        ),
                    )
                    if not depth_ok:
                        preview.show(
                            detected_frame, "rejected", bbox, depth_reason
                        )
                        print(
                            "[Object grounding RETRY] rejected semantic "
                            f"candidate bbox={bbox.tolist()}: {depth_reason}; "
                            "continuing camera-only search",
                            flush=True,
                        )
                        continue
                    if detector_backend == "yolo_world":
                        if not provider.initialize_from_bbox(detected_frame, bbox):
                            retry_reason = (
                                "SAM2/tracker could not initialize this candidate"
                            )
                            preview.show(
                                detected_frame,
                                "rejected",
                                bbox,
                                retry_reason,
                            )
                            print(
                                "[Object grounding RETRY] "
                                f"bbox={bbox.tolist()} {retry_reason}; "
                                "keep the target still and visible",
                                flush=True,
                            )
                            continue
                        initialization = _initialized_roi_evidence(provider)
                        if str(
                            np.asarray(
                                initialization["initialization_mask_source"]
                            ).item()
                        ) != "online_sam2_box":
                            raise RuntimeError(
                                "YOLO-World bbox did not initialize the required "
                                "online SAM2 mask path"
                            )
                    preview.show(detected_frame, "tracked", bbox, depth_reason)
                    print(
                        "[Object grounding TRACKED] "
                        f"bbox={bbox.tolist()} {depth_reason}; "
                        "keep holding naturally until Throw trigger ARMED",
                        flush=True,
                    )
                    break
            finally:
                preview.close()
            roi_array = np.asarray(
                [
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2] - bbox[0]),
                    int(bbox[3] - bbox[1]),
                ],
                dtype=np.int32,
            )
            mask_source = (
                "yolo_world_box_to_online_sam2"
                if detector_backend == "yolo_world"
                else "grounding_dino_sam_replayed"
            )
        else:
            provider, _selection = _initialize_provider(
                pcd_config_path.expanduser().resolve(),
                None,
                disable_online_sam2=False,
            )
            initialization = _initialized_roi_evidence(provider)
            roi_array = np.asarray(initialization["roi_xywh"])
            mask_source = str(
                np.asarray(
                    initialization["initialization_mask_source"]
                ).item()
            )
        if (
            roi_array.shape != (4,)
            or not np.issubdtype(roi_array.dtype, np.integer)
        ):
            raise RuntimeError(
                "object selector returned no integer numeric XYWH"
            )
        roi = tuple(int(value) for value in roi_array.tolist())
        x, y, width, height = roi
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > contract.camera_width
            or y + height > contract.camera_height
        ):
            raise RuntimeError(
                "object selector ROI exceeds the V94 camera frame"
            )
        extrinsics = provider.extrinsics
        if str(extrinsics.camera_serial) != contract.camera_serial:
            raise RuntimeError(
                "object selector camera serial differs from V94"
            )
        if str(extrinsics.calibration_id) != contract.calibration_id:
            raise RuntimeError(
                "object selector calibration differs from V94"
            )
        if not np.allclose(
            np.asarray(extrinsics.T_base_camera, dtype=np.float64),
            contract.T_base_camera_optical,
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise RuntimeError(
                "object selector extrinsics differ from V94"
            )
        return {
            "roi_xywh": list(roi),
            "camera_serial": contract.camera_serial,
            "calibration_id": contract.calibration_id,
            "mask_source": mask_source,
            "object_text": normalized_text,
        }
    finally:
        if provider is not None:
            provider.stop()
        if prompt_manager is not None:
            prompt_manager.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    nonce = str(args.nonce).strip().lower()
    if re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
        raise ValueError("nonce must contain exactly 32 lowercase hex digits")
    descriptor = int(args.result_fd)
    record: dict[str, object]
    exit_code = 0
    try:
        from sim2real.console_output import compact_deployment_console

        with compact_deployment_console(enabled=bool(args.compact_console)):
            selected = _select(
                bundle_path=args.bundle,
                pcd_config_path=args.pcd_config,
                object_text=args.object_text,
            )
        record = {
            "protocol": ROI_SELECTOR_PROTOCOL,
            "nonce": nonce,
            "status": "ok",
            **selected,
        }
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            detail = "selection interrupted or cancelled"
        else:
            detail = f"{type(exc).__name__}: {exc}"
        record = {
            "protocol": ROI_SELECTOR_PROTOCOL,
            "nonce": nonce,
            "status": "error",
            "error": detail,
        }
        exit_code = 1
    try:
        _write_record(descriptor, record)
    except BaseException as exc:
        print(
            "V94 isolated ROI selector could not report its result: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    if exit_code:
        print(
            "V94 isolated ROI selector failed without opening robot interfaces: "
            f"{record.get('error', 'unspecified error')}",
            file=sys.stderr,
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
