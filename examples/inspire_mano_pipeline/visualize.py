from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .model import HARDWARE_JOINTS, ManoDetection, RetargetOutput


HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)


def _mesh_triangles_for_image(
    detection: ManoDetection, image_shape: tuple[int, int]
) -> Optional[np.ndarray]:
    """Return clipped int32 triangles, or None for absent/invalid mesh data."""

    vertices = getattr(detection, "vertices_2d", None)
    faces = getattr(detection, "mesh_faces", None)
    if vertices is None or faces is None:
        return None
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 2
        or vertices.size == 0
        or not np.all(np.isfinite(vertices))
        or faces.ndim != 2
        or faces.shape[1] != 3
        or faces.size == 0
        or not np.all(np.isfinite(faces))
    ):
        return None
    rounded_faces = np.rint(faces)
    if not np.array_equal(faces, rounded_faces):
        return None
    faces = rounded_faces.astype(np.int64, copy=False)
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        return None
    try:
        height, width = int(image_shape[0]), int(image_shape[1])
    except (TypeError, ValueError, IndexError):
        return None
    if height <= 0 or width <= 0:
        return None
    triangles = vertices[faces]
    visible = (
        (triangles[:, :, 0].max(axis=1) >= 0)
        & (triangles[:, :, 0].min(axis=1) < width)
        & (triangles[:, :, 1].max(axis=1) >= 0)
        & (triangles[:, :, 1].min(axis=1) < height)
    )
    triangles = triangles[visible]
    if not len(triangles):
        return None
    # OpenCV clips ordinary off-screen geometry, but bounding coordinates here
    # avoids float-to-int overflow for a corrupt yet finite projection.
    triangles[:, :, 0] = np.clip(triangles[:, :, 0], -width, 2 * width)
    triangles[:, :, 1] = np.clip(triangles[:, :, 1], -height, 2 * height)
    return np.rint(triangles).astype(np.int32)


def _draw_mano_mesh(canvas: np.ndarray, detection: ManoDetection) -> None:
    triangles = _mesh_triangles_for_image(detection, canvas.shape[:2])
    if triangles is None:
        return
    contours = [triangle for triangle in triangles]
    translucent = canvas.copy()
    cv2.fillPoly(
        translucent,
        contours,
        (190, 120, 45),
        lineType=cv2.LINE_AA,
    )
    cv2.addWeighted(translucent, 0.20, canvas, 0.80, 0.0, dst=canvas)
    # MANO has roughly 1.5k faces.  A uniformly sampled wireframe remains
    # legible while keeping the real-time overlay inexpensive.
    max_wire_faces = 450
    step = max(1, (len(contours) + max_wire_faces - 1) // max_wire_faces)
    cv2.polylines(
        canvas,
        contours[::step],
        True,
        (235, 175, 90),
        1,
        cv2.LINE_AA,
    )


def draw_overlay(
    image_bgr: np.ndarray,
    detection: Optional[ManoDetection],
    output: Optional[RetargetOutput],
    inference_fps: float,
    hardware_state: str = "PREVIEW ONLY",
    selected_axes: Optional[Sequence[str]] = None,
    tracking_status: Optional[str] = None,
    sent_targets: Optional[Sequence[int]] = None,
    actual_angles: Optional[Sequence[int]] = None,
    operator_roi: Optional[Sequence[float]] = None,
) -> np.ndarray:
    canvas = image_bgr.copy()
    if operator_roi is not None:
        roi = tuple(float(value) for value in operator_roi)
        if len(roi) == 4 and np.all(np.isfinite(roi)):
            height, width = canvas.shape[:2]
            x1, y1, x2, y2 = (
                int(round(roi[0] * width)),
                int(round(roi[1] * height)),
                int(round(roi[2] * width)),
                int(round(roi[3] * height)),
            )
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (210, 90, 255), 2)
            cv2.putText(
                canvas,
                "OPERATOR ROI",
                (x1 + 6, min(height - 8, y1 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (210, 90, 255),
                1,
                cv2.LINE_AA,
            )
    if detection is not None:
        _draw_mano_mesh(canvas, detection)
        keypoints = np.asarray(detection.keypoints_2d)
        if keypoints.shape == (21, 2) and np.all(np.isfinite(keypoints)):
            points = np.rint(keypoints).astype(np.int32)
            for first, second in HAND_CONNECTIONS:
                cv2.line(
                    canvas,
                    tuple(points[first]),
                    tuple(points[second]),
                    (60, 220, 80),
                    2,
                    cv2.LINE_AA,
                )
            for point in points:
                cv2.circle(
                    canvas, tuple(point), 3, (30, 80, 255), -1, cv2.LINE_AA
                )
        bbox = np.asarray(detection.bbox_xyxy)
        if bbox.shape == (4,) and np.all(np.isfinite(bbox)):
            limit = 4 * max(canvas.shape[:2])
            x1, y1, x2, y2 = np.rint(
                np.clip(bbox, -limit, limit)
            ).astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 180, 40), 2)

    telemetry_text = None
    selected = set(selected_axes or HARDWARE_JOINTS)
    if sent_targets is not None or actual_angles is not None:
        sent = tuple(sent_targets) if sent_targets is not None else (None,) * 6
        actual = tuple(actual_angles) if actual_angles is not None else (None,) * 6
        if len(sent) == 6 and len(actual) == 6:
            parts = []
            for index, name in enumerate(HARDWARE_JOINTS):
                if name not in selected:
                    continue
                target = (
                    output.hardware_targets[index] if output is not None else None
                )
                parts.append(
                    f"{name} T:{target if target is not None else '-'} "
                    f"S:{sent[index] if sent[index] is not None else '-'} "
                    f"A:{actual[index] if actual[index] is not None else '-'}"
                )
            telemetry_text = " | ".join(parts) or None
    header_height = 116 if tracking_status and telemetry_text else (
        90 if tracking_status or telemetry_text else 64
    )
    cv2.rectangle(
        canvas, (0, 0), (canvas.shape[1], header_height), (20, 20, 20), -1
    )
    depth_text = "depth=n/a"
    if detection is not None:
        display_depth = getattr(detection, "control_palm_depth_m", None)
        if display_depth is None:
            display_depth = detection.palm_depth_m
        if display_depth is not None:
            suffix = (
                " held"
                if getattr(detection, "palm_depth_source", "measured") == "held"
                else ""
            )
            depth_text = f"depth={display_depth:.3f}m{suffix}"
    backend = output.backend if output is not None else "no-hand"
    cv2.putText(
        canvas,
        f"WiLoR/MANO -> {backend} | {inference_fps:.1f} FPS | {depth_text}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    if tracking_status:
        tracking_color = (
            (70, 220, 90)
            if tracking_status == "TRACKING OK"
            else (70, 180, 255)
        )
    if telemetry_text:
        cv2.putText(
            canvas,
            telemetry_text,
            (12, 104 if tracking_status else 77),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 210, 80),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            tracking_status,
            (12, 77),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            tracking_color,
            2,
            cv2.LINE_AA,
        )
    state_color = (80, 220, 255) if hardware_state == "PREVIEW ONLY" else (60, 80, 255)
    cv2.putText(
        canvas,
        hardware_state,
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        state_color,
        2,
        cv2.LINE_AA,
    )

    if output is not None:
        bar_width = max(72, canvas.shape[1] // 7)
        base_y = canvas.shape[0] - 16
        for index, (name, target) in enumerate(
            zip(HARDWARE_JOINTS, output.hardware_targets)
        ):
            x = 8 + index * bar_width
            enabled = name in selected and target >= 0
            value = target if enabled else 0
            height = int(70 * value / 1000.0)
            color = (70, 210, 80) if enabled else (100, 100, 100)
            cv2.rectangle(canvas, (x, base_y - 70), (x + 18, base_y), (60, 60, 60), 1)
            if enabled:
                cv2.rectangle(
                    canvas,
                    (x + 1, base_y - height),
                    (x + 17, base_y - 1),
                    color,
                    -1,
                )
            label = name.replace("thumb_", "t_")
            cv2.putText(
                canvas,
                label,
                (x + 22, base_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                str(target) if enabled else "OFF",
                (x + 22, base_y - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas
