from __future__ import annotations

from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Mapping, Optional

import cv2
import numpy as np


def put_lines(img: np.ndarray, lines, org=(12, 24), scale=0.55) -> np.ndarray:
    out = img.copy()
    x, y = org
    for line in lines:
        cv2.putText(out, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
        y += int(26 * scale / 0.55)
    return out


class Open3DLiveViewer:
    """Live object/scene viewer with adjustable point size via keyboard.

    ``update`` retains the original single-cloud API.  ``update_composite``
    displays a dim RGB scene and a separately highlighted segmented object in
    the same robot-base coordinate system.

    Keys:
      ] : increase point size
      [ : decrease point size
      R : reset view on next update
      Q : close window
    """

    def __init__(
        self,
        title="Object point cloud",
        point_size: float = 3.0,
        command_bindings: Optional[Mapping[str, str]] = None,
    ):
        import open3d as o3d

        self.o3d = o3d
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        created = self.vis.create_window(title, width=960, height=720)
        if not created:
            raise RuntimeError("Open3D could not create a visualization window")
        self.pcd = o3d.geometry.PointCloud()
        self.object_pcd = self.pcd
        self.scene_pcd = o3d.geometry.PointCloud()
        self.added = False
        self.scene_added = False
        self.center_marker = None
        self.center_position = None
        self.goal_marker = None
        self.goal_position = None
        self.center_goal_line = o3d.geometry.LineSet()
        self.line_added = False
        self.point_size = point_size
        self.reset_view_next = True
        self.closed = False
        # Open3D invokes key callbacks while ``poll_events`` is running.  Keep
        # those callbacks side-effect free: they only enqueue a command, and
        # the application's perception loop drains and executes it later.  A
        # SimpleQueue is thread-safe as well, so this remains safe if a GUI
        # backend dispatches callbacks from a separate event thread.
        self._command_queue = SimpleQueue()
        self._command_callbacks = []

        self.vis.register_key_callback(ord("Q"), self._close)
        self.vis.register_key_callback(ord("R"), self._reset)
        self.vis.register_key_callback(ord("]"), self._increase)
        self.vis.register_key_callback(ord("["), self._decrease)
        self._register_command_bindings(command_bindings or {})
        self._apply_render_options()

        # The axes are anchored at robot_base: X red, Y green, Z blue.
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        self.vis.add_geometry(axes, reset_bounding_box=False)

    def _apply_render_options(self):
        opt = self.vis.get_render_option()
        opt.point_size = float(self.point_size)
        opt.background_color = np.asarray([0.02, 0.02, 0.02])

    def _close(self, _vis):
        self.closed = True
        return False

    def _reset(self, _vis):
        self.reset_view_next = True
        return False

    def _increase(self, _vis):
        self.point_size = min(20.0, self.point_size + 1.0)
        self._apply_render_options()
        print(f"[Open3D] point_size={self.point_size}")
        return False

    def _decrease(self, _vis):
        self.point_size = max(1.0, self.point_size - 1.0)
        self._apply_render_options()
        print(f"[Open3D] point_size={self.point_size}")
        return False

    def _register_command_bindings(self, bindings: Mapping[str, str]) -> None:
        reserved = {"Q", "R", "[", "]"}
        for key, command in bindings.items():
            normalized = str(key).upper()
            if len(normalized) != 1:
                raise ValueError(f"Open3D command key must be one character, got {key!r}")
            if normalized in reserved:
                raise ValueError(f"Open3D command key {normalized!r} is reserved by the viewer")
            command_name = str(command).strip()
            if not command_name:
                raise ValueError(f"Open3D command for key {normalized!r} is empty")

            callback = self._make_command_callback(normalized, command_name)
            # Keep an explicit reference because some Open3D/pybind versions
            # do not retain Python closures reliably for the window lifetime.
            self._command_callbacks.append(callback)
            self.vis.register_key_callback(ord(normalized), callback)

    def _make_command_callback(self, key: str, command: str):
        def callback(_vis):
            self._command_queue.put(command)
            return False

        return callback

    def drain_commands(self) -> list[str]:
        """Return all queued GUI commands without blocking.

        The returned list preserves key-press order.  Commands are consumed at
        most once and are deliberately not executed inside the GUI callback.
        """

        commands = []
        while True:
            try:
                commands.append(self._command_queue.get_nowait())
            except Empty:
                return commands

    def poll_events(self) -> bool:
        """Process window events without changing any displayed geometry."""

        if self.closed:
            return False
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        if not alive:
            self.closed = True
        return not self.closed

    def update(self, points: np.ndarray, colors: Optional[np.ndarray] = None) -> bool:
        if self.closed:
            return False
        if points is None or len(points) == 0:
            return self.poll_events()
        self.pcd.points = self.o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None and len(colors) == len(points):
            self.pcd.colors = self.o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
        else:
            self.pcd.colors = self.o3d.utility.Vector3dVector(np.tile(np.array([[0.2, 0.8, 1.0]]), (len(points), 1)))

        if not self.added:
            self.vis.add_geometry(self.pcd)
            self.added = True
        else:
            self.vis.update_geometry(self.pcd)

        if self.reset_view_next:
            self.vis.reset_view_point(True)
            self.reset_view_next = False

        return self.poll_events()

    def update_composite(
        self,
        scene_points: Optional[np.ndarray],
        scene_colors: Optional[np.ndarray],
        object_points: Optional[np.ndarray],
        center: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
        scene_brightness: float = 0.32,
        scene_color_floor: float = 0.08,
    ) -> bool:
        """Update scene/object geometries without merging or copying between them.

        ``scene_points=None`` means retain the previous scene cloud.  This lets
        callers update the scene at a lower display rate while refreshing the
        object and processing GUI events on every perception frame.
        """

        if self.closed:
            return False

        if scene_points is not None:
            scene = self._as_points(scene_points)
            self.scene_pcd.points = self.o3d.utility.Vector3dVector(scene)
            if scene_colors is not None and len(scene_colors) == len(scene):
                rgb = np.clip(np.asarray(scene_colors, dtype=np.float64), 0.0, 1.0)
                brightness = float(np.clip(scene_brightness, 0.05, 1.0))
                color_floor = float(np.clip(scene_color_floor, 0.0, 0.4))
                rgb = np.clip(rgb * brightness + color_floor, 0.0, 1.0)
            else:
                rgb = np.tile(np.asarray([[0.22, 0.25, 0.28]]), (len(scene), 1))
            self.scene_pcd.colors = self.o3d.utility.Vector3dVector(rgb)
            if not self.scene_added and len(scene) > 0:
                self.vis.add_geometry(self.scene_pcd, reset_bounding_box=False)
                self.scene_added = True
            elif self.scene_added:
                self.vis.update_geometry(self.scene_pcd)

        obj = self._as_points(object_points)
        self.object_pcd.points = self.o3d.utility.Vector3dVector(obj)
        # Uniform orange-red makes the segmented cloud unmistakable against
        # the dim original scene colors.
        obj_rgb = np.tile(np.asarray([[1.0, 0.12, 0.02]]), (len(obj), 1))
        self.object_pcd.colors = self.o3d.utility.Vector3dVector(obj_rgb)
        if not self.added and len(obj) > 0:
            self.vis.add_geometry(self.object_pcd, reset_bounding_box=False)
            self.added = True
        elif self.added:
            self.vis.update_geometry(self.object_pcd)

        self._update_marker(
            "center_marker", "center_position", center, radius=0.007,
            color=np.asarray([0.1, 1.0, 0.2]),
        )
        self._update_marker(
            "goal_marker", "goal_position", goal, radius=0.009,
            color=np.asarray([1.0, 0.0, 1.0]),
        )
        self._update_center_goal_line(center, goal)

        if self.reset_view_next and (self.scene_added or self.added):
            self.vis.reset_view_point(True)
            self.reset_view_next = False

        return self.poll_events()

    @staticmethod
    def _as_points(points: Optional[np.ndarray]) -> np.ndarray:
        if points is None:
            return np.zeros((0, 3), dtype=np.float64)
        array = np.asarray(points, dtype=np.float64)
        if array.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError(f"point cloud must have shape [N,3+], got {array.shape}")
        return array[:, :3]

    def _update_marker(self, marker_attr, position_attr, position, radius, color):
        marker = getattr(self, marker_attr)
        previous = getattr(self, position_attr)
        if position is None:
            if marker is not None:
                self.vis.remove_geometry(marker, reset_bounding_box=False)
                setattr(self, marker_attr, None)
                setattr(self, position_attr, None)
            return

        position = np.asarray(position, dtype=np.float64).reshape(3)
        if marker is None:
            marker = self.o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
            marker.compute_vertex_normals()
            marker.paint_uniform_color(np.asarray(color, dtype=np.float64))
            marker.translate(position)
            self.vis.add_geometry(marker, reset_bounding_box=False)
            setattr(self, marker_attr, marker)
        else:
            marker.translate(position - previous)
            self.vis.update_geometry(marker)
        setattr(self, position_attr, position)

    def _update_center_goal_line(self, center, goal):
        if center is None or goal is None:
            if self.line_added:
                self.vis.remove_geometry(self.center_goal_line, reset_bounding_box=False)
                self.line_added = False
            return
        points = np.stack(
            [np.asarray(center, dtype=np.float64), np.asarray(goal, dtype=np.float64)]
        )
        self.center_goal_line.points = self.o3d.utility.Vector3dVector(points)
        self.center_goal_line.lines = self.o3d.utility.Vector2iVector([[0, 1]])
        self.center_goal_line.colors = self.o3d.utility.Vector3dVector([[1.0, 0.8, 0.0]])
        if not self.line_added:
            self.vis.add_geometry(self.center_goal_line, reset_bounding_box=False)
            self.line_added = True
        else:
            self.vis.update_geometry(self.center_goal_line)

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception:
            pass


class OpenCVPointCloudViewer:
    """Fallback point cloud viewer when Open3D is unavailable.

    It shows two live 2D projections:
      left: X-Y front projection
      right: X-Z depth projection
    """

    def __init__(self, title="Object point cloud projection", width: int = 960, height: int = 720):
        self.title = title
        self.width = int(width)
        self.height = int(height)
        self.closed = False
        self._scene_points = np.zeros((0, 3), dtype=np.float32)
        self._scene_colors = np.zeros((0, 3), dtype=np.float32)
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)

    def update(self, points: np.ndarray, colors: Optional[np.ndarray] = None) -> bool:
        if self.closed:
            return False

        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 18)

        if points is None or len(points) == 0:
            put_lines(canvas, ["No valid object points"], org=(24, 40), scale=0.6)
            cv2.imshow(self.title, canvas)
            return True

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            put_lines(canvas, ["Invalid point shape"], org=(24, 40), scale=0.6)
            cv2.imshow(self.title, canvas)
            return True

        point_colors = self._colors_for_points(pts, colors)
        margin = 24
        gap = 20
        panel_w = (self.width - 2 * margin - gap) // 2
        panel_h = self.height - 2 * margin
        left = (margin, margin, panel_w, panel_h)
        right = (margin + panel_w + gap, margin, panel_w, panel_h)

        self._draw_projection(canvas, pts, point_colors, dims=(0, 1), panel=left, title="X-Y")
        self._draw_projection(canvas, pts, point_colors, dims=(0, 2), panel=right, title="X-Z")
        cv2.imshow(self.title, canvas)
        return True

    def update_composite(
        self,
        scene_points: Optional[np.ndarray],
        scene_colors: Optional[np.ndarray],
        object_points: Optional[np.ndarray],
        center: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
        scene_brightness: float = 0.32,
        scene_color_floor: float = 0.08,
    ) -> bool:
        if scene_points is not None:
            scene = np.asarray(scene_points, dtype=np.float32).reshape(-1, 3)
            rgb = np.asarray(scene_colors, dtype=np.float32).reshape(-1, 3)
            if len(scene) > 5000:
                idx = np.linspace(0, len(scene) - 1, 5000, dtype=np.int64)
                scene = scene[idx]
                rgb = rgb[idx]
            self._scene_points = scene
            self._scene_colors = np.clip(
                rgb * float(np.clip(scene_brightness, 0.05, 1.0))
                + float(np.clip(scene_color_floor, 0.0, 0.4)),
                0.0,
                1.0,
            )

        obj = (
            np.zeros((0, 3), dtype=np.float32)
            if object_points is None
            else np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
        )
        arrays = [self._scene_points, obj]
        colors = [
            self._scene_colors,
            np.tile(np.asarray([[1.0, 0.12, 0.02]], dtype=np.float32), (len(obj), 1)),
        ]
        if center is not None:
            arrays.append(np.asarray(center, dtype=np.float32).reshape(1, 3))
            colors.append(np.asarray([[0.1, 1.0, 0.2]], dtype=np.float32))
        if goal is not None:
            arrays.append(np.asarray(goal, dtype=np.float32).reshape(1, 3))
            colors.append(np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32))
        return self.update(np.concatenate(arrays), np.concatenate(colors))

    def close(self):
        self.closed = True
        try:
            cv2.destroyWindow(self.title)
        except Exception:
            pass

    def _colors_for_points(self, pts: np.ndarray, colors: Optional[np.ndarray]) -> np.ndarray:
        if colors is not None and len(colors) == len(pts):
            rgb = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
            return (rgb[:, ::-1] * 255.0).astype(np.uint8)

        z = pts[:, 2]
        z_min = float(np.percentile(z, 5))
        z_max = float(np.percentile(z, 95))
        denom = max(1e-6, z_max - z_min)
        norm = np.clip((z - z_min) / denom, 0.0, 1.0)
        vals = (norm * 255.0).astype(np.uint8)
        return cv2.applyColorMap(vals[:, None], cv2.COLORMAP_TURBO)[:, 0, :]

    def _draw_projection(self, canvas: np.ndarray, pts: np.ndarray, colors: np.ndarray, dims, panel, title: str) -> None:
        x, y, w, h = panel
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 80), 1)
        cv2.putText(canvas, title, (x + 12, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

        a = pts[:, dims[0]]
        b = pts[:, dims[1]]
        amin, amax = self._robust_range(a)
        bmin, bmax = self._robust_range(b)
        pad_a = max(0.02, 0.08 * (amax - amin))
        pad_b = max(0.02, 0.08 * (bmax - bmin))
        amin -= pad_a
        amax += pad_a
        bmin -= pad_b
        bmax += pad_b

        px = x + 10 + ((a - amin) / max(1e-6, amax - amin) * (w - 20)).astype(np.int32)
        py = y + h - 10 - ((b - bmin) / max(1e-6, bmax - bmin) * (h - 20)).astype(np.int32)
        inside = (px >= x + 1) & (px < x + w) & (py >= y + 1) & (py < y + h)

        for xi, yi, c in zip(px[inside], py[inside], colors[inside]):
            cv2.circle(canvas, (int(xi), int(yi)), 2, tuple(int(v) for v in c.tolist()), -1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            f"n={len(pts)}  {amin:.2f}..{amax:.2f}, {bmin:.2f}..{bmax:.2f} m",
            (x + 12, y + h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    def _robust_range(self, values: np.ndarray):
        lo = float(np.percentile(values, 2))
        hi = float(np.percentile(values, 98))
        if hi <= lo:
            lo = float(values.min()) - 0.05
            hi = float(values.max()) + 0.05
        if hi <= lo:
            hi = lo + 0.1
        return lo, hi
