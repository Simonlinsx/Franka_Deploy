"""Small GUI environment fixes for OpenCV/Open3D on Ubuntu + conda/venv.

Call :func:`setup_gui_env` before importing cv2. OpenCV 4.13 may overwrite its
font path during import, so call :func:`repair_gui_env_after_cv2_import` once
immediately afterwards as well.
"""
from __future__ import annotations

import os


_SYSTEM_FONT_DIRS = (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts",
)


def _first_system_font_dir() -> str | None:
    for path in _SYSTEM_FONT_DIRS:
        if os.path.isdir(path):
            return path
    return None


def setup_gui_env() -> None:
    # OpenCV's Qt backend sometimes looks for fonts inside cv2/qt/fonts. Use the
    # system DejaVu directory if available. Install with: sudo apt install fonts-dejavu-core fontconfig
    if "QT_QPA_FONTDIR" not in os.environ:
        font_dir = _first_system_font_dir()
        if font_dir is not None:
            os.environ["QT_QPA_FONTDIR"] = font_dir

    # Prefer the XCB backend on Ubuntu desktop/X11. Do not force this if the user
    # already configured Wayland/offscreen/etc.
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    # Mesa driver path for Open3D/OpenCV software GL fallback.
    os.environ.setdefault("LIBGL_DRIVERS_PATH", "/usr/lib/x86_64-linux-gnu/dri")

    # Conda sometimes injects Qt plugin paths that conflict with opencv-python's
    # bundled Qt plugins. Remove only obviously conda/anaconda-managed paths.
    for key in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        value = os.environ.get(key, "")
        if "conda" in value.lower() or "anaconda" in value.lower() or "/.venv/" in value:
            os.environ.pop(key, None)


def repair_gui_env_after_cv2_import() -> None:
    """Repair only paths that the OpenCV wheel overwrote with missing dirs.

    Do not clear Qt plugin paths here: after ``import cv2`` they legitimately
    point at the wheel's bundled XCB plugin. The common 4.13 wheel does however
    replace ``QT_QPA_FONTDIR`` with a non-existent ``cv2/qt/fonts`` directory.
    """

    configured = os.environ.get("QT_QPA_FONTDIR", "")
    if configured and os.path.isdir(configured):
        return
    font_dir = _first_system_font_dir()
    if font_dir is not None:
        os.environ["QT_QPA_FONTDIR"] = font_dir
