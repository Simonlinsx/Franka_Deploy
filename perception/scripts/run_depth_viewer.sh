#!/usr/bin/env bash
set -e
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}
export QT_QPA_FONTDIR=${QT_QPA_FONTDIR:-/usr/share/fonts/truetype/dejavu}
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM_PLUGIN_PATH
python -m dynamic_pcd.apps.realtime_depth_viewer --config configs/d435_default.yaml "$@"
