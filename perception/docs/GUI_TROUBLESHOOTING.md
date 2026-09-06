# GUI troubleshooting

## Qt font warnings from OpenCV

If you see:

```text
QFontDatabase: Cannot find font directory .../site-packages/cv2/qt/fonts
```

install system fonts and run through the provided scripts:

```bash
sudo apt install -y fonts-dejavu-core fontconfig
./scripts/run_depth_viewer.sh
```

The scripts set:

```bash
QT_QPA_FONTDIR=/usr/share/fonts/truetype/dejavu
QT_QPA_PLATFORM=xcb
```

These warnings are usually non-fatal if the window is displayed.

## QObject::moveToThread warnings

If you see:

```text
QObject::moveToThread: Current thread ... is not the object's thread ...
```

this is usually a Qt backend conflict from OpenCV's GUI backend, conda/venv Qt
plugins, or mixing OpenCV windows with Open3D windows in the same process.

Use the provided run scripts first. They unset conflicting Qt plugin paths:

```bash
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM_PLUGIN_PATH
```

If the warning persists but the window works, it is usually safe to ignore.

For the masked point cloud app, the most robust deployment mode is to avoid two
GUI backends in one process:

```bash
./scripts/run_masked_pcd.sh --no_show_pcd
```

or publish via ZMQ and visualize in another process.
