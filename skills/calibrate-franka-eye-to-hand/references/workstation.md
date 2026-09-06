# Franka Workstation Reference

## Contents

- [Authoritative paths](#authoritative-paths)
- [Existing commissioned example](#existing-commissioned-example)
- [Naming](#naming)
- [Environment preflight](#environment-preflight)
- [Collection commands](#collection-commands)
- [Solve and audit commands](#solve-and-audit-commands)
- [Deployment checks](#deployment-checks)

## Authoritative paths

```text
workspace:       /home/qiaoguanren/code/franka
calibration CLI: /home/qiaoguanren/code/franka/beta/dynamic_object_pcd
deployment:      /home/qiaoguanren/code/franka/perception
skill source:    /home/qiaoguanren/code/franka/skills/calibrate-franka-eye-to-hand
```

The reusable collection/solve CLI is:

```text
beta/dynamic_object_pcd/dynamic_pcd/apps/calibrate_eye_to_hand.py
```

The deployment checkout intentionally contains only runtime calibration loading
and transforms. Do not assume it contains the collection CLI.

## Existing commissioned example

Use this only as provenance and a quality reference, never as the transform for
a different camera:

```text
camera:          Intel RealSense D435
serial:          337322072188
stream:          color-aligned depth, 848x480 at 30 Hz
target:          DICT_ARUCO_ORIGINAL, ID 582
black side:      0.100 m
calibration ID:  eye-to-hand-b722bce10485c8a3
final method:    daniilidis
samples:         20
```

Files:

```text
beta/dynamic_object_pcd/calibration_runs/fr3_d435_aruco582_final_v2_dataset.yaml
perception/configs/calibrations/fr3_d435_eye_to_hand.yaml
perception/configs/calibrations/fr3_d435_eye_to_hand.validation.yaml
```

## Naming

Derive a filesystem-safe slug from robot, camera model, serial, and date:

```text
fr3-<camera>-<serial>-<YYYYMMDD-HHMMSS>
```

Recommended artifacts:

```text
calibration_runs/<slug>-dataset.yaml
calibration_runs/<slug>-holdout.yaml
calibration_runs/<slug>-auto-audit.yaml
calibration_runs/<slug>-validation.yaml
configs/calibrations/<slug>-eye-to-hand.yaml
configs/<slug>-runtime.yaml
```

Refuse an existing path unless the user explicitly requests a new versioned
filename. Do not use `--overwrite` during normal commissioning. A headless
`capture` command intentionally appends to its active dataset; ensure that path
was absent before the first training capture and use a different absent path for
the first holdout capture.

## Environment preflight

Run from the calibration CLI root with the Python environment intended for the
hardware session:

```bash
cd /home/qiaoguanren/code/franka/beta/dynamic_object_pcd
python -c "import cv2; assert hasattr(cv2, 'aruco'); import pyrealsense2"
python -m dynamic_pcd.apps.calibrate_eye_to_hand collect --help
```

Confirm no other process owns the RealSense or Franka FCI connection. Enumerate
the camera and compare its physical serial with the requested device. Inspect
the live stream before allowing robot motion.

## Collection commands

For the existing single ArUco target, substitute `NEW_SERIAL` and `SLUG`:

```bash
cd /home/qiaoguanren/code/franka/beta/dynamic_object_pcd
python -m dynamic_pcd.apps.calibrate_eye_to_hand collect \
  --config configs/d435_default.yaml \
  --robot-ip 172.16.0.2 \
  --camera-serial NEW_SERIAL \
  --target-type aruco \
  --dictionary DICT_ARUCO_ORIGINAL \
  --marker-id 582 \
  --marker-length 0.100 \
  --output calibration_runs/SLUG-dataset.yaml
```

Interactive keys:

```text
SPACE/c: accept current stationary pose
d/Backspace: remove the last sample
q/Escape: finish and retain the dataset
```

For a headless, externally supervised pose sequence, append one sample at a
time:

```bash
python -m dynamic_pcd.apps.calibrate_eye_to_hand capture \
  --config configs/d435_default.yaml \
  --robot-ip 172.16.0.2 \
  --camera-serial NEW_SERIAL \
  --target-type aruco \
  --dictionary DICT_ARUCO_ORIGINAL \
  --marker-id 582 \
  --marker-length 0.100 \
  --frames 20 \
  --debug-image calibration_runs/SLUG-sample-NNN.png \
  --output calibration_runs/SLUG-dataset.yaml
```

The `capture` command rejects robot motion above `0.5 mm` or `0.1 deg` during
one capture by default. The collector rejects poor reprojection and poses too
close to the previous sample.

For ChArUco, generate and print an SVG at 100% size, measure physical squares,
then use exactly the same dimensions during collection. See the CLI `--help`
instead of copying the old single-marker parameters.

## Solve and audit commands

First create the algorithm-comparison result:

```bash
python -m dynamic_pcd.apps.calibrate_eye_to_hand solve \
  --dataset calibration_runs/SLUG-dataset.yaml \
  --output calibration_runs/SLUG-auto-audit.yaml \
  --method auto \
  --fail-on-warning
```

Inspect the auto audit and holdout behavior. Then solve the final file with the
selected method, represented here as `SELECTED_METHOD`:

```bash
python -m dynamic_pcd.apps.calibrate_eye_to_hand solve \
  --dataset calibration_runs/SLUG-dataset.yaml \
  --output configs/calibrations/SLUG-eye-to-hand.yaml \
  --method SELECTED_METHOD \
  --fail-on-warning
```

Audit the result from the workspace environment:

```bash
/home/qiaoguanren/code/franka/.venv/bin/python \
  /home/qiaoguanren/code/franka/skills/calibrate-franka-eye-to-hand/scripts/audit_calibration.py \
  /home/qiaoguanren/code/franka/beta/dynamic_object_pcd/configs/calibrations/SLUG-eye-to-hand.yaml \
  --expected-serial NEW_SERIAL \
  --expected-dictionary DICT_ARUCO_ORIGINAL \
  --expected-marker-id 582 \
  --expected-marker-length-m 0.100
```

Collect at least five validation poses into `calibration_runs/SLUG-holdout.yaml`
without ever passing that file to `solve`. After choosing task-specific closure
limits, validate it independently:

```bash
/home/qiaoguanren/code/franka/.venv/bin/python \
  /home/qiaoguanren/code/franka/skills/calibrate-franka-eye-to-hand/scripts/validate_holdout.py \
  configs/calibrations/SLUG-eye-to-hand.yaml \
  calibration_runs/SLUG-holdout.yaml \
  --max-translation-p95-mm TASK_LIMIT_MM \
  --max-rotation-p95-deg TASK_LIMIT_DEG \
  --json
```

## Deployment checks

Copy the accepted final YAML to a new file under:

```text
/home/qiaoguanren/code/franka/perception/configs/calibrations/
```

Create a new runtime YAML with `apply_patch` rather than changing the
commissioned D435 config. Refuse an existing output path. Point
`camera.serial` and `extrinsics.calibration_file` at the new camera and enable
all three fail-closed gates. Set the recorded width, height, and FPS. Reset
camera-view-dependent support-plane/workspace/tracker values until separately
commissioned; leave `runtime.publish_zmq: false` and do not enable robot motion.

Then run:

```bash
cd /home/qiaoguanren/code/franka/perception
../.venv/bin/python \
  ../skills/calibrate-franka-eye-to-hand/scripts/audit_calibration.py \
  configs/calibrations/SLUG-eye-to-hand.yaml \
  --runtime-config configs/SLUG-runtime.yaml \
  --expected-serial NEW_SERIAL \
  --expected-dictionary DICT_ARUCO_ORIGINAL \
  --expected-marker-id 582 \
  --expected-marker-length-m 0.100

env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=/home/qiaoguanren/anaconda3/lib/python3.9/site-packages \
  ../.venv/bin/python -m pytest -q tests/test_calibration_runtime.py
```

The existing runtime test pins the commissioned D435 file. Add serial-specific
tests for a newly deployed camera instead of replacing those assertions. A
static audit PASS is still provisional until holdout and physical validation
evidence also pass.
