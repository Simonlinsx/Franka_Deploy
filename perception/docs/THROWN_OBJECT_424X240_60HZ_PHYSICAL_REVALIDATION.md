# Thrown-object 424x240@60 physical revalidation runbook

Camera: Intel RealSense D435 serial `342222071785`  
Target: single ArUco `DICT_6X6_50`, ID `42`, measured black side `0.190 m`  
Robot: Franka state is read only; none of these commands can command motion  
Decision before completion: provisional; Franka/RH56 execution disabled

## Fixed inputs

```text
camera config:
  beta/dynamic_object_pcd/configs/d435_342222071785_424x240_60hz_revalidation.yaml

frozen extrinsic:
  perception/configs/calibrations/
  fr3_d435_342222071785_eye_to_hand_424x240_60hz_reuse_v1.yaml
```

Do not move the external camera, its bracket, or the Franka base.  Rigidly fix
the marker to the EEF and do not remount or flex it between the reference and
holdout captures.  Because the marker is currently absent and may be remounted
at a different EEF offset, the first capture defines only this session's
`T_ee_target`; it is excluded from all holdout metrics.  `T_base_camera` stays
byte-for-byte unchanged.

Automatic robot motion is outside this runbook and requires fresh explicit
operator authorization.  The operator may stage poses using an independently
approved controller, then run a capture only after the robot is stationary.

## 1. Camera-only visibility

Choose a new session slug.  Every output path must be new.

```bash
cd /home/qiaoguanren/code/franka

SESSION=fr3-d435-342222071785-424x240-60hz-holdout-YYYYMMDD-HHMMSS
RUNS=/home/qiaoguanren/code/franka/beta/dynamic_object_pcd/calibration_runs
CONFIG=/home/qiaoguanren/code/franka/beta/dynamic_object_pcd/configs/d435_342222071785_424x240_60hz_revalidation.yaml
CALIB=/home/qiaoguanren/code/franka/perception/configs/calibrations/fr3_d435_342222071785_eye_to_hand_424x240_60hz_reuse_v1.yaml

.venv/bin/python \
  skills/calibrate-franka-eye-to-hand/scripts/inspect_aruco_frame.py \
  --config "$CONFIG" \
  --camera-serial 342222071785 \
  --raw-output "$RUNS/${SESSION}-visibility-raw.png" \
  --annotated-output "$RUNS/${SESSION}-visibility-annotated.png" \
  --frames 60 \
  --dictionary DICT_6X6_50
```

Continue only if exactly ID 42 is fully visible, sharp, and away from the image
boundary.  This step opens only the camera.

## 2. One mount reference plus at least five independent holdouts

From `beta/dynamic_object_pcd`, run the same read-only capture once per staged
pose.  Use pose labels `R00`, then `H01` through `H05` or more.  The dataset
path remains the same; the debug image must be unique.

```bash
cd /home/qiaoguanren/code/franka/beta/dynamic_object_pcd

POSE_ID=R00
/home/qiaoguanren/code/franka/.venv/bin/python \
  -m dynamic_pcd.apps.calibrate_eye_to_hand capture \
  --config "$CONFIG" \
  --robot-ip 172.16.0.2 \
  --camera-serial 342222071785 \
  --target-type aruco \
  --dictionary DICT_6X6_50 \
  --marker-id 42 \
  --marker-length 0.190 \
  --frames 120 \
  --min-valid-frame-fraction 0.95 \
  --min-valid-frames 114 \
  --max-reprojection-error 0.5 \
  --max-target-translation-jitter 0.001 \
  --max-target-rotation-jitter 0.3 \
  --min-all-reprojection-pass-frames 118 \
  --max-all-reprojection-pass-translation-jitter 0.001 \
  --max-all-reprojection-pass-rotation-jitter 0.3 \
  --max-all-reprojection-pass-reprojection-error 0.5 \
  --max-stationary-translation 0.0005 \
  --max-stationary-rotation 0.1 \
  --min-pose-translation 0.01 \
  --min-pose-rotation 3.0 \
  --debug-image "$RUNS/${SESSION}-${POSE_ID}.png" \
  --output "$RUNS/${SESSION}.yaml"
```

Cover left/centre/right, near/middle/far, and rotations about multiple axes.
After collecting `R00 + H01..H05`, validate without solving:

```bash
cd /home/qiaoguanren/code/franka

.venv/bin/python \
  skills/calibrate-franka-eye-to-hand/scripts/validate_reused_extrinsic_holdout.py \
  "$CALIB" \
  "$RUNS/${SESSION}.yaml" \
  --output "$RUNS/${SESSION}-reused-extrinsic-holdout.json" \
  --minimum-holdout-samples 5 \
  --minimum-translation-span-m 0.08 \
  --minimum-rotation-span-deg 20 \
  --max-translation-p95-mm 10 \
  --max-rotation-p95-deg 1.5
```

This validator fits only `T_ee_target_session` from `R00`.  It never fits,
edits, averages, or inverts `T_base_camera`.

## 3. Multi-distance aligned-depth stations

Capture at least near/left, middle/centre, and far/right stations.  Pass the
successful holdout report so a remounted target uses the session-local mount
transform.  The command opens the camera and a read-only Franka state handle;
it has no motion interface.

```bash
cd /home/qiaoguanren/code/franka

STATION=near-left
.venv/bin/python \
  skills/calibrate-franka-eye-to-hand/scripts/validate_rgbd_physical.py \
  capture-station \
  --config "$CONFIG" \
  --calibration "$CALIB" \
  --camera-serial 342222071785 \
  --station-id "$STATION" \
  --distance-band near \
  --image-region left \
  --frames 120 \
  --depth-min-m 0.25 \
  --depth-max-m 1.50 \
  --pixel-stride 1 \
  --minimum-depth-points 500 \
  --session-target-report "$RUNS/${SESSION}-reused-extrinsic-holdout.json" \
  --output "$RUNS/${SESSION}-${STATION}-rgbd.json" \
  --evidence-dir "$RUNS/${SESSION}-${STATION}-rgbd-evidence" \
  --confirm-stationary-read-only
```

Repeat with unique paths and the corresponding `middle/centre` and `far/right`
labels.  The median target-camera z values must span at least `0.25 m`.

## 4. Independently known robot-base point

At least one station must place the marker centre at a point whose
`[x,y,z]` in `robot_base` is independently established by a surveyed fixture,
metrology fixture, or recorded robot touch-off.  The evidence file must state
the method, units, coordinates, uncertainty, date, and responsible operator.
With a surveyed point, the capture does not open Franka at all:

```bash
.venv/bin/python \
  skills/calibrate-franka-eye-to-hand/scripts/validate_rgbd_physical.py \
  capture-station \
  --config "$CONFIG" \
  --calibration "$CALIB" \
  --camera-serial 342222071785 \
  --station-id known-base-fixture \
  --distance-band middle \
  --image-region center \
  --frames 120 \
  --depth-min-m 0.25 \
  --depth-max-m 1.50 \
  --pixel-stride 1 \
  --minimum-depth-points 500 \
  --known-base-center-m X Y Z \
  --known-base-source surveyed_fixture \
  --known-base-evidence /ABSOLUTE/PATH/known-base-evidence.yaml \
  --output "$RUNS/${SESSION}-known-base-rgbd.json" \
  --evidence-dir "$RUNS/${SESSION}-known-base-rgbd-evidence" \
  --confirm-stationary-read-only
```

Evaluate all station reports with predeclared task gates:

```bash
.venv/bin/python \
  skills/calibrate-franka-eye-to-hand/scripts/validate_rgbd_physical.py \
  evaluate \
  "$RUNS/${SESSION}-near-left-rgbd.json" \
  "$RUNS/${SESSION}-middle-center-rgbd.json" \
  "$RUNS/${SESSION}-far-right-rgbd.json" \
  "$RUNS/${SESSION}-known-base-rgbd.json" \
  --output "$RUNS/${SESSION}-rgbd-validation.json" \
  --expected-serial 342222071785 \
  --expected-camera-name "Intel RealSense D435" \
  --minimum-stations 4 \
  --minimum-image-regions 3 \
  --minimum-distance-span-m 0.25 \
  --minimum-depth-valid-fraction-p05 0.90 \
  --max-plane-residual-p95-mm 5 \
  --max-depth-pnp-z-abs-p95-mm 15 \
  --max-depth-pnp-z-abs-max-mm 25 \
  --max-depth-pnp-3d-p95-mm 20 \
  --max-known-base-z-abs-p95-mm 15 \
  --max-known-base-z-abs-max-mm 25 \
  --max-known-base-3d-p95-mm 20 \
  --max-known-base-3d-max-mm 30 \
  --max-station-signed-bias-range-mm 10 \
  --require-independent-known-base
```

## 5. Catch-workspace point-cloud gate

Only after holdout and physical RGB-D PASS, record throws constrained to the
actual reachable catch volume.  Re-run the frozen text-grounding/current-mask
validator and formal 128-point projector.  Required task gates are:

- current mask on at least 95% of reviewed visible policy ticks;
- zero reviewed complete-absence false positives;
- mask plus projector p95 no more than 50 ms and max no more than 100 ms;
- exactly 128 finite XYZ points in `robot_base` on at least 95% of reviewed
  visible ticks inside the reachable catch volume;
- centre error against the physical reference no more than 15 mm p95 and
  25 mm max;
- no stale cloud relabelled as current-frame evidence.

Any failed physical gate keeps the task provisional.  Do not loosen a limit
after seeing the result.  A systematic holdout failure triggers a full new
hand-eye calibration; isolated dynamic-depth dropout is handled separately and
must preserve current-frame provenance.
