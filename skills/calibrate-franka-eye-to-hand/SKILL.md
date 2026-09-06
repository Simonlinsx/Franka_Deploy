---
name: calibrate-franka-eye-to-hand
description: Commission, recalibrate, validate, and deploy a fixed external RGB-D camera for a Franka/FR3 eye-to-hand setup in this workspace. Use when adding or replacing a RealSense camera, collecting ArUco or ChArUco pose pairs, solving T_base_camera, checking calibration quality and holdout closure, creating serial-specific runtime configs, or deciding whether an existing fixed-camera calibration is still valid. Do not use this as an eye-in-hand procedure for a camera mounted on the EEF.
---

# Calibrate a Franka Eye-to-Hand Camera

Produce a traceable transform from the fixed camera color-optical frame to
`robot_base`, validate it independently, and deploy it without overwriting an
existing commissioned camera.

## Read the relevant resources

- Read [references/workstation.md](references/workstation.md) before running
  commands in this Franka workspace. It defines authoritative roots, file
  naming, target parameters, and command templates.
- Read [references/acceptance.md](references/acceptance.md) before solving,
  validating, or approving a result.
- Run [scripts/audit_calibration.py](scripts/audit_calibration.py) on every
  proposed final calibration and again after deployment. Treat its success as
  a static screening pass, never as full commissioning acceptance.
- Run [scripts/validate_holdout.py](scripts/validate_holdout.py) on a separate
  dataset that was never supplied to the solver.

## Enforce the scope and safety contract

1. Establish the camera topology first.
   - Continue only when the camera is rigidly fixed outside the robot and its
     pose relative to the Franka base is constant.
   - Stop and explain that a camera mounted on the EEF needs an eye-in-hand
     `T_ee_camera` workflow and synchronized robot poses online.
   - For a non-RealSense camera, reuse the hand-eye solver but first implement
     and test a camera adapter that supplies color images, calibrated color
     intrinsics, timestamps, and color-aligned depth. The existing collector
     directly supports RealSense devices only.
2. Treat camera enumeration, frame capture, robot-state reads, offline solving,
   and file inspection as read-only diagnostics.
3. Never command robot motion merely because calibration was requested.
   Require explicit authorization for automatic motion in the current session.
4. Before any authorized motion, verify the emergency stop is reachable,
   clear the full EEF/board/cable swept volume, establish a conservative table
   height and minimum EEF z, bound speed and acceleration, and preview every
   pose. Abort on target loss, unexpected contact, clearance violation, robot
   fault, or operator cancellation.
5. Keep the collector and motion controller separate. The existing collector
   reads `O_T_EE`; it never moves the robot.
6. Never overwrite an old dataset, calibration, validation report, or runtime
   config. Use a serial-specific session slug and preserve provenance.
7. Fail closed. Do not enable robot-facing output until the final calibration
   passes structural, in-sample, independent holdout, serial, and physical
   point-cloud checks.

## Execute the workflow

### 1. Inventory and freeze the setup

Record:

- camera model, serial, color/depth stream, resolution, FPS, depth scale, and
  color intrinsics;
- robot model, base frame, robot IP, and pose source (`O_T_EE`);
- target type, dictionary, ID, and measured physical dimensions;
- camera bracket, robot base, and workcell state;
- whether this camera is new, a replacement, or an additional camera.

Mount the camera and finish all cable routing before collection. Any subsequent
camera/bracket/base movement invalidates the result, even if the replacement
camera occupies the old bracket.

Create a unique slug such as `fr3-d435-337322072188-20260811-143500`. Include
time or an incrementing run number so two sessions on one day cannot collide.
Use it in all dataset, audit, final, validation, and runtime filenames.

### 2. Prepare the target

Prefer a rigid, flat, matte ChArUco board when EEF space permits. Reuse the
commissioned single ArUco only after remeasuring its outer black square. For the
existing marker, the declared target is `DICT_ARUCO_ORIGINAL`, ID 582, outer
black side `0.100 m`; never substitute paper or white-margin size.

Rigidly fix the target to the EEF. It need not align precisely with EEF axes
because `T_ee_target` is solved jointly. It must not flex, slip, or be remounted
between any training or holdout samples.

### 3. Prove camera and target observability

Before moving the robot:

- open the exact requested camera serial;
- verify the selected color profile and reported intrinsics;
- align depth to color when RGB-D validation will be used;
- capture a sharp frame and confirm the complete target is detected;
- verify the configured target dimensions and dictionary produce a plausible
  PnP pose and low reprojection error.

Do not proceed with an ambiguous serial, a partial target, auto-scaled print,
glare, blur, or an unstable camera mount.

### 4. Plan and collect stationary pose pairs

Use either interactive `collect` or repeated headless `capture` from the command
templates. Capture 20–25 accepted stationary training samples, preserving every
debug image when practical. Then collect at least five additional poses into a
separate holdout dataset. Never pass the holdout file to `solve`.

Cover:

- left, center, and right image regions;
- near, middle, and far depths in the intended workspace;
- different heights;
- pronounced rotations around at least two, preferably three axes;
- the center and edges of the commissioned perception volume.

Accept one sample only after the robot is fully stationary and the complete
target is sharp. Reject near-duplicate poses, grazing views, target motion, or
poor PnP. For headless capture, verify the training or holdout path is absent at
session start; subsequent writes may append only within that same active file.

For automatic motion, generate a bounded pose list first, show its base-frame
positions/orientations and clearance checks, then request the explicit motion
authorization. Do not expand the authorized workspace or descend below its
minimum z based only on camera imagery.

### 5. Solve an audit result, then a final result

Solve once with `--method auto` to compare supported algorithms. Inspect method
agreement, cross-validation diagnostics, rejected samples, and motion diversity.
Do not assume the previously selected Daniilidis method must win for a new
camera.

After selecting the method supported by the audit and holdout evidence, solve a
new final file explicitly with that method and `--fail-on-warning`. Do not edit,
transpose, invert, or manually round `T_base_camera`.

Run the bundled audit script. It verifies static metadata, profile, transform,
ID, training metrics, and optional runtime wiring. Its result remains
`provisional`; a structurally valid YAML or successful solver exit is not
acceptance.

### 6. Perform independent validation

Use at least five target poses stored in the separate holdout dataset. For each
pose compare:

```text
T_base_ee @ T_ee_target
T_base_camera @ T_camera_target
```

Report translation and rotation closure per pose plus median, RMS, p95, and max.
Run the bundled holdout validator with task-approved p95 limits and retain its
machine-readable output.
Also perform a mount-rigidity check using a controlled EEF rotation while the
target remains fixed to the EEF.

Independently validate RGB-D scale at several distances. Compare color PnP,
aligned depth, and at least one known base-frame location. Do not infer a global
depth correction from one plane.

### 7. Deploy under a new serial-specific config

Copy the accepted final file into the deployment calibration directory under a
new name. Create a separate runtime config and set:

```yaml
extrinsics:
  calibration_file: calibrations/<new-final>.yaml
  require_calibration: true
  require_quality_pass: true
  strict_camera_serial: true
```

Set `camera.serial`, width, height, FPS, depth limits, and filter/profile choices
for the new device. They must agree with the collected calibration metadata and
live camera report. Keep the previous camera config intact. Do not blindly
inherit the old view's support plane, point-cloud workspace, segmentation
thresholds, goal marker, or hover envelope: reset or recommission each one and
keep `runtime.publish_zmq: false`. Never pass `--enable_robot_motion` during new
camera commissioning.

Run the audit script against the deployed calibration and runtime config. Add a
new serial-specific runtime test; do not replace the test for the first camera.

### 8. Validate the downstream point cloud

With robot motion disabled, confirm every valid packet reports:

- `reference_frame=robot_base`;
- the new calibration ID;
- the new camera serial.

For every masked depth pixel use:

```text
z = depth_raw * depth_scale
p_camera = [(u-cx)z/fx, (v-cy)z/fy, z]
p_base = T_base_camera @ [p_camera, 1]
```

Place a stationary object or validation target at multiple known base-frame
locations and check axis direction, scale, center, and workspace edges. Keep
robot output disabled until task-specific error and clearance margins pass.

### 9. Close the session

Remove the calibration target and temporary brackets from the EEF. Removing it
does not invalidate fixed-camera `T_base_camera`; leaving it attached changes the
collision envelope. Record hashes and paths for the dataset, audit result, final
calibration, validation report, runtime config, camera serial, and evidence
images.

## Report the outcome

Return a concise commissioning report containing:

- topology and hardware identity;
- exact target dimensions;
- dataset path, sample count, and motion diversity;
- selected method and why;
- calibration ID and `T_base_camera` direction;
- in-sample and holdout metrics;
- RGB-D depth observations;
- deployed config path and serial gate;
- physical invalidation conditions;
- an explicit `accepted`, `rejected`, or `provisional` decision.

Return `accepted` only when static audit, separate holdout, mount rigidity,
multi-distance depth, known-base-point, runtime wiring, and task-specific error
and clearance limits all pass. Otherwise return `provisional` or `rejected` and
list the missing evidence.

Never describe solver residuals as guaranteed absolute robot positioning
accuracy. Include depth, segmentation, target-scale, compliance, and task-margin
effects in the decision.
