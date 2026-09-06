# Calibration Acceptance Criteria

## Contents

- [Final-file gates](#final-file-gates)
- [Independent evidence](#independent-evidence)
- [Depth and task accuracy](#depth-and-task-accuracy)
- [Previous commissioning reference](#previous-commissioning-reference)
- [Invalidation conditions](#invalidation-conditions)

## Final-file gates

Apply these screening gates to the proposed final result:

| Metric | Required screening gate |
|---|---:|
| usable samples | at least 15; prefer 20–25 |
| robot translation span | at least 0.08 m |
| robot rotation span | at least 30 deg |
| reprojection error p95 | at most 1.0 px |
| hand-eye translation residual p95 | at most 0.010 m |
| hand-eye rotation residual p95 | at most 2.0 deg |
| quality status | `pass` |
| final warnings | none |
| calibration ID | matches matrix plus camera serial |

Treat these as rejection/screening gates, not a guarantee of absolute
base-frame accuracy.

Review algorithm agreement. If supported solvers differ materially, retain the
auto audit, investigate target scale, mount rigidity, pose diversity, and
outliers, then choose the final method using holdout evidence. Do not average
incompatible transforms.

## Independent evidence

A final deployment requires samples not used by the solver. Report closure for:

```text
T_base_ee @ T_ee_target
T_base_camera @ T_camera_target
```

Include per-pose and aggregate translation/rotation errors. Cover the intended
workspace center and edges. Store at least five holdout poses in a separate
dataset that is never passed to `solve`. Run `scripts/validate_holdout.py` with
limits established for the task. Also check a controlled EEF rotation to show
that the target mount is rigid.

Set task-specific acceptance limits before enabling motion. Hovering, grasping,
and contact tasks need different margins; do not silently adopt the solver
screening thresholds as robot safety limits.

The static audit script deliberately cannot return full commissioning
acceptance. Mark the result `accepted` only after static, holdout, mount-rigidity,
multi-distance depth, known-base-point, runtime, and task-clearance evidence all
pass. Missing physical evidence means `provisional`.

## Depth and task accuracy

Validate aligned depth separately because hand-eye closure can pass while depth
scale or bias remains wrong. At multiple working distances compare:

- marker/board color PnP depth;
- median aligned depth on the target plane;
- a known base-frame point or plane;
- repeatability across frames and image regions.

Do not apply a global z correction from a single plane. Include segmentation
boundary error, D435 noise, target-size measurement, robot compliance, and TCP
uncertainty in the final task margin.

## Previous commissioning reference

The prior D435 commissioning achieved:

| Metric | Value |
|---|---:|
| samples | 20 |
| translation span | 0.147057 m |
| rotation span | 35.804907 deg |
| reprojection p95 | 0.170 px |
| translation residual p95 | 2.653 mm |
| rotation residual p95 | 0.483 deg |
| q7 holdout mean | 1.723 mm / 0.172 deg |
| post-trajectory closure | 0.905 mm / 0.408 deg |

Its marker-center aligned-depth versus color-PnP difference was about 9.994 mm
on one plane. This was a depth/target-scale observation, not a hand-eye closure
failure. It demonstrates why sub-centimeter claims require multi-distance depth
validation.

Use these numbers only as a reference for detecting a clear regression. A new
camera and mount need their own task-derived acceptance limits.

## Invalidation conditions

Recalibrate and revalidate after any of the following:

- camera, lens assembly, bracket, or Franka base moves or is disturbed;
- camera device, resolution, alignment profile, or intrinsics change;
- target dimensions or target mount change during an unfinished session;
- repeated physical checks show systematic or orientation-dependent drift;
- the workcell is rebuilt or transported.

Removing the EEF calibration target after a completed fixed-camera calibration
does not invalidate `T_base_camera`.
