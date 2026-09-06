# Dual-task object mask and point-cloud acceptance

Date: 2026-08-13.

The perception stack has two independent production targets:

1. `tabletop`: object manipulation on the table, including hand approach,
   contact, partial occlusion, complete occlusion, and reappearance.
2. `thrown_object`: a small ball or beanbag entering from outside the image and
   moving quickly through a commissioned catch workspace.

The two targets use separate runtime profiles and are accepted independently.
Passing one target never compensates for a failure in the other. Overall
acceptance is the logical AND of both rows.

## Common hard requirements

- Text grounding is discovery only. A detector bbox is never a policy mask.
- Every published mask and point cloud belongs to the current RGB-D tuple. No
  stale-mask substitution is allowed.
- An absent or completely occluded target publishes no object point cloud.
- Every valid policy observation contains exactly 128 current-frame points in
  `robot_base` and records the camera serial and calibration ID.
- Both trained policies consume aligned RGB-D and masks at exactly `424x240`.
  The tabletop profile obtains this by deterministic stride-2 decimation from
  native `848x480`; the thrown-object profile captures native `424x240` and
  therefore uses the identity adapter.
- The runtime fails closed on a serial, calibration, stream-profile, timestamp,
  mask-provenance, or depth-quality mismatch.
- Provider plus formal 128-point projection must fit the 20 Hz policy budget:
  p95 at most 50 ms, maximum at most 100 ms, and at most 1% of evaluated ticks
  above 50 ms.

## Task-specific gates

| Gate | Tabletop | Thrown object |
|---|---|---|
| Camera profile | Serial `337322072188`, native 848x480 at 30 Hz | Serial `342222071785`, native 424x240 at 60 Hz, automatic exposure |
| Policy RGB-D/mask | 424x240, exact stride-2 without interpolation | 424x240, identity/no resampling |
| Policy cadence | 20 Hz | 20 Hz from fresh 60 Hz RGB-D input |
| Reviewed visible-mask coverage | at least 95% | at least 95% while target is visible in the commissioned catch workspace |
| Longest visible invalid run | at most 2 policy ticks | at most 1 policy tick after entry admission |
| Reviewed absent false-positive rate | at most 5% | at most 1%, including pre-entry and post-exit frames |
| Mask quality | Existing five-case sparse-GT IoU/recall/precision/contamination gates | Human-reviewed masks for every throw plus labelled sparse masks across blur, scale, shape, and hand proximity |
| Entry admission | Existing text/seed contract | Correct text-grounded instance within 50 ms of the first admissible visible frame |
| Point-cloud coverage | at least 95% of reviewed task-valid visible ticks; every valid output has 128 points | at least 95% of visible ticks inside the calibrated and reachable catch workspace; every valid output has 128 points |
| Coordinate frame | `robot_base` | `robot_base` using a 424x240@60 profile validation for serial `342222071785` |

The thrown-object point-cloud denominator deliberately excludes visible flight
outside the commissioned/reachable catch workspace. Those frames must still
have correct 2-D masks, but publishing unreachable or background-depth 3-D
points is a failure, not a way to improve coverage.

## Current evidence snapshot

| Task | Evidence | Current result | Status |
|---|---|---|---|
| Tabletop mask | Current frozen provider, five real-RGB cases, exact 20 Hz replay at `/tmp/guarded-v2-dual-task-tabletop-current-20260813` | 2/5 cases pass. Reviewed coverage: fast entry 89.7%, rolling green 98.7%, rolling red 96.2%, static hand occlusion 91.7%, heavy occlusion 94.0%. | FAIL |
| Tabletop point cloud | `/tmp/guarded-v2-saved-rgbd-final-20260812-14.json`, serial `337322072188`, calibration `eye-to-hand-b722bce10485c8a3` | 60/60 fresh masks and 60/60 full 128-point outputs; provider+projector p95 24.74 ms. | PASS for this saved RGB-D case, not a replacement for the five-case mask gate |
| Thrown-object mask | `data/perception_corpus/beanbag_60hz_validation_20260812/REPORT.md` | 90/90 reviewed visible tracked masks, 29/29 post-exit empty masks, SAM2 p95 16.10 ms. Automatic grounding and full end-to-end entry latency still need formal replay gates. | PARTIAL |
| Thrown-object point cloud | Seven previously reviewed throws | At the previously tested 0.25--1.50 m diagnostic range, 102/112 current masks produce full point clouds. Runtime is now 0.25--1.65 m to cover the SHA-bound V57 `alpha_0_5` target plus object extent; the old trajectories are outside that target volume, so this remains diagnostic rather than catch-workspace acceptance. | FAIL |
| Serial `342222071785` extrinsic | Native-profile reuse report `fr3-d435-342222071785-424x240-60hz-physical-reuse-20260813-reused-extrinsic-holdout-v3.json` | Unchanged `T_base_camera`, 7 holdouts, translation p95 4.022 mm and rotation p95 0.938 deg; native 424x240@60 static/runtime audit passes. | HOLDOUT PASS; known-base and catch-workspace PCD pending |

## Remaining work before overall acceptance

1. Make the current tabletop provider pass all five formal cases without
   regressing empty-frame fail-closed behavior or the 20 Hz deadline.
2. Add automatic text-entry and sparse-GT evaluation to the recorded 60 Hz
   patterned and triangular beanbag throws; manual track initialization alone
   is not end-to-end acceptance.
3. Complete the independently known base-point check for serial
   `342222071785`. Native 424x240@60 holdouts, runtime wiring, and
   multi-distance aligned-depth evidence are already complete.
4. Record new throws through the SHA-bound V57 `alpha_0_5` target-center box
   in `robot_base`, then run the formal mask/point-cloud verifier. Do not widen
   `z_max` merely to count background or flight outside that box.
5. Freeze separate tabletop and thrown-object runtime configs. The thrown
   profile uses camera depth `0.25..1.65 m`; catch admissibility is the separate
   bounded V57 `alpha_0_5` `robot_base` target volume. Keep
   `publish_zmq: false` and robot motion disabled until the corresponding
   profile has passed every gate above.

## Shared task launchers

Both tasks use the same provider, Grounding/SAM2 implementation, point-cloud
projector, policy adapter, and V94 deployment entry point. The small task YAML
overlays own only camera identity/profile, calibration, depth/workspace limits,
default prompt, and the robot-execution commissioning latch. Each invocation
materializes a content-addressed full config under
`dexgrasp/runs/task_configs/`; callers cannot override its policy resolution,
mask mode, or point-cloud config on the command line.

Inspect the two resolved contracts without opening hardware:

```bash
./perception/scripts/run_tabletop_task.sh config
./perception/scripts/run_thrown_object_task.sh config
```

Run the thrown-object camera-only validation and save the rendered mask/point
cloud video (Franka and RH56 remain unopened):

```bash
RUN_ID="thrown-mask-$(date +%Y%m%d-%H%M%S)"
./perception/scripts/run_thrown_object_task.sh perception \
  --run-id "$RUN_ID" \
  --object-text "small patterned beanbag toy" \
  --duration 30 \
  --record-video "/home/qiaoguanren/code/franka/dexgrasp/runs/${RUN_ID}.mp4"
```

The tabletop deployment wrapper accepts the same arguments as
`python -m sim2real.deploy`, while always supplying its task-owned config and
`424x240` policy input. The thrown-object wrapper deliberately rejects
`deploy --execute` while its exact `424x240@60` robot-frame calibration remains
provisional; perception-only validation is available now.
