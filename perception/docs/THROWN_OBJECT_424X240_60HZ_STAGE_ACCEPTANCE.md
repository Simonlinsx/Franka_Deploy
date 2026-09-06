# Thrown-object 424x240@60 stage acceptance

Date: 2026-08-13  
Camera: Intel RealSense D435 serial `342222071785`  
Policy RGB-D: native `424x240`  
Policy cadence: `20 Hz` from a `60 Hz` camera  
Robot access during the recorded acceptance: none

## Decision

The automatic text-grounding and current-frame 2-D mask path is accepted for
this stage.  The current policy-input implementation also supplies 128 points
on every reviewed visible tick by using fresh measured depth when available
and a bounded, explicitly labelled motion-compensated cloud only during short
D435 depth dropouts.  Absolute `robot_base` point-cloud accuracy is **not**
accepted yet.  Franka and RH56 execution therefore remain fail-closed in the
thrown-object task profile.

## Current frozen phase snapshot

The current software/replay snapshot supersedes the older 112-frame depth
envelope diagnostic later in this document.  The older section is retained as
history explaining why the bounded depth-dropout path was added.

| Item | Current result |
|---|---:|
| Resolved config | `dexgrasp/runs/task_configs/thrown_object-c8d70dd28adc44f0.yaml` |
| Config SHA-256 | `c8d70dd28adc44f0118b85509317c255da98f13a88bc652af220d90c826b59ac` |
| Formal replay summary | `data/perception_corpus/v57-triangular-depthfast-final-20260813-201947_thrown_semantic_motionpcd_20260813_v17_final/summary.json` |
| Summary SHA-256 | `2e26917e45d39eb6444c62fd8677f29cac135a1ddd3bc94cc8416081f3767b3a` |
| Reviewed visible masks | `120/120 = 100%` |
| Reviewed absent false positives | `0/20 = 0%` |
| Mask IoU / recall / precision p05 | `0.9937 / 0.9958 / 0.9956` |
| Fresh measured 128-point ticks | `109/120 = 90.83%` |
| Bounded motion-compensated ticks | `11/120 = 9.17%` |
| Usable 128-point policy inputs | `120/120 = 100%` |
| Longest unusable run | `0` ticks |
| Provider plus projector p95 / max | `38.40 / 47.41 ms` |
| Evaluated ticks over 50 ms | `0%` |
| Hardware opened / robot motion | `false / false` |

The bounded fallback never relabels a predicted cloud as fresh.  It translates
only the most recent measured cloud to the exact-current semantic-mask centre,
never compounds predictions, and expires after `0.25 s`, five policy ticks, or
an image speed above `2400 px/s`.  The guarded identity/recovery authority
remains fail-closed and cannot be advanced by this policy-only fallback.

The native-profile reused-extrinsic holdout is also frozen at
`fr3-d435-342222071785-424x240-60hz-physical-reuse-20260813-reused-extrinsic-holdout-v3.json`
(`d2afee61...2646`): seven solver-independent poses, translation p95
`4.022 mm`, rotation p95 `0.938 deg`, translation span `88.3 mm`, and rotation
span `21.19 deg`.  The unchanged calibration bytes remain
`3920aff7...8409`.  A fresh static V57/config audit is saved at
`dexgrasp/runs/thrown_v57_alignment_audit_20260813-231821.json`
(`80812041...130b`) and passes every static alpha=0.5 check.

This is an accepted **camera-only policy-input implementation**, not an
authorization for robot execution.  The physical RGB-D reports use a
robot-kinematic reference derived from the same calibration session and are
therefore not an independently surveyed base-frame truth source.  They also do
not span the full far edge (`1.505 m` camera depth) of the alpha=0.5 target
volume.  One independently known `robot_base` point/fixture and full catch-box
depth/point-cloud closure are still mandatory before changing the task profile
from `provisional` to `accepted`.

## Throw-synchronized rollout boundary

The thrown task now owns a camera-only `object_motion_or_entry` rollout
trigger.  Checkpoint weights and the D435/SAM/tracker services are constructed
first, but policy history and policy inference do not begin until the trigger.
The operator sees three exact milestones: `Throw trigger START`, `Throw trigger
ARMED`, and `Throw trigger DETECTED`.  Only after `DETECTED` may the shared
runtime construct the active Franka and RH56 rollout owners.  The earlier
automatic reset remains a separate, completed hardware transaction.

Two deliberately small trigger paths cover the recorded task:

- three exact empty publications arm entry detection; the first subsequent
  exact object mask triggers;
- three locally stable object masks (centroid speed at most `80 px/s`) arm
  release detection; one locally area-consistent step of at least `6 px` and
  `120 px/s` triggers.

The release threshold is below the reviewed first-flight values
(`140--164 px/s`) and above the stable-tracking median (`78 px/s`).  A trigger
timeout, stale/non-increasing frame, malformed mask, stop request, or invalid
area transition fails before either rollout owner is opened.  The runtime
audit stores the exact trigger frame, camera timestamp, mask kind, measured
speed/displacement, and `trigger_to_first_action_s`; `Rollout START` remains
the exact boundary immediately before the first command is staged.

This trigger has passed software timing/order tests, including proof that the
Franka backend and RH56 owner do not exist at detection time.  It is not yet a
robot-execution acceptance: the task profile remains `provisional` and the
independent base-frame/full catch-workspace physical gates above still apply.

The exact camera-only operator test is exposed through the shared task
launcher with `--test-rollout-trigger`.  It performs the normal automatic
grounding/preflight, records the production mask/point-cloud overlay if
requested, exits immediately after `DETECTED`, and reports zero robot access
and zero policy inference.  It is a trigger diagnostic, not a production
acceptance result.

This distinction is deliberate:

- hot YOLO-World discovery, exact-frame SAM2 initialization, subsequent SAM2
  masks, and mask-plus-projector compute latency passed on all seven recorded
  throws;
- the D435 returned no object depth at all on ten reviewed visible policy
  ticks in the older diagnostic.  Increasing the diagnostic depth ceiling
  could not recover a pixel that the sensor did not measure; the current
  bounded fallback covers those short dropouts without claiming freshness;
- native-profile independent holdout closure has passed, while absolute
  `robot_base` accuracy still needs an independently known physical point and
  full catch-workspace coverage.

## Historical pre-fallback runtime contract

| Item | Value |
|---|---|
| Resolved task config | `dexgrasp/runs/task_configs/thrown_object-040629a221b45dff.yaml` |
| Config SHA-256 | `040629a221b45dfff508c1638650b1231393cbf42a6c84e18f896cf2943314ee` |
| V57 simulation task contract | `perception/v57_real_test_reset_and_throw_ranges.yaml` (`alpha_0_5` only) |
| V57 task contract SHA-256 | `7d0646b1a1592895aeb0a7f591d65d771eb6b67cf14154d4aab1b54b0609f4c4` |
| Profile calibration | `perception/configs/calibrations/fr3_d435_342222071785_eye_to_hand_424x240_60hz_reuse_v1.yaml` |
| Calibration SHA-256 | `3920aff7967b5a7c55f69631af54a1400a3196b5ba186a464228090c8c6f8409` |
| Profile evidence | `perception/configs/calibrations/fr3_d435_342222071785_424x240_60hz_profile_evidence_v1.yaml` |
| Profile evidence SHA-256 | `21efd5faccae671fc7e05b48bfbff1cfd141db7c21ffc805953c89dfb36e3fd4` |
| End-to-end validator SHA-256 | `ad734cda29f799fa1b78a401a8d5210851eb410eb908a9c9e7d7aa36ee314a01` |
| Point-envelope audit SHA-256 | `26f6fe926a297014a1902b957512dfcda1a86a31faf876c4296c846762fbbae7` |
| Catch-workspace recorder SHA-256 | `cf26c702dcb41d4e515f610e924a59ffb6fe1bbeb6beb79462360619bde32272` |
| Formal catch-workspace PCD verifier SHA-256 | `1a7dabf91f4445ae6b5e840f4b5d2f28ac1b7805e3bf50d200f5d70f941d21d1` |
| Static V57/camera audit | `dexgrasp/runs/v57_alpha0p5_424x240_60hz_static_alignment_20260813.json` |
| Static V57/camera audit SHA-256 | `034d7a72d9d583df96eea86b8e8dffb9c560c58f40d7d761d34e8b6566b6f046` |

The projector contract in the frozen validator is the live deployment
contract: 128 XYZ points, at least 16 source points, and a `0.055 m` robust
mask-depth deviation gate.  Prompt and SAM2 compile/model cold start occur
before the cadence measurement.

## Automatic grounding and mask results

Every track starts from text only.  No validation run supplies a manual bbox
to the provider.  YOLO-World finds a compact current-frame bbox; the bbox is
only the SAM2 prompt and is never published as the object mask.

| Object / throws | Text | Entry delay | Visible mask | Absent false positive | IoU p05 | Recall p05 | Precision p05 | Mask+projector p95 / max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Patterned beanbag / 2 | `red patterned ball` | 0 / 0 frames | 40/40 | 0/7 | 0.892 | 0.904 | 0.896 | 21.30 / 21.44 ms |
| Triangular beanbag / 3 | `red triangular object` | 0 / 0 / 0 frames | 50/50 | 0/22 | 0.890 | 0.890 | 0.921 | 20.85 / 21.22 ms |
| Small red ball / 2 | `small red ball` | 3 / 0 frames | 22/22 | 0/7 | 0.931 | 0.963 | 0.943 | 20.33 / 20.39 ms |

Aggregate reviewed evidence is 112/112 visible masks and 0/36 absent false
positives.  The longest invalid visible-mask run is zero.  Grounding is no
later than one 20 Hz tick after the reviewed first-visible frame.

The small-ball operator brought the object into view before throwing it.  Its
entry deadline is therefore based on the actual first-visible frames 303 and
516, not the later release frames 390 and 530.  The tracker then runs through
the hand-held interval, flight, and exit.

### Saved overlays and reports

- Patterned beanbag:
  `data/perception_corpus/beanbag_60hz_validation_20260812/pattern_end_to_end_20260813_v3_aligned/overlay_all.mp4`
  (`cbe07152da1aaa8661d8b2a8c3aaf19aec9a72da326733c0b5491915fd4605a7`)
- Triangular beanbag:
  `data/perception_corpus/beanbag_60hz_validation_20260812/triangle_end_to_end_20260813_v2_aligned/overlay_all.mp4`
  (`98c7bb87728793b859930ec78f8e68d4c0b7dc94dbeb2e7c09c7ce2e50104775`)
- Small ball:
  `data/perception_corpus/ab_blur_20260812/small_ball_end_to_end_20260813_v5_aligned/overlay_all.mp4`
  (`6dc263b361e6d9b2a20ec133dd6404bbe942f4166de3e6b7104a19c0c527d333`)

The corresponding `summary.json`, per-track current-frame masks, and
`frames.csv` live beside each overlay.  Summary SHA-256 values are,
respectively, `d5891f5f...baf7`, `019ff1de...12d3`, and
`e8195c05...0736`.

The patterned/triangular references are the previously reviewed 20 Hz masks.
The small-ball reference is the earlier exact-frame, box-prompted SAM2
baseline, subsampled at 20 Hz; flight masks and seven added empty exit frames
were visually reviewed.  It is an independent-initialization regression
reference, not dense human pixel ground truth.

## 128-point diagnostic

The table-task depth interval remains `0.25..1.20 m`.  The separate thrown-task
profile now uses `0.25..1.65 m`.  The V57 `alpha_0_5` target-center volume
projects to camera depth `1.147..1.505 m`; adding the largest configured object
half-extent reaches `1.547 m`, so `1.65 m` supplies bounded measurement
headroom.  Robot-facing admissibility is independently fixed to the V57
`alpha_0_5` target-center box, so the larger camera depth envelope does not
expand the task volume.

| Object | Production 1.20 m | Diagnostic 1.50 m | Diagnostic 2.00 m | Diagnostic 3.00 m |
|---|---:|---:|---:|---:|
| Patterned beanbag | 20/40 = 50.0% | 37/40 = 92.5% | 37/40 = 92.5% | 36/40 = 90.0% |
| Triangular beanbag | 20/50 = 40.0% | 46/50 = 92.0% | 46/50 = 92.0% | 47/50 = 94.0% |
| Small ball | 0/22 = 0.0% | 19/22 = 86.4% | 19/22 = 86.4% | 18/22 = 81.8% |
| Aggregate | 40/112 = 35.7% | 102/112 = 91.1% | 102/112 = 91.1% | 101/112 = 90.2% |

The prior `1.50 m` diagnostic established that merely increasing the ceiling
does not solve the remaining failures; the V57-aligned runtime now uses
`1.65 m` for geometric coverage.  On every one of the
ten remaining failures, source points inside the exact current mask were zero;
an 8-pixel mask neighbourhood also contained no motion-consistent target
depth.  The failures occur as isolated one- or two-tick D435 depth dropouts,
not as segmentation loss.

The envelope files retain per-frame status and provisional robot-base centre
records.  Their SHA-256 values are:

- patterned: `991573925ff3590ce20ad6af64c34d9b0146d03aa166735b5a4a49b40ea73ab3`;
- triangular: `aa2f32a096cc9071d17209e0c9bf9c03f1654062040419ecc2d6f7c8f4b5c087`;
- small ball: `423815bbe1eca19f7e6c54676e5a3dd774b9602695e219c2cf848d59bb4c8fbc`.

## Remaining commissioning steps

1. Seven independent native-profile holdouts have passed with unchanged
   `T_base_camera` (translation p95 `4.022 mm`, rotation p95 `0.938 deg`).
   Multiple aligned-depth stations span `0.268 m`.  The calibration board has
   therefore been removed from the EEF and is not a runtime dependency.
2. Verify one independently surveyed/metrology/touch-off point in `robot_base`.
   This does not require reinstalling the board on the EEF; a surveyed fixture
   or separately authorized recorded touch-off is acceptable.
3. Record/evaluate throws through the now pinned V57 `alpha_0_5` target-center
   workspace using `scripts/record_thrown_catch_workspace_424x240.sh`.  The
   current videos span large areas outside this volume, so full-flight depth
   coverage is not itself the correct deployment denominator.
4. If current-mask depth still drops for one or two policy ticks inside the
   catch workspace, evaluate a bounded, explicitly labelled depth-dropout
   bridge and train with the same observation contract.  Do not relabel the
   existing stale cloud as fresh, and do not add a generic hallucinated cloud.
5. The physical point-cloud, full-flight perception, correct-reset policy
   shadow, and zero-write lifecycle gates passed on 2026-08-14.  The task is
   enabled only for the first operator-supervised motion stage, capped by the
   task launcher at 20 policy ticks.  Extending that cap requires review of
   the first real-motion audit.

## Next V57 alpha=0.5 recording

This command opens only camera serial `342222071785`.  Its green overlay is
not an independently copied box: the recorder reads the exact V57
`alpha_0_5` target-center bounds from the same SHA-bound contract used by the
runtime reset and verifier.

```bash
cd /home/qiaoguanren/code/franka
THROWN_BALL_OBJECT_TEXT="small red ball" \
THROWN_BALL_DURATION_S=20 \
./perception/scripts/record_thrown_catch_workspace_424x240.sh \
  /home/qiaoguanren/code/franka/object_pcd_testdata \
  "v57-alpha0p5-catch-$(date +%Y%m%d-%H%M%S)"
```

Throw at least three times through the green volume.  The lossless RGB-D files
do not contain the overlay; it is display-only.  After the recording is
reviewed, the formal verifier requires an accepted text-grounding/mask result,
the reviewed per-track masks, and the independent physical RGB-D base-point
report.  It then checks current-frame 128-point coverage and accuracy only in
the V57 target volume expanded by the configured object half-extent.

The shared thrown deployment wrapper fails closed unless both an explicit
V57-compatible `--checkpoint` and the V57 task-specific `--profile` are
provided.  The first-motion commissioning profile additionally refuses more
than 20 supervised policy ticks.
