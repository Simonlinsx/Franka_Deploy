# Installed FR3 + V7 + RH56 collision diagnostics

These tools are offline. They never import `pylibfranka`, open the RH56 serial
port, or authorize motion. The native geometry runtime on this workstation is
ROS Humble Python 3.10 (`/usr/bin/python3`), because that is where Pinocchio
3.9 and HPP-FCL 2.4 are installed. It is intentionally separate from the
Python 3.8/3.9 AnyDexGrasp inference environments and does not use
MinkowskiEngine.

The backend checks the official FR3 collision meshes, the exact V7 STL (with
its millimetre-to-metre scale), and all 13 full-resolution RH56 link meshes.
Installed-return filtering separately uses the eight official FR3 visual DAE
shells, because the coarse collision STLs do not cover the complete white
outer housing. Exact point-cube/mesh intersections seed a second bounded
self-return stage: a point must simultaneously lie inside an HPP-FCL
point-to-mesh cap, match the normalized-sRGB palette of the same installed
component, and connect to a seed through 10 mm 3-D graph edges with at most a
40 mm geodesic. The distance cap is
`min(20 mm, object-alignment maximum + 2 mm)`. Every visual-mesh path,
parameter, index/distance-array digest, and SHA-256 is recorded in evidence.
The fixed adapter/link7-link8 and Link111/adapter mounting interfaces are
explicit exclusions. Positive triangle-mesh clearances and collision states
come from HPP-FCL. Captured scene/object samples are conservative cubes, but
remain non-authoritative because a single D435 frame cannot establish that
occluded space is empty.

## 0. Recommended fresh-evidence entry point

`prepare_fresh_installed_air_audit.sh` is the single camera-only/offline entry
point. It runs deterministic planning first, makes a bootstrap D435
capture/filter pass, and precomputes the eight scene-independent FR3/V7/RH56
mesh checks. It then captures and filters the scene a second time; that final
capture starts the unchanged 120-second clock, and only the ten scene/object
checks plus normal schema-v2 composition run inside it. It never imports a
Franka/RH56 driver and never moves either device.
The seven q values must be copied from a fresh read-only stationary Franka
state; they are metadata at capture time and the executor independently reads
and compares live q again. `capture_q_source=cli_asserted` is deliberately a
weak operator assertion, not an automatic FCI measurement: the full generator
requires and hash-binds that exact label and rejects any field that claims
otherwise. No `pylibfranka` module is mixed into the D435 process.

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

RUN=runs/candidate51_fresh_UTC_UNIQUE_NAME
PROFILE=configs/fr3_rh56_v7_candidate51_commissioned.json

./scripts/prepare_fresh_installed_air_audit.sh \
  --config "$PROFILE" \
  --snapshot runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz \
  --candidate-index 51 \
  --capture-q-rad Q1 Q2 Q3 Q4 Q5 Q6 Q7 \
  --confirm-stationary-q CURRENT_Q_READ_ONLY_AND_STATIONARY \
  --output-dir "$RUN"
```

Append `--dry-run` to validate the bindings and print all seven child commands;
that mode creates no directory, opens no D435, and returns 3 when commissioning
would keep the audit locked.

The intermediate `static_collision_cache.json` is published atomically with
no-replace semantics, fsynced, and mode `0444`. It is accepted only when the
native backend identity, config/snapshot/joint-plan hashes, V7 transform and
mesh, complete q path, all 13 RH56 meshes, FK waypoints, and dense feedback
tubes match exactly. Scene/object points and timestamps are explicitly absent
from that binding. The final audit identifies its backend as the cache/fresh
combiner—not as one monolithic native run—and every cached check embeds the
cache path, file SHA-256 and canonical payload SHA-256. Every dynamic check is
marked as evaluated against the final fresh scene/object point unions. The
same original policy evaluator still requires all 18 observations; a missing,
changed, overlapping or reordered half fails closed.

Offline timing on 2026-07-21 used the real candidate-51 filtered D435 scene
(56,068 retained points), 214 canonical FR3 path samples, and full-resolution
RH56 meshes. The final-code static eight-check cache took 278.59 s, entirely
before the final capture. In a separate process, exact cache validation plus
all ten point-union checks took 15.94 s; deterministic query construction plus
that merge took 21.69 s, and total app setup was 23.46 s. A separate complete
stale schema-v2 construction/validation/write path took 18.00 s; even the
deliberately conservative sum of both whole-app timings is 41.47 s, leaving
78.53 s inside the unchanged 120 s window. This was a timing replay only: the
old scene remained stale, the command returned 3, and no PASS artifact or
motion authorization was produced. A production run still uses its actual
capture timestamp and the post-audit expiry check.

Use a new run directory every time; the workflow has no overwrite option. The
final line reports the scene-expiry Unix time and remaining seconds. If the
profile still lacks q6/exact six-axis commissioning, planning, capture and
filtering are retained as diagnostic evidence, but the workflow exits 3 with
`AUDIT LOCKED` and writes no passing audit. After commissioning, that old
capture is expired and must not be reused.
The `PROFILE` in this command must be the Stage-2 coupled-PASS profile created
by `materialize-config-update` and accepted by `verify-applied
--require-coupled`; the original locked commissioning profile intentionally
returns exit 3.

The two clouds have deliberately different authority:

| Input | Role | What it cannot prove |
| --- | --- | --- |
| official snapshot `object_points` | binds the AnyDex candidate, object pose and RH56 target | complete/occluded object geometry |
| fresh filtered `scene_points` | observed environment obstacles along the audited path | empty unseen camera space or a later unchanged scene |

Before target returns are removed from the fresh scene, the filter requires
saved-object-to-live alignment (median at most 8 mm, p95 at most 15 mm, and at
least 80% coverage within 15 mm). Failure means the object/camera/calibration
changed: rerun the official perception snapshot. Do not enlarge the thresholds
or ICP-shift the old grasp pose. A passing point result stays conditional on
the runtime workspace-clear confirmation.

ROS Python 3.10 on this host has no system `xlrd`. The full-audit wrappers first
try a system import, then add only the reviewed pure-Python cache
`/home/qiaoguanren/anaconda3/pkgs/xlrd-2.0.1-pyhd3eb1b0_0/site-packages`.
Set `DEXGRASP_XLRD_SITE` to another isolated `xlrd` site if needed. No other
Conda packages are added; the official mapping workbook paths and SHA-256
bindings remain part of the audit.

## 1. Filter installed-tool and bound-object returns

The recommended entry point above performs these commands. For diagnostic
replay, use a freshly captured installed scene and the exact official AnyDex
snapshot whose object cloud is being tested:

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

./scripts/filter_installed_scene.sh \
  --live-scene runs/live_scene_installed_FRESH.npz \
  --snapshot runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz \
  --output runs/live_scene_installed_filtered_FRESH.npz
```

The default RH56 open FK is the exact official actuator-to-URDF result for
`[1000]*6`; its generator and both calibration workbooks are hash-bound in the
sidecar evidence. The output contains original point indices for every kept,
installed-return, and object-return point. Filtering is replayable, but its
evidence always says `motion_authorized=false` and
`authoritative_for_unseen_camera_space=false`.

The reviewed pipeline keeps `--installed-inflation-margin-m` at 2 mm. It does
not cure residual returns by widening that unconditional deletion band. The
bounded second stage above is replayable and includes adversarial tests proving
that a differently coloured nearby point or a same-colour disconnected point
is retained. It still cannot promote a scene check to authoritative or become
execution-time freshness proof.

The formal air profile fixes the observed-scene clearance margin at 2 mm. It
is not a CLI relaxation: voxel extent, continuous joint-interval bounds, and
the runtime tracking tube are accounted for separately. A fresh result at or
below 2 mm remains locked. Loaded/contact audits retain 5 mm and can never use
the conditional air exception.

RH56 internal self-collision uses a separate, explicit zero-penetration policy:
the signed distance must remain strictly positive over every commanded interval
and the complete all-six arrival-feedback envelope. All non-adjacent link pairs
remain checked; no RH56 pair is excluded. This local reviewed policy does not
change the 2 mm FR3/adapter/scene margin or the 5 mm loaded-scene margin. Static
cache bindings, audit JSON and the executor all bind the separate `hand_self=0`
value, so evidence made under the former shared 2 mm policy is rejected.

## 2. Replay an exact joint-waypoint path

Supply the reviewed current/transit/default/pregrasp/final joint waypoints.
For an air grasp, the final EEF pose must remain retreated from the candidate
contact pose and the final hand/object check must stay clear:

```bash
./scripts/diagnose_installed_collision.sh \
  --snapshot runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz \
  --filtered-scene runs/live_scene_installed_filtered_candidate51_20260718.npz \
  --candidate-index 51 \
  --mode air_grasp \
  --current-q Q1 Q2 Q3 Q4 Q5 Q6 Q7 \
  --default-transit-q Q1 Q2 Q3 Q4 Q5 Q6 Q7 \
  --default-q Q1 Q2 Q3 Q4 Q5 Q6 Q7 \
  --pregrasp-q Q1 Q2 Q3 Q4 Q5 Q6 Q7 \
  --grasp-q Q1 Q2 Q3 Q4 Q5 Q6 Q7 \
  --output runs/candidate51_air_installed_collision_diagnostic.json
```

Omitting `--current-q` replays from the scene's captured q; it does not make
that old q current. The checked interpolation is piecewise-linear joint space,
not Cartesian interpolation. Every reported minimum deducts a conservative
continuous-interval bound and the configured `0.002 rad` joint tracking tube.

The JSON distinguishes geometry from authority:

- pure mesh checks can be authoritative;
- scene/object checks report signed clearances and pairs but remain
  non-authoritative for unseen camera space;
- `motion_authorized` is always false;
- this diagnostic JSON is not the schema-v2 installed-tool audit accepted by
  the executor.

For a quick offline recheck of selected scopes, repeat `--check-id`, for example
`--check-id fr3_scene_path --check-id rh56_open_scene_path`. Omitting it still
evaluates the complete normal set; subset diagnostics never authorize motion.

## 3. Generate the strict schema-v2 conditional air audit

The generator accepts the deterministic joint-plan manifest, not copied joint
vectors. It verifies the manifest checksum, bound snapshot/config/FR3 URDF,
candidate and hand targets, explicit `start -> default -> pregrasp -> final_air`
path, air poses/distances, exact FK residual result, joint limits, and bare-FR3
self-collision scope. The filtered scene must be no older than 120 seconds.

```bash
./scripts/generate_installed_air_audit.sh \
  --snapshot runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz \
  --filtered-scene runs/live_scene_installed_filtered_FRESH.npz \
  --candidate-index 51 \
  --joint-plan runs/candidate51_installed_air_joint_plan.json \
  --output runs/candidate51_installed_air_audit_v2.json
```

The generator refuses candidate 51 before loading HPP-FCL unless the bound
profile says six-axis coupled closure was commissioned, target q6=646 lies
inside the evidence-backed validated range, and the complete candidate vector
`[0,358,799,911,922,646]` appears verbatim in
`commissioned_air_closure_targets`. A prior arbitrary coupled-closure pass is
not sufficient. Do not edit the boolean, range, or exact-target list by hand;
the commissioning evidence proposal owns those three profile changes.

A passing air artifact is still `motion_authorized=false` and has
`pass_kind=conditional_air`. Captured points and the installed-return filter
remain non-authoritative. The executor additionally requires the exact
`FR3_RH56_WORKSPACE_CLEAR` runtime token, meaning the workspace stayed clear,
the scene did not change, and nobody entered unseen camera space. Loaded mode
cannot consume this condition.

## 4. Required two-capture workflow

After q6 commissioning, use the same bounded process twice:

1. Capture/filter/audit at the current bound q, execute only the audited
   prefix with `air-grasp --trajectory-mode audited-joint --stop-after-default`,
   then verify stop/disable.
2. At the configured cable-friendly default q, capture a new live scene,
   regenerate the object-return filter and deterministic joint plan, generate
   a new audit within 120 seconds, then run the full unloaded air sequence.

The executor replays the audit immediately before motion, compares live q to
the bound `current` waypoint, rechecks scene age, and requires the exact
workspace-clear confirmation. RH56 contact status during the air close is a
fault; there is no lift. Ctrl-C is an emergency request, not the only safety
layer. The operator must retain Franka stop and immediate 24 V cutoff.

## Current candidate-51 offline result

The exact short-direct replay on 2026-07-18 found all five pure mesh checks clear
after the continuous/tracking bound: FR3 self 208.3 mm, V7/FR3 55.1 mm, open
RH56/FR3 57.7 mm, closed RH56/FR3 58.0 mm, and closed RH56/V7 40.4 mm. The air
final state was also clear in the observed data: closed RH56/scene 44.6 mm and
all-link closed RH56/object 40.5 mm.

The 2026-07-21 installed capture replayed 58,415 voxel representatives. The
bounded filter removed 2,053 installed returns and 729 bound-object returns,
keeping 55,633. It improved the continuous FR3/scene minimum from a negative
overlap to `+2.453 mm`, without changing the formal 2 mm air margin. The replay
nevertheless remains **LOCKED**: open RH56/scene is `-17.107 mm` after the
continuous motion bound, with `Link111/scene` and `Link44/scene` pairs. The
decisive retained dark return is only `20.429 mm` from capture-state Link111,
but that is 0.429 mm outside the hard 20 mm self-filter cap; the cap was not
expanded to manufacture a pass. Artifacts:

- `runs/live_scene_installed_filtered_candidate51_selfguard_v6_20260721.npz`
- `runs/candidate51_installed_air_collision_diagnostic_selfguard_v6_20260721.json`

This capture is expired and the single-view unknown space remains
non-authoritative. A new capture or improved calibration/scene geometry is
required; these regression files cannot authorize motion.

Final execution remains blocked until q6 commissioning updates the profile,
the cable-friendly default manifest is regenerated against that exact profile,
and both fresh-scene audit stages pass. The current PLA V7 limits the work to
unloaded, no-contact, no-lift air tests.
