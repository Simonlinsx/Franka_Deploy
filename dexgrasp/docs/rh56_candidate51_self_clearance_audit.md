# Candidate 51 RH56 self-clearance audit

This note records the 2026-07-21 offline investigation of
`rh56_execution_self_final` for the 191-waypoint, no-contact RH56 path of
candidate 51.  Nothing described here imports a hardware driver, opens FCI or
the RH56 serial port, or authorizes motion.

> Policy update (2026-07-27): this file preserves the 2026-07-21 investigation
> under the former shared 2 mm robot margin. The reviewed air-grasp policy now
> treats RH56 internal self-collision separately: every non-adjacent link pair
> must remain strictly non-intersecting (`signed distance > 0`) over the command
> path and all-six feedback envelope. FR3, adapter and observed-scene clearances
> remain 2 mm. No RH56 link pair was excluded. Old artifacts without the
> separate `hand_self=0` binding are invalid and must be regenerated.

## Historical result under the former shared margin

The result was **FAIL / LOCKED** with the former strict `2 mm` robot clearance
policy.

The original diagnostic reported `Link11 / Link22` at interval `81 -> 82` as
the decisive pair, with a `-34.508164 mm` conservative lower bound.  That was
not an observed mesh collision: both endpoint meshes were clear and
`observed_pairs` was empty.  The large deduction came from applying one
directionless serial-chain radius to the complete all-six feedback tube.  It
is especially pessimistic at the official actuator mapping's discontinuous
open row: register `1000` maps to the exact open q12 row, whereas register
`999` maps to a substantially bent q12 state.

The backend now keeps the coarse bound as a fast first pass and, only for a
self-pair that the coarse bound cannot prove, subdivides the exact
component-wise q12 feedback box.  At every box centre it evaluates the two
full-resolution triangle meshes with HPP-FCL.  A leaf is certified only when

```text
exact centre distance
  - sum_j 2 * (R_first,j + R_second,j) * sin(box_half_width_j / 2)
  > 2 mm.
```

Every integer actuator register in the interval endpoint/tolerance union is
mapped through the official XLS conversion, including register `1000`.
Reaching a subdivision depth/node limit is unresolved and remains a failure.
The global margin is unchanged, and no link pair is newly excluded.

For `Link11 / Link22`, interval `81 -> 82`, this gives:

| quantity | result |
| --- | ---: |
| original coarse lower bound | `-34.508164 mm` |
| exact integer-grid minimum (1,326 official-map states) | `7.881501 mm` |
| adaptive-box minimum sampled distance | `7.885844 mm` |
| strict adaptive lower bound | `2.560695 mm` |
| adaptive nodes / maximum depth | `127 / 6` |

Thus the reported `Link11 / Link22` failure was a conservative-bound false
negative, not a collision.  Fixing it does **not** make the complete path pass.

Across all 191 discrete waypoints, the exact nominal global minimum is
`Link111 / Link51` at the fully open waypoint 0:

```text
exact nominal distance = 1.047271 mm
observed mesh collision = false
required policy margin = 2.000000 mm
```

Because the exact endpoint itself is below the policy margin, adaptive
continuous-path refinement cannot remove this blocker.  It is also present at
the mandatory fully-open start, so reshaping the interior closure trajectory
does not solve it.  A full replay with adaptive refinement attempted 2,023
interval/pairs, certified 2,021 and left two unresolved; the final conservative
minimum remained `Link111 / Link51`, and `motion_authorized` remained `false`.

Reference artifacts:

- original diagnostic:
  `runs/candidate51_installed_air_collision_diagnostic_fresh2_20260721.json`,
  SHA-256
  `8ff82b0a5bf5c8e6cafbaa5eb8bb8305cd4564fa6949de530c02553e21817027`
- adaptive replay generated during this investigation:
  `/tmp/candidate51_self_clearance_refined.json`, SHA-256
  `3871887b9561aeb18fec2f4c3a964f74823e4d8918d580ac10de75e0652581d0`
  (diagnostic only; `motion_authorized=false`)

## Link111 / Link51 exclusion audit

No reproducible official basis was found for adding this pair to an allowed
collision list.

The audited AnyDexGrasp source is commit
`c9c4a43df33e40860417c7e2dd02f5122d3b2da2` from
`https://github.com/graspnet/AnyDexGrasp.git`.  Bound source hashes are:

| source | SHA-256 |
| --- | --- |
| `urdf-five3.urdf` | `0bc26c20154bcb5daa0e13fbd05cb7abd895eee3f4289ecfb8df81fdbe17b94f` |
| `meshes/Link111.STL` | `01e66e58f8b8bf95d00f1a99b8700834c569ad6a6357af4fe8bd6ebbf5b02dda` |
| `meshes/Link51.STL` | `6ee10e6278d7771493e118696448e21c7642975af18c9e6f6dbaa01ad705b957` |

The URDF topology is:

```text
Link111 --(revolute Link5)--> Link5 --(revolute Link51)--> Link51
```

Therefore the pair is two kinematic edges apart, not a direct parent/child
pair.  The current backend excludes only the URDF's direct parent/child pairs
from RH56 self testing.  The vendored upstream tree contains no SRDF, allowed
collision matrix, semantic collision file, or pair-specific Link111/Link51
exclusion.

Upstream mesh generation also supplies no such evidence.  Its PyBullet loader
calls `loadURDF(..., useFixedBase=True)` without
`URDF_USE_SELF_COLLISION`, then transforms and merges link meshes.  The
AnyDexGrasp `ModelFreeCollisionDetectorMultifinger` transforms a pregenerated
whole-hand point cloud and tests its voxels against scene points; it does not
evaluate hand-link self collision.  The presence of `CollisionType.SELF` as a
constant is not accompanied by a self-collision implementation in that
detector.

Consequently this audit deliberately does **not**:

- exclude `Link111 / Link51` based only on two-hop topology;
- exclude every two-hop RH56 pair;
- lower the global `2 mm` clearance margin; or
- interpret an upstream pregenerated grasp mesh as self-collision approval.

No pair exclusion was adopted. The later reviewed mechanical policy changed
only the required internal hand clearance to strict non-intersection; candidate
51 still requires a new complete fresh-scene audit before it can pass any
execution gate.

## Runtime feedback-envelope condition

The refined proof covers the commanded interval plus a configured all-six
feedback envelope.  The matching runtime contract is now implemented and is
fail-closed.  After each verified numeric `ANGLE_SET` write, every subsequent
`ANGLE_ACT` sample must remain inside the inclusive component-wise hull of the
previous and current endpoint acceptance bands.  This accepts normal actuator
lag at the previous endpoint and continuous transit toward the new endpoint,
but rejects escape on either side.  An escape enters the driver's existing
fail latch and immediately requests all-six `ANGLE_SET=-1`.

The contract also binds the previously commissioned q6 reverse hysteresis
(`30` register units), the `980` open threshold, and the generic arrival
tolerance.  Its canonical JSON SHA-256 is carried through:

- the runtime driver binding;
- commissioning request/result evidence and per-sample offline replay;
- installed-tool audit policy and backend observation details;
- official-XLS q12 feedback-box hashes; and
- the execution entry gate.

Consequently an older audit or evidence file without this exact policy/hash is
not accepted by the updated executor.  Unit regressions cover normal lag,
lower/upper escape, q6 reverse hysteresis, policy tampering, missing runtime
enforcement and post-fault all-six disable.  This closes the runtime/audit
semantic gap, but it does not affect today's outcome: candidate 51 remains a
geometric FAIL because `Link111 / Link51` is below the strict margin at the
nominal fully-open endpoint.

## Offline replay

The following reproduces the diagnostic geometry only.  It does not authorize
or command hardware:

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

./scripts/diagnose_installed_collision.sh \
  --snapshot runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz \
  --filtered-scene runs/live_scene_installed_filtered_candidate51_fresh2_20260721.npz \
  --candidate-index 51 \
  --mode air_grasp \
  --current-q -0.1210973 -0.1097435 0.0728234 -1.7477672 0.0311030 1.6609629 0.8176927 \
  --default-q -0.1118436 -0.1207545 0.0739457 -1.7431009 0.0463540 1.6809169 0.8117281 \
  --pregrasp-q -0.0066058216 0.5684003987 0.4092758098 -1.5043386777 -0.9253988700 1.7623541953 1.3525 \
  --grasp-q 0.0338131395 0.5310502162 0.3400966609 -1.5574000144 -0.8907144233 1.8046512025 1.3525 \
  --max-joint-step-rad 0.005 \
  --max-q-tracking-error-rad 0.002 \
  --hand-arrival-tolerance-units 25 \
  --output /tmp/candidate51_self_clearance_replay.json
```
