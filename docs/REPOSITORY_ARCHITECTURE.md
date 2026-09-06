# Repository architecture and maintenance rules

## Current boundary

The maintained robot stack consists of five cooperating domains:

```text
                         sim2real
                deployment / task runtime
              /          /   \           \
             v          v       v          v
 motion_planning   dexgrasp  perception  robot_control
    trajectories    grasps  (`dynamic_pcd`) device boundary
```

`sim2real` owns checkpoint/policy admission, task sequencing, and supervised
runtime composition. `motion_planning` owns pure kinematics and interception.
`perception` owns camera-to-mask-to-point-cloud behavior; its retained Python
package name is `dynamic_pcd`. `dexgrasp` owns
grasp and installed-hand semantics. `robot_control` owns the authorized FR3
and RH56 device boundary. New lower-layer runtime modules must not import task
orchestration from `sim2real`. Retired flat implementation imports have been
removed after all maintained callers migrated to domain packages.

## Measured maintenance risks

The September 2026 inventory found:

| Area | Evidence | Maintenance impact |
| --- | ---: | --- |
| `sim2real` root | 16 Python modules, about 4,100 lines | Public commands and compatibility facades remain; replay and closed-loop implementations now live in focused subpackages |
| object point-cloud provider | about 26,200 lines in the transactional owner | State records, runtime config, and pure mask geometry are separate; mask policy and I/O remain tightly coupled |
| online tabletop planner | about 3,000 lines after the first kinematics extraction | Phase transitions, safety, and diagnostics still change together |
| README/runbook | previously over 1,300 lines with repeated shell history | Stale commands could be mistaken for supported workflows |
| root Git state | maintained source tracked; generated/runtime/external trees ignored | Reviews and diffs now focus on maintained integration code |

These numbers are indicators, not refactoring targets by themselves.  A
smaller file is useful only when the extracted module has one stable role and
the same safety tests still protect every transaction boundary.

## Target source layout

The following is the direction for incremental moves.  Compatibility modules
may remain at their old import paths while callers migrate.

```text
sim2real/
  deployment/      checkpoint admission and stable deploy API
  tasks/           tabletop and thrown-object task composition
  runtime/         supervised scheduling, owners, and observation/action flow
  observation/     camera profiles, policy RGB-D/point-cloud, ROI, and display
  diagnostics/     saved-data audit, comparison, replay, and export tools
  commissioning/   explicit calibration, identification, and hardware probes
  policy/          NumPy inference, exact I/O evidence, and rate contracts
  contracts/       observation schemas and V94 action/runtime contracts
  replay/          validated replay models, loaders, and policy transaction
  closed_loop/     authorization, commands, dual-ACK ledger, and action mapper

motion_planning/
  kinematics.py    pure Panda/palm FK and bounded IK corrections
  tabletop.py      public moving-tabletop interception API

robot_control/
  franka/          FR3 persistent session, backend, and native protocol
  rh56/            RH56 transport, transaction, watchdog, and stop API
  safety.py        shared authorization and commit primitives
  reference/       pure NumPy mapping/interpolation regression references

perception/dynamic_pcd/
  camera/          RGB-D acquisition and timestamp contracts
  segmentation/    prompt, image, temporal, and appearance segmentation
  tracking/        temporal history and recovery state
  pointcloud/      mask/depth projection and filtering
  provider/        public facade plus small state/policy/transaction modules
  evaluation/      offline replay and acceptance metrics

dexgrasp/
  src/anydex_pipeline/  reusable grasp and installed-hand library
  apps/                 thin CLI adapters
  scripts/              thin shell wrappers only
```

## Workspace storage layout

Source modules do not own generated recordings, model checkpoints, external
Git histories, or historical release bundles:

```text
data/                  curated demos and small test fixtures
  checkpoints/         active large checkpoints
  perception_corpus/   active RGB-D and replay corpus
  runs/                active deployment evidence
third_party/           independent external source checkouts and environments
data/archives/         inactive bundles and release snapshots
docs/controller_reference/ commissioned controller contracts and legacy references
```

Maintained code uses `data/checkpoints`, `data/perception_corpus`, and
`data/runs` directly. The internal `dexgrasp/runs` compatibility link remains
while external operator tooling migrates. New historical bundles must go
directly under `data/archives/`, not the workspace root.

Project-specific external code and Python environments likewise live under
`third_party/`; compatibility links inside `dexgrasp/` and
`examples/inspire_mano_pipeline/` preserve their established tool paths.

## Naming rules

1. Prefer domain names over experiment numbers.  New code should use
   `live_observation_owner`, `supervised_runtime`, or `tabletop_planner`, not a
   new `v95_*` name.  Existing `v94` names are compatibility contracts and are
   not renamed without import/config shims.
2. Use `*_config` for validated immutable configuration, `*_state` for mutable
   episode state, `*_evidence` for observed facts, `*_proof` for a validated
   decision input, and `*_commit` for a transaction that crosses a publication
   or hardware boundary.
3. Boolean names state the positive fact: `tracking_committed`,
   `authority_eligible`, `stop_confirmed`.  Avoid generic `ok`, `flag`, and
   double negatives.
4. Include units in physical scalar names (`_m`, `_s`, `_hz`, `_rad`, `_px`,
   `_g`, `_ma`).  Use explicit frame suffixes such as `_base`, `_camera`, and
   `_policy_palm` for vectors/transforms.
5. CLI modules parse/validate and call a library function.  They should not
   contain the control implementation.  Shell scripts resolve the workspace
   and delegate; they must not duplicate safety policy.
6. Private experiment helpers start with `_`; public reusable objects receive
   a docstring and are exported deliberately through the package API.

## Incremental refactoring sequence

### Phase 0 — repository hygiene

- Ignore environments, caches, recordings, checkpoints, and external trees.
- Establish this root navigation and one canonical command per task.
- In a separate reviewed Git operation, remove `.venv/` from the index and add
  the maintained source/config/test files.  Do not delete the local environment.

### Phase 1 — provider state extraction

- Move the provider's evidence/proof dataclasses into
  `dynamic_pcd.provider.state` with compatibility re-exports.
- Pure mask geometry and provider-local config validation now live in
  `dynamic_pcd.provider.mask_geometry` and `dynamic_pcd.provider.config`.
- Split proof validation from camera/tracker mutation in later guarded steps.
- Keep `ObjectPCDProvider` as the public facade; require byte-identical replay
  and the saved-provenance suite before each move.

### Phase 2 — tabletop controller extraction

- Pure Panda/palm kinematics now live in `motion_planning.kinematics`.
- The transactional tabletop controller now lives at
  `motion_planning.online_tabletop`; prediction and phase extraction remain
  future internal refactors behind `motion_planning.tabletop`.
- Keep one transactional controller that owns pending/commit/discard state.
- Add transition-table tests so `approach -> closure -> lift -> hold` is
  auditable independently from hardware I/O.

### Phase 3 — runtime and CLI separation

- Keep only stable public commands at the package root; domain implementation
  imports use the owning subpackage.
- Bounded execution, policy-tick, live-observation, and supervised composition
  implementations live under `sim2real.runtime`; historical flat aliases have
  been retired.
- Offline replay, audit, comparison, and export implementations live under
  `sim2real.diagnostics`.
- Calibration, identification, camera timing, and RH56 physical probe tools
  live under `sim2real.commissioning`.
- Remove lower-layer imports of `sim2real` by moving shared projection and task
  contract types to the layer that owns them.
- FR3 session/backend/native-session and RH56 transport/actuator/watchdog
  implementations live under `robot_control.franka` and
  `robot_control.rh56`.
- `sim2real.policy` and `sim2real.contracts` own policy support and stable
  schemas; former flat support modules have been retired.

## Required refactoring gates

For every runtime refactor:

- no hardware access during tests;
- unchanged configuration defaults and content-addressed provenance unless the
  source manifest is intentionally refreshed;
- exact mask/action/target equality for compatibility paths;
- fail-closed behavior for missing, stale, tampered, or out-of-envelope input;
- focused tests first, then the full owning package suite;
- one semantic change per patch—file movement and behavior changes do not share
  a review boundary.

This sequence keeps the working deployment usable while replacing the current
large modules with testable components over time.
