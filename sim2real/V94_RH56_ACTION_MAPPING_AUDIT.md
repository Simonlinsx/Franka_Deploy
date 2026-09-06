# V94 RH56 action mapping independent audit

Date: 2026-07-22

Result: **software/mathematical mapping PASS; installed-hand physical mapping NOT YET COMMISSIONED**.

This audit was offline-only. It did not import a hardware adapter, open the
RH56 serial port, connect to Franka, or write any register.

## Bound inputs

- `data/test_fixtures/sim2real/deploy.zip` SHA-256:
  `a2d0a1e319f2ae577016bb1762b7ccb2b48d24fdd9d229002196884988de7ee4`
- primary checkpoint SHA-256 from the verified bundle manifest:
  `e79dd574953f91a161fe0d05b2b6abb14007b6828451cdcf92337beb0cbf22df`
- `dexgrasp/configs/fr3_rh56_v94_commissioning.json` SHA-256:
  `5bb8d4e3756b6020a5d88a1a2e44a6830d09e5e9bfe2d6041fff96e93ae5e0da`
- optional real read-only shadow `_20260722_21.npz` SHA-256:
  `3590cda5b32f8f9c0b886aa6c42870538466a63916f35205d8c5b8f14ac03a8a`

The bundle verifier checked all 16 manifest-bound files before the mapping
comparison.

## Exact semantic and register order

| policy action | policy semantic axis | semantic close q (rad) | RH56 register index | manufacturer axis |
| ---: | --- | ---: | ---: | --- |
| 7 | thumb rotation | 1.25 | 5 | thumb rotation |
| 8 | thumb bending | 0.599 | 4 | thumb bending |
| 9 | index | 0.95 | 3 | index |
| 10 | middle | 0.95 | 2 | middle |
| 11 | ring | 1.05 | 1 | ring |
| 12 | little | 1.10 | 0 | little |

Therefore the policy order

```text
[thumb_rotation, thumb_bending, index, middle, ring, little]
```

is permuted into the manufacturer `ANGLE_SET(0..5)` order

```text
[little, ring, middle, index, thumb_bending, thumb_rotation].
```

Synthetic one-hot tests showed that each policy axis changes only the named
virtual joint and the corresponding manufacturer register. No cross-axis or
off-by-one permutation was found.

## Numeric mapping

For the six RH56 outputs, the transferred simulator contract and runtime use:

```text
a       = clip(action[7:13], -1, 1)
fraction = (a + 1) / 2
q_raw    = fraction * q_semantic_close
q_filter = 0.20 * q_raw + 0.80 * previous_committed_target
q_next   = previous_committed_target
           + clip(q_filter - previous_committed_target, -0.05, 0.05)
r_policy = rint(1000 * (1 - q_next / q_semantic_close))
r_device = r_policy permuted into manufacturer order
```

The hand action is an **absolute semantic closure setpoint**, not an increment:

- `-1` means fully open (`q=0`, nominal register `1000`);
- `0` means half closed (`q=0.5*q_close`, nominal register `500`);
- `+1` means fully closed (`q=q_close`, nominal register `0`).

The command sent on one tick is nevertheless stateful because it is filtered
from the previous committed target and limited to `0.05 rad/tick`. Repeated
zero action converged to half closure rather than accumulating past it. Integer
conversion uses NumPy `rint` (nearest integer, including its tie-to-even rule),
not truncation or independent per-driver rounding.

## Reference and shadow results

| source | rows | virtual target max abs error | integer register comparison |
| --- | ---: | ---: | --- |
| bundled reset/idle-open stream | 12 | `0 rad` | exact |
| bundled executed simulator closed loop | 11 | `0 rad` | exact |
| latest real read-only shadow | 599 | `0 rad` | exact |

The latest real shadow also had `0` error for clipped/executed policy action.
It used `one_step_from_measured_idle` proposals, so it proves the live action
and observation are passed through the expected one-step mapper; it does not
prove accumulated physical tracking.

The default command mapping and the observation-side `ANGLE_ACT` feedback
mapping were also checked as inverse permutations. The sampled round-trip
maximum error was `1.49e-8 rad`, below the worst one-register quantisation bound
of `0.000625 rad`.

The bundled simulator closed-loop stream's proprioception slice `54:67`
equals the previous executed action exactly on every available transition.
The local transactional protocol was exercised independently: one Franka or
RH56 ACK alone advanced neither `previous_executed_action13` nor the mapper
target; both ACKs for the same sequence were required before both states
advanced.

## Remaining physical uncertainty

The above evidence confirms the software equations and wiring, but not the
installed RH56's physical response. Current commissioning evidence says:

- profile mode is `commissioning_locked`;
- `six_axis_coupled_closure_commissioned=false`;
- `rh56_six_axis_policy_motion_commissioned=false`;
- thumb-rotation real-time range is validated only for registers `900..1000`;
- the transferred deployment document itself requires an unloaded low-speed
  check of thumb-rotation direction because firmware endpoint direction can
  differ.

Before checkpoint-driven hand motion, every manufacturer axis must therefore
receive a bounded, one-at-a-time, unloaded microprobe. Each probe must verify
command/readback identity, `ANGLE_ACT` direction, allowed travel, tracking,
current, temperature, status, and final all-six `ANGLE_SET=-1` disable. Thumb
rotation needs special attention and must not be extrapolated from its current
`900..1000` validated interval to the full `0..1000` range.

## Reproduction

```bash
.venv/bin/python -m sim2real.deployment.rh56_action_audit \
  --shadow dexgrasp/runs/v94_live_readonly_clean_dkms_power_on_reuse_guard_20260722_21.npz

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  sim2real/tests/test_audit_v94_rh56_action_mapping.py \
  sim2real/tests/test_v94_actions.py \
  sim2real/tests/test_closed_loop_core.py \
  sim2real/tests/test_rh56_transactional_actuator.py
```

Observed result: `52 passed`.
