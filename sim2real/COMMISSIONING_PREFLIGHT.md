# V94 C2 commissioning preflight

`commissioning_preflight.py` is a pure offline evidence gate.  It does not
import a camera/Franka/RH56 adapter, open a device, create a
`MotionAuthorization`, or expose an execute/force option.  Its only successful
outcome is eligibility to present the exact run to a separate, trusted operator
authorization boundary:

```text
python3 -m sim2real.deployment.preflight SHADOW.npz \
  --config sim2real/v94_deploy_config.json \
  --evidence COMMISSIONING_EVIDENCE.json
```

The command always emits one JSON object.  Exit code `0` means offline C2
preflight `PASS`; exit code `2` means `FAIL`.  Both outcomes keep
`arming_state="DISARMED"`, `physical_motion_authorized=false`,
`hardware_writes=false`, and `robot_command_writes=false`.  There is no CLI
threshold override.  A PASS is neither C3/task readiness nor permission to
move hardware.

## Evidence manifest

The manifest is strict JSON with this top-level shape:

```json
{
  "schema_version": 1,
  "kind": "v94_c2_commissioning_evidence",
  "run_id": "unique-commissioning-run-id",
  "bindings": {
    "shadow_npz_sha256": "64 lowercase hex characters",
    "bundle_sha256": "64 lowercase hex characters",
    "deploy_config_sha256": "64 lowercase hex characters",
    "commissioning_profile_sha256": "64 lowercase hex characters"
  },
  "items": {
    "physical_deadman_acceptance": {
      "result": "PASS",
      "artifact": "relative/or/absolute/report-path.json",
      "sha256": "sha256 of that regular file",
      "reviewed_by": "non-empty reviewer identity"
    }
  },
  "metrics": {
    "franka_control_loop_rate_hz": 1000.0,
    "rh56_sustained_command_rate_hz": 60.0,
    "dual_device_sustained_ack_rate_hz": 60.0,
    "dual_device_max_interaction_gap_s": 0.03333333333333333,
    "policy_command_watchdog_timeout_s": 0.05
  }
}
```

Every item listed in `REQUIRED_EVIDENCE_ITEMS` in
`commissioning_preflight.py` is mandatory.  They cover installed hardware and
payload identity, reset/workspace/tool collision, physical deadman and E-stop,
persistent single-owner 1 kHz FCI operation and verified stop behavior, RH56
single ownership/full range/60 Hz readback/disable behavior, exact dual-device
sequence acknowledgement, and camera/time/mask/scene commissioning.  An item
passes only when its result is exactly `PASS`, its reviewer is non-empty, and
its artifact is a non-empty regular file with the declared SHA-256.

The config/profile are independent gates, not substitutes for artifacts.  C2
requires an explicit `c2_bounded_closed_loop_commissioned` profile, training
reset agreement, a commissioned Franka rate covering the policy's nominal
rate, seven positive acceleration/jerk/tracking-error limits, collision model
acceptance, unrestricted six-axis RH56 commissioning, and full thumb register
range `0..1000`.  The shadow NPZ independently has to pass strict freshness,
rate/frame-reuse, no-write, reset, and action replay checks.  Missing or stale
hash-bound evidence always produces named machine-readable blockers.

`v94_deploy_config.json` deliberately points to the dedicated
`fr3_rh56_v94_commissioning.json` profile.  That profile records the V94
training reset and the matching read-only shadow artifact, so a generic
cable-friendly default pose is not misreported as a live reset failure.  It is
still `commissioning_locked`: motion authorization, reset/path collision,
0.18 rad/s policy velocity coverage, acceleration/jerk/tracking limits, the
installed collision model, RH56 full range, and all physical interlock/control
evidence remain false or absent.  The generic V7 profile is intentionally not
modified.
