# Diagnostics

This directory owns offline-only tools: checkpoint audits, action/observation
comparisons, replay analysis, calibration reports, and export utilities.

Available tools include:

- `analyze_v94_pointcloud_ab`;
- `audit_thrown_v60_candidate` and `audit_thrown_v61_bundle`;
- `audit_v57_thrown_alignment`;
- `compare_policy_io` and `compare_v94_action_trends`;
- `export_inference_checkpoint` and `franka_shaper_demo`;
- `export_policy_io_action_replay`;
- `replay_v94`;
- `audit_v94_preview`;
- `build_tabletop_intercept_replays`.

Runtime code must not import diagnostics. Run tools through their canonical
module path, for example:

```bash
.venv/bin/python -m sim2real.diagnostics.compare_policy_io --help
```
