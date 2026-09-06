# Sim-to-real support configuration

This directory holds shared configuration consumed by the `sim2real` library:

- `config.json`: default camera, calibration, and observation settings;
- `v94_deploy_config.json`: V94 deployment and commissioning defaults;
- `rh56_speed600_identified_dynamics.yaml`: identified RH56 dynamics evidence.

Task-specific camera and perception profiles remain under
`perception/configs/tasks/`. Hardware execution profiles remain under
`dexgrasp/configs/`. Runtime-generated materialized configurations remain
under `data/runs/` (with the legacy `dexgrasp/runs/` compatibility path).

Code should resolve these files relative to the package or workspace root;
new machine-specific absolute paths do not belong here.
