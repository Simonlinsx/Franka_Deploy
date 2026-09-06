# Supervised runtime

This package owns live observation/action scheduling and transactional runtime
composition after deployment admission:

- `bounded_c2_orchestrator`: sealed admission and bounded-run lifecycle;
- `bounded_c2_runtime`: dual-device transaction and stop coordination;
- `v94_policy_tick_source`: observation/history/inference proposals;
- `v94_live_observation_owner`: single-source camera, arm, and hand snapshots;
- `supervised_v94_runtime`: production composition of the above components.

The package does not own task selection, calibration tools, offline analysis,
or reusable device transports. Import runtime primitives from this package;
the retired flat aliases are no longer part of the maintained API.
