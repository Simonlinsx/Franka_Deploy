# Robot control

This module is the hardware boundary for the FR3 and Inspire RH56:

- `robot_control.franka`: persistent FR3 session, lazy pylibfranka adapter,
  and supervised native-session protocol;
- `robot_control.rh56`: serial transport, transactional actuation, physical
  stop verification, and single-owner watchdog;
- `robot_control.safety`: authorization, command ledger, and fail-closed
  transaction primitives shared by both devices;
- `robot_control.reference`: pure NumPy action-mapping and command-shaping
  references used for regression, never as a hardware backend.

The implementations live inside the `franka/` and `rh56/` subpackages.
Importing these modules never opens hardware. Device access remains possible
only after the deployment runtime has completed its commissioning and
supervision checks.
