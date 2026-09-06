# Motion planning

This module owns hardware-inert planning logic:

- Panda/palm kinematics and inverse-kinematics corrections;
- moving-object prediction and interception;
- tabletop approach, closure, lift, and hold planning;
- trajectory construction and validation.

It does **not** own cameras, model inference, serial/FCI communication, or
hardware authorization.  Those responsibilities belong to perception,
`sim2real`, and `robot_control` respectively.

Import pure geometry from `motion_planning.kinematics` and the tabletop planner
API from `motion_planning.tabletop`. The controller implementation lives in
`motion_planning.online_tabletop`.
