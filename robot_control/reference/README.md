# Franka control references

This package contains pure NumPy reference implementations used to verify
policy-action mapping, interpolation, and derivative envelopes. They do not
open a robot and are not deployment backends.

Runtime hardware ownership remains in `robot_control.franka`. Reference
implementations are imported only through `robot_control.reference`.
