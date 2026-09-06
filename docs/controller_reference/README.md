# Controller references

This directory contains controller contracts and reference implementations,
not the live runtime owner.

- root files: current V258 reference configuration and example implementations;
- `contracts/`: simulator/control contracts consumed by maintained code;
- `legacy/`: older commissioned controller handoffs retained for compatibility.

Runtime code must pin the exact referenced configuration bytes before hardware
access.
