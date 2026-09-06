# Runtime contracts

- `__init__.py`: general observation/action data contracts, preserving the
  public `sim2real.contracts` API;
- `v94.py`: immutable V94 bundle and observation contract;
- `actions.py`: V94 Franka/RH56 action mapping and transactional proposals.

Contracts validate data and construct proposals; they do not own hardware or
task sequencing.
