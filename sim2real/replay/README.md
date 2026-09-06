# Transactional action replay

This package separates the replay contract from payload decoding and runtime
state:

- `models.py`: immutable replay/config/result contracts and summaries;
- `config.py`: strict planner metadata validation;
- `loading.py`: bounded JSON/CSV/NumPy/ZIP decoding without extraction;
- `policy.py`: transactional replay and visual-intercept state machine.

Use `sim2real.action_replay` as the stable compatibility import. Runtime code
must not bypass the loader or proposal commit/discard boundary.
