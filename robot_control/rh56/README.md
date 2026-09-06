# RH56 device boundary

- `linux_transport.py`: lazy serial transport;
- `actuator.py`: transactional actuation and verified stop;
- `watchdog.py`: single-owner command and feedback watchdog.

Importing the package never opens RH56. Hardware writes remain guarded by the
existing preflight, authorization, feedback, and stop contracts.
