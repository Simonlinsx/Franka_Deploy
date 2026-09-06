"""Hardware-boundary interfaces for the FR3 and Inspire RH56.

Importing :mod:`robot_control` itself is hardware-inert.  Concrete adapters
remain lazy and open devices only through explicitly authorized runtime calls.
"""

__all__ = ["franka", "reference", "rh56", "safety"]
