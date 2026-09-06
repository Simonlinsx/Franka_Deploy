"""Closed-loop protocol constants."""

from typing import Tuple

AUTHORIZATION_SCOPE = "v94_closed_loop_franka_rh56"
COMMAND_CONSUMERS: Tuple[str, str] = ("franka", "rh56")

__all__ = ["AUTHORIZATION_SCOPE", "COMMAND_CONSUMERS"]
