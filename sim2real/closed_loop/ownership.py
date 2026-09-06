"""One-thread ownership guard for hardware endpoints."""

from __future__ import annotations

import threading
from typing import Optional

from .authorization import ClosedLoopProtocolError


class SingleThreadOwner:
    """Bind a hardware endpoint (notably RH56 serial) to one thread forever."""

    def __init__(self, endpoint_name: str) -> None:
        name = str(endpoint_name).strip()
        if not name:
            raise ValueError("endpoint_name must be non-empty")
        self.endpoint_name = name
        self._lock = threading.Lock()
        self._owner_thread: Optional[threading.Thread] = None

    @property
    def owner_ident(self) -> Optional[int]:
        with self._lock:
            return None if self._owner_thread is None else self._owner_thread.ident

    def claim_for_current_thread(self) -> None:
        thread = threading.current_thread()
        with self._lock:
            if self._owner_thread is None:
                self._owner_thread = thread
                return
            if self._owner_thread is not thread:
                raise ClosedLoopProtocolError(
                    f"{self.endpoint_name} already has a different owner thread"
                )

    def require_current_thread(self) -> None:
        thread = threading.current_thread()
        with self._lock:
            if self._owner_thread is None:
                raise ClosedLoopProtocolError(
                    f"{self.endpoint_name} has no bound owner thread"
                )
            if self._owner_thread is not thread:
                raise ClosedLoopProtocolError(
                    f"{self.endpoint_name} access attempted by a non-owner thread"
                )

__all__ = ["SingleThreadOwner"]
