"""Lazy, read-only adapters for Franka, RH56, and the object-PCD publisher."""

from __future__ import annotations

import importlib
import struct
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from .contracts import FrankaObservation, InspireObservation, ObjectPCDObservation

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _prepend_local_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _ensure_workspace_imports() -> None:
    # These are source roots already used by the repository's own launchers.
    _prepend_local_path(WORKSPACE_ROOT)
    _prepend_local_path(WORKSPACE_ROOT / "perception")
    _prepend_local_path(WORKSPACE_ROOT / "dexgrasp" / "src")


def _midpoint_timestamp(started_at_s: float, finished_at_s: float) -> float:
    return 0.5 * (float(started_at_s) + float(finished_at_s))


class FrankaStateReader:
    """One explicit pylibfranka connection used only for read_once()."""

    def __init__(
        self,
        robot_ip: str,
        *,
        enforce_realtime: bool = False,
        pylibfranka_module: Any = None,
        robot: Any = None,
    ) -> None:
        self.robot_ip = str(robot_ip)
        self.enforce_realtime = bool(enforce_realtime)
        self._pylibfranka = pylibfranka_module
        self._robot = robot

    def start(self) -> None:
        if self._robot is not None:
            return
        module = self._pylibfranka
        if module is None:
            module = importlib.import_module("pylibfranka")
            self._pylibfranka = module
        realtime = (
            module.RealtimeConfig.kEnforce
            if self.enforce_realtime
            else module.RealtimeConfig.kIgnore
        )
        self._robot = module.Robot(self.robot_ip, realtime)

    def read(self) -> FrankaObservation:
        self.start()
        started = time.time()
        state = self._robot.read_once()
        finished = time.time()
        return FrankaObservation.from_state(
            state, _midpoint_timestamp(started, finished)
        )

    def close(self) -> None:
        # pylibfranka Robot has no close method.  Do not call stop() from a
        # read-only reader: this process never created a control handle.
        self._robot = None

    def __enter__(self) -> "FrankaStateReader":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class InspireStateReader:
    """Exclusive RH56 serial context that never writes a register."""

    SNAPSHOT_MODES = ("full", "compact_policy")

    def __init__(
        self,
        *,
        port: Optional[str],
        baud: int = 115200,
        hand_id: int = 1,
        timeout_s: float = 0.5,
        debug: bool = False,
        snapshot_mode: str = "full",
        api: Any = None,
        serial_context: Any = None,
        hand: Any = None,
    ) -> None:
        self.port = port
        self.baud = int(baud)
        self.hand_id = int(hand_id)
        self.timeout_s = float(timeout_s)
        self.debug = bool(debug)
        self.snapshot_mode = str(snapshot_mode)
        if self.snapshot_mode not in self.SNAPSHOT_MODES:
            raise ValueError(
                "snapshot_mode must be one of "
                f"{self.SNAPSHOT_MODES}, got {self.snapshot_mode!r}"
            )
        self._api = api
        self._serial_context = serial_context
        self._hand = hand
        self.resolved_port: Optional[str] = None
        self._owns_context = False

    def start(self) -> None:
        if self._hand is not None:
            return
        _ensure_workspace_imports()
        api = self._api or importlib.import_module("examples.inspire_rh56_test")
        self._api = api
        resolved_port = self.port or api.find_serial_port()
        context = self._serial_context or api.LinuxSerial(
            resolved_port, self.baud, self.timeout_s, self.debug
        )
        entered = context.__enter__()
        serial_port = context if entered is None else entered
        try:
            self._hand = api.RH56Hand(serial_port, self.hand_id)
        except BaseException:
            context.__exit__(*sys.exc_info())
            raise
        self._serial_context = context
        self._owns_context = True
        self.resolved_port = str(resolved_port)

    def read(self) -> InspireObservation:
        self.start()
        started = time.time()
        if self.snapshot_mode == "compact_policy":
            snapshot = self._read_compact_policy_snapshot()
        else:
            snapshot = self._hand.snapshot()
        finished = time.time()
        return InspireObservation.from_snapshot(
            snapshot, _midpoint_timestamp(started, finished)
        )

    def _read_compact_policy_snapshot(self) -> Mapping[str, Any]:
        """Read the exact V94 feedback/safety fields in three transactions.

        The RH56 map contains three useful contiguous blocks: ANGLE_SET;
        POS_ACT+ANGLE_ACT; and FORCE_ACT through TEMP.  The generic diagnostic
        ``snapshot()`` also reads configuration registers one-by-one and costs
        roughly fifteen serial round trips.  This path changes no register and
        preserves every field consumed by ``InspireObservation``.
        """

        api = self._api
        hand = self._hand
        if api is None or hand is None:
            raise RuntimeError("RH56 compact snapshot reader is not started")
        required = (
            "REG_ANGLE_SET",
            "REG_POS_ACT",
            "REG_ANGLE_ACT",
            "REG_FORCE_ACT",
            "REG_CURRENT",
            "REG_ERROR",
            "REG_STATUS",
            "REG_TEMP",
        )
        missing = [name for name in required if not hasattr(api, name)]
        if missing:
            raise RuntimeError(
                "RH56 compact snapshot register map is incomplete: "
                + ", ".join(missing)
            )
        expected_adjacency = (
            ("REG_POS_ACT", "REG_ANGLE_ACT", 12),
            ("REG_FORCE_ACT", "REG_CURRENT", 12),
            ("REG_CURRENT", "REG_ERROR", 12),
            ("REG_ERROR", "REG_STATUS", 6),
            ("REG_STATUS", "REG_TEMP", 6),
        )
        for first, second, width in expected_adjacency:
            if int(getattr(api, second)) != int(getattr(api, first)) + width:
                raise RuntimeError(
                    "RH56 compact snapshot register map is not contiguous: "
                    f"{first}->{second}"
                )

        angle_targets_raw = hand.read(
            int(api.REG_ANGLE_SET), 12, retries=0
        )
        position_angle_raw = hand.read(
            int(api.REG_POS_ACT), 24, retries=0
        )
        safety_raw = hand.read(
            int(api.REG_FORCE_ACT), 42, retries=0
        )
        if len(angle_targets_raw) != 12:
            raise RuntimeError("RH56 compact ANGLE_SET block is incomplete")
        if len(position_angle_raw) != 24:
            raise RuntimeError("RH56 compact POS/ANGLE_ACT block is incomplete")
        if len(safety_raw) != 42:
            raise RuntimeError("RH56 compact safety block is incomplete")

        position_angle = struct.unpack("<12h", position_angle_raw)
        return {
            "angle_targets": struct.unpack("<6h", angle_targets_raw),
            "positions": position_angle[:6],
            "angles": position_angle[6:],
            "forces": struct.unpack("<6h", safety_raw[:12]),
            "currents": struct.unpack("<6h", safety_raw[12:24]),
            "errors": tuple(safety_raw[24:30]),
            "statuses": tuple(safety_raw[30:36]),
            "temperatures": tuple(safety_raw[36:42]),
        }

    def close(self) -> None:
        context = self._serial_context
        if self._owns_context and context is not None:
            context.__exit__(None, None, None)
        self._owns_context = False
        self._serial_context = None
        self._hand = None

    def __enter__(self) -> "InspireStateReader":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _require_trusted_pcd_endpoint(addr: str) -> None:
    """Reject remote pickle endpoints; the bundled publisher is local-only."""

    if addr.startswith("ipc://") or addr.startswith("inproc://"):
        return
    parsed = urlparse(addr)
    if parsed.scheme != "tcp" or parsed.hostname not in (
        "127.0.0.1",
        "localhost",
        "::1",
    ):
        raise ValueError(
            "object-PCD packets use pickle; only loopback tcp://, ipc://, or "
            "inproc:// endpoints are accepted"
        )


class ObjectPCDReader:
    """Latest-value adapter around dynamic_pcd's ZMQ subscriber."""

    def __init__(
        self,
        addr: str,
        *,
        timeout_ms: int = 1000,
        subscriber: Any = None,
    ) -> None:
        self.addr = str(addr)
        self.timeout_ms = int(timeout_ms)
        self._subscriber = subscriber

    def start(self) -> None:
        if self._subscriber is not None:
            return
        _require_trusted_pcd_endpoint(self.addr)
        _ensure_workspace_imports()
        module = importlib.import_module("dynamic_pcd.ipc.zmq_pubsub")
        self._subscriber = module.ZMQObjectPCDSubscriber(
            self.addr, timeout_ms=self.timeout_ms
        )

    def read(self) -> Optional[ObjectPCDObservation]:
        self.start()
        packet = self._subscriber.recv_latest(timeout_ms=self.timeout_ms)
        if packet is None:
            return None
        return ObjectPCDObservation.from_packet(packet, received_at_s=time.time())

    def close(self) -> None:
        if self._subscriber is not None:
            self._subscriber.close()
        self._subscriber = None

    def __enter__(self) -> "ObjectPCDReader":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class ObservationReader:
    """Sequential diagnostic sampler for all enabled sources.

    This class is intentionally not the final active-control observation path.
    During FCI motion the controlling process must reuse the state returned by
    that control handle's ``readOnce()`` instead of opening this second Robot.
    """

    def __init__(
        self,
        runtime: Mapping[str, Any],
        profile: Mapping[str, Any],
        *,
        use_franka: bool,
        use_inspire: bool,
        use_object_pcd: bool,
        franka_reader: Any = None,
        inspire_reader: Any = None,
        object_reader: Any = None,
    ) -> None:
        self.use_franka = bool(use_franka)
        self.use_inspire = bool(use_inspire)
        self.use_object_pcd = bool(use_object_pcd)
        franka_cfg = runtime["franka"]
        hand_cfg = runtime["inspire"]
        pcd_cfg = runtime["object_pcd"]
        profile_franka = profile["franka"]
        profile_hand = profile["inspire"]
        self.franka_reader = franka_reader or FrankaStateReader(
            str(profile_franka["ip"]),
            enforce_realtime=bool(franka_cfg.get("enforce_realtime", False)),
        )
        configured_port = profile_hand.get("port")
        self.inspire_reader = inspire_reader or InspireStateReader(
            port=(None if configured_port in (None, "") else str(configured_port)),
            baud=int(profile_hand["baud"]),
            hand_id=int(profile_hand["hand_id"]),
            timeout_s=float(hand_cfg.get("serial_timeout_s", 0.5)),
            debug=bool(hand_cfg.get("debug", False)),
        )
        self.object_reader = object_reader or ObjectPCDReader(
            str(pcd_cfg["addr"]), timeout_ms=int(pcd_cfg.get("timeout_ms", 1000))
        )

    def start(self) -> None:
        started = []
        try:
            if self.use_franka:
                self.franka_reader.start()
                started.append(self.franka_reader)
            if self.use_inspire:
                self.inspire_reader.start()
                started.append(self.inspire_reader)
            if self.use_object_pcd:
                self.object_reader.start()
                started.append(self.object_reader)
        except BaseException:
            for reader in reversed(started):
                try:
                    reader.close()
                except Exception:
                    pass
            raise

    def read_raw(self):
        franka = self.franka_reader.read() if self.use_franka else None
        inspire = self.inspire_reader.read() if self.use_inspire else None
        object_pcd = self.object_reader.read() if self.use_object_pcd else None
        return franka, inspire, object_pcd

    def close(self) -> None:
        errors = []
        for enabled, reader in (
            (self.use_object_pcd, self.object_reader),
            (self.use_inspire, self.inspire_reader),
            (self.use_franka, self.franka_reader),
        ):
            if enabled:
                try:
                    reader.close()
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise RuntimeError(
                "observation reader cleanup failed: "
                + "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
            )

    def __enter__(self) -> "ObservationReader":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "FrankaStateReader",
    "InspireStateReader",
    "ObjectPCDReader",
    "ObservationReader",
]
