from __future__ import annotations

import pickle
import time
from typing import Optional

import zmq

from dynamic_pcd.types import ObjectPCDPacket


def serialize_object_pcd_packet(packet: ObjectPCDPacket) -> bytes:
    """Serialize exactly the payload used by the live ZMQ publisher."""

    return pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)


class ZMQObjectPCDPublisher:
    def __init__(self, addr: str = "tcp://127.0.0.1:5556"):
        self.addr = addr
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 8)
        self.sock.bind(addr)
        # Give subscribers a moment if the app starts them quickly.
        time.sleep(0.1)
        print(f"[ZMQ] publishing object pcd packets on {addr}")

    def publish(self, packet: ObjectPCDPacket) -> None:
        payload = serialize_object_pcd_packet(packet)
        self.sock.send_multipart([b"object_pcd", payload])

    def close(self):
        self.sock.close(linger=0)


class ZMQObjectPCDSubscriber:
    def __init__(self, addr: str = "tcp://127.0.0.1:5556", timeout_ms: int = 1000):
        self.addr = addr
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 8)
        self.sock.connect(addr)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"object_pcd")
        self.sock.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        print(f"[ZMQ] subscribed to {addr}")

    def recv(self, timeout_ms: Optional[int] = None) -> Optional[ObjectPCDPacket]:
        try:
            if timeout_ms is not None and not self.sock.poll(
                timeout=max(0, int(timeout_ms)), flags=zmq.POLLIN
            ):
                return None
            _topic, payload = self.sock.recv_multipart()
            return pickle.loads(payload)
        except zmq.Again:
            return None

    def recv_latest(
        self, timeout_ms: int = 0, max_drain: int = 16
    ) -> Optional[ObjectPCDPacket]:
        """Return a recent packet while bounding queue-drain work."""

        latest = self.recv(timeout_ms=timeout_ms)
        if latest is None:
            return None
        for _ in range(max(0, int(max_drain) - 1)):
            newer = self.recv(timeout_ms=0)
            if newer is None:
                return latest
            latest = newer
        return latest

    def close(self):
        self.sock.close(linger=0)
