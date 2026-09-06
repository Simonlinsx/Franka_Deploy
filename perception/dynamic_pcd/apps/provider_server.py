from __future__ import annotations

import argparse
import time

from dynamic_pcd.config import load_config
from dynamic_pcd.provider.object_pcd_provider import ObjectPCDProvider
from dynamic_pcd.ipc.zmq_pubsub import ZMQObjectPCDPublisher


def main():
    parser = argparse.ArgumentParser(description="Headless-ish object point cloud publisher. ROI selection still uses OpenCV at startup.")
    parser.add_argument("--config", type=str, default="configs/d435_default.yaml")
    parser.add_argument("--addr", type=str, default="tcp://127.0.0.1:5556")
    parser.add_argument("--sam2", action="store_true")
    parser.add_argument("--mode", type=str, default=None, choices=["roi_depth", "sam2_reinit"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["runtime"]["publish_zmq"] = True
    cfg["runtime"]["zmq_addr"] = args.addr
    cfg["runtime"]["show_pcd"] = False
    cfg["runtime"]["vis"] = True  # still show 2D overlay for debug
    if args.sam2:
        cfg["sam2"]["enabled"] = True
        cfg["tracker"]["mode"] = "sam2_reinit"
    if args.mode:
        cfg["tracker"]["mode"] = args.mode

    provider = ObjectPCDProvider(cfg)
    pub = ZMQObjectPCDPublisher(args.addr)
    try:
        provider.start()
        if not provider.select_and_initialize():
            return
        max_hz = float(cfg["runtime"].get("max_packet_hz", 30))
        last_pub = 0.0
        while True:
            _frame, _mask, _obj, packet = provider.step()
            now = time.time()
            if now - last_pub >= 1.0 / max(1e-6, max_hz):
                pub.publish(packet)
                last_pub = now
    except KeyboardInterrupt:
        pass
    finally:
        pub.close()
        provider.stop()


if __name__ == "__main__":
    main()
