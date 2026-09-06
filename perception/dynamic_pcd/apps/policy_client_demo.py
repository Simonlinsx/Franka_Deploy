from __future__ import annotations

import argparse
import time

import numpy as np

from dynamic_pcd.ipc.zmq_pubsub import ZMQObjectPCDSubscriber


def fake_policy(obs):
    """Replace this with your real policy forward pass."""
    if not obs["valid"] or obs["object_pcd_history"] is None:
        return {"mode": "hold", "action": np.zeros(7, dtype=np.float32)}
    center = obs["object_center"]
    velocity = obs["object_velocity"]
    # Example only: policy would use robot_state + pcd history.
    return {"mode": "policy", "center": center, "velocity": velocity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", type=str, default="tcp://127.0.0.1:5556")
    parser.add_argument("--timeout_ms", type=int, default=1000)
    args = parser.parse_args()

    sub = ZMQObjectPCDSubscriber(args.addr, timeout_ms=args.timeout_ms)
    last_t = time.time()
    try:
        while True:
            packet = sub.recv()
            if packet is None:
                print("[policy_client] timeout waiting for object_pcd")
                continue
            obs = packet.to_policy_obs()
            action = fake_policy(obs)
            now = time.time()
            hz = 1.0 / max(1e-6, now - last_t)
            last_t = now
            shape = None if obs["object_pcd_history"] is None else obs["object_pcd_history"].shape
            print(
                f"frame={packet.frame_id} valid={packet.valid} recv_hz={hz:.1f} "
                f"hist_shape={shape} center={packet.center} vel={packet.velocity} "
                f"reference_frame={packet.reference_frame} "
                f"point_frame={packet.point_frame} "
                f"calibration_id={packet.calibration_id} action={action['mode']}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()


if __name__ == "__main__":
    main()
