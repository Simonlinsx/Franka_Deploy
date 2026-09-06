# Deployment notes

## Recommended launch order

1. Verify D435 is USB3:

```bash
lsusb -t
rs-enumerate-devices
```

2. Inspect depth:

```bash
./scripts/run_depth_viewer.sh
```

3. Run object point cloud provider without SAM2:

```bash
./scripts/run_masked_pcd.sh --mode roi_depth
```

4. Publish packets to a policy process:

```bash
./scripts/run_masked_pcd.sh --publish_zmq --no_show_pcd
python -m dynamic_pcd.apps.policy_client_demo
```

## Policy-side interface

The consumer receives `ObjectPCDPacket`:

```python
obs = packet.to_policy_obs()
pcd_history = obs["object_pcd_history"]  # [T, N, 3] or [T, N, 6]
center = obs["object_center"]            # [3]
velocity = obs["object_velocity"]        # [3]
valid = obs["valid"]
```

A safe policy loop should reject stale or invalid perception:

```python
if not packet.valid or time.time() - packet.timestamp > 0.15:
    action = hold_or_retract()
else:
    action = policy(pcd_history, center, velocity, robot_state)
```

## Transform to robot base

Replace `extrinsics.T_base_camera` in `configs/d435_default.yaml` after hand-eye calibration.

If `center_policy_points: true`, each point in `pcd_history` is local to the object center, while `object_center` remains in base frame. This is usually better for policy generalization.

## SAM2 mode

SAM2 is optional. Enable it only after the ROI-depth pipeline works:

```yaml
sam2:
  enabled: true
  checkpoint: /path/to/sam2.1_hiera_tiny.pt
  model_cfg: configs/sam2.1/sam2.1_hiera_t.yaml
  device: cuda

tracker:
  mode: sam2_reinit
```

Run:

```bash
./scripts/run_masked_pcd.sh --sam2
```

SAM2 is used for initialization/re-initialization. The high-frequency loop still uses ROI-depth tracking.
