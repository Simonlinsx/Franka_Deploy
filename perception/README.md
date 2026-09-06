# Perception

This module provides the deployable object-mask and partial-point-cloud stack
for **dynamic grasping**. The installed Python package remains `dynamic_pcd`
for import compatibility; `perception/` is the owning workspace domain.

The former top-level directory name `dynamic_object_pcd_v1` has been retired.
New code, documentation, and scripts use `perception/`; the installed Python
package remains `dynamic_pcd` so application imports stay stable.

```text
Intel RealSense D435 RGB-D
  -> aligned RGB-D
  -> manual ROI box (only once, used as a SAM2 prompt)
  -> SAM2 first-frame semantic mask (the rectangle is never the mask)
  -> stateful SAM2 Video exact-frame semantic mask
  -> same-frame masked depth -> object partial point cloud
  -> robust depth cleaning + fixed-N sampling
  -> policy-friendly packet
```

The design keeps heavyweight open-vocabulary detection outside the frame
deadline while running the compact stateful SAM2 video model online:

```text
D435 depth/color:          30 Hz in the calibrated deployment
SAM2-video semantic mask: target >=20 Hz
policy inference:          10-30 Hz
robot low-level control:   50-100 Hz+
```

## Tested environment target

- Ubuntu 22.04
- Intel RealSense D435 / D435i / D455-style D400 devices
- Python 3.9-3.11
- CUDA GPU for the default online SAM2 path (disable with `--no_online_sam2`)

Your Open3D environment may require the same workaround you already found:

```bash
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

## Install

```bash
cd /home/qiaoguanren/code/franka/perception
python -m pip install -e .
```

Or with a mirror:

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
```

Optional SAM2 support:

```bash
git clone https://github.com/facebookresearch/sam2.git ../third_party/sam2
cd ../third_party/sam2
python -m pip install -e .
# Download a SAM2/SAM2.1 checkpoint and set sam2.checkpoint/model_cfg in configs/d435_default.yaml
```

## Quick start

### 1. Live depth viewer

```bash
./scripts/run_depth_viewer.sh
```

Keys:

```text
q / ESC  quit
s        save current depth images
r        reset display range
1        toggle spatial filter
2        toggle temporal filter
3        toggle hole filling
```

### 2. Text-prompted live object point cloud (recommended)

The default prompt service uses YOLO-World to search exact D435 frames at
20 Hz. Its text-conditioned rectangle is used only as a native box prompt for
the second service, an official SAM2.1 tiny video predictor. The rectangle is
never published as the object mask or used to build policy points. Both GPU
services are launched and stopped automatically:

```bash
./scripts/run_masked_pcd.sh \
  --prompt "pink cylinder" \
  --online_sam2 \
  --publish_zmq \
  --show_scene_pcd \
  --print_center --print_fps --print_every 30
```

Use the most specific known phrase, for example `green ball` rather than only
`ball`. On the recorded fast-entry case, warmed YOLO-World inference measured
about 9.4--10.5 ms; the first correct `green ball` bbox appeared one frame
after the object entered. SAM2 then owns the complete binary silhouette.

The deployed per-frame path (`online_sam2.mask_publication_mode:
semantic_sam2`) runs SAM2.1 tiny at 512 px with BF16/TF32. SAM2 alone owns the
binary object silhouette. Depth, a fixed support-plane assumption, workspace
cropping, and the legacy adaptive RGB-D tracker cannot delete mask pixels or
invalidate a healthy semantic mask. This matters when the object sits on a
raised box instead of the table.

The mask is kept as the complete SAM2 object-id output rather than reduced to
its largest connected component. If the hand splits the visible object into
multiple pieces, all visible pieces remain available. The mask and depth always
come from the same D435 frame. A late GPU result is dropped instead of being
paired with newer depth; the policy observation owner holds its last complete
point-cloud sample until the next fresh exact-frame result.

For manual ROI acquisition, select the object tightly but leave a small amount
of background around it. The selected box is sent directly to the GPU SAM2
service as a native box prompt. The returned semantic silhouette initializes
temporal memory. Initialization fails clearly if SAM2 is unavailable or returns
an invalid mask; it does not silently fall back to GrabCut.

The older `adaptive_fusion` mode remains available for experiments, but it is
not selected by `configs/d435_default.yaml` and is not in the deployed mask
publication path.

### Legacy adaptive-fusion mode

The recovery logic below applies only when
`online_sam2.mask_publication_mode: adaptive_fusion` is selected explicitly.

Complete occlusion therefore does not require KLT to survive. While LOST, the
fast RGB-D path first searches around the last valid bbox; after two rejected
frames it scans the complete image using the appearance, depth, and original
size learned from the prompt mask. A unique RGB-D candidate must remain
consistent for two internal observations. Every recovery source (adaptive,
online SAM2, or semantic prompt replay) then enters a provider-level quarantine:
three additional exact RGB-D frames must agree in bbox overlap, mask area,
depth, and 3D center before any mask/PCD/center becomes public. A passing hand,
one- or two-frame false recovery, or near-tied duplicate remains fail-closed.

Recovery scale is checked against the original complete prompt mask, not the
small fragment left by a gradually advancing occluder. Nearby same-depth colour
islands are reconstructed through a tight depth-supported hull before ranking,
so highlights and small holes on one object do not become multiple ambiguous
targets. Distant same-category objects and a closer hand are not merged.

After three lost frames (about 0.1 s at 30 Hz), semantic re-acquisition also
runs in a background worker. Automatic retries search globally and do not bias
the text detector toward the old bbox. The camera retains about two seconds of
exact RGB-D frames. A returned semantic mask initializes a fresh generic
adaptive tracker on its own source frame and is replayed, in strict frame-ID
order, to the exact newest buffered frame. At most 12 uniformly sampled frames
are processed while always retaining the source and newest endpoints. Any
invalid intermediate update fails closed; fixed-scale template translation is
not used as a production fallback. Because a text prompt names a category and
the user may carry the object while it is hidden, this category-level fallback
may relocate across raw camera depth. The immutable appearance and original-
size gates remain mandatory. A result is discarded only if the provider already
has a committed current packet; a raw tracker state still in recovery probation
cannot suppress semantic recovery. The packet computed before a successful
reset is suppressed. Errors, evicted source frames, invalid replay updates, and
rejected masks automatically schedule a newest-frame retry with a 0.5 s start-
to-start cooldown; no `T` press is required.

If several identical objects are visible, the default is the highest text
score. Provide an image-space hint when necessary:

```bash
./scripts/run_masked_pcd.sh \
  --prompt "pink cylinder" \
  --prompt_reference_roi X1 Y1 X2 Y2
```

When tracking is `LOST`, `T`/`r` can still force a non-blocking re-run of the
same prompt; normal failed acquisitions retry automatically. A key press is
ignored while the tracker is healthy so an old semantic result cannot replace
a valid track. Automatic semantic re-acquisition searches globally and then
applies strict current-frame appearance/depth/size gates. Those gates reduce accidental switching, but
text and RGB-D observations cannot prove physical identity between truly
indistinguishable instances; use `--prompt_reference_roi` and operator
verification whenever duplicates are visible.

The deployment config enables only the Open3D scene/object window by default.
This avoids loading OpenCV-Qt and Open3D/GLFW GUI backends together, which can
produce harmless `QObject::moveToThread` warnings. For the RGB/mask window only,
run with `--vis --no_show_pcd`; for headless publishing use
`--no_vis --no_show_pcd`.

The prompt service can also be run manually; see
[`docs/prompt_segmentation_service.md`](docs/prompt_segmentation_service.md).

### 3. Manual-ROI fallback

```bash
./scripts/run_masked_pcd.sh --no_sam2 --no_online_sam2
```

A color window opens. Drag a box around the target object and press Enter/Space. The provider will track the object using ROI + depth and show:

- RGB overlay with mask and bbox
- Optional Open3D object point cloud
- Fixed-N object point cloud history internally

Keys in the RGB window:

```text
q / ESC  quit
T / r    reselect target ROI (use this after moving/losing the object)
L        toggle LOCKED/FOLLOW ROI mode
s        save current object point cloud as PLY/NPY
p        pause/resume
H        print key help
D        launch a dry-run hover plan (no motion)
M        arm one frozen one-shot hover
Y        confirm an armed hover within 5 seconds
X        cancel/disarm the active plan or motion child
```

Keys in the Open3D point cloud window:

```text
T        reselect target ROI in the RGB selector
L        toggle LOCKED/FOLLOW ROI mode
P        pause/resume perception and publishing
S        save current object point cloud
H        print key help
D        launch a dry-run hover plan (no motion)
M        arm one frozen one-shot hover
Y        confirm an armed hover within 5 seconds
X        cancel/disarm the active plan or motion child
]        increase point size
[        decrease point size
R        reset camera view
Q        close only the Open3D window
```

`LOCKED` is appropriate only while the object remains stationary. Switch to
`FOLLOW` before moving it; if tracking is already lost, press `T` and draw a
new tight box around the object. Open3D callbacks only enqueue requests—the
perception loop performs tracker changes. While the ROI selector is open no new
packets are published, so do not reselect during an active robot trajectory;
the motion consumer must also keep its stale-packet watchdog enabled.

To enable the robot keys together with the RGB and scene/object point-cloud
windows, start the app with all fail-closed confirmations:

```bash
./scripts/run_masked_pcd.sh \
  --no_sam2 --publish_zmq --show_scene_pcd \
  --goal_clearance_m 0.115 --hover_clearance_m 0.10 \
  --enable_robot_motion \
  --confirm_calibration_id eye-to-hand-b722bce10485c8a3 \
  --confirm_workspace_clear --confirm_eef_clear --confirm_descent_clear
```

Use `T` to reacquire a moved object, wait until its mask and center are stable,
then press `L` so the overlay says `LOCKED`. Press `D` first to inspect the
printed plan. `M` only arms the request; a separate `Y` press within five
seconds launches the isolated robot process. The path always rises to the
commissioned transit height, translates in XY, and only then descends to the
requested clearance. `X` sends `SIGINT` so the control-owning process calls
`robot.stop()`; it is not a replacement for the physical emergency stop.

This is manual reacquisition plus a frozen one-shot hover, not continuous visual
servoing. Do not move the object after `Y`. A true moving-target follower needs
a semantic/color tracker, latency-bounded goal filtering, and a separately
commissioned receding-horizon controller.

### 4. Publish object point cloud for policy

Terminal 1:

```bash
./scripts/run_masked_pcd.sh --no_sam2 --publish_zmq --no_show_pcd --print_center --print_every 10
```

Draw a tight ROI around only the target object and press Enter/Space. The RGB
overlay and terminal then report the cleaned, visible partial-cloud center in
meters, for example:

```text
[Center] frame=120 valid=True frame_name=robot_base visible_median=[0.5481 0.0613 0.0842] z95=0.1126 calibration_id=eye-to-hand-b722bce10485c8a3
```

`visible_median` is the robust center of the currently visible object points,
not necessarily the object's CAD/geometric center. Use `r` if the tracker has
latched onto the table, gripper, calibration board, or another object. Remove
`--no_show_pcd` if an interactive 3D view is useful.

Terminal 2:

```bash
python -m dynamic_pcd.apps.policy_client_demo --addr tcp://127.0.0.1:5556
```

The policy packet contains:

```python
{
    "pcd_history": np.ndarray,   # [T, N, 3] or [T, N, 6]
    "pcd_current": np.ndarray,   # [N, 3] or [N, 6]
    "center": np.ndarray,        # [3], in base frame or camera frame if extrinsics is identity
    "velocity": np.ndarray,      # [3]
    "bbox_xyxy": np.ndarray,     # [4]
    "timestamp": float,
    "valid": bool,
    "frame_id": int,
    "pcd_reference": np.ndarray, # absolute sampled points used for geometry
    "reference_frame": str,      # robot_base for the calibrated deployment
    "point_frame": str,
    "calibration_id": str,
    "camera_serial": str,
    "T_base_camera": np.ndarray,
}
```

When `valid=False`, `pcd_current`, `pcd_history`, `pcd_reference`, `center`,
`velocity`, and `bbox_xyxy` are all `None`; an old object payload is never
re-stamped as a current frame. The last search bbox is diagnostic-only under
`packet.debug.tracker_search_bbox_xyxy`. Live status reports total publish rate,
valid object-PCD rate, valid ratio, and `LOST/NO_VALID` separately.

## FR3 fixed-camera eye-to-hand deployment

This checkout contains the completed fixed-camera calibration:

```text
config:          configs/calibrations/fr3_d435_eye_to_hand.yaml
calibration ID:  eye-to-hand-b722bce10485c8a3
camera serial:   337322072188
transform:       T_base_camera (camera_color_optical_frame -> robot_base)
quality:         pass
```

`configs/d435_default.yaml` loads that file and requires both a quality pass and
the matching physical camera serial. The provider fails closed instead of
silently publishing camera-frame points if the file, ID, transform, frame names,
quality result, or serial is inconsistent.

The ArUco board was rigidly attached to the EEF **only while collecting the
eye-to-hand calibration poses**. Before object-cloud or robot-hover operation:

1. Remove the calibration board and all temporary tape/brackets from the EEF.
2. Do not move or re-mount the RealSense camera, the Franka base, or the table
   fixtures that define their relative pose. If either camera or robot base has
   moved, run a new eye-to-hand calibration before using `robot_base` output.
3. Keep the emergency stop reachable, verify the EEF starts clear of the table,
   and clear the complete rise-and-translate corridor.
4. Place one stationary target in the commissioned region and select a tight ROI
   around it. This first validation tool deliberately rejects a moving/noisy
   target; it is not a dynamic following controller.

Removing the board after calibration does not invalidate a fixed-camera
eye-to-hand transform. Leaving it attached is unsafe here because it changes the
EEF collision envelope, can obscure the target, and is not represented in the
hover workspace checks.

### 1. Start calibrated perception and display the center

Keep this running in terminal 1:

```bash
cd /home/qiaoguanren/code/franka/perception
./scripts/run_masked_pcd.sh \
  --no_sam2 \
  --publish_zmq \
  --no_show_pcd \
  --print_center \
  --print_every 10
```

After selecting the ROI, first verify all valid center lines say:

```text
frame_name=robot_base
calibration_id=eye-to-hand-b722bce10485c8a3
```

The coordinates are meters in the Franka base frame. Do not proceed if the
center jumps, follows the gripper/table instead of the object, or reports another
frame/calibration ID.

### 2. Plan the hover first (default dry-run)

With terminal 1 still publishing, run in terminal 2:

```bash
cd /home/qiaoguanren/code/franka/perception
python -m dynamic_pcd.apps.hover_over_object \
  --config configs/d435_default.yaml \
  --addr tcp://127.0.0.1:5556 \
  --robot-ip 172.16.0.2
```

This command connects to the Franka to read its state, waits for a stable object
center, and prints the frozen goal and every waypoint. It sends **no motion
command** unless `--execute` is present. Review the printed current EEF, target,
goal, and waypoints before continuing.

### 3. Execute one conservative hover

Only after the dry-run plan is correct, the board is removed, and the whole
workspace is visibly clear, run:

```bash
python -m dynamic_pcd.apps.hover_over_object \
  --config configs/d435_default.yaml \
  --addr tcp://127.0.0.1:5556 \
  --robot-ip 172.16.0.2 \
  --allow-descent --clearance-m 0.10 \
  --execute \
  --confirm-calibration-id eye-to-hand-b722bce10485c8a3 \
  --confirm-workspace-clear \
  --confirm-eef-clear \
  --confirm-descent-clear
```

The motion tool freezes one stable target and performs a single
**rise, high XY translate, then guarded vertical descent** plan. Without
`--allow-descent`, it never commands a downward move. The commissioned limits
in `configs/d435_default.yaml` are:

```text
object center: x=[0.505, 0.650], y=[0.008, 0.104], z=[-0.05, 0.30] m
EEF motion:    x=[0.480, 0.660], y=[-0.020, 0.130], z=[0.300, 0.360] m
hover z:       0.320 to 0.345 m, with 0.150 m cloud-top clearance
motion:        <=0.030 m per segment, <=0.010 m/s
```

It refuses motion for stale/invalid perception, insufficient points, unstable
center, wrong frame/calibration/camera serial, an out-of-workspace plan, robot
errors/contact/collision, or insufficient joint margin. During execution it
stops on stale perception or more than 20 mm target drift, and it does not retry
automatically. Keep a hand at the emergency stop throughout this physical
validation.

## Coordinate frame

The provided FR3 configuration is calibrated, so `pcd_reference`, `center`, and
the uncentered policy point clouds are expressed in `robot_base`. Other
deployments may use an identity transform only for camera-frame visualization;
never command a robot from identity/fallback extrinsics.

Recommended policy input:

```text
object_pcd_local = object_pcd_xyz - object_center
object_center_base
object_velocity_base
robot_state
```

This repo can center the policy points automatically with `pointcloud.center_policy_points: true`.

## Why not run the text detector after acquisition?

YOLO-World is used only in SEARCHING/LOST states. Running it after acquisition
would duplicate SAM2 work and increase GPU contention. The normal path is:

```text
text prompt -> YOLO-World exact-frame bbox -> online SAM2 box mask
            -> online SAM2 video memory || adaptive RGB-D/KLT tracker
            -> stable adaptive tracked mask + strict SAM watchdog
            -> local/full-image recovery + three-frame publication quarantine
            -> object point cloud only after provider commit
            -> asynchronous category re-detection after three lost frames
```

The appearance model is learned from the prompt mask; it is not a
`pink-cylinder` heuristic. Failed identity/size/depth/motion gates immediately
produce an invalid empty payload.

## Important limitations

保存的真实 RGB-D 可在完全不打开 D435、Franka 或 RH56 的条件下，按当前
`guarded_sam2_primary` production mask owner 重新运行，并把每个最终 mask 送入正式 128 点
projector。命令、严格退出阈值和逐帧 provenance 字段见
[`docs/GUARDED_V2_SAVED_RGBD_ACCEPTANCE.md`](docs/GUARDED_V2_SAVED_RGBD_ACCEPTANCE.md)。

- D435 depth has edge shadows and flying pixels near object boundaries.
- The mask is eroded before point projection to avoid boundary artifacts.
- Temporal depth filtering is off by default because it can create motion ghosts.
- YOLO-World is the category acquisition/re-detection path; the separate
  SAM2.1 video predictor is the per-frame temporal path.
- A text prompt names a category, not a unique physical instance. Continuity
  gates and an optional reference ROI are required when duplicates are visible.
- After full occlusion, a unique learned-appearance/depth candidate can recover
  anywhere in the camera image after two consistent observations. Two visually
  indistinguishable category instances remain fundamentally ambiguous and are
  kept fail-closed instead of silently switching identity.
- You still need a calibrated `T_base_camera` before commanding a robot.

## File layout

```text
configs/d435_default.yaml                 main config
scripts/run_depth_viewer.sh               depth viewer launcher
scripts/run_masked_pcd.sh                 object pcd launcher
scripts/run_prompt_segmentation_service.sh persistent Grounded-SAM launcher
scripts/run_yolo_world_prompt_service.sh persistent YOLO-World launcher
scripts/run_sam2_video_service.sh         persistent online SAM2 launcher
dynamic_pcd/camera/realsense_camera.py    D435 RGB-D capture
dynamic_pcd/segmentation/grounded_sam.py  prompt detector/segmenter backend
dynamic_pcd/segmentation/prompt_runtime.py service lifecycle + async recovery
dynamic_pcd/segmentation/sam2_video_backend.py official online video-memory adapter
dynamic_pcd/segmentation/sam2_video_runtime.py online service lifecycle manager
dynamic_pcd/segmentation/adaptive_color_depth_tracker.py high-rate generic tracker
dynamic_pcd/segmentation/roi_depth_tracker.py fast tracker
dynamic_pcd/segmentation/sam2_image.py    optional SAM2 image predictor wrapper
dynamic_pcd/pointcloud/extractor.py       mask-depth -> object pcd
dynamic_pcd/provider/object_pcd_provider.py perception provider
dynamic_pcd/ipc/zmq_pubsub.py             policy IPC
dynamic_pcd/apps/*.py                     runnable apps
```
