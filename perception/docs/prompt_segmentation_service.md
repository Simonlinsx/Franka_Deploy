# Prompt segmentation service

The prompt service keeps Grounding DINO and SAM resident in a separate process.
The camera process sends a BGR frame and an arbitrary text prompt such as
`striped ceramic mug`; no object label is hard-coded in the service.

The default launcher uses the existing offline FoundationPose environment,
native GroundingDINO Swin-T weights, and SAM ViT-B weights.  `--device auto`
selects CUDA on the RTX 3060 and retains a CPU fallback when CUDA is not
available:

The normal camera application manages this service automatically:

```bash
./scripts/run_masked_pcd.sh \
  --prompt "pink cylinder" \
  --publish_zmq --show_scene_pcd --print_fps --print_center
```

Use the manual launcher below only when sharing one already-warmed service
between camera-process restarts or debugging its model backend.

```bash
cd /home/qiaoguanren/code/franka/perception
./scripts/run_prompt_segmentation_service.sh
```

It binds `tcp://127.0.0.1:5557`. Models load once and remain resident. Offline
Hugging Face mode is the default; pass `--allow-network` only when a required
tokenizer/model is intentionally not cached.

The Python 3.9 camera-side API has no Torch, Transformers, GroundingDINO, SAM,
or Pillow dependency:

```python
from dynamic_pcd.segmentation.prompt_client import ZMQPromptSegmentationClient

client = ZMQPromptSegmentationClient("tcp://127.0.0.1:5557")
result = client.segment(
    color_bgr,
    prompt="striped ceramic mug",
    previous_bbox_xyxy=last_bbox,
    request_id="recovery-generation-12",
    frame_id=frame.frame_id,
    frame_timestamp=frame.timestamp,
    frame_metadata={"tracker_generation": 12},
)
```

`result.mask` is a binary `uint8` image. `result.candidates` contains all ranked
detector candidates (up to `top_k`) with detector score, grounded label, box,
previous-box IoU/distance, rank score, and SAM validation metadata. The request
and frame identifiers are echoed so an asynchronous caller can discard stale
re-acquisition results.

GroundingDINO remains the heavyweight category detector even when it runs on
CUDA. Use this service for initial acquisition and asynchronous semantic
fallback. A separate persistent SAM2.1 video service supplies the per-frame
temporal mask, while the learned appearance/depth tracker supplies independent
RGB-D recovery and validates every SAM2 result. A prompt mask belongs to its
captured source image, so the camera keeps a bounded recent RGB-D buffer. A
fresh generic adaptive tracker is initialized on that exact source frame and
replayed in strict frame-ID order to the exact current buffer head before the
current-frame gates run. Source and head are always retained in a uniformly
sampled replay of at most 12 frames. An evicted source, an invalid intermediate
update, or a head-frame mismatch is rejected instead of translating stale
pixels. The response reports detector, segmenter, total, and service inference
milliseconds; `client.health()` reports resolved device, persistent model load
time, and request count.

After the high-rate tracker is confirmed lost, the camera process keeps at most
one asynchronous prompt request in flight. An error, an invalid detection, or a
mask rejected by current-frame appearance/size/category checks schedules another
request from the newest camera frame automatically. The configured reacquisition
cooldown is start-to-start. Retries stay fail-closed and preserve request ID, source
frame, tracker generation, prompt, and reason echoes; no point cloud resumes
until a fresh mask passes all current-frame gates and the provider's three-frame
RGB-D publication quarantine. A raw tracker `valid` flag does not cancel a
semantic response while that quarantine is still active.

Alternative combinations are configurable, for example a Transformers detector
or the local SAM2 image predictor:

```bash
./scripts/run_prompt_segmentation_service.sh \
  --detector-backend transformers \
  --mask-backend sam2 \
  --local-files-only
```

The alternative backend requires its optional packages and cached weights. The
default native GroundingDINO + SAM1 path requires no download on this machine.
