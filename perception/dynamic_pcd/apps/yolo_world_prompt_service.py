from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from dynamic_pcd.apps.prompt_segmentation_service import run_server
from dynamic_pcd.segmentation.yolo_world_prompt import YOLOWorldPromptBackend


DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "long_vos_clean"
    / "weights"
    / "yolov8s-worldv2.pt"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persistent YOLO-World text bbox service"
    )
    parser.add_argument("--addr", default="tcp://127.0.0.1:5557")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence", type=float, default=0.03)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--preload-prompt", default=None)
    parser.add_argument("--max-image-mb", type=float, default=16.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    device = str(args.device).strip().lower()
    if device == "cuda":
        device = "0"
    elif device.startswith("cuda:"):
        device = device.split(":", 1)[1]
    backend = YOLOWorldPromptBackend(
        weights=str(args.weights.expanduser().resolve()),
        device=device,
        confidence=float(args.confidence),
        image_size=int(args.image_size),
        preload_prompt=args.preload_prompt,
    )
    run_server(
        backend=backend,
        addr=str(args.addr),
        max_image_bytes=int(float(args.max_image_mb) * 1024 * 1024),
        preload=True,
    )


if __name__ == "__main__":
    main()
