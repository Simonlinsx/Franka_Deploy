#!/usr/bin/env python3
"""Read-only runtime preflight for the official GPU inference stage."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check official AnyDex GPU runtime without opening camera/robot hardware"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "weights/logs/model/checkpoint.tar.18"
    )
    parser.add_argument(
        "--inspire-model-dir",
        type=Path,
        default=ROOT / "weights/logs/model/inspire_model/obj140",
    )
    parser.add_argument(
        "--upstream-root", type=Path, default=ROOT / "third_party/AnyDexGrasp"
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def _decision_paths(model_dir: Path) -> tuple[Path, ...]:
    root = model_dir / "480" if (model_dir / "480").is_dir() else model_dir
    output = []
    for grasp_type in range(1, 9):
        candidates = sorted((root / str(grasp_type)).glob("*.pth"))
        if len(candidates) != 1:
            raise ValueError(
                f"Inspire type {grasp_type} requires exactly one .pth, found {len(candidates)}"
            )
        output.append(candidates[0].resolve())
    return tuple(output)


def check_runtime(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    model_dir = args.inspire_model_dir.expanduser().resolve()
    upstream = args.upstream_root.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"representation checkpoint is missing: {checkpoint}")
    decisions = _decision_paths(model_dir)
    required_source = (
        upstream / "models/minkowski_graspnet_single_point.py",
        upstream / "generate_mesh_and_pointcloud/inspire_urdf/width_12Dangle_6Dangle.json",
    )
    missing = [str(path) for path in required_source if not path.is_file()]
    if missing:
        raise FileNotFoundError("official source assets are missing: " + "; ".join(missing))

    torch = importlib.import_module("torch")
    importlib.import_module("MinkowskiEngine")
    importlib.import_module("pointnet2._ext")
    importlib.import_module("knn_pytorch.knn_pytorch")
    device = torch.device(str(args.device))
    if device.type != "cuda":
        raise ValueError("official AnyDex workflow requires a CUDA device")
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("torch.cuda.is_available() is false in the official environment")
    index = int(torch.cuda.current_device() if device.index is None else device.index)
    if index < 0 or index >= int(torch.cuda.device_count()):
        raise ValueError(
            f"requested {args.device} but CUDA device_count={torch.cuda.device_count()}"
        )
    name = str(torch.cuda.get_device_name(index))
    print(
        "[official runtime] READY python={} torch={} compiled_cuda={} device={} ({})".format(
            sys.version.split()[0],
            getattr(torch, "__version__", "unknown"),
            getattr(torch.version, "cuda", "unknown"),
            args.device,
            name,
        )
    )
    print(
        f"[official assets] representation={checkpoint} "
        f"decision_heads={len(decisions)} source={upstream}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        check_runtime(args)
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"[official runtime][FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
