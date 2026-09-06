#!/usr/bin/env python3
"""Export a training checkpoint as a compact inference-only checkpoint.

Canonical CLI: ``python -m sim2real.diagnostics.export_inference_checkpoint``.

The supervised runtime deliberately refuses checkpoint files larger than
64 MiB. PPO checkpoints may exceed that limit because they retain optimizer,
critic, and replay state even though deployment only consumes the student
model, normalization, metadata, spec, and training progress value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Optional, Sequence

import torch


MAX_SOURCE_BYTES = 256 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_mapping(value: object, name: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    result: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must contain only string-keyed tensors")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name}.{key} contains NaN or infinity")
        result[key] = tensor.detach().cpu().contiguous()
    return result


def export_inference_checkpoint(source: Path, output: Path) -> dict[str, object]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source checkpoint is missing: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {output}")
    source_size = source.stat().st_size
    if source_size > MAX_SOURCE_BYTES:
        raise ValueError(f"source checkpoint exceeds {MAX_SOURCE_BYTES} bytes")

    # weights_only uses PyTorch's restricted checkpoint loader and cannot
    # execute arbitrary globals from the training artifact.
    decoded = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(decoded, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    metadata = decoded.get("metadata")
    spec = decoded.get("spec")
    if not isinstance(metadata, Mapping) or not isinstance(spec, Mapping):
        raise ValueError("checkpoint metadata/spec must be mappings")
    progress_key = "iteration" if "iteration" in decoded else "epoch"
    progress = decoded.get(progress_key)
    if isinstance(progress, bool) or not isinstance(progress, int):
        raise ValueError("checkpoint must contain an integer iteration or epoch")

    compact = {
        "model_state_dict": _tensor_mapping(
            decoded.get("model_state_dict"), "model_state_dict"
        ),
        "normalization": _tensor_mapping(
            decoded.get("normalization"), "normalization"
        ),
        "metadata": dict(metadata),
        "spec": dict(spec),
        progress_key: int(progress),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".partial", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(compact, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "source": str(source),
        "source_bytes": source_size,
        "source_sha256": _sha256(source),
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": _sha256(output),
        "model_tensors": len(compact["model_state_dict"]),
        "normalization_tensors": len(compact["normalization"]),
        "progress_key": progress_key,
        "progress": int(progress),
        "removed_training_sections": sorted(
            key
            for key in decoded
            if key not in compact
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                export_inference_checkpoint(args.source, args.output),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"inference checkpoint export: REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
