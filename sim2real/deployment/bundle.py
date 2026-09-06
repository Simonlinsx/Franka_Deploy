"""Read and verify the supplied V94 deployment bundle without extracting it.

The outer handoff and the PyTorch checkpoint are both ZIP containers.  This
module treats both as untrusted structured data: paths are checked, every
manifest hash can be verified, and checkpoint metadata is decoded with a very
small allow-list unpickler that cannot import arbitrary application classes.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import pickle
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple
import zipfile

import numpy as np

BUNDLE_CONTRACT = "inspire_v94_rgbd_student_sim2real_handoff_v1"
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return path


def _json_object(data: bytes, name: str) -> Dict[str, Any]:
    if len(data) > MAX_JSON_BYTES:
        raise ValueError(f"{name} is unexpectedly large")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON number in {name}: {value}")

    try:
        value = json.loads(data.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _hash_entries(value: Any) -> Iterator[Tuple[str, str, Optional[int]]]:
    if isinstance(value, Mapping):
        path = value.get("path")
        digest = value.get("sha256")
        byte_count = value.get("bytes")
        if isinstance(path, str) and isinstance(digest, str):
            yield path, digest, int(byte_count) if byte_count is not None else None
        for child in value.values():
            yield from _hash_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from _hash_entries(child)


@dataclass(frozen=True)
class BundleVerification:
    bundle_contract: str
    checked_files: int
    root: str
    primary_checkpoint_sha256: str


class DeployBundle:
    """Random-access view of one deployment bundle inside an outer ZIP."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"deployment bundle not found: {self.path}")
        try:
            with zipfile.ZipFile(self.path, "r") as archive:
                names = archive.namelist()
                for name in names:
                    _safe_member(name)
                candidates = [
                    name
                    for name in names
                    if name.endswith("/manifest.json")
                    and not name.startswith("__MACOSX/")
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "deployment ZIP must contain exactly one manifest.json"
                    )
                manifest_name = candidates[0]
                self.root = manifest_name[: -len("manifest.json")]
                self.manifest = _json_object(archive.read(manifest_name), manifest_name)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"invalid deployment ZIP {self.path}: {exc}") from exc
        if self.manifest.get("bundle_contract") != BUNDLE_CONTRACT:
            raise ValueError(
                "unsupported deployment bundle contract: "
                f"{self.manifest.get('bundle_contract')!r}"
            )

    def _member(self, relative_path: str) -> str:
        relative = _safe_member(relative_path)
        if str(relative) in (".", ""):
            raise ValueError("bundle relative path cannot be empty")
        return self.root + relative.as_posix()

    def read_bytes(
        self, relative_path: str, *, max_bytes: Optional[int] = None
    ) -> bytes:
        member = self._member(relative_path)
        try:
            with zipfile.ZipFile(self.path, "r") as archive:
                info = archive.getinfo(member)
                if max_bytes is not None and info.file_size > int(max_bytes):
                    raise ValueError(
                        f"bundle member {relative_path} exceeds {int(max_bytes)} bytes"
                    )
                return archive.read(info)
        except KeyError as exc:
            raise FileNotFoundError(
                f"bundle member is missing: {relative_path}"
            ) from exc
        except zipfile.BadZipFile as exc:
            raise ValueError(f"invalid deployment ZIP {self.path}: {exc}") from exc

    def read_json(self, relative_path: str) -> Dict[str, Any]:
        return _json_object(
            self.read_bytes(relative_path, max_bytes=MAX_JSON_BYTES), relative_path
        )

    def load_npz(self, relative_path: str) -> Dict[str, np.ndarray]:
        payload = self.read_bytes(relative_path, max_bytes=128 * 1024 * 1024)
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                return {name: archive[name].copy() for name in archive.files}
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid safe NPZ {relative_path}: {exc}") from exc

    def checkpoint_bytes(self, *, fallback: bool = False) -> bytes:
        section = "fallback_checkpoint" if fallback else "primary_checkpoint"
        entry = self.manifest.get(section)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ValueError(f"manifest has no valid {section}")
        return self.read_bytes(str(entry["path"]), max_bytes=MAX_CHECKPOINT_BYTES)

    def verify(self) -> BundleVerification:
        expected_contract = str(self.manifest["bundle_contract"])
        unique: Dict[str, Tuple[str, Optional[int]]] = {}
        for relative, expected_hash, expected_bytes in _hash_entries(self.manifest):
            if len(expected_hash) != 64:
                raise ValueError(f"invalid SHA-256 in manifest for {relative}")
            previous = unique.get(relative)
            current = (expected_hash.lower(), expected_bytes)
            if previous is not None and previous != current:
                raise ValueError(f"conflicting manifest hashes for {relative}")
            unique[relative] = current

        for relative, (expected_hash, expected_bytes) in unique.items():
            payload = self.read_bytes(relative)
            if expected_bytes is not None and len(payload) != expected_bytes:
                raise ValueError(
                    f"bundle size mismatch for {relative}: "
                    f"expected={expected_bytes}, actual={len(payload)}"
                )
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected_hash:
                raise ValueError(
                    f"bundle SHA-256 mismatch for {relative}: "
                    f"expected={expected_hash}, actual={actual}"
                )

        primary = self.manifest.get("primary_checkpoint")
        if not isinstance(primary, Mapping):
            raise ValueError("manifest primary_checkpoint is invalid")
        return BundleVerification(
            bundle_contract=expected_contract,
            checked_files=len(unique),
            root=self.root.rstrip("/"),
            primary_checkpoint_sha256=str(primary["sha256"]),
        )


@dataclass(frozen=True)
class _TensorRef:
    storage_key: str
    dtype: np.dtype
    storage_size: int
    offset: int
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]


_STORAGE_DTYPES = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ByteStorage": np.dtype("u1"),
    "BoolStorage": np.dtype("?"),
}


def _storage_marker(name: str) -> Tuple[str, str]:
    return ("storage_type", name)


def _rebuild_tensor(
    storage: Any,
    offset: Any,
    shape: Any,
    stride: Any,
    requires_grad: Any = False,
    hooks: Any = None,
) -> _TensorRef:
    del requires_grad, hooks
    if (
        not isinstance(storage, tuple)
        or len(storage) < 5
        or storage[0] != "storage"
        or not isinstance(storage[1], tuple)
        or storage[1][0] != "storage_type"
    ):
        raise pickle.UnpicklingError("invalid checkpoint tensor storage reference")
    storage_name = str(storage[1][1])
    dtype = _STORAGE_DTYPES.get(storage_name)
    if dtype is None:
        raise pickle.UnpicklingError(
            f"unsupported checkpoint storage type {storage_name}"
        )
    try:
        result = _TensorRef(
            storage_key=str(storage[2]),
            dtype=dtype,
            storage_size=int(storage[4]),
            offset=int(offset),
            shape=tuple(int(value) for value in shape),
            stride=tuple(int(value) for value in stride),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise pickle.UnpicklingError("invalid checkpoint tensor metadata") from exc
    if result.offset < 0 or result.storage_size < 0:
        raise pickle.UnpicklingError("negative checkpoint tensor bounds")
    if len(result.shape) != len(result.stride) or any(
        value < 0 for value in result.shape
    ):
        raise pickle.UnpicklingError("invalid checkpoint tensor shape")
    return result


class _RestrictedCheckpointUnpickler(pickle.Unpickler):
    """Allow only the primitives emitted by torch.save(state_dict)."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if module == "torch._utils" and name in {
            "_rebuild_tensor",
            "_rebuild_tensor_v2",
        }:
            return _rebuild_tensor
        if module == "torch" and name in _STORAGE_DTYPES:
            return _storage_marker(name)
        raise pickle.UnpicklingError(f"forbidden checkpoint global {module}.{name}")

    def persistent_load(self, persistent_id: Any) -> Any:
        if not isinstance(persistent_id, tuple) or not persistent_id:
            raise pickle.UnpicklingError("invalid checkpoint persistent ID")
        if persistent_id[0] != "storage":
            raise pickle.UnpicklingError(
                f"unsupported checkpoint persistent ID {persistent_id[0]!r}"
            )
        return persistent_id


@dataclass(frozen=True)
class CheckpointData:
    model_state_dict: Mapping[str, np.ndarray]
    ppo_state_dict: Mapping[str, np.ndarray]
    normalization: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    spec: Mapping[str, Any]
    iteration: int


def _contiguous_stride(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    result = []
    stride = 1
    for size in reversed(shape):
        result.append(stride)
        stride *= max(1, int(size))
    return tuple(reversed(result))


def load_checkpoint_safely(payload: bytes) -> CheckpointData:
    """Decode tensor weights without importing torch or executing checkpoint code."""

    if not isinstance(payload, bytes) or len(payload) > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint payload is missing or unexpectedly large")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            names = archive.namelist()
            for name in names:
                _safe_member(name)
            pickle_names = [name for name in names if name.endswith("/data.pkl")]
            if len(pickle_names) != 1:
                raise ValueError("checkpoint must contain exactly one data.pkl")
            pickle_name = pickle_names[0]
            prefix = pickle_name[: -len("data.pkl")]
            raw = archive.read(pickle_name)
            decoded = _RestrictedCheckpointUnpickler(io.BytesIO(raw)).load()
            if not isinstance(decoded, Mapping):
                raise ValueError("checkpoint root must be a mapping")

            storage_cache: Dict[Tuple[str, str], np.ndarray] = {}

            def tensor(ref: _TensorRef) -> np.ndarray:
                if not isinstance(ref, _TensorRef):
                    raise ValueError("checkpoint state contains a non-tensor value")
                if any(stride < 0 for stride in ref.stride):
                    raise ValueError("negative checkpoint tensor stride is unsupported")
                count = math.prod(ref.shape)
                if count == 0:
                    if ref.offset > ref.storage_size:
                        raise ValueError("checkpoint tensor exceeds its storage")
                else:
                    maximum_index = ref.offset + sum(
                        (size - 1) * stride
                        for size, stride in zip(ref.shape, ref.stride)
                    )
                    if ref.offset >= ref.storage_size or maximum_index >= ref.storage_size:
                        raise ValueError("checkpoint tensor exceeds its storage")
                cache_key = (ref.storage_key, ref.dtype.str)
                storage = storage_cache.get(cache_key)
                if storage is None:
                    member = prefix + "data/" + ref.storage_key
                    try:
                        storage_bytes = archive.read(member)
                    except KeyError as exc:
                        raise ValueError(
                            f"checkpoint storage is missing: {ref.storage_key}"
                        ) from exc
                    expected = ref.storage_size * ref.dtype.itemsize
                    if len(storage_bytes) != expected:
                        raise ValueError(
                            f"checkpoint storage {ref.storage_key} size mismatch"
                        )
                    storage = np.frombuffer(storage_bytes, dtype=ref.dtype)
                    storage_cache[cache_key] = storage
                if count == 0:
                    return np.empty(ref.shape, dtype=ref.dtype)
                if ref.stride == _contiguous_stride(ref.shape):
                    return (
                        storage[ref.offset : ref.offset + count]
                        .reshape(ref.shape)
                        .copy()
                    )
                # PyTorch commonly saves broadcast normalizers with a zero
                # stride.  Bounds were proven above, so materializing the
                # read-only strided view into a contiguous NumPy array is safe.
                view = np.lib.stride_tricks.as_strided(
                    storage[ref.offset :],
                    shape=ref.shape,
                    strides=tuple(
                        stride * ref.dtype.itemsize for stride in ref.stride
                    ),
                    writeable=False,
                )
                return np.array(view, dtype=ref.dtype, copy=True, order="C")

            def tensor_mapping(name: str) -> Dict[str, np.ndarray]:
                section = decoded.get(name)
                if not isinstance(section, Mapping):
                    raise ValueError(f"checkpoint {name} must be a mapping")
                return {str(key): tensor(value) for key, value in section.items()}

            def optional_tensor_mapping(name: str) -> Dict[str, np.ndarray]:
                section = decoded.get(name)
                if section is None:
                    return {}
                if not isinstance(section, Mapping):
                    raise ValueError(f"checkpoint {name} must be a mapping")
                return {str(key): tensor(value) for key, value in section.items()}

            metadata = decoded.get("metadata")
            spec = decoded.get("spec")
            if not isinstance(metadata, Mapping) or not isinstance(spec, Mapping):
                raise ValueError("checkpoint metadata/spec must be mappings")
            progress = decoded.get("iteration", decoded.get("epoch"))
            if isinstance(progress, bool) or not isinstance(progress, int):
                raise ValueError(
                    "checkpoint must contain an integer iteration or epoch"
                )
            return CheckpointData(
                model_state_dict=tensor_mapping("model_state_dict"),
                # Offline student-pretrain exports contain everything needed
                # for deterministic deployment inference, but intentionally
                # omit the PPO optimizer/state section.  Keep rejecting a
                # malformed section when present while accepting its absence.
                ppo_state_dict=optional_tensor_mapping("ppo_state_dict"),
                normalization=tensor_mapping("normalization"),
                metadata=dict(metadata),
                spec=dict(spec),
                # PPO exports use ``iteration``; student-pretrain exports use
                # ``epoch``.  This field is provenance only and never changes
                # the forward pass.
                iteration=int(progress),
            )
    except (zipfile.BadZipFile, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f"invalid or unsafe checkpoint: {exc}") from exc


__all__ = [
    "BUNDLE_CONTRACT",
    "MAX_CHECKPOINT_BYTES",
    "BundleVerification",
    "CheckpointData",
    "DeployBundle",
    "load_checkpoint_safely",
]
