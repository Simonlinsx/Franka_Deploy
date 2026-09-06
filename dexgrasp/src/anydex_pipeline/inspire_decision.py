"""Lazy wrapper for the released AnyDexGrasp Inspire decision models."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .official_backend import (
    OFFICIAL_NO_FLIP_FEATURE_WIDTH,
    OfficialBackendUnavailable,
    OfficialCheckpointError,
    RepresentationFeatureLayout,
    RepresentationGraspBatch,
    default_anydex_repo,
)


@dataclass(frozen=True)
class InspireDecisionBatch:
    source_indices: np.ndarray
    grasp_types: np.ndarray
    depth_offsets_m: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(value) for value in (
            self.source_indices,
            self.grasp_types,
            self.depth_offsets_m,
            self.scores,
        ))
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("decision outputs must be one-dimensional")
        if len({len(array) for array in arrays}) != 1:
            raise ValueError("decision outputs must have the same length")
        if len(arrays[0]) and not np.all((arrays[1] >= 1) & (arrays[1] <= 8)):
            raise ValueError("grasp type must be in [1, 8]")

    def __len__(self) -> int:
        return len(self.scores)


class OfficialInspireDecisionBackend:
    """Rank representation candidates using the eight released Inspire heads.

    The released ``.pth`` files contain serialized module objects rather than
    plain state dictionaries.  Loading them with PyTorch 1.13 therefore uses
    pickle.  This wrapper refuses to do so unless ``trust_checkpoints=True`` is
    explicitly supplied by the caller.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        anydex_repo: str | Path | None = None,
        device: str | None = None,
        trust_checkpoints: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.anydex_repo = Path(anydex_repo or default_anydex_repo()).expanduser().resolve()
        self.requested_device = device
        self.trust_checkpoints = bool(trust_checkpoints)
        self._torch: Any | None = None
        self._device: Any | None = None
        self._models: list[Any] = []

    @property
    def loaded(self) -> bool:
        return len(self._models) == 8

    def load(self) -> "OfficialInspireDecisionBackend":
        if self.loaded:
            return self
        if not self.trust_checkpoints:
            raise OfficialCheckpointError(
                "The released Inspire .pth files are pickle-backed module objects. "
                "Pass trust_checkpoints=True only after obtaining them from the "
                "official AnyDexGrasp folder."
            )
        root = self.model_dir
        if (root / "480").is_dir():
            root = root / "480"
        if not root.is_dir():
            raise OfficialCheckpointError(f"Inspire decision model directory not found: {root}")

        for path in (self.anydex_repo / "models", self.anydex_repo):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        try:
            torch = importlib.import_module("torch")
            model_module = importlib.import_module("models.minkowski_graspnet_single_point")
        except Exception as exc:
            raise OfficialBackendUnavailable(
                "Could not import PyTorch/official Inspire decision model class"
            ) from exc
        device_name = self.requested_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_name)

        models: list[Any] = []
        for grasp_type in range(1, 9):
            candidates = sorted((root / str(grasp_type)).glob("*.pth"))
            if len(candidates) != 1:
                raise OfficialCheckpointError(
                    f"expected exactly one .pth for Inspire type {grasp_type} in "
                    f"{root / str(grasp_type)}, found {len(candidates)}"
                )
            target = model_module.MinkowskiGraspNetMultifingerType1Inference(input_num=480)
            try:
                try:
                    saved = torch.load(
                        str(candidates[0]), map_location="cpu", weights_only=False
                    )
                except TypeError:
                    # PyTorch 1.13 has no weights_only argument.
                    saved = torch.load(str(candidates[0]), map_location="cpu")
                state_dict = saved.state_dict() if hasattr(saved, "state_dict") else saved
                target.load_state_dict(state_dict)
                target.to(device)
                target.eval()
            except Exception as exc:
                raise OfficialCheckpointError(
                    f"failed to load Inspire type {grasp_type}: {candidates[0]}"
                ) from exc
            models.append(target)
        self._torch = torch
        self._device = device
        self._models = models
        return self

    @staticmethod
    def _decision_features(
        representation: RepresentationGraspBatch,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reproduce upstream 480-D rotation normalization and depth classes.

        This accepts the batch, rather than a bare matrix, so the versioned
        feature layout cannot be discarded between representation and decision
        inference.
        """

        if (
            representation.feature_layout
            is not RepresentationFeatureLayout.OFFICIAL_RIGHT_HAND_NO_FLIP_V1
        ):
            raise ValueError(
                "Inspire decision requires official_right_hand_no_flip_v1 "
                "representation features"
            )
        values = np.asarray(representation.features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != OFFICIAL_NO_FLIP_FEATURE_WIDTH:
            raise ValueError(
                "official_right_hand_no_flip_v1 representation features must have shape "
                f"(N, {OFFICIAL_NO_FLIP_FEATURE_WIDTH})"
            )
        model_input = values[:, 240:720].copy()
        # official_right_hand_no_flip_v1 tail:
        # view_index, view_score, seed_index, paired_angle_class, depth_m.
        angle_classes = np.rint(values[:, -2]).astype(np.int64)
        depth_classes = np.rint(values[:, -1] * 100.0).astype(np.int64)
        if np.any((angle_classes < 0) | (angle_classes >= 24)):
            raise ValueError("representation paired angle class is outside [0, 23]")
        if np.any((depth_classes < 0) | (depth_classes >= 5)):
            raise ValueError("representation depth class is outside [0, 4]")

        rotated = np.empty_like(model_input)
        # Each angle contributes five depth values.  Rotate score and width
        # halves independently so the selected angle becomes class zero.
        for index, angle in enumerate(angle_classes):
            offset = int(angle) * 5
            rotated[index, :240] = np.roll(model_input[index, :240], -offset)
            rotated[index, 240:] = np.roll(model_input[index, 240:], -offset)
        return rotated, depth_classes

    def infer(
        self,
        representation: RepresentationGraspBatch,
        *,
        top_k: int = 10,
        score_threshold: float | None = None,
    ) -> InspireDecisionBatch:
        if not self.loaded:
            self.load()
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if len(representation) == 0:
            return InspireDecisionBatch(
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )
        model_input, depth_classes = self._decision_features(representation)
        torch = self._torch
        assert torch is not None and self._device is not None
        tensor = torch.as_tensor(model_input, device=self._device)
        depth_tensor = torch.as_tensor(depth_classes, device=self._device, dtype=torch.long)
        base = torch.arange(4, device=self._device).view(1, 4)

        per_type = []
        with torch.no_grad():
            for model in self._models:
                prediction, _ = model(tensor)
                prediction = prediction.reshape(len(representation), 20)
                indices = depth_tensor.view(-1, 1) * 4 + base
                per_type.append(prediction.gather(1, indices))
        scores_matrix = torch.stack(per_type, dim=1)  # [candidate, type, depth]
        flat = scores_matrix.reshape(-1)
        keep = min(int(top_k), int(flat.numel()))
        scores, flat_indices = torch.topk(flat, keep)
        candidate_stride = 8 * 4
        source = torch.div(flat_indices, candidate_stride, rounding_mode="floor")
        remainder = flat_indices % candidate_stride
        type_zero = torch.div(remainder, 4, rounding_mode="floor")
        depth_zero = remainder % 4

        scores_np = scores.detach().cpu().numpy().astype(np.float32, copy=False)
        source_np = source.detach().cpu().numpy().astype(np.int64, copy=False)
        type_np = (type_zero.detach().cpu().numpy() + 1).astype(np.int32, copy=False)
        depth_np = depth_zero.detach().cpu().numpy().astype(np.float32, copy=False)
        # Upstream special case: zero-based type 5 == public Inspire type 6.
        depth_np[type_np == 6] += 2.0
        depth_np *= 0.01

        if score_threshold is not None:
            mask = scores_np >= float(score_threshold)
            scores_np = scores_np[mask]
            source_np = source_np[mask]
            type_np = type_np[mask]
            depth_np = depth_np[mask]
        return InspireDecisionBatch(source_np, type_np, depth_np, scores_np)
