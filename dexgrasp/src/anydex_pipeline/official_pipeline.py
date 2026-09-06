"""End-to-end, robot-free AnyDexGrasp adapter for calibrated observations."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from .frames import change_pose_reference, inverse_transform, transform_points
from .inspire_decision import OfficialInspireDecisionBackend
from .inspire_mapping import load_inspire_mapping, map_two_finger_grasps
from .official_backend import OfficialRepresentationBackend
from .types import GraspCandidate, GraspResult, PointCloudObservation


_DECISION_OVERSAMPLE_FACTOR = 16
_NEAR_DUPLICATE_TRANSLATION_M = 0.001
_NEAR_DUPLICATE_ROTATION_RAD = np.deg2rad(1.0)
_NEAR_DUPLICATE_SCALAR_M = 0.001
_NEAR_DUPLICATE_POLICY = (
    "same_type_target_pose_1mm_1deg_width_depth_1mm_v1"
)
_DEDUP_IDENTITY_METADATA_KEYS = (
    "representation_checkpoint_sha256",
    "decision_checkpoint_sha256s",
    "official_source_commit",
    "pose_frame_before_calibration",
    "collision_checked",
    "collision_check",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_GIT_OBJECT_ID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def _git_head_commit(repository: Path) -> str:
    """Return the exact checked-out Git object ID, or fail closed."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(
            f"cannot resolve official AnyDexGrasp source commit: {repository}"
        ) from exc
    commit = completed.stdout.strip().lower()
    if not _GIT_OBJECT_ID_RE.fullmatch(commit):
        raise ValueError("official AnyDexGrasp source commit is malformed")
    return commit


def _decision_checkpoint_paths(model_dir: Path) -> tuple[Path, ...]:
    root = model_dir / "480" if (model_dir / "480").is_dir() else model_dir
    paths = []
    for grasp_type in range(1, 9):
        candidates = sorted((root / str(grasp_type)).glob("*.pth"))
        if len(candidates) != 1:
            raise ValueError(
                "expected exactly one decision checkpoint for Inspire type "
                f"{grasp_type}, found {len(candidates)}"
            )
        paths.append(candidates[0].resolve())
    return tuple(paths)


def _strict_positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be >= 1")
    return normalized


def _rotation_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(
        second, dtype=np.float64
    )
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def _poses_near(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(
        np.linalg.norm(first[:3, 3] - second[:3, 3])
        <= _NEAR_DUPLICATE_TRANSLATION_M
        and _rotation_distance_rad(first[:3, :3], second[:3, :3])
        <= _NEAR_DUPLICATE_ROTATION_RAD
    )


def _scalars_near(first: float, second: float) -> bool:
    if np.isnan(first) or np.isnan(second):
        return bool(np.isnan(first) and np.isnan(second))
    return bool(abs(float(first) - float(second)) <= _NEAR_DUPLICATE_SCALAR_M)


def _target_key(candidate: GraspCandidate) -> tuple[object, ...] | None:
    if candidate.hand_angles is None:
        return None
    return (
        int(candidate.grasp_type_id),
        *tuple(float(value) for value in candidate.hand_angles),
    )


def _same_dedup_identity(first: GraspCandidate, second: GraspCandidate) -> bool:
    if _target_key(first) != _target_key(second):
        return False
    if first.T_reference_hand is None or second.T_reference_hand is None:
        return False
    if bool(first.collision_free) != bool(second.collision_free):
        return False
    if any(
        first.metadata.get(key) != second.metadata.get(key)
        for key in _DEDUP_IDENTITY_METADATA_KEYS
    ):
        return False
    return bool(
        _poses_near(first.T_reference_grasp, second.T_reference_grasp)
        and _poses_near(first.T_reference_hand, second.T_reference_hand)
        and _scalars_near(first.width_m, second.width_m)
        and _scalars_near(first.depth_m, second.depth_m)
    )


def _suppress_near_duplicate_candidates(
    candidates: list[GraspCandidate] | tuple[GraspCandidate, ...],
    *,
    max_candidates: int | None = None,
) -> tuple[GraspCandidate, ...]:
    """Stable, provenance-preserving NMS for mapped Inspire candidates."""

    if max_candidates is not None:
        max_candidates = _strict_positive_int(max_candidates, "max_candidates")
    values = tuple(candidates)
    if not values:
        return ()

    priority = np.argsort(
        -np.asarray([candidate.score for candidate in values], dtype=np.float64),
        kind="stable",
    )
    kept: list[GraspCandidate] = []
    clusters: list[list[tuple[int, GraspCandidate]]] = []
    cluster_indices_by_target: dict[tuple[object, ...], list[int]] = {}

    for original_index in priority:
        candidate = values[int(original_index)]
        target = _target_key(candidate)
        matched_cluster = None
        if target is not None:
            for cluster_index in cluster_indices_by_target.get(target, ()):
                if _same_dedup_identity(kept[cluster_index], candidate):
                    matched_cluster = cluster_index
                    break
        if matched_cluster is None:
            cluster_index = len(kept)
            kept.append(candidate)
            clusters.append([(int(original_index), candidate)])
            if target is not None:
                cluster_indices_by_target.setdefault(target, []).append(cluster_index)
        else:
            clusters[matched_cluster].append((int(original_index), candidate))

    annotated = []
    for winner, cluster in zip(kept, clusters):
        metadata = dict(winner.metadata)
        metadata.update(
            {
                "near_duplicate_suppression_policy": _NEAR_DUPLICATE_POLICY,
                "near_duplicate_cluster_size": len(cluster),
                "near_duplicate_source_indices": tuple(
                    int(candidate.source_index) for _, candidate in cluster
                ),
                "near_duplicate_decision_ranks": tuple(
                    int(index) for index, _ in cluster
                ),
                "near_duplicate_scores": tuple(
                    float(candidate.score) for _, candidate in cluster
                ),
            }
        )
        annotated.append(replace(winner, metadata=metadata))
    if max_candidates is not None:
        annotated = annotated[:max_candidates]
    return tuple(annotated)


class OfficialAnyDexBackend:
    """Representation + Inspire decision/mapping, with no hardware control.

    Collision checking is intentionally outside this first perception-only
    milestone because the released repository does not include the generated
    Inspire mesh point-cloud library.  Every returned candidate records that
    fact in metadata.
    """

    name = "AnyDexGrasp_official_inspire"

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        upstream_root: str | Path,
        inspire_model_dir: str | Path | None,
        device: str = "cuda:0",
        top_k: int = 10,
        representation_top_k: int = 1000,
        decision_score_threshold: float | None = None,
        trust_official_checkpoints: bool = False,
        min_approach_camera_z: float | None = 0.92,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.upstream_root = Path(upstream_root).expanduser().resolve()
        if inspire_model_dir is None:
            raise ValueError("official AnyDexGrasp requires inspire_model_dir")
        self.inspire_model_dir = Path(inspire_model_dir).expanduser().resolve()
        self.top_k = _strict_positive_int(top_k, "top_k")
        self.representation_top_k = _strict_positive_int(
            representation_top_k, "representation_top_k"
        )
        self.decision_score_threshold = decision_score_threshold
        self.mapping_path = (
            self.upstream_root
            / "generate_mesh_and_pointcloud/inspire_urdf/width_12Dangle_6Dangle.json"
        )
        if not isinstance(trust_official_checkpoints, (bool, np.bool_)):
            raise ValueError("trust_official_checkpoints must be boolean")
        if self.representation_top_k < self.top_k:
            raise ValueError("require representation_top_k >= top_k >= 1")
        self.representation = OfficialRepresentationBackend(
            self.checkpoint_path,
            anydex_repo=self.upstream_root,
            device=device,
            use_graspnet_v2=True,
            min_approach_camera_z=min_approach_camera_z,
            allow_unsafe_checkpoint=trust_official_checkpoints,
        )
        self.decision = OfficialInspireDecisionBackend(
            self.inspire_model_dir,
            anydex_repo=self.upstream_root,
            device=device,
            trust_checkpoints=trust_official_checkpoints,
        )
        self._checkpoint_sha256 = ""
        self._decision_checkpoint_sha256s: tuple[str, ...] = ()
        self._official_source_commit = ""

    def _model_provenance(self) -> tuple[str, tuple[str, ...], str]:
        if not self._checkpoint_sha256:
            self._checkpoint_sha256 = _sha256(self.checkpoint_path)
        if not self._decision_checkpoint_sha256s:
            self._decision_checkpoint_sha256s = tuple(
                _sha256(path)
                for path in _decision_checkpoint_paths(self.inspire_model_dir)
            )
        if not self._official_source_commit:
            self._official_source_commit = _git_head_commit(self.upstream_root)
        return (
            self._checkpoint_sha256,
            self._decision_checkpoint_sha256s,
            self._official_source_commit,
        )

    def infer(self, observation: PointCloudObservation) -> GraspResult:
        started = time.perf_counter()
        representation_sha256, decision_sha256s, source_commit = (
            self._model_provenance()
        )
        T_reference_camera = observation.T_reference_camera
        T_camera_reference = inverse_transform(T_reference_camera)
        object_camera = transform_points(T_camera_reference, observation.object_points)
        representation = self.representation.infer(
            object_camera,
            max_grasps=self.representation_top_k,
            sort_by_score=True,
        )
        if len(representation) == 0:
            return GraspResult(
                candidates=(),
                backend_name=self.name,
                reference_frame=observation.reference_frame,
                inference_time_s=time.perf_counter() - started,
                model_name="AnyDexGrasp representation + Inspire decision",
                checkpoint_sha256=representation_sha256,
                inference_points=observation.object_points,
            )
        decision = self.decision.infer(
            representation,
            top_k=self.top_k * _DECISION_OVERSAMPLE_FACTOR,
            score_threshold=self.decision_score_threshold,
        )
        if len(decision) == 0:
            return GraspResult(
                candidates=(),
                backend_name=self.name,
                reference_frame=observation.reference_frame,
                inference_time_s=time.perf_counter() - started,
                model_name="AnyDexGrasp representation + Inspire decision",
                checkpoint_sha256=representation_sha256,
                inference_points=transform_points(
                    T_reference_camera, representation.voxel_points_camera
                ),
            )

        selected_two_finger = representation.grasps[decision.source_indices]
        inspire = map_two_finger_grasps(
            selected_two_finger,
            decision.grasp_types,
            load_inspire_mapping(self.mapping_path),
            scores=decision.scores,
            depth_offsets=decision.depth_offsets_m,
        )
        hand_camera = inspire.pose_matrices(apply_depth=True)
        candidates = []
        for index, source_index in enumerate(decision.source_indices):
            two_finger = selected_two_finger[index]
            canonical_camera = np.eye(4, dtype=np.float64)
            canonical_camera[:3, :3] = two_finger[4:13].reshape(3, 3)
            canonical_camera[:3, 3] = two_finger[13:16]
            canonical_reference = change_pose_reference(
                T_reference_camera, canonical_camera
            )
            hand_reference = change_pose_reference(
                T_reference_camera, hand_camera[index]
            )
            candidates.append(
                GraspCandidate(
                    T_reference_grasp=canonical_reference,
                    T_reference_hand=hand_reference,
                    hand_angles=inspire.angles[index],
                    score=float(decision.scores[index]),
                    width_m=float(inspire.widths[index]),
                    depth_m=float(inspire.depths[index]),
                    collision_free=False,
                    grasp_type_id=int(inspire.grasp_types[index]),
                    source_index=int(source_index),
                    metadata={
                        "decision_rank": index,
                        "representation_score": float(two_finger[0]),
                        "collision_check": "not_run_missing_generated_inspire_mesh_clouds",
                        "collision_checked": False,
                        "representation_checkpoint_sha256": representation_sha256,
                        "decision_checkpoint_sha256s": decision_sha256s,
                        "official_source_commit": source_commit,
                        "pose_frame_before_calibration": "camera_color_optical_frame",
                    },
                )
            )
        candidates = _suppress_near_duplicate_candidates(
            candidates, max_candidates=self.top_k
        )
        return GraspResult(
            candidates=candidates,
            backend_name=self.name,
            reference_frame=observation.reference_frame,
            inference_time_s=time.perf_counter() - started,
            selected_index=0,
            model_name="AnyDexGrasp official representation + Inspire obj140 decision",
            checkpoint_sha256=representation_sha256,
            inference_points=transform_points(
                T_reference_camera, representation.voxel_points_camera
            ),
        )
