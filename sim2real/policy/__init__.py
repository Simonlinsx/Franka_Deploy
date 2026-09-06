"""NumPy inference for the V94 rolling-student checkpoint.

Keeping the deployment forward pass NumPy-only makes bundle verification and
CPU dry-runs possible in the repository's hardware environment, where PyTorch
is intentionally not a dependency.  All operations mirror the checkpoint's
small SiLU MLP + one-layer LSTM architecture and remain float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import numpy as np

from sim2real.deployment.bundle import CheckpointData
from sim2real.tasks.ballistics import (
    CONTRACT_DIM as ANALYTIC_FUTURE_CONTRACT_DIM,
    SUPPORTED_17D_CONTRACTS,
    build_17d_future_contract,
)

EXPECTED_SPEC = {
    "num_object_points": 128,
    "action_dim": 13,
    "arm_dim": 7,
    "hand_dim": 6,
    "geometric_summary": True,
    "temporal_encoder": "lstm",
    "temporal_point_frame": "native_palm",
    "temporal_hidden_dim": 256,
    "temporal_layers": 1,
    "privileged_action_conditioning": True,
    "object_memory_encoder": "none",
    "future_motion_horizons": 4,
    "future_motion_streams": 1,
    "future_motion_action_conditioning": True,
}

SUPPORTED_POINT_FEATURE_MODES = {3: "xyz", 6: "xyzrgb"}

LEGACY_ACTION_CONTROLLER_CONTRACT_ID = "v94_inspire_semantic_13d"
LEGACY_ACTION_CONTROLLER_METADATA_SCHEMA = "joint_target_action_adapter_v1"
QD_G015_ACTION_CONTROLLER_CONTRACT_ID = (
    "franka_inspire_qd_g015_student_controller_v1"
)

_LEGACY_INITIAL_PREVIOUS_ACTION13 = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
)
_ZERO_INITIAL_PREVIOUS_ACTION13 = (0.0,) * 13


def _finite_float32(value: np.ndarray, shape: Tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape {shape} with finite values")
    return result


def _sigmoid(value: np.ndarray) -> np.ndarray:
    # Clipping avoids overflow while matching float32 sigmoid to far beyond the
    # packaged replay tolerance.
    clipped = np.clip(value, np.float32(-40.0), np.float32(40.0))
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-clipped))


def _silu(value: np.ndarray) -> np.ndarray:
    return value * _sigmoid(value)


@dataclass(frozen=True)
class PolicyOutput:
    action13: np.ndarray
    predicted_privileged32: np.ndarray
    future_motion24: np.ndarray
    predicted_hold6: np.ndarray
    predicted_hold_logit: float


@dataclass(frozen=True)
class ActionControllerParameters:
    """Checkpoint-owned normalized-action to simulator-target contract."""

    arm_raw_gain_rad: float
    arm_target_filter_alpha: float
    maximum_arm_target_step_rad: float
    hand_target_filter_alpha: float
    maximum_hand_target_step_rad: float
    contract_id: str
    initial_previous_action13: tuple[float, ...]

    @staticmethod
    def _is_qd_g015_controller(controller: object) -> bool:
        """Recognize the exporter marker used by early V75/V76 checkpoints.

        The first frozen q_d-g015 exports retained the generic
        ``joint_target_action_adapter_v1`` schema and placed the discriminating
        contract in ``arm.incremental_reference`` plus its exact numeric
        parameters.  Treat that complete tuple as an explicit contract marker;
        a partial or numerically different tuple must stay on the legacy path
        and fail its legacy validation instead of being guessed as q_d-g015.
        """

        if not isinstance(controller, Mapping):
            return False
        arm = controller.get("arm")
        hand = controller.get("hand")
        if not isinstance(arm, Mapping) or not isinstance(hand, Mapping):
            return False
        if arm.get("incremental_reference") not in {
            "shaper_q_d",
            "current_shaper_q_d",
        }:
            return False

        def exact(section: Mapping[str, Any], key: str, expected: float) -> bool:
            value = section.get(key)
            if isinstance(value, (bool, np.bool_)):
                return False
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError):
                return False
            return bool(np.isfinite(numeric) and numeric == expected)

        return bool(
            controller.get("schema") == LEGACY_ACTION_CONTROLLER_METADATA_SCHEMA
            and controller.get("target_update_clock") == "policy"
            and arm.get("dimensions") == 7
            and arm.get("semantics") == "incremental_joint_target"
            and hand.get("dimensions") == 6
            and hand.get("semantics") == "absolute_physical_motor_target"
            and exact(controller, "control_dt_s", 0.05)
            and exact(controller, "policy_frequency_hz", 20.0)
            and exact(controller, "physics_hold_substeps", 6.0)
            and exact(arm, "delta_scale_rad_per_policy_step", 0.15)
            and exact(arm, "moving_average", 0.40)
            and exact(
                arm,
                "effective_delta_scale_after_ema_rad_per_policy_step",
                0.06,
            )
            and exact(arm, "max_target_delta_rad_per_policy_step", 0.0)
            and exact(arm, "tracking_error_limit_rad", 0.0)
            and exact(hand, "moving_average", 0.737856)
            and exact(hand, "max_target_delta_rad_per_policy_step", 0.30)
        )

    @staticmethod
    def _metadata_contract_id(
        metadata: Mapping[str, Any], controller: object
    ) -> str:
        """Resolve the checkpoint-owned controller contract identifier.

        New exports may put the identifier directly in
        ``action_controller.schema`` or retain the legacy action-controller
        block and add a top-level controller-contract identifier.  Supporting
        both layouts keeps the runtime compatible with the two exporter
        variants without guessing from a checkpoint filename or dimensions.
        """

        if ActionControllerParameters._is_qd_g015_controller(controller):
            return QD_G015_ACTION_CONTROLLER_CONTRACT_ID

        candidates: list[object] = []
        if isinstance(controller, Mapping):
            candidates.extend(
                [controller.get("contract_id"), controller.get("schema")]
            )
        candidates.extend(
            [
                metadata.get("controller_contract_id"),
                metadata.get("controller_contract"),
                metadata.get("franka_controller_contract"),
            ]
        )
        flattened: list[object] = []
        for value in candidates:
            if isinstance(value, Mapping):
                flattened.extend([value.get("schema"), value.get("contract")])
            else:
                flattened.append(value)
        resolved = {
            str(value).strip()
            for value in flattened
            if isinstance(value, str) and str(value).strip()
        }
        if LEGACY_ACTION_CONTROLLER_METADATA_SCHEMA in resolved:
            resolved.remove(LEGACY_ACTION_CONTROLLER_METADATA_SCHEMA)
            resolved.add(LEGACY_ACTION_CONTROLLER_CONTRACT_ID)
        supported = resolved.intersection(
            {
                LEGACY_ACTION_CONTROLLER_CONTRACT_ID,
                QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
            }
        )
        if QD_G015_ACTION_CONTROLLER_CONTRACT_ID in supported:
            return QD_G015_ACTION_CONTROLLER_CONTRACT_ID
        if supported == {LEGACY_ACTION_CONTROLLER_CONTRACT_ID}:
            return LEGACY_ACTION_CONTROLLER_CONTRACT_ID
        if resolved:
            raise ValueError(
                "unsupported checkpoint action-controller contract identifier: "
                + ", ".join(sorted(resolved))
            )
        return LEGACY_ACTION_CONTROLLER_CONTRACT_ID

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "ActionControllerParameters":
        controller = metadata.get("action_controller")
        contract_id = cls._metadata_contract_id(metadata, controller)
        if contract_id == QD_G015_ACTION_CONTROLLER_CONTRACT_ID:
            # The q_d-relative contract is frozen by the handoff.  These are
            # intentionally not inferred from the old measured/previous-target
            # adapter fields: q_cmd = current q_d + (0.15 * 0.40) * action.
            return cls(
                arm_raw_gain_rad=0.15,
                arm_target_filter_alpha=0.40,
                maximum_arm_target_step_rad=0.0,
                hand_target_filter_alpha=0.737856,
                maximum_hand_target_step_rad=0.30,
                contract_id=QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
                initial_previous_action13=_ZERO_INITIAL_PREVIOUS_ACTION13,
            )
        if controller is None:
            # Original V94 exports predate the explicit action-controller
            # metadata but have this frozen, golden-tested mapping.
            return cls(
                0.015,
                0.20,
                0.015,
                0.20,
                0.05,
                LEGACY_ACTION_CONTROLLER_CONTRACT_ID,
                _LEGACY_INITIAL_PREVIOUS_ACTION13,
            )
        if not isinstance(controller, Mapping):
            raise ValueError("checkpoint metadata.action_controller must be a mapping")
        arm = controller.get("arm")
        hand = controller.get("hand")
        if (
            controller.get("schema") != LEGACY_ACTION_CONTROLLER_METADATA_SCHEMA
            or not isinstance(arm, Mapping)
            or not isinstance(hand, Mapping)
            or arm.get("dimensions") != 7
            or hand.get("dimensions") != 6
            or arm.get("semantics") != "incremental_joint_target"
            or hand.get("semantics") != "absolute_physical_motor_target"
        ):
            raise ValueError("unsupported checkpoint action_controller contract")

        def positive(section: Mapping[str, Any], key: str) -> float:
            try:
                value = float(section[key])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"checkpoint action_controller {key} must be finite and positive"
                ) from exc
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"checkpoint action_controller {key} must be finite and positive"
                )
            return value

        arm_alpha = positive(arm, "moving_average")
        hand_alpha = positive(hand, "moving_average")
        if arm_alpha > 1.0 or hand_alpha > 1.0:
            raise ValueError("checkpoint action-controller EMA must not exceed one")
        return cls(
            arm_raw_gain_rad=positive(arm, "delta_scale_rad_per_policy_step"),
            arm_target_filter_alpha=arm_alpha,
            maximum_arm_target_step_rad=positive(
                arm, "max_target_delta_rad_per_policy_step"
            ),
            hand_target_filter_alpha=hand_alpha,
            maximum_hand_target_step_rad=positive(
                hand, "max_target_delta_rad_per_policy_step"
            ),
            contract_id=LEGACY_ACTION_CONTROLLER_CONTRACT_ID,
            initial_previous_action13=_LEGACY_INITIAL_PREVIOUS_ACTION13,
        )


class RollingStudentPolicy:
    """Deterministic action-mean implementation of the packaged V94 student."""

    # The LSTM hidden/cell arrays are recreated inside every ``act`` call.
    # Consequently the complete temporal state is the caller-owned frame
    # history, and a rejected proposal cannot mutate state inside this object.
    # The transactional C2 tick source requires this exact contract instead of
    # assuming that an arbitrary policy wrapper is rollback-safe.
    # Historical wire name retained for compatibility.  The same rollback
    # property holds for all supported external histories (4, 8 and 16 frames).
    transactional_state_contract = (
        "external_four_frame_history_lstm_recomputed_from_zero_v1"
    )

    def __init__(self, checkpoint: CheckpointData) -> None:
        for key, expected in EXPECTED_SPEC.items():
            if checkpoint.spec.get(key) != expected:
                raise ValueError(
                    f"checkpoint spec mismatch for {key}: "
                    f"expected={expected!r}, actual={checkpoint.spec.get(key)!r}"
                )
        history = checkpoint.spec.get("history")
        if (
            isinstance(history, bool)
            or not isinstance(history, int)
            or history not in (4, 8, 16)
        ):
            raise ValueError(
                "checkpoint spec history must be 4, 8 or 16, "
                f"actual={history!r}"
            )
        proprio_dim = checkpoint.spec.get("proprio_dim")
        if (
            isinstance(proprio_dim, bool)
            or not isinstance(proprio_dim, int)
            or proprio_dim not in (67, 96)
        ):
            raise ValueError(
                "checkpoint spec proprio_dim must be 67 or 96, "
                f"actual={proprio_dim!r}"
            )
        if (history, proprio_dim) not in (
            (4, 67),
            (8, 67),
            (8, 96),
            (16, 96),
        ):
            raise ValueError(
                "unsupported checkpoint history/proprio pair: "
                f"history={history}, proprio_dim={proprio_dim}"
            )
        self.history_length = int(history)
        self.proprio_dim = int(proprio_dim)
        point_feature_dim = checkpoint.spec.get("point_feature_dim")
        if (
            isinstance(point_feature_dim, bool)
            or not isinstance(point_feature_dim, int)
            or point_feature_dim not in SUPPORTED_POINT_FEATURE_MODES
        ):
            raise ValueError(
                "checkpoint spec point_feature_dim must be 3 (XYZ) or 6 "
                f"(XYZRGB), actual={point_feature_dim!r}"
            )
        self.point_feature_dim = int(point_feature_dim)
        self.point_feature_mode = SUPPORTED_POINT_FEATURE_MODES[self.point_feature_dim]
        compact_privileged_dim = checkpoint.spec.get("compact_privileged_dim")
        if (
            isinstance(compact_privileged_dim, bool)
            or not isinstance(compact_privileged_dim, int)
            or compact_privileged_dim not in (32, 33)
        ):
            raise ValueError(
                "checkpoint spec compact_privileged_dim must be 32 or 33, "
                f"actual={compact_privileged_dim!r}"
            )
        self.compact_privileged_dim = int(compact_privileged_dim)
        analytic_adapter_enabled = checkpoint.spec.get(
            "analytic_future_contract_action_adapter_enabled", False
        )
        if not isinstance(analytic_adapter_enabled, (bool, np.bool_)):
            raise ValueError(
                "checkpoint analytic_future_contract_action_adapter_enabled "
                "must be boolean"
            )
        self.analytic_future_contract_action_adapter_enabled = bool(
            analytic_adapter_enabled
        )
        self.analytic_future_contract: str | None = None
        self.analytic_future_contract_action_adapter_scale = np.float32(0.0)
        if self.analytic_future_contract_action_adapter_enabled:
            contract = checkpoint.spec.get("analytic_future_contract")
            if contract not in SUPPORTED_17D_CONTRACTS:
                raise ValueError(
                    "checkpoint analytic_future_contract must be one of "
                    f"{sorted(SUPPORTED_17D_CONTRACTS)}, actual={contract!r}"
                )
            required_adapter_spec = {
                "history": 16,
                "proprio_dim": 96,
                "point_feature_dim": 3,
                "compact_privileged_dim": 33,
                "action_chunk_size": 1,
                "analytic_future_contract_action_adapter_hidden_dim": 128,
                "analytic_future_contract_stop_gradient": True,
            }
            for name, expected in required_adapter_spec.items():
                if checkpoint.spec.get(name) != expected:
                    raise ValueError(
                        f"analytic future adapter requires {name}={expected!r}, "
                        f"actual={checkpoint.spec.get(name)!r}"
                    )
            scale = checkpoint.spec.get(
                "analytic_future_contract_action_adapter_scale"
            )
            if isinstance(scale, (bool, np.bool_)):
                raise ValueError("analytic future adapter scale must be exactly 1.0")
            try:
                numeric_scale = float(scale)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "analytic future adapter scale must be exactly 1.0"
                ) from exc
            if not np.isfinite(numeric_scale) or numeric_scale != 1.0:
                raise ValueError("analytic future adapter scale must be exactly 1.0")
            self.analytic_future_contract = str(contract)
            self.analytic_future_contract_action_adapter_scale = np.float32(
                numeric_scale
            )
        self.action_controller = ActionControllerParameters.from_metadata(
            checkpoint.metadata
        )
        history_proprio = (self.history_length, self.proprio_dim)
        if (
            self.action_controller.contract_id
            == QD_G015_ACTION_CONTROLLER_CONTRACT_ID
        ):
            if (
                self.history_length != 8
                or self.proprio_dim not in (67, 96)
                or self.point_feature_dim != 3
            ):
                raise ValueError(
                    "q_d g015 V75/V76 checkpoint requires XYZ points, "
                    "history=8, and proprio_dim=67 or 96"
                )
        elif history_proprio not in ((4, 67), (8, 96), (16, 96)):
            raise ValueError(
                "legacy checkpoint history/proprio contract must be "
                "(4,67), (8,96) or (16,96)"
            )
        self.initial_previous_action13 = _finite_float32(
            np.asarray(
                self.action_controller.initial_previous_action13,
                dtype=np.float32,
            ),
            (13,),
            "action_controller.initial_previous_action13",
        ).copy()
        self.initial_previous_action13.setflags(write=False)
        metadata = checkpoint.metadata
        required_metadata = {
            "history_bootstrap": "repeat_initial",
            "point_features": self.point_feature_mode,
            "proprio_source": "deployable_robot",
            "action_contract": "inspire_semantic_13d",
            "adapter_contract": "fr3_rh56_adapter_v7_shoulder_10mm",
            "sample_timing": "pre_action",
        }
        for key, expected in required_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"checkpoint metadata mismatch for {key}: "
                    f"expected={expected!r}, actual={metadata.get(key)!r}"
                )
        self.weights = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in checkpoint.model_state_dict.items()
        }
        normalization = checkpoint.normalization
        self.point_mean = self._normalizer(
            normalization,
            "pointcloud_mean",
            (1, 1, 1, self.point_feature_dim),
        )
        self.point_std = self._normalizer(
            normalization,
            "pointcloud_std",
            (1, 1, 1, self.point_feature_dim),
            positive=True,
        )
        self.proprio_mean = self._normalizer(
            normalization, "proprio_mean", (1, 1, self.proprio_dim)
        )
        self.proprio_std = self._normalizer(
            normalization,
            "proprio_std",
            (1, 1, self.proprio_dim),
            positive=True,
        )
        self._validate_weight_shapes()

    @staticmethod
    def _normalizer(
        values: Mapping[str, np.ndarray],
        name: str,
        shape: Tuple[int, ...],
        *,
        positive: bool = False,
    ) -> np.ndarray:
        if name not in values:
            raise ValueError(f"checkpoint normalization is missing {name}")
        result = _finite_float32(values[name], shape, f"normalization.{name}")
        if positive and np.any(result <= 0.0):
            raise ValueError(f"normalization.{name} must be positive")
        return result.copy()

    def _validate_weight_shapes(self) -> None:
        expected = {
            "point_encoder.0.weight": (128, self.point_feature_dim + 1),
            "point_encoder.0.bias": (128,),
            "point_encoder.2.weight": (128, 128),
            "point_encoder.2.bias": (128,),
            "frame_encoder.0.weight": (512, 266 + self.proprio_dim),
            "frame_encoder.0.bias": (512,),
            "frame_encoder.2.weight": (256, 512),
            "frame_encoder.2.bias": (256,),
            "temporal_lstm.weight_ih_l0": (1024, 256),
            "temporal_lstm.weight_hh_l0": (1024, 256),
            "temporal_lstm.bias_ih_l0": (1024,),
            "temporal_lstm.bias_hh_l0": (1024,),
            "global_encoder.0.weight": (512, 256),
            "global_encoder.0.bias": (512,),
            "global_encoder.2.weight": (256, 512),
            "global_encoder.2.bias": (256,),
            "privileged_head.0.weight": (512, 256),
            "privileged_head.0.bias": (512,),
            "privileged_head.2.weight": (self.compact_privileged_dim, 512),
            "privileged_head.2.bias": (self.compact_privileged_dim,),
            "future_motion_head.0.weight": (512, 256),
            "future_motion_head.0.bias": (512,),
            "future_motion_head.2.weight": (24, 512),
            "future_motion_head.2.bias": (24,),
            "future_motion_action_encoder.0.weight": (256, 24),
            "future_motion_action_encoder.0.bias": (256,),
            "future_motion_action_encoder.2.weight": (128, 256),
            "future_motion_action_encoder.2.bias": (128,),
            "action_head.0.weight": (
                512,
                384 + self.compact_privileged_dim,
            ),
            "action_head.0.bias": (512,),
            "action_head.2.weight": (13, 512),
            "action_head.2.bias": (13,),
            "hold_head.0.weight": (512, 256),
            "hold_head.0.bias": (512,),
            "hold_head.2.weight": (6, 512),
            "hold_head.2.bias": (6,),
            "hold_gate_head.0.weight": (256, 256),
            "hold_gate_head.0.bias": (256,),
            "hold_gate_head.2.weight": (1, 256),
            "hold_gate_head.2.bias": (1,),
            "flow_head.0.weight": (512, 387),
            "flow_head.0.bias": (512,),
            "flow_head.2.weight": (256, 512),
            "flow_head.2.bias": (256,),
            "flow_head.4.weight": (3, 256),
            "flow_head.4.bias": (3,),
            "affordance_head.0.weight": (512, 387),
            "affordance_head.0.bias": (512,),
            "affordance_head.2.weight": (256, 512),
            "affordance_head.2.bias": (256,),
            "affordance_head.4.weight": (1, 256),
            "affordance_head.4.bias": (1,),
        }
        if self.analytic_future_contract_action_adapter_enabled:
            expected.update(
                {
                    "analytic_future_contract_action_adapter.0.weight": (
                        128,
                        384
                        + self.compact_privileged_dim
                        + ANALYTIC_FUTURE_CONTRACT_DIM,
                    ),
                    "analytic_future_contract_action_adapter.0.bias": (128,),
                    "analytic_future_contract_action_adapter.2.weight": (13, 128),
                    "analytic_future_contract_action_adapter.2.bias": (13,),
                }
            )
        else:
            unexpected_adapter = sorted(
                name
                for name in self.weights
                if name.startswith("analytic_future_contract_action_adapter.")
            )
            if unexpected_adapter:
                raise ValueError(
                    "checkpoint contains analytic future adapter weights while "
                    "the adapter is disabled"
                )
        missing = sorted(set(expected) - set(self.weights))
        if missing:
            raise ValueError(
                f"checkpoint model_state_dict is missing weights: {missing}"
            )
        for name, shape in expected.items():
            value = self.weights[name]
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"checkpoint weight {name} must have shape {shape}")
        action_distribution = self.weights.get("action_log_std_head.0.weight")
        if action_distribution is not None:
            gaussian_expected = {
                "action_log_std_head.0.weight": (
                    256,
                    384 + self.compact_privileged_dim,
                ),
                "action_log_std_head.0.bias": (256,),
                "action_log_std_head.2.weight": (13, 256),
                "action_log_std_head.2.bias": (13,),
            }
            for name, shape in gaussian_expected.items():
                value = self.weights.get(name)
                if value is None or value.shape != shape or not np.all(np.isfinite(value)):
                    raise ValueError(
                        f"checkpoint Gaussian weight {name} must have shape {shape}"
                    )
        # The export also carries unused training-only flow/affordance heads.
        # Validate them here even though deterministic deployment does not
        # execute them, so malformed external checkpoints cannot hide behind
        # a matching aggregate parameter count.
        self.expected_model_parameter_count = int(
            2_522_192
            - 128 * (6 - self.point_feature_dim)
            + 1_025 * (self.compact_privileged_dim - 32)
            + 512 * (self.proprio_dim - 67)
            + (110_349 if action_distribution is not None else 0)
            + (
                57_357
                if self.analytic_future_contract_action_adapter_enabled
                else 0
            )
        )

    def _linear(self, value: np.ndarray, prefix: str) -> np.ndarray:
        return (
            value @ self.weights[prefix + ".weight"].T + self.weights[prefix + ".bias"]
        )

    def _mlp2(self, value: np.ndarray, prefix: str) -> np.ndarray:
        return self._linear(_silu(self._linear(value, prefix + ".0")), prefix + ".2")

    @staticmethod
    def _geometry(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
        mask = valid[..., None]
        count = np.maximum(np.sum(valid, axis=2, keepdims=True), 1.0)
        centroid = np.sum(points[..., :3] * mask, axis=2) / count
        low = np.min(np.where(mask > 0.5, points[..., :3], np.inf), axis=2)
        high = np.max(np.where(mask > 0.5, points[..., :3], -np.inf), axis=2)
        extent = high - low
        no_points = np.sum(valid, axis=2) < 0.5
        centroid[no_points] = 0.0
        extent[no_points] = 0.0
        fraction = np.mean(valid, axis=2, keepdims=True)
        displacement = np.zeros_like(centroid)
        displacement[:, 1:] = centroid[:, 1:] - centroid[:, :-1]
        return np.concatenate([centroid, extent, fraction, displacement], axis=-1)

    def _lstm(self, frames: np.ndarray) -> np.ndarray:
        batch = frames.shape[0]
        hidden = np.zeros((batch, 256), dtype=np.float32)
        cell = np.zeros((batch, 256), dtype=np.float32)
        weight_ih = self.weights["temporal_lstm.weight_ih_l0"]
        weight_hh = self.weights["temporal_lstm.weight_hh_l0"]
        bias = (
            self.weights["temporal_lstm.bias_ih_l0"]
            + self.weights["temporal_lstm.bias_hh_l0"]
        )
        for index in range(frames.shape[1]):
            gates = frames[:, index] @ weight_ih.T + hidden @ weight_hh.T + bias
            input_gate, forget_gate, candidate, output_gate = np.split(gates, 4, axis=1)
            input_gate = _sigmoid(input_gate)
            forget_gate = _sigmoid(forget_gate)
            candidate = np.tanh(candidate)
            output_gate = _sigmoid(output_gate)
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * np.tanh(cell)
        return hidden

    def act(
        self,
        pointcloud: np.ndarray,
        valid: np.ndarray,
        proprio: np.ndarray,
    ) -> PolicyOutput:
        """Run one deterministic inference call.

        Inputs may omit the leading batch dimension.  History frames are
        native-palm observations ordered oldest to newest.
        """

        points = np.asarray(pointcloud, dtype=np.float32)
        validity = np.asarray(valid, dtype=np.float32)
        robot = np.asarray(proprio, dtype=np.float32)
        point_shape = (self.history_length, 128, self.point_feature_dim)
        if points.shape == point_shape:
            points = points[None]
        if validity.shape == (self.history_length, 128):
            validity = validity[None]
        if robot.shape == (self.history_length, self.proprio_dim):
            robot = robot[None]
        if points.ndim != 4 or points.shape[1:] != point_shape:
            raise ValueError(
                "pointcloud must have shape "
                f"[B,{self.history_length},128,{self.point_feature_dim}] for "
                f"{self.point_feature_mode.upper()} mode"
            )
        batch = points.shape[0]
        if validity.shape != (batch, self.history_length, 128):
            raise ValueError(
                f"valid must have shape [B,{self.history_length},128]"
            )
        if robot.shape != (batch, self.history_length, self.proprio_dim):
            raise ValueError(
                "proprio must have shape "
                f"[B,{self.history_length},{self.proprio_dim}]"
            )
        if not (
            np.all(np.isfinite(points))
            and np.all(np.isfinite(validity))
            and np.all(np.isfinite(robot))
        ):
            raise ValueError("policy inputs must be finite")
        if np.any(validity < 0.0) or np.any(validity > 1.0):
            raise ValueError("validity values must lie in [0,1]")
        if np.any(np.sum(validity, axis=2) < 1.0):
            raise ValueError("every policy frame must contain at least one valid point")

        normalized_points = (points - self.point_mean) / self.point_std
        normalized_robot = (robot - self.proprio_mean) / self.proprio_std
        point_input = np.concatenate([normalized_points, validity[..., None]], axis=-1)
        encoded = _silu(self._linear(point_input, "point_encoder.0"))
        encoded = self._linear(encoded, "point_encoder.2")
        mask = validity[..., None]
        count = np.maximum(np.sum(validity, axis=2, keepdims=True), 1.0)
        mean_pool = np.sum(encoded * mask, axis=2) / count
        max_pool = np.max(np.where(mask > 0.5, encoded, -np.inf), axis=2)
        geometry = self._geometry(normalized_points, validity)
        frame_input = np.concatenate(
            [mean_pool, max_pool, normalized_robot, geometry], axis=-1
        )
        frames = _silu(self._linear(frame_input, "frame_encoder.0"))
        frames = self._linear(frames, "frame_encoder.2")
        temporal = self._lstm(frames)
        latent = _silu(self._linear(temporal, "global_encoder.0"))
        latent = self._linear(latent, "global_encoder.2")

        privileged = self._mlp2(latent, "privileged_head")
        future_motion = self._mlp2(latent, "future_motion_head")
        future_features = _silu(
            self._linear(future_motion, "future_motion_action_encoder.0")
        )
        future_features = self._linear(
            future_features, "future_motion_action_encoder.2"
        )
        action_features = np.concatenate([latent, privileged, future_features], axis=-1)
        action_hidden = _silu(self._linear(action_features, "action_head.0"))
        action = self._linear(action_hidden, "action_head.2")
        if self.analytic_future_contract_action_adapter_enabled:
            if self.analytic_future_contract is None:
                raise RuntimeError("analytic future adapter contract disappeared")
            analytic_future = build_17d_future_contract(
                self.analytic_future_contract,
                metric_pointcloud_seq=points,
                valid_seq=validity,
                metric_proprio_seq=robot,
                predicted_compact_privileged=privileged,
            )
            if analytic_future.shape != (batch, ANALYTIC_FUTURE_CONTRACT_DIM):
                raise RuntimeError("analytic future contract returned wrong shape")
            adapter_input = np.concatenate(
                (action_features, analytic_future), axis=-1
            )
            action = action + self.analytic_future_contract_action_adapter_scale * (
                self._mlp2(
                    adapter_input,
                    "analytic_future_contract_action_adapter",
                )
            )
        action = np.clip(
            action,
            np.float32(-1.0),
            np.float32(1.0),
        )
        hold = self._mlp2(latent, "hold_head")
        hold_logit = self._mlp2(latent, "hold_gate_head")

        if batch != 1:
            raise ValueError("deployment currently requires batch size 1")
        return PolicyOutput(
            action13=action[0].astype(np.float32, copy=True),
            predicted_privileged32=privileged[0].astype(np.float32, copy=True),
            future_motion24=future_motion[0].astype(np.float32, copy=True),
            predicted_hold6=hold[0].astype(np.float32, copy=True),
            predicted_hold_logit=float(hold_logit[0, 0]),
        )


__all__ = [
    "ActionControllerParameters",
    "EXPECTED_SPEC",
    "LEGACY_ACTION_CONTROLLER_CONTRACT_ID",
    "LEGACY_ACTION_CONTROLLER_METADATA_SCHEMA",
    "QD_G015_ACTION_CONTROLLER_CONTRACT_ID",
    "SUPPORTED_POINT_FEATURE_MODES",
    "PolicyOutput",
    "RollingStudentPolicy",
]
