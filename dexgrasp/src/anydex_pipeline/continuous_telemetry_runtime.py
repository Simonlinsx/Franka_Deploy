"""Opt-in execution-side wiring for native continuous telemetry.

Importing this module is hardware-free: it does not import pylibfranka, the
native producer extension, a serial backend, a camera backend, or a viewer.
The native extension is imported only by :meth:`ContinuousTelemetryRuntime.start`
after the executor has completed its existing offline and live pre-motion
gates.

The runtime never creates a second robot or RH56 connection.  It installs the
native fused ``readOnce`` tap into the already-connected Franka driver and an
observer into the already-connected RH56 driver's *validated* feedback path.
Consequently it adds neither a second FCI read nor a second serial read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import importlib.machinery
import os
from pathlib import Path
import secrets
import sys
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from .telemetry_session_manifest import (
    LoadedTelemetrySessionManifest,
    canonical_json_sha256,
    load_telemetry_session_manifest,
    sha256_file,
)


PathLike = Union[str, Path]
NATIVE_PRODUCER_MODULE = "_anydex_franka_telemetry"
DEFAULT_ARM_PUBLISH_DECIMATION = 20
INITIAL_STAGE_NAME = "disarmed"


@dataclass(frozen=True)
class ExecutionTelemetryRequest:
    """One replayed manifest bound to one not-yet-created mapping path."""

    mapping_path: Path
    manifest: LoadedTelemetrySessionManifest
    producer_build_path: Path
    python_dir: Path


class ContinuousTelemetryStartError(RuntimeError):
    """Start failed and a still-attached runtime requires cleanup retry."""

    def __init__(
        self,
        primary: BaseException,
        rollback_errors: Tuple[str, ...],
        runtime: "ContinuousTelemetryRuntime",
    ) -> None:
        self.primary = primary
        self.rollback_errors = tuple(rollback_errors)
        self.runtime = runtime
        super().__init__(
            "continuous telemetry start failed: {}; rollback incomplete: {}".format(
                primary, "; ".join(self.rollback_errors)
            )
        )


def _resolved_file(path: PathLike, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError("{} does not exist: {}".format(name, resolved))
    return resolved


def _same_path(actual: Any, expected: PathLike, name: str) -> None:
    actual_path = Path(str(actual)).expanduser().resolve()
    expected_path = Path(expected).expanduser().resolve()
    if actual_path != expected_path:
        raise ValueError(
            "telemetry manifest {} path differs from this execution: {} != {}".format(
                name, actual_path, expected_path
            )
        )


def _mapping_output_path(
    value: PathLike, manifest: LoadedTelemetrySessionManifest
) -> Path:
    raw = Path(value).expanduser()
    # ``Path.exists`` is false for a broken symlink; native O_NOFOLLOW would
    # reject it anyway, so expose that error during hardware-free validation.
    if os.path.lexists(str(raw)):
        raise ValueError("continuous telemetry mapping already exists: {}".format(raw))
    output = raw.resolve(strict=False)
    if os.path.lexists(str(output)):
        raise ValueError(
            "continuous telemetry mapping already exists: {}".format(output)
        )
    if not output.parent.is_dir():
        raise ValueError(
            "continuous telemetry mapping parent does not exist: {}".format(
                output.parent
            )
        )
    if not os.access(str(output.parent), os.W_OK | os.X_OK):
        raise ValueError(
            "continuous telemetry mapping parent is not writable: {}".format(
                output.parent
            )
        )
    protected = []
    if manifest.path is not None:
        protected.append(manifest.path.resolve())
    for binding in manifest.payload["sources"].values():
        protected.append(Path(str(binding["path"])).expanduser().resolve())
    if output in protected:
        raise ValueError("continuous telemetry mapping would overwrite a bound artifact")
    return output


def _audit_path_for_command(
    command: str,
    *,
    installed_tool_audit_path: Optional[PathLike],
    pregrasp_only_audit_path: Optional[PathLike],
    loaded_lift_audit_path: Optional[PathLike],
) -> PathLike:
    if command == "pregrasp":
        selected = pregrasp_only_audit_path
    elif command in ("grasp", "air-grasp"):
        selected = installed_tool_audit_path
    elif command == "grasp-lift":
        selected = loaded_lift_audit_path
    else:
        raise ValueError(
            "continuous telemetry session manifests do not support command {!r}".format(
                command
            )
        )
    if selected is None:
        raise ValueError(
            "{} continuous telemetry requires its command-specific audit path".format(
                command
            )
        )
    return selected


def load_execution_telemetry_request(
    *,
    mapping_path: PathLike,
    manifest_path: PathLike,
    python_dir: PathLike,
    expected_command: Optional[str] = None,
    expected_control_config_path: Optional[PathLike] = None,
    expected_snapshot_path: Optional[PathLike] = None,
    expected_installed_tool_audit_path: Optional[PathLike] = None,
    expected_pregrasp_only_audit_path: Optional[PathLike] = None,
    expected_loaded_lift_audit_path: Optional[PathLike] = None,
    expected_selected_index: Optional[int] = None,
) -> ExecutionTelemetryRequest:
    """Replay a session manifest without importing the native producer.

    When ``expected_command`` is omitted this is a pure manifest inspection.
    A formal or dry-run executor supplies it and every command-specific source
    so that a valid manifest for a different run cannot be attached by mistake.
    """

    loaded = load_telemetry_session_manifest(manifest_path, verify_files=True)
    sources = loaded.payload["sources"]
    execution = loaded.payload["execution"]
    producer_path = _resolved_file(
        sources["producer_build"]["path"], "manifest producer build"
    )
    resolved_python_dir = Path(python_dir).expanduser().resolve()
    if not resolved_python_dir.is_dir():
        raise ValueError(
            "continuous telemetry Python directory does not exist: {}".format(
                resolved_python_dir
            )
        )
    if producer_path.parent != resolved_python_dir:
        raise ValueError(
            "manifest producer build is not directly in the requested Python directory"
        )
    if not producer_path.name.startswith(NATIVE_PRODUCER_MODULE) or not any(
        producer_path.name.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise ValueError(
            "manifest producer build is not a loadable {} extension".format(
                NATIVE_PRODUCER_MODULE
            )
        )
    output = _mapping_output_path(mapping_path, loaded)

    if expected_command is not None:
        command = str(expected_command)
        if execution["command"] != command:
            raise ValueError(
                "telemetry manifest command {!r} differs from executor {!r}".format(
                    execution["command"], command
                )
            )
        if expected_control_config_path is None or expected_snapshot_path is None:
            raise ValueError(
                "execution-bound telemetry validation requires config and snapshot paths"
            )
        _same_path(
            sources["control_config"]["path"],
            expected_control_config_path,
            "control config",
        )
        _same_path(
            sources["snapshot"]["path"], expected_snapshot_path, "snapshot"
        )
        audit_path = _audit_path_for_command(
            command,
            installed_tool_audit_path=expected_installed_tool_audit_path,
            pregrasp_only_audit_path=expected_pregrasp_only_audit_path,
            loaded_lift_audit_path=expected_loaded_lift_audit_path,
        )
        _same_path(sources["audit_artifact"]["path"], audit_path, "audit")
        if isinstance(expected_selected_index, bool) or not isinstance(
            expected_selected_index, int
        ):
            raise ValueError(
                "execution-bound telemetry validation requires an integer selected index"
            )
        if int(execution["selected_index"]) != expected_selected_index:
            raise ValueError(
                "telemetry manifest selected_index={} differs from executor {}".format(
                    execution["selected_index"], expected_selected_index
                )
            )

    return ExecutionTelemetryRequest(
        mapping_path=output,
        manifest=loaded,
        producer_build_path=producer_path,
        python_dir=resolved_python_dir,
    )


def _replay_unchanged_request(
    request: ExecutionTelemetryRequest,
) -> LoadedTelemetrySessionManifest:
    source = request.manifest.path
    if source is None or request.manifest.manifest_file_sha256 is None:
        raise RuntimeError("telemetry request has no file-backed manifest")
    replayed = load_telemetry_session_manifest(source, verify_files=True)
    if replayed.manifest_file_sha256 != request.manifest.manifest_file_sha256:
        raise RuntimeError("telemetry session manifest changed before producer start")
    if replayed.identity != request.manifest.identity:
        raise RuntimeError("telemetry session identity changed before producer start")
    if canonical_json_sha256(replayed.payload) != canonical_json_sha256(
        request.manifest.payload
    ):
        raise RuntimeError("telemetry session content changed before producer start")
    if os.path.lexists(str(request.mapping_path)):
        raise RuntimeError(
            "continuous telemetry mapping appeared before producer start: {}".format(
                request.mapping_path
            )
        )
    return replayed


def _writer_tokens(token_source: Callable[[int], int]) -> Tuple[int, int]:
    first = 0
    second = 0
    for _ in range(16):
        candidate = int(token_source(64))
        if 0 < candidate <= (1 << 64) - 1:
            if first == 0:
                first = candidate
            elif candidate != first:
                second = candidate
                break
    if first == 0 or second == 0:
        raise RuntimeError("could not generate distinct nonzero telemetry writer tokens")
    return first, second


def _dependency_entry(value: Any, name: str) -> Tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise RuntimeError("native {} dependency metadata is malformed".format(name))
    path = _resolved_file(value["path"], "native {} dependency".format(name))
    digest = value["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise RuntimeError("native {} dependency SHA-256 is malformed".format(name))
    if sha256_file(path) != digest:
        raise RuntimeError("native {} dependency hash mismatch".format(name))
    return path, digest


def _verify_native_dependencies(
    module: Any, *, proc_maps_path: PathLike = "/proc/self/maps"
) -> None:
    """Verify the exact wheel extension and sole loaded libfranka DSO."""

    metadata = getattr(module, "BUILD_DEPENDENCIES", None)
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "schema_version",
        "pylibfranka",
        "libfranka",
    }:
        raise RuntimeError("native producer build dependency metadata is missing")
    if metadata["schema_version"] != 1:
        raise RuntimeError("native producer dependency metadata schema is unsupported")
    pylibfranka_path, _ = _dependency_entry(
        metadata["pylibfranka"], "pylibfranka"
    )
    libfranka_path, _ = _dependency_entry(metadata["libfranka"], "libfranka")

    wheel_module = importlib.import_module("pylibfranka._pylibfranka")
    loaded_wheel_path = getattr(wheel_module, "__file__", None)
    if loaded_wheel_path is None or Path(str(loaded_wheel_path)).resolve() != pylibfranka_path:
        raise RuntimeError(
            "loaded pylibfranka extension differs from producer build metadata"
        )

    maps_source = Path(proc_maps_path)
    try:
        lines = maps_source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("cannot audit loaded libfranka mappings: {}".format(exc)) from exc
    loaded_paths = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        candidate_text = fields[5]
        candidate_name = Path(candidate_text).name
        if candidate_name.startswith("libfranka"):
            loaded_paths.add(Path(candidate_text).resolve())
    if loaded_paths != {libfranka_path}:
        raise RuntimeError(
            "process must map exactly one manifest-built libfranka DSO; loaded={}".format(
                sorted(str(path) for path in loaded_paths)
            )
        )


class ContinuousTelemetryRuntime:
    """Installed native producer owned by the existing two device drivers."""

    def __init__(
        self,
        *,
        request: ExecutionTelemetryRequest,
        producer: Any,
        arm: Any,
        hand: Any,
        observer_identity: str,
        stage_epoch: int,
    ) -> None:
        self.request = request
        self.producer = producer
        self.arm = arm
        self.hand = hand
        self.observer_identity = observer_identity
        self._stage_epoch = int(stage_epoch)
        self._arm_installed = False
        self._hand_installed = False
        self._closed = False

    @classmethod
    def start(
        cls,
        request: ExecutionTelemetryRequest,
        *,
        arm: Any,
        hand: Any,
        initial_validated_arm_state: Any,
        initial_arm_timestamp_unix_ns: int,
        initial_arm_timestamp_monotonic_ns: int,
        robot_id: str,
        arm_publish_decimation: int = DEFAULT_ARM_PUBLISH_DECIMATION,
        native_module_loader: Optional[Callable[[], Any]] = None,
        token_source: Callable[[int], int] = secrets.randbits,
        unix_time_ns: Callable[[], int] = time.time_ns,
        monotonic_time_ns: Callable[[], int] = time.monotonic_ns,
        dependency_checker: Callable[[Any], None] = _verify_native_dependencies,
    ) -> "ContinuousTelemetryRuntime":
        """Revalidate, import the exact bound extension, and install observers.

        The caller must pass an arm state already read and validated by the
        executor's existing installed-end-effector gate.  This method does not
        read either device.
        """

        if not isinstance(request, ExecutionTelemetryRequest):
            raise TypeError("request must be ExecutionTelemetryRequest")
        replayed = _replay_unchanged_request(request)
        python_dir_text = str(request.python_dir)
        if python_dir_text not in sys.path:
            sys.path.insert(0, python_dir_text)
        loader = native_module_loader
        if loader is None:
            loader = lambda: importlib.import_module(NATIVE_PRODUCER_MODULE)
        module = loader()
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError("native telemetry module has no file provenance")
        loaded_binary = Path(str(module_file)).expanduser().resolve()
        if loaded_binary != request.producer_build_path:
            raise RuntimeError(
                "loaded native telemetry module differs from manifest producer build: "
                "{} != {}".format(loaded_binary, request.producer_build_path)
            )
        if sha256_file(loaded_binary) != replayed.identity.producer_build_sha256:
            raise RuntimeError("loaded native telemetry module hash differs from manifest")
        dependency_checker(module)
        producer_type = getattr(module, "NativeTelemetryProducer", None)
        creator = getattr(producer_type, "create", None)
        if not callable(creator):
            raise RuntimeError("native telemetry module exposes no producer factory")

        created_unix_ns = int(unix_time_ns())
        created_monotonic_ns = int(monotonic_time_ns())
        if created_unix_ns <= 0 or created_monotonic_ns <= 0:
            raise RuntimeError("telemetry clocks returned a nonpositive timestamp")
        sampled_unix_ns = int(initial_arm_timestamp_unix_ns)
        sampled_monotonic_ns = int(initial_arm_timestamp_monotonic_ns)
        if sampled_unix_ns <= 0 or sampled_monotonic_ns <= 0:
            raise RuntimeError(
                "initial validated arm sample timestamps must be positive"
            )
        identity = asdict(replayed.identity)
        provenance = dict(
            identity,
            created_monotonic_ns=created_monotonic_ns,
            created_unix_ns=created_unix_ns,
            producer_name="execute_control_sequence",
            robot_id=str(robot_id),
        )
        arm_token, hand_token = _writer_tokens(token_source)
        producer = creator(
            str(request.mapping_path),
            provenance,
            arm_token,
            hand_token,
            int(arm_publish_decimation),
        )
        observer_identity = "continuous-telemetry:{}".format(
            replayed.identity.run_uuid
        )
        runtime = cls(
            request=request,
            producer=producer,
            arm=arm,
            hand=hand,
            observer_identity=observer_identity,
            stage_epoch=1,
        )
        try:
            producer.set_stage(INITIAL_STAGE_NAME, 1, 0)
            producer.synchronize_and_publish_arm(
                initial_validated_arm_state,
                sampled_unix_ns,
                sampled_monotonic_ns,
            )
            arm_installer = getattr(arm, "install_native_telemetry_tap", None)
            if not callable(arm_installer):
                raise RuntimeError("Franka driver exposes no native telemetry tap installer")
            arm_installer(producer)
            runtime._arm_installed = True
            hand_installer = getattr(
                hand, "install_validated_feedback_observer", None
            )
            if not callable(hand_installer):
                raise RuntimeError(
                    "RH56 driver exposes no validated-feedback observer installer"
                )
            hand_installer(
                runtime.publish_validated_hand_feedback,
                identity=observer_identity,
            )
            runtime._hand_installed = True
        except BaseException as primary:
            rollback_errors = runtime._detach_and_close()
            if rollback_errors:
                raise ContinuousTelemetryStartError(
                    primary, rollback_errors, runtime
                ) from primary
            raise
        return runtime

    @property
    def stage_epoch(self) -> int:
        return self._stage_epoch

    @property
    def closed(self) -> bool:
        return self._closed

    def observe_transition(self, state: Any) -> None:
        """Publish an exact sequence-state tag before that stage performs work."""

        if self._closed:
            raise RuntimeError("continuous telemetry runtime is closed")
        name = str(getattr(state, "value", state))
        if not name or len(name.encode("utf-8")) > 31 or "\x00" in name:
            raise RuntimeError("sequence state does not fit native stage ABI")
        if self._stage_epoch >= (1 << 64) - 1:
            raise RuntimeError("continuous telemetry stage epoch exhausted")
        self._stage_epoch += 1
        self.producer.set_stage(name, self._stage_epoch, 0)

    def publish_validated_hand_feedback(self, observation: Any) -> int:
        """Adapt one already-read, already-validated RH56 feedback event."""

        if self._closed:
            raise RuntimeError("continuous telemetry runtime is closed")
        if str(getattr(observation, "observer_identity", "")) != self.observer_identity:
            raise RuntimeError("RH56 feedback observer identity mismatch")
        return int(
            self.producer.publish_hand(
                tuple(getattr(observation, "angles")),
                tuple(getattr(observation, "angle_targets")),
                tuple(getattr(observation, "currents")),
                (
                    None
                    if getattr(observation, "forces") is None
                    else tuple(getattr(observation, "forces"))
                ),
                tuple(getattr(observation, "temperatures")),
                tuple(getattr(observation, "statuses")),
                tuple(getattr(observation, "errors")),
                int(getattr(observation, "timestamp_unix_ns")),
                int(getattr(observation, "timestamp_monotonic_ns")),
            )
        )

    def _detach_and_close(self) -> Tuple[str, ...]:
        errors = []
        if self._hand_installed:
            remover = getattr(
                self.hand, "remove_validated_feedback_observer", None
            )
            try:
                if not callable(remover):
                    raise RuntimeError(
                        "RH56 driver exposes no validated-feedback observer remover"
                    )
                remover(identity=self.observer_identity)
                self._hand_installed = False
            except BaseException as exc:
                errors.append("RH56 observer detach: {}".format(exc))
        if self._arm_installed:
            remover = getattr(self.arm, "remove_native_telemetry_tap", None)
            try:
                if not callable(remover):
                    raise RuntimeError("Franka driver exposes no telemetry tap remover")
                remover(self.producer)
                self._arm_installed = False
            except BaseException as exc:
                errors.append("Franka tap detach: {}".format(exc))
        # A driver must never retain a pointer to a closed native producer.
        # Leave the producer alive and make ``close`` retryable until both
        # identity-bound detach operations have succeeded.
        if not self._hand_installed and not self._arm_installed:
            try:
                self.producer.close()
            except BaseException as exc:
                errors.append("native producer close: {}".format(exc))
            else:
                self._closed = True
        return tuple(errors)

    def close(self) -> None:
        """Detach both observers and release the mapping writer claims."""

        if self._closed:
            return
        errors = self._detach_and_close()
        if errors:
            raise RuntimeError("; ".join(errors))


__all__ = [
    "ContinuousTelemetryStartError",
    "ContinuousTelemetryRuntime",
    "DEFAULT_ARM_PUBLISH_DECIMATION",
    "ExecutionTelemetryRequest",
    "INITIAL_STAGE_NAME",
    "NATIVE_PRODUCER_MODULE",
    "_verify_native_dependencies",
    "load_execution_telemetry_request",
]
