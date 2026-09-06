"""Shared launcher for the tabletop and thrown-object V94 task profiles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TASK_CONFIG_ROOT = WORKSPACE_ROOT / "perception" / "configs" / "tasks"
RESOLVED_CONFIG_ROOT = WORKSPACE_ROOT / "dexgrasp" / "runs" / "task_configs"
THROWN_TASK_NAMES = frozenset(
    ("thrown_object", "thrown_object_v60", "thrown_object_v61")
)
TASK_NAMES = ("tabletop", *sorted(THROWN_TASK_NAMES))
MODES = ("config", "perception", "shadow", "deploy")
PROTECTED_OPTIONS = (
    "--pcd-config",
    "--policy-rgbd-resolution",
    "--object-mask-mode",
)
TARGET_OPTIONS = ("--object-text", "--select-object-roi", "--object-roi")


class TaskLauncherError(ValueError):
    pass


def _is_thrown_task(task_name: str) -> bool:
    return task_name in THROWN_TASK_NAMES


def _task_allows_robot_execution(task_name: str, task: Mapping[str, Any]) -> bool:
    """Return the exact task-owned execution latch.

    Both launch-time and runtime gates independently enforce the declared
    task-owned supervised step cap.
    """

    if task.get("robot_execution_enabled") is not True:
        return False
    status = str(task.get("commissioning_status", "")).strip()
    if status == "accepted":
        return True
    limits = task.get("first_motion_execution_limits")
    v60_first_motion = bool(
        task_name == "thrown_object_v60"
        and status == "first_motion_accepted"
        and task.get("maximum_supervised_execute_steps") == 1
        and isinstance(limits, Mapping)
        and limits.get("maximum_commanded_policy_ticks") == 1
        and limits.get("require_current_policy_points_inside_processing_depth")
        is True
        and limits.get("robot_hardware_writes") is True
    )
    limits = task.get("supervised_execution_limits")
    v61_forty_tick = bool(
        task_name == "thrown_object_v61"
        and status == "supervised_40_tick_authorized"
        and task.get("maximum_supervised_execute_steps") == 40
        and isinstance(limits, Mapping)
        and limits.get("maximum_commanded_policy_ticks") == 40
        and limits.get("require_current_policy_points_inside_processing_depth")
        is True
        and limits.get("robot_hardware_writes") is True
    )
    return v60_first_motion or v61_forty_tick


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TaskLauncherError(f"cannot load task YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TaskLauncherError(f"task YAML root must be a mapping: {path}")
    return value


def _deep_update(
    destination: MutableMapping[str, Any], source: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(destination.get(key), dict):
            _deep_update(destination[key], value)
        else:
            destination[key] = value
    return destination


def _task_source_path(task_name: str) -> Path:
    if task_name not in TASK_NAMES:
        raise TaskLauncherError(
            f"task must be one of {', '.join(TASK_NAMES)}, got {task_name!r}"
        )
    return (TASK_CONFIG_ROOT / f"{task_name}.yaml").resolve()


def _verify_accepted_thrown_evidence(task: Mapping[str, Any]) -> None:
    task_name = str(task.get("name", "")).strip()
    if not _task_allows_robot_execution(task_name, task):
        return
    limited_shadow_gate = bool(
        task_name == "thrown_object_v60"
        and str(task.get("commissioning_status", "")).strip()
        == "first_motion_accepted"
        or task_name == "thrown_object_v61"
        and str(task.get("commissioning_status", "")).strip()
        == "supervised_40_tick_authorized"
    )
    evidence = task.get("commissioning_evidence")
    if not isinstance(evidence, Mapping):
        raise TaskLauncherError(
            "accepted thrown task has no commissioning_evidence"
        )
    records: list[tuple[str, str, str]] = []
    checkpoint_path = str(evidence.get("checkpoint_path", "")).strip()
    checkpoint_sha = str(evidence.get("checkpoint_sha256", "")).strip().lower()
    records.append(("checkpoint", checkpoint_path, checkpoint_sha))
    for label in ("correct_reset_policy_shadow_report", "correct_reset_policy_io"):
        value = evidence.get(label)
        if not isinstance(value, Mapping):
            raise TaskLauncherError(f"accepted thrown task lacks {label}")
        records.append(
            (
                label,
                str(value.get("path", "")).strip(),
                str(value.get("sha256", "")).strip().lower(),
            )
        )
    if not limited_shadow_gate:
        full_flight = evidence.get("full_flight_reports")
        if not isinstance(full_flight, list) or len(full_flight) < 3:
            raise TaskLauncherError(
                "accepted thrown task requires three full-flight reports"
            )
        for index, value in enumerate(full_flight):
            if not isinstance(value, Mapping):
                raise TaskLauncherError("full-flight evidence entry is malformed")
            records.append(
                (
                    f"full_flight_reports[{index}]",
                    str(value.get("path", "")).strip(),
                    str(value.get("sha256", "")).strip().lower(),
                )
            )
    for label, path_value, expected_sha in records:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            raise TaskLauncherError(f"{label} path must be absolute")
        path = path.resolve()
        if not path.is_file():
            raise TaskLauncherError(f"{label} evidence is missing: {path}")
        if len(expected_sha) != 64 or _sha256_file(path) != expected_sha:
            raise TaskLauncherError(f"{label} evidence SHA-256 differs")
    if limited_shadow_gate:
        report_path = Path(
            str(evidence["correct_reset_policy_shadow_report"]["path"])
        ).expanduser().resolve()
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskLauncherError(
                f"limited supervised shadow report is unreadable: {exc}"
            ) from exc
        shadow = report.get("policy_shadow")
        task_profile = report.get("task_profile")
        if not (
            report.get("accepted") is True
            and report.get("result") == "PASS"
            and report.get("robot_hardware_writes") is False
            and isinstance(shadow, Mapping)
            and shadow.get("all_outputs_finite") is True
            and shadow.get("robot_hardware_writes") is False
            and int(shadow.get("inference_count", 0)) >= 20
            and float(shadow.get("policy_inference_p95_ms", float("inf"))) <= 10.0
            and isinstance(task_profile, Mapping)
            and task_profile.get("name") == task_name
            and shadow.get("checkpoint_sha256") == checkpoint_sha
        ):
            raise TaskLauncherError(
                "shadow evidence does not satisfy the limited supervised gate"
            )


def materialize_task_config(task_name: str) -> tuple[Path, Mapping[str, Any]]:
    """Create a content-addressed, complete provider config for one task."""

    task_path = _task_source_path(task_name)
    task = _yaml_mapping(task_path)
    if task.get("schema_version") != 1 or task.get("kind") != "v94_task_profile":
        raise TaskLauncherError(f"unsupported task profile schema: {task_path}")
    if str(task.get("name", "")).strip() != task_name:
        raise TaskLauncherError("task profile name differs from its filename")
    if _is_thrown_task(task_name):
        _verify_accepted_thrown_evidence(task)
    if str(task.get("policy_rgbd_resolution", "")).strip().lower() != "424x240":
        raise TaskLauncherError(
            "tabletop and thrown-object task profiles must both use 424x240 policy RGB-D"
        )
    base_value = str(task.get("base_config", "")).strip()
    if not base_value:
        raise TaskLauncherError("task profile has no base_config")
    base_path = (task_path.parent / base_value).resolve()
    if not base_path.is_file():
        raise TaskLauncherError(f"task base config is missing: {base_path}")
    base = _yaml_mapping(base_path)
    overrides = task.get("config_overrides", {})
    if not isinstance(overrides, Mapping):
        raise TaskLauncherError("task config_overrides must be a mapping")
    resolved: dict[str, Any] = dict(base)
    _deep_update(resolved, overrides)

    extrinsics = resolved.get("extrinsics")
    if not isinstance(extrinsics, dict):
        raise TaskLauncherError("resolved task config has no extrinsics mapping")
    calibration_value = str(extrinsics.get("calibration_file", "")).strip()
    if not calibration_value:
        raise TaskLauncherError("resolved task config has no calibration_file")
    calibration_path = Path(calibration_value).expanduser()
    if not calibration_path.is_absolute():
        calibration_path = (base_path.parent / calibration_path).resolve()
    else:
        calibration_path = calibration_path.resolve()
    if not calibration_path.is_file():
        raise TaskLauncherError(f"task calibration is missing: {calibration_path}")
    actual_calibration_sha256 = _sha256_file(calibration_path)
    expected_calibration_sha256 = str(task.get("calibration_sha256", "")).strip()
    if (
        expected_calibration_sha256
        and expected_calibration_sha256 != actual_calibration_sha256
    ):
        raise TaskLauncherError(
            "task calibration bytes differ from calibration_sha256"
        )
    extrinsics["calibration_file"] = str(calibration_path)

    task_metadata = {
        key: value
        for key, value in task.items()
        if key not in ("config_overrides", "base_config", "schema_version", "kind")
    }
    if _is_thrown_task(task_name):
        from .thrown_contract import load_v57_thrown_task_contract

        contract_value = str(
            task_metadata.get("simulation_task_contract_file", "")
        ).strip()
        contract_sha256 = str(
            task_metadata.get("simulation_task_contract_sha256", "")
        ).strip().lower()
        curriculum = str(task_metadata.get("simulation_curriculum", "")).strip()
        if not contract_value or len(contract_sha256) != 64 or not curriculum:
            raise TaskLauncherError(
                "thrown task profile must pin a simulation task contract"
            )
        contract_path = Path(contract_value).expanduser()
        if not contract_path.is_absolute():
            contract_path = (task_path.parent / contract_path).resolve()
        else:
            contract_path = contract_path.resolve()
        try:
            task_contract = load_v57_thrown_task_contract(
                contract_path,
                selected_curriculum=curriculum,
                expected_sha256=contract_sha256,
            )
        except (OSError, ValueError) as exc:
            raise TaskLauncherError(f"invalid thrown task contract: {exc}") from exc
        task_metadata["simulation_task_contract_file"] = str(
            task_contract.source_path
        )
        task_metadata["simulation_task_contract_sha256"] = (
            task_contract.source_sha256
        )
        task_metadata["simulation_task_contract_control_hz"] = (
            task_contract.control_hz
        )
    task_metadata.update(
        {
            "resolved": True,
            "schema_version": 1,
            "source_task_config": str(task_path),
            "source_task_config_sha256": _sha256_file(task_path),
            "base_config": str(base_path),
            "base_config_sha256": _sha256_file(base_path),
            "calibration_file": str(calibration_path),
            "calibration_sha256": actual_calibration_sha256,
        }
    )
    resolved["task_profile"] = task_metadata
    encoded = yaml.safe_dump(
        resolved,
        sort_keys=False,
        allow_unicode=True,
    ).encode("utf-8")
    digest = _sha256_bytes(encoded)
    RESOLVED_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    output = RESOLVED_CONFIG_ROOT / f"{task_name}-{digest[:16]}.yaml"
    if output.exists():
        if output.read_bytes() != encoded:
            raise TaskLauncherError(
                f"content-addressed task config collision: {output}"
            )
    else:
        descriptor = os.open(
            output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                output.unlink()
            except OSError:
                pass
            raise
    return output.resolve(), task_metadata


def _option_present(arguments: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(option + "=") for value in arguments)


def _single_option_value(
    arguments: Sequence[str], option: str
) -> Optional[str]:
    values: list[str] = []
    index = 0
    while index < len(arguments):
        value = str(arguments[index])
        if value == option:
            if index + 1 >= len(arguments):
                raise TaskLauncherError(f"{option} requires a value")
            values.append(str(arguments[index + 1]))
            index += 2
            continue
        if value.startswith(option + "="):
            values.append(value.split("=", 1)[1])
        index += 1
    if len(values) > 1:
        raise TaskLauncherError(f"{option} may be supplied only once")
    return None if not values else values[0]


def build_task_command(
    task_name: str,
    mode: str,
    forwarded: Sequence[str],
) -> tuple[list[str], Path, Mapping[str, Any]]:
    if mode not in MODES:
        raise TaskLauncherError(
            f"mode must be one of {', '.join(MODES)}, got {mode!r}"
        )
    if mode == "shadow" and not _is_thrown_task(task_name):
        raise TaskLauncherError("shadow mode is defined only for thrown tasks")
    arguments = [str(value) for value in forwarded]
    for option in PROTECTED_OPTIONS:
        if _option_present(arguments, option):
            raise TaskLauncherError(
                f"{option} is task-owned and cannot be overridden"
            )
    if _is_thrown_task(task_name) and _option_present(
        arguments, "--policy-rate-hz"
    ):
        raise TaskLauncherError(
            "--policy-rate-hz is owned by the thrown task contract"
        )
    config_path, metadata = materialize_task_config(task_name)
    if mode == "config":
        return [], config_path, metadata
    if (
        mode == "deploy"
        and _option_present(arguments, "--execute")
        and not _task_allows_robot_execution(task_name, metadata)
    ):
        raise TaskLauncherError(
            f"{task_name} robot execution is disabled: commissioning_status="
            f"{metadata.get('commissioning_status')!r}"
        )
    if _is_thrown_task(task_name) and mode == "deploy" and _option_present(
        arguments, "--execute"
    ):
        maximum_value = metadata.get("maximum_supervised_execute_steps")
        if isinstance(maximum_value, bool):
            raise TaskLauncherError(
                f"{task_name} maximum_supervised_execute_steps is invalid"
            )
        try:
            maximum_steps = int(maximum_value)
        except (TypeError, ValueError) as exc:
            raise TaskLauncherError(
                f"{task_name} has no valid supervised step cap"
            ) from exc
        if maximum_steps < 1:
            raise TaskLauncherError(
                f"{task_name} supervised step cap must be positive"
            )
        requested_value = _single_option_value(arguments, "--steps")
        if requested_value is None:
            raise TaskLauncherError(
                f"{task_name} --execute requires an explicit --steps value"
            )
        try:
            requested_steps = int(requested_value)
        except (TypeError, ValueError) as exc:
            raise TaskLauncherError("--steps must be an integer") from exc
        if not 1 <= requested_steps <= maximum_steps:
            raise TaskLauncherError(
                f"{task_name} commissioning permits only "
                f"1..{maximum_steps} supervised steps"
            )
    if (
        _is_thrown_task(task_name)
        and mode in {"shadow", "deploy"}
        and not _option_present(arguments, "--checkpoint")
    ):
        compatibility_label = {
            "thrown_object": "V57-compatible",
            "thrown_object_v60": "V60-compatible",
            "thrown_object_v61": "V61-compatible",
        }[task_name]
        raise TaskLauncherError(
            f"{task_name} {mode} requires an explicit {compatibility_label} "
            "--checkpoint; the shared bundle checkpoint is not a task model"
        )
    if (
        _is_thrown_task(task_name)
        and mode in {"shadow", "deploy"}
        and not _option_present(arguments, "--profile")
    ):
        raise TaskLauncherError(
            f"{task_name} {mode} requires an explicit task-specific "
            "--profile binding camera 342222071785 and the task reset; it will "
            "be commissioned only after the physical point-cloud gates pass"
        )
    if mode in {"perception", "shadow"} and not any(
        _option_present(arguments, option) for option in TARGET_OPTIONS
    ):
        default_text = str(metadata.get("default_object_text", "")).strip()
        if not default_text:
            raise TaskLauncherError(
                "perception requires --object-text, --select-object-roi, or --object-roi"
            )
        arguments = ["--object-text", default_text, *arguments]
    module = (
        "sim2real.perception"
        if mode == "perception"
        else (
            "sim2real.tasks.thrown_shadow"
            if mode == "shadow"
            else "sim2real.deploy"
        )
    )
    if _is_thrown_task(task_name) and mode == "deploy":
        arguments = [
            "--policy-rate-hz",
            str(int(float(metadata["simulation_task_contract_control_hz"]))),
            *arguments,
        ]
    command = [
        sys.executable,
        "-m",
        module,
        "--pcd-config",
        str(config_path),
        "--policy-rgbd-resolution",
        "424x240",
        "--object-mask-mode",
        "guarded_v2",
        *arguments,
    ]
    return command, config_path, metadata


def _usage() -> str:
    return (
        "usage: python -m sim2real.tasks "
        "{tabletop,thrown_object,thrown_object_v60,thrown_object_v61} "
        "{config,perception,shadow,deploy} [ARGS...]"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values in (["-h"], ["--help"]):
        print(_usage())
        return 0
    if (
        len(values) == 2
        and values[0] in TASK_NAMES
        and values[1] in ("-h", "--help")
    ):
        print(
            f"usage: python -m sim2real.tasks {values[0]} "
            "{config,perception,shadow,deploy} [ARGS...]"
        )
        return 0
    try:
        if len(values) < 2:
            raise TaskLauncherError(_usage())
        task_name, mode = values[:2]
        command, config_path, metadata = build_task_command(
            task_name, mode, values[2:]
        )
        if mode == "config":
            print(
                json.dumps(
                    {
                        "task": task_name,
                        "config": str(config_path),
                        "policy_rgbd_resolution": "424x240",
                        "commissioning_status": metadata.get(
                            "commissioning_status"
                        ),
                        "robot_execution_enabled": metadata.get(
                            "robot_execution_enabled"
                        ),
                        "camera_contract_override": metadata.get(
                            "camera_contract_override"
                        ),
                        "calibration_sha256": metadata.get("calibration_sha256"),
                        "simulation_curriculum": metadata.get(
                            "simulation_curriculum"
                        ),
                        "simulation_task_contract_sha256": metadata.get(
                            "simulation_task_contract_sha256"
                        ),
                        "simulation_task_contract_control_hz": metadata.get(
                            "simulation_task_contract_control_hz"
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        print(
            f"[Task launcher] task={task_name} mode={mode} "
            f"policy_rgbd=424x240 config={config_path}",
            flush=True,
        )
        print(
            "[Task launcher] exec=" + shlex.join(command),
            flush=True,
        )
        os.execv(sys.executable, command)
        raise AssertionError("os.execv returned unexpectedly")
    except TaskLauncherError as exc:
        print(f"task launcher: REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TaskLauncherError",
    "build_task_command",
    "materialize_task_config",
]
