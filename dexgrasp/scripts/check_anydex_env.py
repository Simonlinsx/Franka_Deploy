#!/usr/bin/env python3
"""Read-only environment checker for the dexgrasp visualization and AnyDex stacks.

The checker imports installed modules and inspects files, but never installs packages,
builds extensions, downloads checkpoints, or accesses robot hardware.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANYDEX_ROOT = PROJECT_ROOT / "third_party" / "AnyDexGrasp"
DEFAULT_WEIGHT_ROOT = PROJECT_ROOT / "weights" / "logs" / "model"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    hint: str = ""


def _version_tuple(value: object) -> Tuple[int, ...]:
    match = re.match(r"\s*(\d+(?:\.\d+)*)", str(value))
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _dist_version(*names: str) -> Optional[str]:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _import(name: str):
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            module = importlib.import_module(name)
        return module, None
    except Exception as exc:  # Binary-extension failures are part of the report.
        message = f"{type(exc).__name__}: {exc}"
        noise = captured.getvalue().strip()
        if noise:
            message = f"{message}; import output: {noise[-240:]}"
        return None, message


def _package_check(
    import_name: str,
    display_name: str,
    *,
    distributions: Sequence[str] = (),
    expected: str = "",
    predicate=None,
    optional: bool = False,
) -> Check:
    module, error = _import(import_name)
    if error:
        status = "WARN" if optional else "FAIL"
        return Check(display_name, status, f"not importable ({error})", expected)

    version = getattr(module, "__version__", None)
    if not version:
        version = _dist_version(*(distributions or (display_name, import_name)))
    version_text = str(version or "unknown")
    if predicate is not None and not predicate(_version_tuple(version_text)):
        status = "WARN" if optional else "FAIL"
        return Check(display_name, status, f"import OK, version {version_text}", expected)
    return Check(display_name, "PASS", f"import OK, version {version_text}")


def _find_nvcc() -> Optional[Path]:
    candidates = []
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidates.append(Path(cuda_home) / "bin" / "nvcc")
    path_nvcc = shutil.which("nvcc")
    if path_nvcc:
        candidates.append(Path(path_nvcc))
    candidates.extend((Path("/usr/local/cuda/bin/nvcc"), Path("/usr/local/cuda-12.1/bin/nvcc")))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def _nvcc_check() -> Check:
    nvcc = _find_nvcc()
    if nvcc is None:
        return Check(
            "CUDA compiler",
            "FAIL",
            "nvcc not found",
            "Install the complete CUDA 11.7 toolkit; a CUDA runtime alone cannot build the extensions.",
        )
    try:
        result = subprocess.run(
            [str(nvcc), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check("CUDA compiler", "FAIL", f"cannot run {nvcc}: {exc}")
    output = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"release\s+(\d+\.\d+)", output)
    version = match.group(1) if match else "unknown"
    if result.returncode != 0 or version != "11.7":
        return Check(
            "CUDA compiler",
            "FAIL",
            f"{nvcc} reports CUDA {version}",
            "Official AnyDex requires nvcc 11.7, matching torch.version.cuda == '11.7'.",
        )
    return Check("CUDA compiler", "PASS", f"{nvcc} reports CUDA 11.7")


def shell_checks(anydex_root: Path) -> list[Check]:
    py = sys.version_info[:2]
    checks = [
        Check(
            "Python",
            "PASS" if py >= (3, 8) else "FAIL",
            f"{sys.version.split()[0]} at {sys.executable}",
            "The visualization shell supports Python 3.8 or newer.",
        ),
        _package_check(
            "numpy",
            "NumPy",
            expected="Use NumPy >=1.23,<3 for the visualization shell.",
            predicate=lambda v: v >= (1, 23) and v < (3,),
        ),
        _package_check(
            "open3d",
            "Open3D",
            expected="Use Open3D >=0.18,<0.20.",
            predicate=lambda v: v >= (0, 18) and v < (0, 20),
        ),
        _package_check(
            "cv2",
            "OpenCV",
            distributions=("opencv-python", "opencv-python-headless"),
            expected="Optional for image overlays and live RGB display.",
            optional=True,
        ),
        _package_check(
            "pyrealsense2",
            "pyrealsense2",
            expected="Optional for live RealSense acquisition; not needed for saved PCD input.",
            optional=True,
        ),
        _package_check(
            "scipy",
            "SciPy",
            expected="Optional for some geometric helpers.",
            optional=True,
        ),
    ]
    source_ok = (anydex_root / "models" / "minkowski_graspnet_single_point.py").is_file()
    checks.append(
        Check(
            "AnyDex source checkout",
            "PASS" if source_ok else "WARN",
            str(anydex_root),
            "The visualization shell can run without inference, but the official backend needs this checkout.",
        )
    )
    checks.append(
        Check(
            "Torch/ME backend",
            "INFO",
            "not required for point-cloud loading and grasp-pose visualization",
        )
    )
    return checks


def official_checks(
    anydex_root: Path,
    representation_checkpoint: Path,
    inspire_model_dir: Path,
) -> list[Check]:
    py = sys.version_info[:2]
    checks = [
        Check(
            "Python",
            "PASS" if py == (3, 8) else "FAIL",
            f"{sys.version.split()[0]} at {sys.executable}",
            "Official AnyDex targets Python 3.8.",
        ),
        _package_check(
            "numpy",
            "NumPy",
            expected="Use NumPy 1.23.x (1.23.5 recommended).",
            predicate=lambda v: v[:2] == (1, 23),
        ),
        _package_check(
            "open3d",
            "Open3D",
            expected="The reproducible environment pins Open3D 0.18.x.",
            predicate=lambda v: v[:2] == (0, 18),
        ),
        _package_check(
            "graspnetAPI",
            "graspnetAPI",
            distributions=("graspnetAPI",),
            expected="Optional in this robot-free adapter; upstream robot/collision utilities use it.",
            optional=True,
        ),
        _package_check(
            "MinkowskiEngine",
            "MinkowskiEngine",
            distributions=("MinkowskiEngine",),
            expected="Build MinkowskiEngine 0.5.x against PyTorch 1.13 and CUDA 11.7.",
            predicate=lambda v: v[:2] == (0, 5),
        ),
        _package_check(
            "pointnet2._ext",
            "pointnet2 CUDA extension",
            expected="Build third_party/AnyDexGrasp/pointnet2 in this exact environment.",
        ),
        _package_check(
            "knn_pytorch.knn_pytorch",
            "knn CUDA extension",
            expected=(
                "Build third_party/AnyDexGrasp/knn while torch.cuda.is_available() is true; "
                "otherwise setup.py silently builds CPU-only."
            ),
        ),
    ]

    torch, torch_error = _import("torch")
    if torch_error:
        checks.extend(
            (
                Check("PyTorch", "FAIL", f"not importable ({torch_error})", "Install PyTorch 1.13.1."),
                Check("PyTorch CUDA build", "FAIL", "unavailable because PyTorch did not import"),
                Check("GPU runtime", "FAIL", "unavailable because PyTorch did not import"),
            )
        )
    else:
        torch_version = str(getattr(torch, "__version__", "unknown"))
        compiled_cuda = str(getattr(getattr(torch, "version", None), "cuda", None))
        checks.append(
            Check(
                "PyTorch",
                "PASS" if _version_tuple(torch_version)[:2] == (1, 13) else "FAIL",
                f"version {torch_version}",
                "Official AnyDex targets PyTorch 1.13.x; 1.13.1 is recommended.",
            )
        )
        checks.append(
            Check(
                "PyTorch CUDA build",
                "PASS" if compiled_cuda == "11.7" else "FAIL",
                f"torch.version.cuda = {compiled_cuda}",
                "Install the cu117 build, not a CPU, cu118, or cu121 build.",
            )
        )
        cuda_available = bool(torch.cuda.is_available())
        device_detail = "CUDA unavailable"
        if cuda_available:
            try:
                device_detail = f"CUDA available: {torch.cuda.get_device_name(0)}"
            except Exception:
                device_detail = "CUDA available"
        checks.append(
            Check(
                "GPU runtime",
                "PASS" if cuda_available else "FAIL",
                device_detail,
                "AnyDex pointnet2 and several inference paths require CUDA. In a sandbox, rerun on the host.",
            )
        )

    checks.append(_nvcc_check())

    source_files = (
        anydex_root / "models" / "minkowski_graspnet_single_point.py",
        anydex_root / "pointnet2" / "setup.py",
        anydex_root / "knn" / "setup.py",
    )
    missing_source = [str(path) for path in source_files if not path.is_file()]
    checks.append(
        Check(
            "AnyDex source checkout",
            "FAIL" if missing_source else "PASS",
            f"missing: {', '.join(missing_source)}" if missing_source else str(anydex_root),
        )
    )

    checks.append(
        Check(
            "Representation checkpoint",
            "PASS" if representation_checkpoint.is_file() else "FAIL",
            str(representation_checkpoint),
            "Download the official model assets; the source repository does not include weights.",
        )
    )
    decision_models = list(inspire_model_dir.rglob("*.pth")) if inspire_model_dir.is_dir() else []
    checks.append(
        Check(
            "Inspire decision models",
            "PASS" if decision_models else "WARN",
            f"{len(decision_models)} .pth file(s) under {inspire_model_dir}",
            "Needed for AnyDex Inspire-hand type/depth scoring; not needed for two-finger proposal inspection.",
        )
    )
    return checks


def _print_profile(title: str, checks: Iterable[Check]) -> bool:
    checks = list(checks)
    print(f"\n[{title}]")
    for check in checks:
        print(f"{check.status:>4}  {check.name}: {check.detail}")
        if check.hint and check.status != "PASS":
            print(f"      -> {check.hint}")
    failed = sum(check.status == "FAIL" for check in checks)
    warnings = sum(check.status == "WARN" for check in checks)
    print(f"Result: {'NOT READY' if failed else 'READY'} ({failed} failure(s), {warnings} warning(s))")
    return failed == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("shell", "official", "all"),
        default="all",
        help="Check the visualization shell, the official AnyDex inference stack, or both.",
    )
    parser.add_argument("--anydex-root", type=Path, default=DEFAULT_ANYDEX_ROOT)
    parser.add_argument(
        "--representation-checkpoint",
        type=Path,
        default=None,
        help="Default: <project>/weights/logs/model/checkpoint.tar.18",
    )
    parser.add_argument(
        "--inspire-model-dir",
        type=Path,
        default=None,
        help="Default: <project>/weights/logs/model/inspire_model/obj140",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    anydex_root = args.anydex_root.expanduser().resolve()
    representation_checkpoint = (
        args.representation_checkpoint.expanduser().resolve()
        if args.representation_checkpoint
        else DEFAULT_WEIGHT_ROOT / "checkpoint.tar.18"
    )
    inspire_model_dir = (
        args.inspire_model_dir.expanduser().resolve()
        if args.inspire_model_dir
        else DEFAULT_WEIGHT_ROOT / "inspire_model" / "obj140"
    )

    print("AnyDex environment check (read-only)")
    print(f"Project: {PROJECT_ROOT}")
    print(f"AnyDex:  {anydex_root}")
    print("Profiles are independent: the visualization shell intentionally does not require Torch or CUDA.")

    ready = []
    if args.profile in ("shell", "all"):
        ready.append(_print_profile("visualization-shell", shell_checks(anydex_root)))
    if args.profile in ("official", "all"):
        ready.append(
            _print_profile(
                "official-anydex-inference",
                official_checks(anydex_root, representation_checkpoint, inspire_model_dir),
            )
        )
    return 0 if all(ready) else 1


if __name__ == "__main__":
    raise SystemExit(main())
