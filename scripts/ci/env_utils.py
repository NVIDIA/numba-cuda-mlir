# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for CI test environments."""

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from typing import Sequence

logger = logging.getLogger(__name__)

CUSIMT_ROOT = Path(__file__).resolve().parent.parent.parent

CUDA_CONFIGS = {
    "12.8": {
        "version_pin": "12.8.*",
        "extra": "cu12",
        "cupy": "cupy-cuda12x",
        "torch_index": "https://download.pytorch.org/whl/cu128",
        "nvmath": "nvmath-python[cu12]",
        "mathdx": "nvidia-libmathdx-cu12",
    },
    "12.9": {
        "version_pin": "12.9.*",
        "extra": "cu12",
        "cupy": "cupy-cuda12x",
        "torch_index": "https://download.pytorch.org/whl/cu129",
        "nvmath": "nvmath-python[cu12]",
        "mathdx": "nvidia-libmathdx-cu12",
    },
    "13.0": {
        "version_pin": "13.0.*",
        "extra": "cu13",
        "cupy": "cupy-cuda13x",
        "torch_index": "https://download.pytorch.org/whl/cu130",
        "nvmath": "nvmath-python[cu13]",
        "mathdx": "nvidia-libmathdx-cu13",
    },
    "13.1": {
        "version_pin": "13.1.*",
        "extra": "cu13",
        "cupy": "cupy-cuda13x",
        "torch_index": "https://download.pytorch.org/whl/cu130",
        "nvmath": "nvmath-python[cu13]",
        "mathdx": "nvidia-libmathdx-cu13",
    },
    "13.2": {
        "version_pin": "13.2.*",
        "extra": "cu13",
        "cupy": "cupy-cuda13x",
        "torch_index": "https://download.pytorch.org/whl/cu130",
        "nvmath": "nvmath-python[cu13]",
        "mathdx": "nvidia-libmathdx-cu13",
    },
}


def get_cuda_version() -> str:
    if v := os.environ.get("CUDA_VERSION"):
        return v
    cuda_bin = _find_cuda_bin()
    if cuda_bin:
        try:
            result = subprocess.run(
                [os.path.join(cuda_bin, "nvcc"), "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in result.stdout.splitlines():
                if "release" in line:
                    # e.g. "Cuda compilation tools, release 13.0, V13.0.80"
                    parts = line.split("release")
                    if len(parts) > 1:
                        ver = parts[1].split(",")[0].strip()
                        # Normalize: "12.9" stays, "13.0" stays
                        major, minor = ver.split(".")[:2]
                        return f"{major}.{minor}"
        except Exception:
            logger.exception("Failed to probe nvcc")
    return "13.0"


def get_cuda_config(cuda_version: str | None = None) -> dict:
    version = cuda_version or get_cuda_version()
    if version not in CUDA_CONFIGS:
        raise ValueError(
            f"Unsupported CUDA version: {version}. Supported: {list(CUDA_CONFIGS)}"
        )
    return CUDA_CONFIGS[version]


def _find_cuda_bin(cuda_version: str | None = None) -> str | None:
    """Find CUDA bin directory, preferring one matching cuda_version if specified."""
    cuda_version = cuda_version or os.environ.get("CUDA_VERSION")
    candidates = [
        "/usr/local/cuda/bin",  # Container/system default
        os.environ.get("CUDA_HOME", "") + "/bin",
        os.environ.get("CUDA_PATH", "") + "/bin",
        "/proj/cuda/13.2/Linux_x86_64/bin",  # NVIDIA internal
        "/proj/cuda/13.1/Linux_x86_64/bin",  # NVIDIA internal
        "/proj/cuda/13.0/Linux_x86_64/bin",  # NVIDIA internal
        "/proj/cuda/12.9/Linux_x86_64/bin",  # NVIDIA internal
        "/proj/cuda/12.8/Linux_x86_64/bin",  # NVIDIA internal
    ]
    if cuda_version:
        version_specific = f"/proj/cuda/{cuda_version}/Linux_x86_64/bin"
        if os.path.isdir(version_specific) and os.path.isfile(
            os.path.join(version_specific, "nvcc")
        ):
            return version_specific
    for path in candidates:
        if path and os.path.isdir(path) and os.path.isfile(os.path.join(path, "nvcc")):
            return path
    return None


def _default_env() -> dict:
    """Get default environment with CUDA in PATH."""
    env = os.environ.copy()
    cuda_bin = _find_cuda_bin()
    if cuda_bin:
        env["PATH"] = f"{cuda_bin}:{env.get('PATH', '')}"
        env.setdefault("CUDA_HOME", str(Path(cuda_bin).parent))
    env.setdefault("CUSIMT_ICE_FULL_TB", "1")
    return env


def run(
    cmd: str | Sequence[str],
    *,
    check: bool = True,
    capture: bool = False,
    cwd: Path | None = None,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run a shell command."""
    if isinstance(cmd, str):
        print(f"+ {cmd}", flush=True)
        cmd = ["bash", "-c", cmd]
    else:
        print(f"+ {' '.join(cmd)}", flush=True)
    if env is None:
        env = _default_env()
    kwargs = {"check": check, "cwd": cwd, "env": env}
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    else:
        kwargs["stdout"] = sys.stdout
        kwargs["stderr"] = sys.stderr
    return subprocess.run(cmd, **kwargs)


class VEnv:
    """A virtual environment context for running commands."""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.bin = self.path / "bin"
        self.python = self.bin / "python"
        self.pip = self.bin / "pip"
        self.history: list[str] = []

    def _record(self, cmd: str) -> None:
        self.history.append(cmd)

    def _run_with_history(
        self, cmd: list | str, **kwargs
    ) -> subprocess.CompletedProcess:
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        self._record(cmd_str)
        try:
            return run(cmd, **kwargs)
        except subprocess.CalledProcessError:
            self.print_history()
            raise

    def print_history(self) -> None:
        print("\n# To reproduce, run:\n", file=sys.stderr)
        print(f"python -m venv {self.path}", file=sys.stderr)
        cuda_bin = _find_cuda_bin()
        if cuda_bin:
            print(f"export PATH={cuda_bin}:$PATH", file=sys.stderr)
        for cmd in self.history:
            print(cmd, file=sys.stderr)
        print(file=sys.stderr)

    @classmethod
    def create(cls, path: Path, python: str = sys.executable) -> "VEnv":
        """Create a new virtual environment."""
        path = Path(path)
        run([python, "-m", "venv", str(path)])
        venv = cls(path)
        venv.run_pip("install", "--upgrade", "pip")
        return venv

    def run(self, cmd: str, **kwargs) -> subprocess.CompletedProcess:
        """Run a shell command with the venv activated."""
        activate = f"source {self.bin}/activate && {cmd}"
        return self._run_with_history(activate, **kwargs)

    def run_pip(self, *args: str, **kwargs) -> subprocess.CompletedProcess:
        """Run pip in this environment."""
        return self._run_with_history([str(self.pip), *args], **kwargs)

    def run_python(self, *args: str, **kwargs) -> subprocess.CompletedProcess:
        """Run python in this environment."""
        return self._run_with_history([str(self.python), *args], **kwargs)

    def install(self, *packages: str, **pip_kwargs) -> subprocess.CompletedProcess:
        """Install packages via pip."""
        return self.run_pip("install", *packages, **pip_kwargs)

    def install_project(self, extras: str = "") -> subprocess.CompletedProcess:
        """Install the current project (cusimt) in editable mode."""
        spec = f"-e .[{extras}]" if extras else "-e ."
        return self.run_pip("install", spec, cwd=CUSIMT_ROOT)


@contextmanager
def temp_venv(prefix: str = "ci_test_"):
    """Context manager for a temporary virtual environment."""
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    venv_path = Path(tmpdir) / "venv"
    try:
        yield VEnv.create(venv_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def add_standard_parser_args(
    parser: argparse.ArgumentParser,
    errors_file_default: Path | None = None,
) -> None:
    parser.add_argument(
        "--venv",
        type=Path,
        help="Path to existing venv (creates temp venv if not provided)",
    )
    parser.add_argument(
        "--keep-venv",
        action="store_true",
        help="Keep the temp venv after running (prints path)",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Additional arguments to pass to pytest (after --)",
    )
    if errors_file_default is not None:
        parser.add_argument(
            "--errors-file",
            type=Path,
            default=errors_file_default,
            help=f"File to save error details (default: {errors_file_default.name})",
        )


@contextmanager
def resolve_venv(args: argparse.Namespace, keep_venv_path: str, temp_prefix: str):
    if args.venv:
        yield VEnv(args.venv)
    elif getattr(args, "keep_venv", False):
        venv = VEnv.create(CUSIMT_ROOT / keep_venv_path)
        yield venv
        print(f"Venv kept at: {venv.path}")
    else:
        with temp_venv(prefix=temp_prefix) as venv:
            yield venv


MLIR_PACKAGES = [
    "mlir-native-tools",
    "mlir-python-bindings",
]
MLIR_FIND_LINKS = "https://llvm.github.io/eudsl"

# Pinned nvidia package versions for each CUDA version.
# Use pin_nvidia_packages() to ensure consistent versions after installing
# packages (like torch) that may pull in incompatible nvidia library versions.
NVIDIA_PACKAGES_FOR_CUDA = {
    "12.8": [
        "nvidia-cublas-cu12==12.8.*",
        "nvidia-cuda-cccl-cu12==12.8.*",
        "nvidia-cuda-cupti-cu12==12.8.*",
        "nvidia-cuda-nvcc-cu12==12.8.*",
        "nvidia-cuda-nvrtc-cu12==12.8.*",
        "nvidia-cuda-runtime-cu12==12.8.*",
        "nvidia-nvjitlink-cu12==12.8.*",
        "nvidia-nvtx-cu12==12.8.*",
    ],
    "12.9": [
        "nvidia-cublas-cu12==12.9.*",
        "nvidia-cuda-cccl-cu12==12.9.*",
        "nvidia-cuda-cupti-cu12==12.9.*",
        "nvidia-cuda-nvcc-cu12==12.9.*",
        "nvidia-cuda-nvrtc-cu12==12.9.*",
        "nvidia-cuda-runtime-cu12==12.9.*",
        "nvidia-nvjitlink-cu12==12.9.*",
        "nvidia-nvtx-cu12==12.9.*",
    ],
    "13.0": [
        "nvidia-cublas==13.0.*",
        "nvidia-cuda-cccl==13.0.*",
        "nvidia-cuda-cupti==13.0.*",
        "nvidia-cuda-nvrtc==13.0.*",
        "nvidia-cuda-runtime==13.0.*",
        "nvidia-nvjitlink==13.0.*",
        "nvidia-nvtx==13.0.*",
        "nvidia-nvvm==13.0.*",
    ],
    "13.1": [
        "nvidia-cublas==13.1.*",
        "nvidia-cuda-cccl==13.1.*",
        "nvidia-cuda-cupti==13.1.*",
        "nvidia-cuda-nvrtc==13.1.*",
        "nvidia-cuda-runtime==13.1.*",
        "nvidia-nvjitlink==13.1.*",
        "nvidia-nvtx==13.1.*",
        "nvidia-nvvm==13.1.*",
    ],
    "13.2": [
        "nvidia-cublas==13.2.*",
        "nvidia-cuda-cccl==13.2.*",
        "nvidia-cuda-cupti==13.2.*",
        "nvidia-cuda-nvrtc==13.2.*",
        "nvidia-cuda-runtime==13.2.*",
        "nvidia-nvjitlink==13.2.*",
        "nvidia-nvtx==13.2.*",
        "nvidia-nvvm==13.2.*",
    ],
}

# Packages that torch cu130 installs but are only available at CUDA 12.x.
# These must be removed so system CUDA libraries are used instead.
NVIDIA_PACKAGES_TO_REMOVE = {
    "12.9": [],
    "13.0": [
        "nvidia-cufft",
        "nvidia-cusolver",
        "nvidia-cusparse",
    ],
    "13.1": [
        "nvidia-cufft",
        "nvidia-cusolver",
        "nvidia-cusparse",
    ],
    "13.2": [
        "nvidia-cufft",
        "nvidia-cusolver",
        "nvidia-cusparse",
    ],
}

# Torch-specific nvidia packages that should be upgraded to latest after
# removing the CUDA 12.x packages. These have their own version schemes
# (not CUDA-aligned) and the latest versions work with CUDA 13.0.
NVIDIA_PACKAGES_TO_UPGRADE = {
    "12.9": [],
    "13.0": [
        "nvidia-cudnn-cu13",
        "nvidia-cufile",
        "nvidia-curand",
        "nvidia-cusparselt-cu13",
        "nvidia-nccl-cu13",
        "nvidia-nvshmem-cu13",
    ],
    "13.1": [
        "nvidia-cudnn-cu13",
        "nvidia-cufile",
        "nvidia-curand",
        "nvidia-cusparselt-cu13",
        "nvidia-nccl-cu13",
        "nvidia-nvshmem-cu13",
    ],
    "13.2": [
        "nvidia-cudnn-cu13",
        "nvidia-cufile",
        "nvidia-curand",
        "nvidia-cusparselt-cu13",
        "nvidia-nccl-cu13",
        "nvidia-nvshmem-cu13",
    ],
}


def pin_nvidia_packages(venv: VEnv, cuda_version: str = "13.0") -> None:
    """Pin nvidia packages to versions matching the CUDA version."""
    packages = NVIDIA_PACKAGES_FOR_CUDA.get(cuda_version, [])
    if packages:
        venv.run_pip("install", "--upgrade", *packages)
    to_remove = NVIDIA_PACKAGES_TO_REMOVE.get(cuda_version, [])
    if to_remove:
        venv.run_pip("uninstall", "-y", *to_remove, check=False)
    to_upgrade = NVIDIA_PACKAGES_TO_UPGRADE.get(cuda_version, [])
    if to_upgrade:
        venv.run_pip("install", "--upgrade", *to_upgrade)


def _mlir_python_packages_from_mlir_dir() -> Path | None:
    """Derive MLIR Python packages path from MLIR_DIR env var.

    MLIR_DIR points to <install>/lib/cmake/mlir.  The Python packages
    live at <install>/python_packages/mlir_core/.
    """
    mlir_dir = os.environ.get("MLIR_DIR")
    if not mlir_dir:
        return None
    install_root = Path(mlir_dir).resolve().parent.parent.parent
    pkg = install_root / "python_packages" / "mlir_core"
    return pkg if pkg.is_dir() else None


def install_mlir(venv: VEnv) -> None:
    """Install MLIR packages into a venv.

    When MLIR_DIR is set, uses the MLIR Python packages from the local
    LLVM build tree via a .pth file so that libMLIRPythonCAPI.so matches
    what libMLIRToLLVM70.so was compiled against.  Otherwise uses eudsl.
    """
    local_mlir = _mlir_python_packages_from_mlir_dir()
    if local_mlir:
        print(f"Using MLIR Python bindings from local build: {local_mlir}")
        venv.run_pip("uninstall", "-y", "mlir-python-bindings", check=False)
        result = venv.run_python(
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
            capture=True,
        )
        site_packages = Path(result.stdout.strip())
        pth_file = site_packages / "mlir_local_build.pth"
        pth_file.write_text(str(local_mlir) + "\n")
    else:
        venv.run_pip("install", "--upgrade", *MLIR_PACKAGES, "-f", MLIR_FIND_LINKS)


def install_cusimt_editable(
    venv: VEnv, extras: str | None = None, cusimt_root: Path | None = None
) -> None:
    """Install cusimt into a venv.

    If a pre-built wheel is available in <cusimt_root>/dist/, install that
    (preferred in CI — no rebuild, no cmake or CUDA Toolkit required).
    Otherwise fall back to an editable install from source (local dev /
    no wheel).
    """
    if extras is None:
        extras = get_cuda_config()["extra"]
    root = cusimt_root or CUSIMT_ROOT

    # Prefer a pre-built wheel if one exists — avoids rebuilding the
    # C extension + MLIR bindings in every test job.
    wheels = sorted((root / "dist").glob("cusimt-*.whl"))
    if wheels:
        wheel = wheels[-1]
        spec = f"{wheel}[{extras}]" if extras else str(wheel)
        print(f"Installing cusimt from pre-built wheel: {wheel.name}")
        venv.run_pip("install", spec)
        venv.run_pip("list")
        return

    # No wheel available — do a full editable install from source.
    install_mlir(venv)
    spec = f"{root}[{extras}]" if extras else str(root)
    venv.run_pip("install", "-e", spec)
    venv.run_pip("list")


@dataclass
class JUnitResults:
    """Test results parsed from JUnit XML."""

    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0

    @property
    def has_failures(self) -> bool:
        return self.failed > 0 or self.errors > 0 or self.total == 0

    def print_summary(self) -> None:
        """Print test results summary to stdout."""
        print("\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"  Passed:  {self.passed}")
        print(f"  Failed:  {self.failed}")
        print(f"  Errors:  {self.errors}")
        print(f"  Skipped: {self.skipped}")
        print("=" * 50)
        if self.total > 0:
            print(f"  Pass rate: {self.pass_rate:.1f}%")
        print("=" * 50)


def parse_junit_xml(junit_xml: Path) -> JUnitResults:
    """Parse JUnit XML file and return test results."""
    results = JUnitResults()
    if not junit_xml.exists():
        return results
    tree = ET.parse(junit_xml)
    root = tree.getroot()
    for testsuite in root.iter("testsuite"):
        tests = int(testsuite.get("tests", 0))
        failures = int(testsuite.get("failures", 0))
        errors = int(testsuite.get("errors", 0))
        skipped = int(testsuite.get("skipped", 0))
        results.passed += tests - failures - errors - skipped
        results.failed += failures
        results.errors += errors
        results.skipped += skipped
    return results


def save_junit_errors(junit_xml: Path, output_file: Path) -> None:
    """Save all test failures/errors from JUnit XML to a file for analysis."""
    if not junit_xml.exists():
        return
    tree = ET.parse(junit_xml)
    with open(output_file, "w") as f:
        for testcase in tree.iter("testcase"):
            name = f"{testcase.get('classname')}::{testcase.get('name')}"
            failure = testcase.find("failure")
            error = testcase.find("error")
            if failure is not None:
                f.write(f"FAILURE: {name}\n")
                f.write(f"  message: {failure.get('message', '')}\n")
                if failure.text:
                    f.write(f"  {failure.text}\n")
                f.write("\n")
            elif error is not None:
                f.write(f"ERROR: {name}\n")
                f.write(f"  message: {error.get('message', '')}\n")
                if error.text:
                    f.write(f"  {error.text}\n")
                f.write("\n")


def count_failures_by_file(junit_xml: Path) -> dict[str, int]:
    """
    Count test failures/errors by test file.

    Returns dict mapping test file name (e.g. 'test_caching.py') to failure count.
    """
    if not junit_xml.exists():
        return {}

    counts: dict[str, int] = {}
    tree = ET.parse(junit_xml)
    for testcase in tree.iter("testcase"):
        failure = testcase.find("failure")
        error = testcase.find("error")
        if failure is not None or error is not None:
            # classname is like 'numba.cuda.tests.cudapy.test_caching.CUDACachingTest'
            # We want the module name before the class: 'test_caching'
            classname = testcase.get("classname", "")
            parts = classname.split(".")
            # Find the test file - it's typically 'test_*' pattern
            test_file = None
            for part in reversed(parts):
                if part.startswith("test_"):
                    test_file = f"{part}.py"
                    break
            if test_file is None and len(parts) >= 2:
                # Fallback: use second-to-last component as module name
                test_file = f"{parts[-2]}.py"
            if test_file:
                counts[test_file] = counts.get(test_file, 0) + 1

    return counts


def print_failures_by_file(junit_xml: Path) -> None:
    """Print failures by test file in descending order."""
    counts = count_failures_by_file(junit_xml)
    if not counts:
        print("No failures found.")
        return

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    total = sum(counts.values())
    print("\n" + "=" * 50)
    print("FAILURES BY TEST FILE")
    print("=" * 50)
    for test_file, count in sorted_counts:
        print(f"  {count:4d}  {test_file}")
    print("=" * 50)
    print(f"  {total:4d}  TOTAL")
    print("=" * 50)


def filter_junit_failures(
    junit_xml: Path, patterns: list[str], output_xml: Path | None = None
) -> tuple[JUnitResults, int]:
    """
    Filter failures/errors from JUnit XML that match any of the given patterns.

    Patterns are matched against the test name (classname::name), failure/error
    message, and failure/error text using regex.

    Returns (updated results, number of filtered failures).
    """
    if not junit_xml.exists():
        return JUnitResults(), 0

    tree = ET.parse(junit_xml)
    root = tree.getroot()
    filtered_count = 0
    compiled = [re.compile(p) for p in patterns]

    def matches(text: str | None) -> bool:
        return text is not None and any(p.search(text) for p in compiled)

    for testsuite in root.iter("testsuite"):
        for testcase in testsuite.findall("testcase"):
            name = f"{testcase.get('classname')}::{testcase.get('name')}"
            failure = testcase.find("failure")
            error = testcase.find("error")
            if failure is not None or error is not None:
                # Match against name, message, or text content
                should_filter = matches(name)
                if failure is not None:
                    should_filter = should_filter or matches(failure.get("message"))
                    should_filter = should_filter or matches(failure.text)
                if error is not None:
                    should_filter = should_filter or matches(error.get("message"))
                    should_filter = should_filter or matches(error.text)

                if should_filter:
                    if failure is not None:
                        testcase.remove(failure)
                        testsuite.set(
                            "failures", str(int(testsuite.get("failures", 0)) - 1)
                        )
                    if error is not None:
                        testcase.remove(error)
                        testsuite.set(
                            "errors", str(int(testsuite.get("errors", 0)) - 1)
                        )
                    filtered_count += 1

    out = output_xml or junit_xml
    tree.write(out, encoding="unicode")
    return parse_junit_xml(out), filtered_count
