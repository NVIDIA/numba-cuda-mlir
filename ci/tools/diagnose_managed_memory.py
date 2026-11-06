# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary diagnostics for #430; failures are evidence, not expected passes."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time


PAIR = [
    "tests/numba_cuda_tests/cudapy/test_cuda_array_interface.py::TestCudaArrayInterface::test_launch_sync_two_streams",
    "tests/test_managed_memory.py::TestManagedMemory::test_managed_array_kernel_read",
]
VARIANTS = ("baseline", "stream-sync", "managed-sync", "both")


def checked(result):
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"CUDA call failed: {result!r}")
    return values[0] if len(values) == 1 else values


def environment():
    from cuda.bindings import driver
    import numba_cuda_mlir

    checked(driver.cuInit(0))
    attributes = {}
    for name in ("MANAGED_MEMORY", "CONCURRENT_MANAGED_ACCESS"):
        attr = getattr(driver.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_" + name)
        attributes[name] = checked(driver.cuDeviceGetAttribute(attr, 0))
    expected = {
        "numba-cuda-mlir": "0.6.0",
        "cuda-toolkit": "13.4.2",
        "cuda-bindings": "13.4.",
        "nvidia-cuda-nvrtc": "13.4.",
        "nvidia-cuda-runtime": "13.4.",
        "nvidia-nvjitlink": "13.4.",
    }
    versions = {package: version(package) for package in expected}
    deviations = {
        package: actual
        for package, actual in versions.items()
        if not (
            actual.startswith(expected[package])
            if expected[package].endswith(".")
            else actual == expected[package]
        )
    }
    if sys.platform != "win32":
        deviations["platform"] = sys.platform
    if attributes["CONCURRENT_MANAGED_ACCESS"]:
        deviations["CONCURRENT_MANAGED_ACCESS"] = 1
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": sys.version,
                "executable": sys.executable,
                "package": numba_cuda_mlir.__file__,
                "attributes": attributes,
                "versions": versions,
                "environment_deviations": deviations,
            },
            indent=2,
        ),
        flush=True,
    )


def probe(synchronize):
    import numpy as np
    from cuda.bindings import driver, nvrtc
    from numba_cuda_mlir import cuda

    cuda.current_context()
    # Allocate and initialize before launching any work. No managed allocation
    # or compiler work may accidentally synchronize the measured overlap.
    managed = cuda.managed_array(1, dtype=np.int32)
    managed[0] = 0
    major = checked(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0
        )
    )
    minor = checked(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0
        )
    )
    source = b"""extern "C" __global__ void delay() {
        unsigned long long start = clock64();
        while (clock64() - start < 200000000ULL) {}
    }"""
    program = checked(nvrtc.nvrtcCreateProgram(source, b"delay.cu", 0, [], []))
    module = None
    try:
        result = nvrtc.nvrtcCompileProgram(
            program, 1, [f"--gpu-architecture=sm_{major}{minor}".encode()]
        )
        if int(result[0]) != 0:
            size = checked(nvrtc.nvrtcGetProgramLogSize(program))
            log = bytearray(size)
            checked(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(log.decode(errors="replace"))
        cubin = bytearray(checked(nvrtc.nvrtcGetCUBINSize(program)))
        checked(nvrtc.nvrtcGetCUBIN(program, cubin))
        module = checked(driver.cuModuleLoadData(bytes(cubin)))
        function = checked(driver.cuModuleGetFunction(module, b"delay"))
        checked(driver.cuCtxSynchronize())
        checked(driver.cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0))
        if synchronize:
            checked(driver.cuCtxSynchronize())
        status = driver.cuStreamQuery(0)[0]
        expected = (
            driver.CUresult.CUDA_SUCCESS if synchronize else driver.CUresult.CUDA_ERROR_NOT_READY
        )
        print(f"before CPU access: {status!s}; synchronized={synchronize}", flush=True)
        if status != expected:
            raise RuntimeError(f"Probe precondition not satisfied: expected {expected!s}")
        managed[0] = 42
        checked(driver.cuCtxSynchronize())
        assert managed[0] == 42
        print("CPU access completed", flush=True)
    finally:
        if module is not None:
            checked(driver.cuCtxSynchronize())
            checked(driver.cuModuleUnload(module))
        checked(nvrtc.nvrtcDestroyProgram(program))


def prepare(source, destination, variant):
    shutil.copytree(source / "tests", destination / "tests")
    shutil.copy2(source / "pyproject.toml", destination / "pyproject.toml")
    if variant in ("stream-sync", "both"):
        path = destination / "tests/numba_cuda_tests/cudapy/test_cuda_array_interface.py"
        text = path.read_text()
        anchor = "        mock_sync.assert_has_calls([call(), call()])"
        if text.count(anchor) != 1:
            raise RuntimeError("Released stream test anchor changed")
        path.write_text(text.replace(anchor, anchor + "\n        cuda.synchronize()"))
    if variant in ("managed-sync", "both"):
        path = destination / "tests/test_managed_memory.py"
        text = path.read_text()
        anchor = "    ctx = cuda.current_context()"
        if text.count(anchor) != 1:
            raise RuntimeError("Released managed-memory helper anchor changed")
        # Every managed-memory test calls this helper before CPU initialization.
        path.write_text(text.replace(anchor, anchor + "\n    ctx.synchronize()"))


def run(args):
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    deadline = time.monotonic() + args.budget_seconds
    active_deadline = deadline
    results = []
    env = os.environ.copy()
    env.update(PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1")

    def execute(name, command, cwd, timeout, child_env=None):
        remaining = active_deadline - time.monotonic()
        record = {"name": name, "command": command, "cwd": str(cwd)}
        if remaining <= 1:
            record["status"] = "not-run-budget-exhausted"
        else:
            start = time.monotonic()
            with (output / f"{name}.log").open("wb") as log:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=cwd,
                        env=child_env or env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=min(timeout, remaining),
                        check=False,
                    )
                    code = completed.returncode
                    record.update(returncode=code, windows_status=f"0x{code & 0xFFFFFFFF:08X}")
                    record["status"] = "completed" if code == 0 else "failed"
                except subprocess.TimeoutExpired:
                    record["status"] = "timeout"
            record["seconds"] = round(time.monotonic() - start, 3)
        results.append(record)
        (output / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(record), flush=True)
        return record["status"] == "completed"

    python = [sys.executable, "-X", "faulthandler"]
    if not execute("environment", python + [str(script), "--child", "environment"], source, 60):
        return 1
    if not args.skip_suite:
        # Reserve half the allocation (at most ten minutes) for seeded suites.
        active_deadline -= min(600, args.budget_seconds / 2)
    with tempfile.TemporaryDirectory(prefix="managed-memory-diagnostic-") as temporary:
        root = Path(temporary)
        # Suppress Windows crash dialogs in children; retain faulthandler output.
        (root / "sitecustomize.py").write_text(
            "import sys\nif sys.platform == 'win32':\n"
            "    import ctypes\n    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)\n"
        )
        variants = {}
        for variant in args.variants:
            stage = root / variant
            prepare(source, stage, variant)
            variants[variant] = stage
        for variant, stage in variants.items():
            child_env = env.copy()
            child_env["PYTHONPATH"] = os.pathsep.join((str(root), str(stage / "tests")))
            for attempt in range(args.attempts):
                ok = execute(
                    f"pair-{variant}-{attempt + 1}",
                    python + ["-m", "pytest", "-vv", "-s", "-p", "no:randomly", *PAIR],
                    stage,
                    90,
                    child_env,
                )
                if not ok:
                    break
            if variant == "baseline":
                for mode in ("overlap", "synchronized"):
                    execute(
                        f"probe-{mode}",
                        python + [str(script), "--child", mode],
                        stage,
                        60,
                        child_env,
                    )
        if not args.skip_suite:
            active_deadline = deadline
            for variant in ("baseline", "both"):
                if variant not in variants:
                    continue
                stage = variants[variant]
                child_env = env.copy()
                child_env["PYTHONPATH"] = os.pathsep.join((str(root), str(stage / "tests")))
                execute(
                    f"suite-{variant}",
                    python
                    + [
                        "-m",
                        "pytest",
                        "-vv",
                        "--randomly-seed=3613105221",
                        "--ignore=tests/benchmarks",
                        "--ignore=tests/doc_examples",
                        "--deselect=tests/numba_cuda_tests/cudadrv/test_cuda_devicerecord.py::TestCudaDeviceRecordWithRecord::test_device_record_copy",
                        "tests/",
                    ],
                    stage,
                    300,
                    child_env,
                )
    # Crashes and timeouts must leave the workflow red, even if candidates pass.
    return int(any(result["status"] != "completed" for result in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("released-tests"))
    parser.add_argument("--output", type=Path, default=Path("diagnostic-results"))
    parser.add_argument("--attempts", type=int, choices=range(1, 21), default=5)
    parser.add_argument("--budget-seconds", type=int, default=1200)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--skip-suite", action="store_true")
    parser.add_argument("--child", choices=("environment", "overlap", "synchronized"))
    args = parser.parse_args()
    if args.child == "environment":
        environment()
    elif args.child:
        probe(args.child == "synchronized")
    else:
        return run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
