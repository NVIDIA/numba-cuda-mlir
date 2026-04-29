#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import subprocess
import sys
import re
import os
from pathlib import Path

try:
    from tabulate import tabulate
except ImportError:

    def tabulate(data, headers, tablefmt="grid"):
        col_widths = [len(h) for h in headers]
        for row in data:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        separator = "-+-".join("-" * w for w in col_widths)
        result = [header_line, separator]
        for row in data:
            row_line = " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
            result.append(row_line)
        return "\n".join(result)


def pytest_addoption(parser):
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run benchmark tests with NCU profiling",
    )


def pytest_configure(config):
    config._benchmark_results = []


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "_benchmark_results") and session.config._benchmark_results:
        _print_consolidated_results(session.config._benchmark_results)


@pytest.fixture
def benchmark_runner(request):
    if not request.config.getoption("--benchmark"):
        pytest.skip("Use --benchmark to run profiling")

    def run_benchmark(script, mode=None):
        script_path = Path(script).resolve()
        benchmark_name = script_path.stem.replace("test_", "").replace("_", " ").title()
        if mode:
            benchmark_name = f"{benchmark_name} ({mode})"

        compile_times = _run_compile_time_measurement(script_path, mode)
        ncu_kernel_times = _run_ncu_profiling(script_path, mode)

        request.config._benchmark_results.append(
            {
                "name": benchmark_name,
                "compile_times": compile_times,
                "ncu_times": ncu_kernel_times,
            }
        )

        return {"compile_times": compile_times, "ncu_times": ncu_kernel_times}

    return run_benchmark


def _run_compile_time_measurement(script_path, mode=None):
    env = os.environ.copy()
    cmd = [sys.executable, str(script_path)]
    if mode:
        cmd.append(mode)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, env=env, timeout=300
        )
        return _parse_compile_times(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Compile time measurement failed:")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        print("Compile time measurement timed out (>5 minutes)")
        raise


def _run_ncu_profiling(script_path, mode=None):
    try:
        subprocess.run(["ncu", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: NCU not found. Skipping NCU profiling.")
        return {}

    env = os.environ.copy()
    cmd = [
        "ncu",
        "--metrics",
        "gpu__time_duration.sum",
        "--target-processes",
        "all",
        "--csv",
        sys.executable,
        str(script_path),
    ]
    if mode:
        cmd.append(mode)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        return _parse_ncu_csv(result.stdout)
    except subprocess.TimeoutExpired:
        print("NCU profiling timed out (>10 minutes)")
        return {}
    except Exception as e:
        print(f"NCU profiling failed: {e}")
        return {}


def _parse_compile_times(stdout):
    compile_times = {}
    in_section = False

    for line in stdout.split("\n"):
        if "=== COMPILE TIMES ===" in line:
            in_section = True
            continue
        if in_section:
            if line.startswith("==="):
                break
            match = re.search(r"(Numba-CUDA|cuSIMT).*?:\s*([\d.]+)\s*ms", line, re.IGNORECASE)
            if match:
                variant = match.group(1).lower()
                time_ms = float(match.group(2))
                compile_times[variant] = time_ms

    return compile_times


def _parse_ncu_csv(csv_text):
    kernel_times = {"numba-cuda": [], "cusimt": []}
    lines = csv_text.split("\n")

    for line in lines:
        if not line.strip():
            continue
        line_lower = line.lower()
        if "numba_cuda" in line_lower or "cusimt" in line_lower:
            try:
                parts = line.split('","')
                if len(parts) < 2:
                    parts = line.split(",")
                parts = [p.strip('"').strip() for p in parts]

                time_ns = None
                for p in reversed(parts):
                    try:
                        time_ns = float(p.replace(",", ""))
                        break
                    except ValueError:
                        continue

                if time_ns is None:
                    continue

                time_ms = time_ns / 1e6
                if "cusimt" in line_lower:
                    kernel_times["cusimt"].append(time_ms)
                elif "numba_cuda" in line_lower:
                    kernel_times["numba-cuda"].append(time_ms)
            except (ValueError, IndexError):
                continue

    result = {}
    for variant, times in kernel_times.items():
        if times:
            result[variant] = sum(times)

    return result


def _print_consolidated_results(all_results):
    headers = [
        "Benchmark",
        "Numba-CUDA Compile (ms)",
        "cuSIMT Compile (ms)",
        "Compile Speedup",
        "Numba-CUDA Kernel (ms)",
        "cuSIMT Kernel (ms)",
        "Kernel Speedup",
    ]
    table_data = []

    for result in all_results:
        benchmark_name = result["name"]
        compile_times = result["compile_times"]
        ncu_times = result["ncu_times"]

        numba_cuda_compile = compile_times.get("numba-cuda")
        cusimt_compile = compile_times.get("cusimt")
        numba_cuda_kernel = ncu_times.get("numba-cuda")
        cusimt_kernel = ncu_times.get("cusimt")

        if numba_cuda_compile and cusimt_compile:
            compile_speedup = f"{numba_cuda_compile / cusimt_compile:.2f}x"
        else:
            compile_speedup = "N/A"

        if numba_cuda_kernel and cusimt_kernel:
            kernel_speedup = f"{numba_cuda_kernel / cusimt_kernel:.2f}x"
        else:
            kernel_speedup = "N/A"

        row = [
            benchmark_name,
            f"{numba_cuda_compile:.2f}" if numba_cuda_compile else "N/A",
            f"{cusimt_compile:.2f}" if cusimt_compile else "N/A",
            compile_speedup,
            f"{numba_cuda_kernel:.4f}" if numba_cuda_kernel else "N/A",
            f"{cusimt_kernel:.4f}" if cusimt_kernel else "N/A",
            kernel_speedup,
        ]
        table_data.append(row)

    print(f"\n{'=' * 100}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'=' * 100}")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"{'=' * 100}\n")


def verify_against_reference(reference, implementation, tolerance=1e-5, name=""):
    import numpy as np

    if isinstance(reference, tuple):
        for i, (ref, impl) in enumerate(zip(reference, implementation)):
            max_err = np.max(np.abs(ref - impl))
            assert max_err < tolerance, (
                f"{name} output {i}: max error {max_err:.2e} exceeds tolerance {tolerance:.2e}"
            )
        max_err = max(np.max(np.abs(r - i)) for r, i in zip(reference, implementation))
    else:
        max_err = np.max(np.abs(reference - implementation))
        assert max_err < tolerance, (
            f"{name}: max error {max_err:.2e} exceeds tolerance {tolerance:.2e}"
        )
