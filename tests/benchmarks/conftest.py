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

        compile_times = _run_compile_time_measurements(script_path, mode)
        ncu_kernel_times = _run_ncu_profiling(script_path, mode)

        request.config._benchmark_results.append(
            {
                "name": benchmark_name,
                "compile_times": compile_times,
                "ncu_times": ncu_kernel_times,
            }
        )

        return {
            "compile_times": compile_times,
            "ncu_times": ncu_kernel_times,
        }

    return run_benchmark


_NUM_SAMPLES = int(os.environ.get("BENCH_SAMPLES", "3"))


def _median(values):
    """Return the median of a list of numbers."""
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _run_compile_time_measurements(script_path, mode=None):
    return {
        "cold": _run_compile_time_measurement(script_path, mode, "cold"),
        "warm": _run_compile_time_measurement(script_path, mode, "warm"),
    }


def _run_compile_time_measurement(script_path, mode=None, compile_mode="cold"):
    """Measure compile time for each backend in its own isolated subprocess.

    Runs _NUM_SAMPLES iterations per backend and returns the median.
    The first iteration is a discarded warmup (OS page-cache priming).
    """
    env = os.environ.copy()
    results = {}

    for backend in ("numba-cuda", "numba-cuda-mlir"):
        samples = []
        for i in range(_NUM_SAMPLES + 1):
            cmd = [sys.executable, str(script_path)]
            if mode:
                cmd.append(mode)
            cmd.extend(["--compile-mode", compile_mode, "--backend", backend])

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=True, env=env, timeout=300
                )
                times = _parse_compile_times(result.stdout)
                key = "numba_cuda_mlir" if backend == "numba-cuda-mlir" else "numba-cuda"
                val = times.get(key)
                if val is not None:
                    if i > 0:
                        samples.append(val)
            except subprocess.CalledProcessError as e:
                print(f"Compile time measurement failed for {backend}:")
                print(f"  stdout: {e.stdout}")
                print(f"  stderr: {e.stderr}")
                raise
            except subprocess.TimeoutExpired:
                print(f"Compile time measurement timed out for {backend} (>5 minutes)")
                raise

        key = "numba_cuda_mlir" if backend == "numba-cuda-mlir" else "numba-cuda"
        if samples:
            results[key] = _median(samples)

    return results


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
            match = re.search(r"^\s*([^:]+):\s*([\d.]+)\s*ms", line, re.IGNORECASE)
            if match:
                variant = {
                    "numba-cuda": "numba-cuda",
                    "numba-cuda-mlir": "numba_cuda_mlir",
                }.get(match.group(1).strip().lower())
                if variant is not None:
                    compile_times[variant] = float(match.group(2))

    return compile_times


def _parse_ncu_csv(csv_text):
    kernel_times = {"numba-cuda": [], "numba_cuda_mlir": []}
    lines = csv_text.split("\n")

    for line in lines:
        if not line.strip():
            continue
        line_lower = line.lower()
        if "numba_cuda" in line_lower or "numba_cuda_mlir" in line_lower:
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
                if "numba_cuda_mlir" in line_lower:
                    kernel_times["numba_cuda_mlir"].append(time_ms)
                elif "numba_cuda" in line_lower:
                    kernel_times["numba-cuda"].append(time_ms)
            except (ValueError, IndexError):
                continue

    result = {}
    for variant, times in kernel_times.items():
        if times:
            result[variant] = sum(times)

    return result


def _geomean(values):
    """Compute geometric mean of positive values, ignoring None/zero."""
    vals = [v for v in values if v and v > 0]
    if not vals:
        return None
    import math

    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _geomean_row(all_results):
    """Build a 'Geomean' summary row matching the column layout of _print_consolidated_results."""
    cold_nc, cold_mlir = [], []
    warm_nc, warm_mlir = [], []
    kern_nc, kern_mlir = [], []

    for r in all_results:
        cold = r["compile_times"].get("cold", {})
        warm = r["compile_times"].get("warm", {})
        cold_nc.append(cold.get("numba-cuda"))
        cold_mlir.append(cold.get("numba_cuda_mlir"))
        warm_nc.append(warm.get("numba-cuda"))
        warm_mlir.append(warm.get("numba_cuda_mlir"))
        kern_nc.append(r["ncu_times"].get("numba-cuda"))
        kern_mlir.append(r["ncu_times"].get("numba_cuda_mlir"))

    def _fmt(val, precision=2):
        return f"{val:.{precision}f}" if val else "N/A"

    def _speedup(a, b):
        return f"{a / b:.2f}x" if a and b else "N/A"

    g_cold_nc, g_cold_mlir = _geomean(cold_nc), _geomean(cold_mlir)
    g_warm_nc, g_warm_mlir = _geomean(warm_nc), _geomean(warm_mlir)
    g_kern_nc, g_kern_mlir = _geomean(kern_nc), _geomean(kern_mlir)

    return [
        "GEOMEAN",
        _fmt(g_cold_nc),
        _fmt(g_cold_mlir),
        _speedup(g_cold_nc, g_cold_mlir),
        _fmt(g_warm_nc),
        _fmt(g_warm_mlir),
        _speedup(g_warm_nc, g_warm_mlir),
        _fmt(g_kern_nc, 4),
        _fmt(g_kern_mlir, 4),
        _speedup(g_kern_nc, g_kern_mlir),
    ]


def _print_consolidated_results(all_results):
    headers = [
        "Benchmark",
        "Numba-CUDA Cold Compile (ms)",
        "numba-cuda-mlir Cold Compile (ms)",
        "Cold Compile Speedup",
        "Numba-CUDA Warm Compile (ms)",
        "numba-cuda-mlir Warm Compile (ms)",
        "Warm Compile Speedup",
        "Numba-CUDA Kernel (ms)",
        "numba-cuda-mlir Kernel (ms)",
        "Kernel Speedup",
    ]
    table_data = []

    for result in all_results:
        benchmark_name = result["name"]
        compile_times = result["compile_times"]
        ncu_times = result["ncu_times"]

        cold_compile_times = compile_times.get("cold", {})
        warm_compile_times = compile_times.get("warm", {})
        numba_cuda_cold_compile = cold_compile_times.get("numba-cuda")
        numba_cuda_mlir_cold_compile = cold_compile_times.get("numba_cuda_mlir")
        numba_cuda_warm_compile = warm_compile_times.get("numba-cuda")
        numba_cuda_mlir_warm_compile = warm_compile_times.get("numba_cuda_mlir")
        numba_cuda_kernel = ncu_times.get("numba-cuda")
        numba_cuda_mlir_kernel = ncu_times.get("numba_cuda_mlir")

        if numba_cuda_cold_compile and numba_cuda_mlir_cold_compile:
            cold_compile_speedup = f"{numba_cuda_cold_compile / numba_cuda_mlir_cold_compile:.2f}x"
        else:
            cold_compile_speedup = "N/A"

        if numba_cuda_warm_compile and numba_cuda_mlir_warm_compile:
            warm_compile_speedup = f"{numba_cuda_warm_compile / numba_cuda_mlir_warm_compile:.2f}x"
        else:
            warm_compile_speedup = "N/A"

        if numba_cuda_kernel and numba_cuda_mlir_kernel:
            kernel_speedup = f"{numba_cuda_kernel / numba_cuda_mlir_kernel:.2f}x"
        else:
            kernel_speedup = "N/A"

        row = [
            benchmark_name,
            f"{numba_cuda_cold_compile:.2f}" if numba_cuda_cold_compile else "N/A",
            (f"{numba_cuda_mlir_cold_compile:.2f}" if numba_cuda_mlir_cold_compile else "N/A"),
            cold_compile_speedup,
            f"{numba_cuda_warm_compile:.2f}" if numba_cuda_warm_compile else "N/A",
            (f"{numba_cuda_mlir_warm_compile:.2f}" if numba_cuda_mlir_warm_compile else "N/A"),
            warm_compile_speedup,
            f"{numba_cuda_kernel:.4f}" if numba_cuda_kernel else "N/A",
            f"{numba_cuda_mlir_kernel:.4f}" if numba_cuda_mlir_kernel else "N/A",
            kernel_speedup,
        ]
        table_data.append(row)

    if len(table_data) > 1:
        table_data.append(_geomean_row(all_results))

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
