#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kernel launch-latency microbenchmark comparing numba-cuda vs numba-cuda-mlir.

Usage:
    python launch_latency_ubench.py
"""

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

GRID = (1,)
BLOCK = (1,)
WARMUP = 500
ITERATIONS = 5000
KERNEL_NAMES = ["empty", "1_array_arg", "16_scalar_args", "16_array_args", "256_scalar_args"]


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------


def _build_empty_kernel(cuda):
    @cuda.jit("void()")
    def kernel():
        pass

    return kernel[GRID, BLOCK], ()


def _build_1_array_arg_kernel(cuda):
    @cuda.jit("void(float32[::1])")
    def kernel(arr):
        pass

    buf = cuda.device_array(1, dtype=np.float32)
    return kernel[GRID, BLOCK], (buf,)


def _build_multi_scalar_kernel(cuda, n):
    sig = "void(" + ", ".join(["float32"] * n) + ")"
    params = ", ".join(f"a{i}" for i in range(n))
    ns = {}
    exec(f"def kernel({params}):\n    pass\n", ns)  # noqa: S102
    kernel = cuda.jit(sig)(ns["kernel"])
    args = tuple(np.float32(i) for i in range(n))
    return kernel[GRID, BLOCK], args


def _build_multi_array_kernel(cuda, n):
    sig = "void(" + ", ".join(["float32[::1]"] * n) + ")"
    params = ", ".join(f"a{i}" for i in range(n))
    ns = {}
    exec(f"def kernel({params}):\n    pass\n", ns)  # noqa: S102
    kernel = cuda.jit(sig)(ns["kernel"])
    args = tuple(cuda.device_array(1, dtype=np.float32) for _ in range(n))
    return kernel[GRID, BLOCK], args


def setup(cuda):
    kernels = {
        "empty": _build_empty_kernel(cuda),
        "1_array_arg": _build_1_array_arg_kernel(cuda),
        "16_scalar_args": _build_multi_scalar_kernel(cuda, 16),
        "16_array_args": _build_multi_array_kernel(cuda, 16),
        "256_scalar_args": _build_multi_scalar_kernel(cuda, 256),
    }
    for configured_kernel, args in kernels.values():
        configured_kernel(*args)
    cuda.synchronize()
    return kernels


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure(cuda, configured_kernel, args):
    for _ in range(WARMUP):
        configured_kernel(*args)
    cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        configured_kernel(*args)
    cuda.synchronize()
    return (time.perf_counter() - t0) / ITERATIONS * 1e9


def bench(label, cuda):
    kernels = setup(cuda)
    results = {}
    for name in KERNEL_NAMES:
        k, kargs = kernels[name]
        results[name] = measure(cuda, k, kargs)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import numba.cuda as nc_cuda
    from numba_cuda_mlir import cuda as mlir_cuda

    nc_results = bench("numba_cuda", nc_cuda)
    mlir_results = bench("numba_cuda_mlir", mlir_cuda)

    sep = "-" * 86
    print(sep)
    print(
        f"{'Benchmark':<24} | {'numba_cuda (ns)':>16} | {'numba_cuda_mlir (ns)':>21} | {'Speedup':>10}"
    )
    print(sep)
    for name in KERNEL_NAMES:
        n, m = nc_results[name], mlir_results[name]
        print(f"{'launch_' + name:<24} | {n:>16.1f} | {m:>21.1f} | {n / m:>9.2f}x")
    print(sep)


if __name__ == "__main__":
    main()
