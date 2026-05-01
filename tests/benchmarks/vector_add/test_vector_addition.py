# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
import numpy as np
from numba_cuda_mlir.numba_cuda import types

import numba.cuda as numba_cuda
from numba_cuda_mlir import cuda

N = 0x1 << 24
TEST_N = 1000000


@numba_cuda.jit
def numba_cuda_vector_add(a, b, c, n):
    idx = numba_cuda.blockIdx.x * numba_cuda.blockDim.x + numba_cuda.threadIdx.x
    if idx < n:
        c[idx] = a[idx] + b[idx]


@numba_cuda.jit
def numba_cuda_vector_add_vectorized(a, b, c, n):
    base_idx = (numba_cuda.blockIdx.x * numba_cuda.blockDim.x + numba_cuda.threadIdx.x) * 4
    if base_idx + 3 < n:
        c[base_idx] = a[base_idx] + b[base_idx]
        c[base_idx + 1] = a[base_idx + 1] + b[base_idx + 1]
        c[base_idx + 2] = a[base_idx + 2] + b[base_idx + 2]
        c[base_idx + 3] = a[base_idx + 3] + b[base_idx + 3]
    else:
        for i in range(4):
            if base_idx + i < n:
                c[base_idx + i] = a[base_idx + i] + b[base_idx + i]


@cuda.jit
def numba_cuda_mlir_vector_add(a, b, c, n):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idx < n:
        c[idx] = a[idx] + b[idx]


@cuda.jit
def numba_cuda_mlir_vector_add_vectorized(a, b, c, n):
    base_idx = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) * 4
    if base_idx + 3 < n:
        c[base_idx] = a[base_idx] + b[base_idx]
        c[base_idx + 1] = a[base_idx + 1] + b[base_idx + 1]
        c[base_idx + 2] = a[base_idx + 2] + b[base_idx + 2]
        c[base_idx + 3] = a[base_idx + 3] + b[base_idx + 3]
    else:
        for i in range(4):
            if base_idx + i < n:
                c[base_idx + i] = a[base_idx + i] + b[base_idx + i]


def get_input_data(size):
    a = np.ones(size, dtype=np.float32)
    b = np.full(size, 2.0, dtype=np.float32)
    return a, b


def run_numba_cuda_mlir_scalar(a, b):
    n = len(a)
    a_device = cuda.to_device(a)
    b_device = cuda.to_device(b)
    c_device = cuda.device_array(n, dtype=np.float32)
    threads = 1024
    blocks = (n + threads - 1) // threads
    numba_cuda_mlir_vector_add[blocks, threads](a_device, b_device, c_device, n)
    cuda.synchronize()
    return c_device.copy_to_host()


def run_numba_cuda_mlir_vectorized(a, b):
    n = len(a)
    a_device = cuda.to_device(a)
    b_device = cuda.to_device(b)
    c_device = cuda.device_array(n, dtype=np.float32)
    threads = 1024
    elements_per_thread = 4
    total_threads = (n + elements_per_thread - 1) // elements_per_thread
    blocks = (total_threads + threads - 1) // threads
    numba_cuda_mlir_vector_add_vectorized[blocks, threads](a_device, b_device, c_device, n)
    cuda.synchronize()
    return c_device.copy_to_host()


def test_vector_addition():
    a, b = get_input_data(TEST_N)
    expected = a + b
    numba_cuda_mlir_result = run_numba_cuda_mlir_scalar(a, b)
    assert np.allclose(numba_cuda_mlir_result, expected), (
        "numba-cuda-mlir scalar verification failed"
    )


def test_vector_addition_vectorized():
    a, b = get_input_data(TEST_N)
    expected = a + b
    numba_cuda_mlir_result = run_numba_cuda_mlir_vectorized(a, b)
    assert np.allclose(numba_cuda_mlir_result, expected), (
        "numba-cuda-mlir vectorized verification failed"
    )


def test_vector_addition_scalar_benchmark(benchmark_runner):
    benchmark_runner(script=__file__, mode="scalar")


def test_vector_addition_vectorized_benchmark(benchmark_runner):
    benchmark_runner(script=__file__, mode="vectorized")


def run_benchmark_scalar():
    sig = types.void(types.float32[::1], types.float32[::1], types.float32[::1], types.int64)

    start = time.perf_counter()
    numba_cuda_vector_add.compile(sig)
    numba_compile_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    numba_cuda_mlir_vector_add.compile(sig)
    numba_cuda_mlir_compile_time = (time.perf_counter() - start) * 1000

    print("\n=== COMPILE TIMES ===")
    print(f"Numba-CUDA: {numba_compile_time:.3f} ms")
    print(f"numba-cuda-mlir: {numba_cuda_mlir_compile_time:.3f} ms")

    n = N
    a, b = get_input_data(n)
    threads = 1024
    blocks = (n + threads - 1) // threads

    a_device = numba_cuda.to_device(a)
    b_device = numba_cuda.to_device(b)
    c_device = numba_cuda.device_array(n, dtype=np.float32)
    numba_cuda_vector_add[blocks, threads](a_device, b_device, c_device, n)
    numba_cuda.synchronize()

    a_device = cuda.to_device(a)
    b_device = cuda.to_device(b)
    c_device = cuda.device_array(n, dtype=np.float32)
    numba_cuda_mlir_vector_add[blocks, threads](a_device, b_device, c_device, n)
    cuda.synchronize()


def run_benchmark_vectorized():
    sig = types.void(types.float32[::1], types.float32[::1], types.float32[::1], types.int64)

    start = time.perf_counter()
    numba_cuda_vector_add_vectorized.compile(sig)
    numba_compile_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    numba_cuda_mlir_vector_add_vectorized.compile(sig)
    numba_cuda_mlir_compile_time = (time.perf_counter() - start) * 1000

    print("\n=== COMPILE TIMES ===")
    print(f"Numba-CUDA: {numba_compile_time:.3f} ms")
    print(f"numba-cuda-mlir: {numba_cuda_mlir_compile_time:.3f} ms")

    n = N
    a, b = get_input_data(n)
    threads = 1024
    elements_per_thread = 4
    total_threads = (n + elements_per_thread - 1) // elements_per_thread
    blocks = (total_threads + threads - 1) // threads

    a_device = numba_cuda.to_device(a)
    b_device = numba_cuda.to_device(b)
    c_device = numba_cuda.device_array(n, dtype=np.float32)
    numba_cuda_vector_add_vectorized[blocks, threads](a_device, b_device, c_device, n)
    numba_cuda.synchronize()

    a_device = cuda.to_device(a)
    b_device = cuda.to_device(b)
    c_device = cuda.device_array(n, dtype=np.float32)
    numba_cuda_mlir_vector_add_vectorized[blocks, threads](a_device, b_device, c_device, n)
    cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector addition benchmark")
    parser.add_argument(
        "mode",
        nargs="?",
        default="scalar",
        choices=["scalar", "vectorized"],
        help="Benchmark mode: scalar or vectorized (default: scalar)",
    )
    args = parser.parse_args()

    if args.mode == "vectorized":
        run_benchmark_vectorized()
    else:
        run_benchmark_scalar()
