# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from pathlib import Path

import numpy as np
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_utils import (
    BACKEND_BOTH,
    BACKEND_NUMBA_CUDA,
    BACKEND_NUMBA_CUDA_MLIR,
    add_backend_arg,
    add_compile_mode_arg,
    prepare_compile_measurement,
    print_compile_times,
    selected_backend_from_argv,
    should_run_backend,
    time_compile,
)

SELECTED_BACKEND = selected_backend_from_argv()
if should_run_backend(SELECTED_BACKEND, BACKEND_NUMBA_CUDA):
    import numba.cuda as numba_cuda
if should_run_backend(SELECTED_BACKEND, BACKEND_NUMBA_CUDA_MLIR):
    from numba_cuda_mlir import cuda

DEFAULT_SIZE = 1 << 20
SEED = 42


def softmax_numpy(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


if should_run_backend(SELECTED_BACKEND, BACKEND_NUMBA_CUDA):

    @numba_cuda.jit
    def softmax_kernel_numba_cuda(input_arr, output_arr, n):
        sdata = numba_cuda.shared.array(256, dtype=numba_cuda.float32)
        tid = numba_cuda.threadIdx.x
        block_size = numba_cuda.blockDim.x

        local_max = -1e38
        for i in range(tid, n, block_size):
            local_max = max(local_max, input_arr[i])
        sdata[tid] = local_max
        numba_cuda.syncthreads()

        s = block_size >> 1
        while s > 0:
            if tid < s:
                sdata[tid] = max(sdata[tid], sdata[tid + s])
            numba_cuda.syncthreads()
            s >>= 1

        max_val = sdata[0]
        numba_cuda.syncthreads()

        local_sum = 0.0
        for i in range(tid, n, block_size):
            exp_val = math.exp(input_arr[i] - max_val)
            output_arr[i] = exp_val
            local_sum += exp_val
        sdata[tid] = local_sum
        numba_cuda.syncthreads()

        s = block_size >> 1
        while s > 0:
            if tid < s:
                sdata[tid] = sdata[tid] + sdata[tid + s]
            numba_cuda.syncthreads()
            s >>= 1

        sum_exp = sdata[0]
        numba_cuda.syncthreads()

        for i in range(tid, n, block_size):
            output_arr[i] = output_arr[i] / sum_exp


if should_run_backend(SELECTED_BACKEND, BACKEND_NUMBA_CUDA_MLIR):

    @cuda.jit
    def softmax_kernel_numba_cuda_mlir(input_arr, output_arr, n):
        sdata = cuda.shared.array(256, dtype=cuda.float32)
        tid = cuda.threadIdx.x
        block_size = cuda.blockDim.x

        local_max = -1e38
        for i in range(tid, n, block_size):
            local_max = max(local_max, input_arr[i])
        sdata[tid] = local_max
        cuda.syncthreads()

        s = block_size >> 1
        while s > 0:
            if tid < s:
                sdata[tid] = max(sdata[tid], sdata[tid + s])
            cuda.syncthreads()
            s >>= 1

        max_val = sdata[0]
        cuda.syncthreads()

        local_sum = 0.0
        for i in range(tid, n, block_size):
            exp_val = math.exp(input_arr[i] - max_val)
            output_arr[i] = exp_val
            local_sum += exp_val
        sdata[tid] = local_sum
        cuda.syncthreads()

        s = block_size >> 1
        while s > 0:
            if tid < s:
                sdata[tid] = sdata[tid] + sdata[tid + s]
            cuda.syncthreads()
            s >>= 1

        sum_exp = sdata[0]
        cuda.syncthreads()

        for i in range(tid, n, block_size):
            output_arr[i] = output_arr[i] / sum_exp


def get_input_data(size=DEFAULT_SIZE):
    np.random.seed(SEED)
    return np.random.randn(size).astype(np.float32)


def run_numba_cuda_mlir_version(input_array):
    n = len(input_array)
    d_input = cuda.to_device(input_array)
    d_output = cuda.device_array(n, dtype=np.float32)
    softmax_kernel_numba_cuda_mlir[1, 256](d_input, d_output, n)
    cuda.synchronize()
    return d_output.copy_to_host()


def test_softmax_large():
    from conftest import verify_against_reference

    test_size = 100000
    input_array = get_input_data(test_size)
    reference = softmax_numpy(input_array)
    numba_cuda_mlir_output = run_numba_cuda_mlir_version(input_array)
    verify_against_reference(
        reference, numba_cuda_mlir_output, tolerance=1e-5, name="numba-cuda-mlir"
    )


def test_softmax_large_benchmark(benchmark_runner):
    benchmark_runner(script=__file__)


def run_benchmark_main(compile_mode="cold", backend=BACKEND_BOTH):
    sig = "void(float32[::1], float32[::1], int64)"
    prepare_compile_measurement(compile_mode, backend)

    compile_times = {}
    if should_run_backend(backend, BACKEND_NUMBA_CUDA):
        compile_times[BACKEND_NUMBA_CUDA] = time_compile(softmax_kernel_numba_cuda.compile, sig)
    if should_run_backend(backend, BACKEND_NUMBA_CUDA_MLIR):
        compile_times[BACKEND_NUMBA_CUDA_MLIR] = time_compile(
            softmax_kernel_numba_cuda_mlir.compile, sig
        )
    print_compile_times(compile_times)

    n = DEFAULT_SIZE
    input_array = get_input_data(n)

    if should_run_backend(backend, BACKEND_NUMBA_CUDA):
        d_input = numba_cuda.to_device(input_array)
        d_output = numba_cuda.device_array(n, dtype=np.float32)
        softmax_kernel_numba_cuda[1, 256](d_input, d_output, n)
        numba_cuda.synchronize()

    if should_run_backend(backend, BACKEND_NUMBA_CUDA_MLIR):
        d_input = cuda.to_device(input_array)
        d_output = cuda.device_array(n, dtype=np.float32)
        softmax_kernel_numba_cuda_mlir[1, 256](d_input, d_output, n)
        cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Large softmax benchmark")
    add_compile_mode_arg(parser)
    add_backend_arg(parser)
    args = parser.parse_args()
    run_benchmark_main(args.compile_mode, args.backend)
