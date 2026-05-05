# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import numpy as np
import math
import numba.cuda as numba_cuda
from numba_cuda_mlir import cuda

INPUT_SIZE = 128
SEED = 42


def softmax_numpy(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


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


def get_input_data():
    np.random.seed(SEED)
    return np.random.randn(INPUT_SIZE).astype(np.float32)


def run_numba_cuda_mlir_version(input_array):
    d_input = cuda.to_device(input_array)
    d_output = cuda.device_array(INPUT_SIZE, dtype=np.float32)
    softmax_kernel_numba_cuda_mlir[1, 256](d_input, d_output, INPUT_SIZE)
    cuda.synchronize()
    return d_output.copy_to_host()


def test_softmax():
    from conftest import verify_against_reference

    input_array = get_input_data()
    reference = softmax_numpy(input_array)
    numba_cuda_mlir_output = run_numba_cuda_mlir_version(input_array)
    verify_against_reference(
        reference, numba_cuda_mlir_output, tolerance=1e-5, name="numba-cuda-mlir"
    )


def test_softmax_benchmark(benchmark_runner):
    benchmark_runner(script=__file__)


def run_benchmark_main():
    sig = "void(float32[::1], float32[::1], int64)"

    start = time.perf_counter()
    softmax_kernel_numba_cuda.compile(sig)
    numba_compile_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    softmax_kernel_numba_cuda_mlir.compile(sig)
    numba_cuda_mlir_compile_time = (time.perf_counter() - start) * 1000

    print("\n=== COMPILE TIMES ===")
    print(f"Numba-CUDA: {numba_compile_time:.3f} ms")
    print(f"numba-cuda-mlir: {numba_cuda_mlir_compile_time:.3f} ms")

    input_array = get_input_data()

    d_input = numba_cuda.to_device(input_array)
    d_output = numba_cuda.device_array(INPUT_SIZE, dtype=np.float32)
    softmax_kernel_numba_cuda[1, 256](d_input, d_output, INPUT_SIZE)
    numba_cuda.synchronize()

    d_input = cuda.to_device(input_array)
    d_output = cuda.device_array(INPUT_SIZE, dtype=np.float32)
    softmax_kernel_numba_cuda_mlir[1, 256](d_input, d_output, INPUT_SIZE)
    cuda.synchronize()


if __name__ == "__main__":
    run_benchmark_main()
