# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    skipped_backend,
    time_compile,
)

SELECTED_BACKEND = selected_backend_from_argv()
if should_run_backend(SELECTED_BACKEND, BACKEND_NUMBA_CUDA):
    import numba.cuda as numba_cuda
else:
    numba_cuda = skipped_backend()
if should_run_backend(SELECTED_BACKEND, BACKEND_NUMBA_CUDA_MLIR):
    from numba_cuda_mlir import cuda
else:
    cuda = skipped_backend()

TPB = 16
M = 512
N = 512
K = 512


def matmul_numpy(A, B):
    return A @ B


@numba_cuda.jit
def numba_cuda_matmul_smem(A, B, C):
    sA = numba_cuda.shared.array(shape=(TPB, TPB), dtype=numba_cuda.float32)
    sB = numba_cuda.shared.array(shape=(TPB, TPB), dtype=numba_cuda.float32)

    tx = numba_cuda.threadIdx.x
    ty = numba_cuda.threadIdx.y
    x = numba_cuda.blockIdx.x * numba_cuda.blockDim.x + tx
    y = numba_cuda.blockIdx.y * numba_cuda.blockDim.y + ty
    bpg = math.ceil(A.shape[1] / TPB)

    if x >= C.shape[0] or y >= C.shape[1]:
        return

    tmp = 0.0
    for i in range(bpg):
        ax = x
        ay = i * TPB + ty
        bx = i * TPB + tx
        by = y
        if ax < A.shape[0] and ay < A.shape[1]:
            sA[tx, ty] = A[ax, ay]
        else:
            sA[tx, ty] = 0.0

        if bx < B.shape[0] and by < B.shape[1]:
            sB[tx, ty] = B[bx, by]
        else:
            sB[tx, ty] = 0.0

        numba_cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]
        numba_cuda.syncthreads()

    C[x, y] = tmp


@cuda.jit
def numba_cuda_mlir_matmul_smem(A, B, C):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=cuda.float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=cuda.float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    x = cuda.blockIdx.x * cuda.blockDim.x + tx
    y = cuda.blockIdx.y * cuda.blockDim.y + ty
    bpg = math.ceil(A.shape[1] / TPB)

    if x >= C.shape[0] or y >= C.shape[1]:
        return

    tmp = 0.0
    for i in range(bpg):
        ax = x
        ay = i * TPB + ty
        bx = i * TPB + tx
        by = y
        if ax < A.shape[0] and ay < A.shape[1]:
            sA[tx, ty] = A[ax, ay]
        else:
            sA[tx, ty] = 0.0

        if bx < B.shape[0] and by < B.shape[1]:
            sB[tx, ty] = B[bx, by]
        else:
            sB[tx, ty] = 0.0

        cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]
        cuda.syncthreads()

    C[x, y] = tmp


def get_input_data():
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    return A, B


def run_numba_cuda_mlir_version(A, B):
    A_dev = cuda.to_device(A)
    B_dev = cuda.to_device(B)
    C_dev = cuda.device_array((M, N), dtype=np.float32)
    threads = (TPB, TPB)
    blocks = (math.ceil(M / TPB), math.ceil(N / TPB))
    numba_cuda_mlir_matmul_smem[blocks, threads](A_dev, B_dev, C_dev)
    cuda.synchronize()
    return C_dev.copy_to_host()


def test_matmul_smem():
    from conftest import verify_against_reference

    A, B = get_input_data()
    reference = matmul_numpy(A, B)
    numba_cuda_mlir_output = run_numba_cuda_mlir_version(A, B)
    verify_against_reference(
        reference, numba_cuda_mlir_output, tolerance=1e-3, name="numba-cuda-mlir"
    )


def test_matmul_smem_benchmark(benchmark_runner):
    benchmark_runner(script=__file__)


def run_benchmark_main(compile_mode="cold", backend=BACKEND_BOTH):
    sig = "void(float32[:, ::1], float32[:, ::1], float32[:, ::1])"
    prepare_compile_measurement(compile_mode, backend)

    compile_times = {}
    if should_run_backend(backend, BACKEND_NUMBA_CUDA):
        compile_times[BACKEND_NUMBA_CUDA] = time_compile(numba_cuda_matmul_smem.compile, sig)
    if should_run_backend(backend, BACKEND_NUMBA_CUDA_MLIR):
        compile_times[BACKEND_NUMBA_CUDA_MLIR] = time_compile(
            numba_cuda_mlir_matmul_smem.compile, sig
        )

    print_compile_times(compile_times)

    A, B = get_input_data()
    threads = (TPB, TPB)
    blocks = (math.ceil(M / TPB), math.ceil(N / TPB))

    if should_run_backend(backend, BACKEND_NUMBA_CUDA):
        A_dev = numba_cuda.to_device(A)
        B_dev = numba_cuda.to_device(B)
        C_dev = numba_cuda.device_array((M, N), dtype=np.float32)
        numba_cuda_matmul_smem[blocks, threads](A_dev, B_dev, C_dev)
        numba_cuda.synchronize()

    if should_run_backend(backend, BACKEND_NUMBA_CUDA_MLIR):
        A_dev = cuda.to_device(A)
        B_dev = cuda.to_device(B)
        C_dev = cuda.device_array((M, N), dtype=np.float32)
        numba_cuda_mlir_matmul_smem[blocks, threads](A_dev, B_dev, C_dev)
        cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shared-memory matmul benchmark")
    add_compile_mode_arg(parser)
    add_backend_arg(parser)
    args = parser.parse_args()
    run_benchmark_main(args.compile_mode, args.backend)
