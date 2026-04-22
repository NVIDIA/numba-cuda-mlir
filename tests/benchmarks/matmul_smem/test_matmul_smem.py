# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import numpy as np
import math
from numba import types

import numba.cuda as numba_cuda
import cuda.simt as cusimt_cuda

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


@cusimt_cuda.jit
def cusimt_matmul_smem(A, B, C):
    sA = cusimt_cuda.shared.array(shape=(TPB, TPB), dtype=cusimt_cuda.float32)
    sB = cusimt_cuda.shared.array(shape=(TPB, TPB), dtype=cusimt_cuda.float32)

    tx = cusimt_cuda.threadIdx.x
    ty = cusimt_cuda.threadIdx.y
    x = cusimt_cuda.blockIdx.x * cusimt_cuda.blockDim.x + tx
    y = cusimt_cuda.blockIdx.y * cusimt_cuda.blockDim.y + ty
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

        cusimt_cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]
        cusimt_cuda.syncthreads()

    C[x, y] = tmp


def get_input_data():
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    return A, B


def run_cusimt_version(A, B):
    A_dev = cusimt_cuda.to_device(A)
    B_dev = cusimt_cuda.to_device(B)
    C_dev = cusimt_cuda.device_array((M, N), dtype=np.float32)
    threads = (TPB, TPB)
    blocks = (math.ceil(M / TPB), math.ceil(N / TPB))
    cusimt_matmul_smem[blocks, threads](A_dev, B_dev, C_dev)
    cusimt_cuda.synchronize()
    return C_dev.copy_to_host()


def test_matmul_smem():
    from conftest import verify_against_reference

    A, B = get_input_data()
    reference = matmul_numpy(A, B)
    cusimt_output = run_cusimt_version(A, B)
    verify_against_reference(reference, cusimt_output, tolerance=1e-3, name="cuSIMT")


def test_matmul_smem_benchmark(benchmark_runner):
    benchmark_runner(script=__file__)


def run_benchmark_main():
    sig = types.void(
        types.float32[:, ::1],
        types.float32[:, ::1],
        types.float32[:, ::1],
    )

    start = time.perf_counter()
    numba_cuda_matmul_smem.compile(sig)
    numba_compile_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    cusimt_matmul_smem.compile(sig)
    cusimt_compile_time = (time.perf_counter() - start) * 1000

    print("\n=== COMPILE TIMES ===")
    print(f"Numba-CUDA: {numba_compile_time:.3f} ms")
    print(f"cuSIMT: {cusimt_compile_time:.3f} ms")

    A, B = get_input_data()
    threads = (TPB, TPB)
    blocks = (math.ceil(M / TPB), math.ceil(N / TPB))

    A_dev = numba_cuda.to_device(A)
    B_dev = numba_cuda.to_device(B)
    C_dev = numba_cuda.device_array((M, N), dtype=np.float32)
    numba_cuda_matmul_smem[blocks, threads](A_dev, B_dev, C_dev)
    numba_cuda.synchronize()

    A_dev = cusimt_cuda.to_device(A)
    B_dev = cusimt_cuda.to_device(B)
    C_dev = cusimt_cuda.device_array((M, N), dtype=np.float32)
    cusimt_matmul_smem[blocks, threads](A_dev, B_dev, C_dev)
    cusimt_cuda.synchronize()


if __name__ == "__main__":
    run_benchmark_main()
