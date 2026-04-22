# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import numpy as np
import math
from numba import types

import numba.cuda as numba_cuda
import cuda.simt as cusimt_cuda

N = 512
SEED = 42


def generate_spd_matrix(n, seed=42):
    np.random.seed(seed)
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        row_sum = 0.0
        for j in range(i + 1, n):
            val = np.random.uniform(-0.5, 0.5)
            A[i, j] = val
            A[j, i] = val
            row_sum += abs(val)
        A[i, i] = 2.0 * row_sum + n * 0.01 + 1.0
    return A


def cholesky_numpy(A):
    return np.linalg.cholesky(A)


def verify_cholesky(A_orig, L, tolerance=None):
    n = A_orig.shape[0]
    if tolerance is None:
        tolerance = 1e-12 * np.sqrt(n)
    A_reconstructed = L @ L.T
    diff = A_orig - A_reconstructed
    error = np.linalg.norm(diff, "fro") / np.linalg.norm(A_orig, "fro")
    return error < tolerance, error


@numba_cuda.jit
def chol_compute_diag_numba_cuda(d_A, d_L, n, k, info):
    ssum = numba_cuda.shared.array(256, dtype=numba_cuda.float64)
    tid = numba_cuda.threadIdx.x
    block_dim = numba_cuda.blockDim.x

    local = 0.0
    for j in range(tid, k, block_dim):
        idx = k * n + j
        v = d_L[idx]
        local += v * v
    ssum[tid] = local
    numba_cuda.syncthreads()

    s = block_dim >> 1
    while s > 0:
        if tid < s:
            ssum[tid] = ssum[tid] + ssum[tid + s]
        numba_cuda.syncthreads()
        s >>= 1

    if tid == 0:
        sumsq = ssum[0]
        idx_kk = k * n + k
        diag = d_A[idx_kk] - sumsq
        if diag <= 0.0:
            info[0] = k + 1
            d_L[idx_kk] = 0.0
        else:
            d_L[idx_kk] = math.sqrt(diag)


@numba_cuda.jit
def chol_compute_column_numba_cuda(d_A, d_L, n, k):
    t = numba_cuda.blockIdx.x * numba_cuda.blockDim.x + numba_cuda.threadIdx.x
    i = k + 1 + t
    if i >= n:
        return

    s = 0.0
    for j in range(k):
        idx_ij = i * n + j
        idx_kj = k * n + j
        s += d_L[idx_ij] * d_L[idx_kj]

    idx_kk = k * n + k
    idx_ik = i * n + k
    Lkk = d_L[idx_kk]
    d_L[idx_ik] = (d_A[idx_ik] - s) / Lkk


@cusimt_cuda.jit
def chol_compute_diag_cusimt(d_A, d_L, n, k, info):
    ssum = cusimt_cuda.shared.array(256, dtype=cusimt_cuda.float64)
    tid = cusimt_cuda.threadIdx.x
    block_dim = cusimt_cuda.blockDim.x

    local = 0.0
    for j in range(tid, k, block_dim):
        idx = k * n + j
        v = d_L[idx]
        local += v * v
    ssum[tid] = local
    cusimt_cuda.syncthreads()

    s = block_dim >> 1
    while s > 0:
        if tid < s:
            ssum[tid] = ssum[tid] + ssum[tid + s]
        cusimt_cuda.syncthreads()
        s >>= 1

    if tid == 0:
        sumsq = ssum[0]
        idx_kk = k * n + k
        diag = d_A[idx_kk] - sumsq
        if diag <= 0.0:
            info[0] = k + 1
            d_L[idx_kk] = 0.0
        else:
            d_L[idx_kk] = math.sqrt(diag)


@cusimt_cuda.jit
def chol_compute_column_cusimt(d_A, d_L, n, k):
    t = cusimt_cuda.blockIdx.x * cusimt_cuda.blockDim.x + cusimt_cuda.threadIdx.x
    i = k + 1 + t
    if i >= n:
        return

    s = 0.0
    for j in range(k):
        idx_ij = i * n + j
        idx_kj = k * n + j
        s += d_L[idx_ij] * d_L[idx_kj]

    idx_kk = k * n + k
    idx_ik = i * n + k
    Lkk = d_L[idx_kk]
    d_L[idx_ik] = (d_A[idx_ik] - s) / Lkk


def get_input_matrix():
    return generate_spd_matrix(N, seed=SEED)


def run_cusimt_cholesky(A):
    n = N
    d_A = cusimt_cuda.to_device(A.flatten())
    h_L = np.zeros(n * n, dtype=np.float64)
    d_L = cusimt_cuda.to_device(h_L)
    h_info = np.zeros(1, dtype=np.int32)
    d_info = cusimt_cuda.to_device(h_info)

    for k in range(n):
        chol_compute_diag_cusimt[1, 256](d_A, d_L, n, k, d_info)
        rows = n - (k + 1)
        if rows > 0:
            grid = (rows + 255) // 256
            chol_compute_column_cusimt[grid, 256](d_A, d_L, n, k)

    cusimt_cuda.synchronize()
    return d_L.copy_to_host().reshape((n, n))


def test_cholesky():
    A = get_input_matrix()
    L_cusimt = run_cusimt_cholesky(A)

    cusimt_ok, cusimt_err = verify_cholesky(A, L_cusimt)
    assert cusimt_ok, f"cuSIMT verification failed with error {cusimt_err}"


def test_cholesky_benchmark(benchmark_runner):
    benchmark_runner(script=__file__)


def run_benchmark_main():
    diag_sig = types.void(
        types.float64[::1],
        types.float64[::1],
        types.int64,
        types.int64,
        types.int32[::1],
    )
    col_sig = types.void(
        types.float64[::1], types.float64[::1], types.int64, types.int64
    )

    start = time.perf_counter()
    chol_compute_diag_numba_cuda.compile(diag_sig)
    chol_compute_column_numba_cuda.compile(col_sig)
    numba_compile_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    chol_compute_diag_cusimt.compile(diag_sig)
    chol_compute_column_cusimt.compile(col_sig)
    cusimt_compile_time = (time.perf_counter() - start) * 1000

    print("\n=== COMPILE TIMES ===")
    print(f"Numba-CUDA: {numba_compile_time:.3f} ms")
    print(f"cuSIMT: {cusimt_compile_time:.3f} ms")

    n = N
    A = get_input_matrix()

    d_A = numba_cuda.to_device(A.flatten())
    h_L = np.zeros(n * n, dtype=np.float64)
    d_L = numba_cuda.to_device(h_L)
    h_info = np.zeros(1, dtype=np.int32)
    d_info = numba_cuda.to_device(h_info)

    for k in range(n):
        chol_compute_diag_numba_cuda[1, 256](d_A, d_L, n, k, d_info)
        rows = n - (k + 1)
        if rows > 0:
            grid = (rows + 255) // 256
            chol_compute_column_numba_cuda[grid, 256](d_A, d_L, n, k)
    numba_cuda.synchronize()

    d_A = cusimt_cuda.to_device(A.flatten())
    d_L = cusimt_cuda.to_device(h_L)
    d_info = cusimt_cuda.to_device(h_info)

    for k in range(n):
        chol_compute_diag_cusimt[1, 256](d_A, d_L, n, k, d_info)
        rows = n - (k + 1)
        if rows > 0:
            grid = (rows + 255) // 256
            chol_compute_column_cusimt[grid, 256](d_A, d_L, n, k)
    cusimt_cuda.synchronize()


if __name__ == "__main__":
    run_benchmark_main()
