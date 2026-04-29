# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import numpy as np
import math
from cusimt.numba_cuda import types

import numba.cuda as numba_cuda
import cuda.simt as cusimt_cuda

N = 512
B = 64
SEED = 42


def generate_spd_matrix_colmajor(n, seed=42):
    np.random.seed(seed)
    A = np.zeros(n * n, dtype=np.float64)

    def col_maj(i, j, ld):
        return i + j * ld

    for i in range(n):
        row_sum = 0.0
        for j in range(i):
            val = 2.0 * np.random.rand() - 1.0
            A[col_maj(i, j, n)] = val
            A[col_maj(j, i, n)] = val
            row_sum += abs(val)
        for j in range(i + 1, n):
            row_sum += abs(A[col_maj(i, j, n)])
        A[col_maj(i, i, n)] = row_sum + n * 2.0

    return A


def verify_cholesky_colmajor(A_orig, L, n, tolerance=None):
    if tolerance is None:
        tolerance = 1e-12 * np.sqrt(n)

    def col_maj(i, j, ld):
        return i + j * ld

    A_reconstructed = np.zeros_like(A_orig)
    for j in range(n):
        for i in range(j, n):
            s = 0.0
            for k in range(j + 1):
                s += L[col_maj(i, k, n)] * L[col_maj(j, k, n)]
            A_reconstructed[col_maj(i, j, n)] = s
            A_reconstructed[col_maj(j, i, n)] = s

    diff_sq = np.sum((A_orig - A_reconstructed) ** 2)
    a_sq = np.sum(A_orig**2)
    error = np.sqrt(diff_sq / a_sq)

    return error < tolerance, error


@numba_cuda.jit
def chol_panel_kernel_numba_cuda(A, ld, k, bk, n, info):
    swork = numba_cuda.shared.array(256, dtype=numba_cuda.float64)
    TID = numba_cuda.threadIdx.x
    NT = numba_cuda.blockDim.x

    failed = numba_cuda.shared.array(1, dtype=numba_cuda.int32)
    if TID == 0:
        failed[0] = 0
    numba_cuda.syncthreads()

    for c in range(bk):
        if failed[0]:
            break
        kk = k + c

        local = 0.0
        for j in range(k + TID, kk, NT):
            idx = kk + j * ld
            v = A[idx]
            local += v * v
        swork[TID] = local
        numba_cuda.syncthreads()

        s = NT >> 1
        while s > 0:
            if TID < s:
                swork[TID] = swork[TID] + swork[TID + s]
            numba_cuda.syncthreads()
            s >>= 1

        if TID == 0:
            sumsq = swork[0]
            idx_kk = kk + kk * ld
            Akk = A[idx_kk]
            diag = Akk - sumsq
            if diag <= 0.0 or not math.isfinite(diag):
                info[0] = kk + 1
                failed[0] = 1
            else:
                A[idx_kk] = math.sqrt(diag)
        numba_cuda.syncthreads()
        if failed[0]:
            break

        Lkk = A[kk + kk * ld]
        for i in range(kk + 1 + TID, k + bk, NT):
            dot = 0.0
            for j in range(k, kk):
                dot += A[i + j * ld] * A[kk + j * ld]
            idx_ik = i + kk * ld
            Aik = A[idx_ik]
            A[idx_ik] = (Aik - dot) / Lkk
        numba_cuda.syncthreads()


@numba_cuda.jit
def trsm_right_lower_trans_rows_numba_cuda(A, ld, k, bk, m):
    row = numba_cuda.blockIdx.x * numba_cuda.blockDim.x + numba_cuda.threadIdx.x
    if row >= m:
        return
    row_global = k + bk + row

    for c in range(bk):
        s = 0.0
        for t in range(c):
            B_idx = row_global + (k + t) * ld
            Lkk_idx = (k + c) + (k + t) * ld
            s += A[B_idx] * A[Lkk_idx]
        Ucc_idx = (k + c) + (k + c) * ld
        Ucc = A[Ucc_idx]
        bc_idx = row_global + (k + c) * ld
        bc = A[bc_idx]
        A[bc_idx] = (bc - s) / Ucc


@numba_cuda.jit
def syrk_rankk_lower_numba_cuda(A, ld, k, bk, m):
    i = numba_cuda.blockIdx.y * numba_cuda.blockDim.y + numba_cuda.threadIdx.y
    j = numba_cuda.blockIdx.x * numba_cuda.blockDim.x + numba_cuda.threadIdx.x
    if i >= m or j >= m or i < j:
        return

    i_global = k + bk + i
    j_global = k + bk + j

    s = 0.0
    for p in range(bk):
        B_i_idx = i_global + (k + p) * ld
        B_j_idx = j_global + (k + p) * ld
        s += A[B_i_idx] * A[B_j_idx]

    C_idx = i_global + j_global * ld
    A[C_idx] -= s


@cusimt_cuda.jit
def chol_panel_kernel_cusimt(A, ld, k, bk, n, info):
    swork = cusimt_cuda.shared.array(256, dtype=cusimt_cuda.float64)
    TID = cusimt_cuda.threadIdx.x
    NT = cusimt_cuda.blockDim.x

    failed = cusimt_cuda.shared.array(1, dtype=cusimt_cuda.int32)
    if TID == 0:
        failed[0] = 0
    cusimt_cuda.syncthreads()

    for c in range(bk):
        if failed[0]:
            break
        kk = k + c

        local = 0.0
        for j in range(k + TID, kk, NT):
            idx = kk + j * ld
            v = A[idx]
            local += v * v
        swork[TID] = local
        cusimt_cuda.syncthreads()

        s = NT >> 1
        while s > 0:
            if TID < s:
                swork[TID] = swork[TID] + swork[TID + s]
            cusimt_cuda.syncthreads()
            s >>= 1

        if TID == 0:
            sumsq = swork[0]
            idx_kk = kk + kk * ld
            Akk = A[idx_kk]
            diag = Akk - sumsq
            if diag <= 0.0 or not math.isfinite(diag):
                info[0] = kk + 1
                failed[0] = 1
            else:
                A[idx_kk] = math.sqrt(diag)
        cusimt_cuda.syncthreads()
        if failed[0]:
            break

        Lkk = A[kk + kk * ld]
        for i in range(kk + 1 + TID, k + bk, NT):
            dot = 0.0
            for j in range(k, kk):
                dot += A[i + j * ld] * A[kk + j * ld]
            idx_ik = i + kk * ld
            Aik = A[idx_ik]
            A[idx_ik] = (Aik - dot) / Lkk
        cusimt_cuda.syncthreads()


@cusimt_cuda.jit
def trsm_right_lower_trans_rows_cusimt(A, ld, k, bk, m):
    row = cusimt_cuda.blockIdx.x * cusimt_cuda.blockDim.x + cusimt_cuda.threadIdx.x
    if row >= m:
        return
    row_global = k + bk + row

    for c in range(bk):
        s = 0.0
        for t in range(c):
            B_idx = row_global + (k + t) * ld
            Lkk_idx = (k + c) + (k + t) * ld
            s += A[B_idx] * A[Lkk_idx]
        Ucc_idx = (k + c) + (k + c) * ld
        Ucc = A[Ucc_idx]
        bc_idx = row_global + (k + c) * ld
        bc = A[bc_idx]
        A[bc_idx] = (bc - s) / Ucc


@cusimt_cuda.jit
def syrk_rankk_lower_cusimt(A, ld, k, bk, m):
    i = cusimt_cuda.blockIdx.y * cusimt_cuda.blockDim.y + cusimt_cuda.threadIdx.y
    j = cusimt_cuda.blockIdx.x * cusimt_cuda.blockDim.x + cusimt_cuda.threadIdx.x
    if i >= m or j >= m or i < j:
        return

    i_global = k + bk + i
    j_global = k + bk + j

    s = 0.0
    for p in range(bk):
        B_i_idx = i_global + (k + p) * ld
        B_j_idx = j_global + (k + p) * ld
        s += A[B_i_idx] * A[B_j_idx]

    C_idx = i_global + j_global * ld
    A[C_idx] = A[C_idx] - s


def get_input_matrix():
    return generate_spd_matrix_colmajor(N, seed=SEED)


def run_cusimt_cholesky_blocked(A):
    n, b = N, B
    d_A = cusimt_cuda.to_device(A.copy())
    h_info = np.zeros(1, dtype=np.int32)
    d_info = cusimt_cuda.to_device(h_info)

    for k in range(0, n, b):
        bk = min(b, n - k)
        rows_below = n - (k + bk)
        cusimt_cuda.to_device(np.zeros(1, dtype=np.int32), to=d_info)
        chol_panel_kernel_cusimt[1, 256](d_A, n, k, bk, n, d_info)
        if rows_below > 0:
            m = rows_below
            grid = (m + 255) // 256
            trsm_right_lower_trans_rows_cusimt[grid, 256](d_A, n, k, bk, m)
            block = (16, 16)
            grid = ((rows_below + 15) // 16, (rows_below + 15) // 16)
            syrk_rankk_lower_cusimt[grid, block](d_A, n, k, bk, rows_below)

    cusimt_cuda.synchronize()
    return d_A.copy_to_host()


def test_cholesky_blocked():
    A = get_input_matrix()
    cusimt_L = run_cusimt_cholesky_blocked(A)

    cusimt_ok, cusimt_err = verify_cholesky_colmajor(A, cusimt_L, N)
    assert cusimt_ok, f"cuSIMT verification failed with error {cusimt_err}"


def test_cholesky_blocked_benchmark(benchmark_runner):
    benchmark_runner(script=__file__)


def run_benchmark_main():
    panel_sig = types.void(
        types.float64[::1],
        types.int64,
        types.int64,
        types.int64,
        types.int64,
        types.int32[::1],
    )
    trsm_sig = types.void(types.float64[::1], types.int64, types.int64, types.int64, types.int64)
    syrk_sig = types.void(types.float64[::1], types.int64, types.int64, types.int64, types.int64)

    start = time.perf_counter()
    chol_panel_kernel_numba_cuda.compile(panel_sig)
    trsm_right_lower_trans_rows_numba_cuda.compile(trsm_sig)
    syrk_rankk_lower_numba_cuda.compile(syrk_sig)
    numba_compile_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    chol_panel_kernel_cusimt.compile(panel_sig)
    trsm_right_lower_trans_rows_cusimt.compile(trsm_sig)
    syrk_rankk_lower_cusimt.compile(syrk_sig)
    cusimt_compile_time = (time.perf_counter() - start) * 1000

    print("\n=== COMPILE TIMES ===")
    print(f"Numba-CUDA: {numba_compile_time:.3f} ms")
    print(f"cuSIMT: {cusimt_compile_time:.3f} ms")

    n, b = N, B
    A = get_input_matrix()
    h_info = np.zeros(1, dtype=np.int32)
    d_info = numba_cuda.to_device(h_info)

    d_A = numba_cuda.to_device(A.copy())
    for k in range(0, n, b):
        bk = min(b, n - k)
        rows_below = n - (k + bk)
        numba_cuda.to_device(np.zeros(1, dtype=np.int32), to=d_info)
        chol_panel_kernel_numba_cuda[1, 256](d_A, n, k, bk, n, d_info)
        if rows_below > 0:
            m = rows_below
            grid = (m + 255) // 256
            trsm_right_lower_trans_rows_numba_cuda[grid, 256](d_A, n, k, bk, m)
            block = (16, 16)
            grid = ((rows_below + 15) // 16, (rows_below + 15) // 16)
            syrk_rankk_lower_numba_cuda[grid, block](d_A, n, k, bk, rows_below)
    numba_cuda.synchronize()

    d_info = cusimt_cuda.to_device(h_info)
    d_A = cusimt_cuda.to_device(A.copy())
    for k in range(0, n, b):
        bk = min(b, n - k)
        rows_below = n - (k + bk)
        cusimt_cuda.to_device(np.zeros(1, dtype=np.int32), to=d_info)
        chol_panel_kernel_cusimt[1, 256](d_A, n, k, bk, n, d_info)
        if rows_below > 0:
            m = rows_below
            grid = (m + 255) // 256
            trsm_right_lower_trans_rows_cusimt[grid, 256](d_A, n, k, bk, m)
            block = (16, 16)
            grid = ((rows_below + 15) // 16, (rows_below + 15) // 16)
            syrk_rankk_lower_cusimt[grid, block](d_A, n, k, bk, rows_below)
    cusimt_cuda.synchronize()


if __name__ == "__main__":
    run_benchmark_main()
