# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from numba_cuda_mlir import cuda
from numba_cuda_mlir.types import float32


def test_dynamic_shared_after_control_flow():
    """shared.array(0) lowers after a branch without a compiler crash."""

    @cuda.jit
    def k(inp, out):
        if cuda.threadIdx.x != 0 or cuda.blockIdx.x != 0:
            return
        shared = cuda.shared.array(0, dtype=float32)
        for i in range(inp.size):
            shared[i] = inp[i]
        for i in range(out.size):
            out[i] = shared[i]

    inp = np.array([7.0, 8.0], dtype=np.float32)
    out = cuda.to_device(np.zeros(2, dtype=np.float32))
    k[1, 1, 0, 8](cuda.to_device(inp), out)
    np.testing.assert_allclose(out.copy_to_host(), inp)


def test_runtime_shaped_shared_after_control_flow():
    """Runtime-shaped shared arrays lower after a branch without a
    compiler crash."""

    @cuda.jit
    def k(n_arr, inp, out):
        if cuda.threadIdx.x != 0 or cuda.blockIdx.x != 0:
            return
        n = n_arr[0]
        shared = cuda.shared.array(n, dtype=float32)
        for i in range(inp.size):
            shared[i] = inp[i]
        for i in range(out.size):
            out[i] = shared[i]

    n_arr = np.array([2], dtype=np.int32)
    inp = np.array([9.0, 10.0], dtype=np.float32)
    out = cuda.to_device(np.zeros(2, dtype=np.float32))
    k[1, 1, 0, 8](cuda.to_device(n_arr), cuda.to_device(inp), out)
    np.testing.assert_allclose(out.copy_to_host(), inp)


def test_dynamic_shared_after_conditional_runtime_shaped_shared():
    """Multiple shared arrays across control flow keep offset values valid."""

    @cuda.jit
    def k(flag, n_arr, out):
        if flag[0] != 0:
            n = n_arr[0]
            scratch = cuda.shared.array(n, dtype=float32)
            scratch[0] = 13.0

        shared = cuda.shared.array(0, dtype=float32)
        shared[0] = 21.0

        if flag[0] != 0:
            out[0] = scratch[0]
        out[1] = shared[0]

    flag = np.array([1], dtype=np.int32)
    n_arr = np.array([2], dtype=np.int32)
    out = cuda.to_device(np.zeros(2, dtype=np.float32))
    k[1, 1, 0, 16](cuda.to_device(flag), cuda.to_device(n_arr), out)
    np.testing.assert_allclose(out.copy_to_host(), np.array([13.0, 21.0], dtype=np.float32))
