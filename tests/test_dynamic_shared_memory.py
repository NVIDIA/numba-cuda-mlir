# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda.core import errors
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


def test_runtime_shaped_then_dynamic_shared_after_control_flow():
    """A runtime-shaped request after a branch still provides a
    dominating offset for a dynamic request in a later block, and the
    two views do not overlap."""

    @cuda.jit
    def k(n_arr, inp, out):
        if cuda.threadIdx.x != 0 or cuda.blockIdx.x != 0:
            return
        n = n_arr[0]
        a = cuda.shared.array(n, dtype=float32)
        for i in range(n):
            a[i] = inp[i]
        b = cuda.shared.array(0, dtype=float32)
        for i in range(n):
            b[i] = inp[n + i]
        for i in range(n):
            out[i] = a[i]
            out[n + i] = b[i]

    n_arr = np.array([2], dtype=np.int32)
    inp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = cuda.to_device(np.zeros(4, dtype=np.float32))
    k[1, 1, 0, 16](cuda.to_device(n_arr), cuda.to_device(inp), out)
    np.testing.assert_allclose(out.copy_to_host(), inp)


def test_dynamic_shared_in_branch_then_later_request():
    """A dynamic request inside a branch arm resets the running offset
    to the entry-block total size, so a request after the join still
    lowers to valid IR."""

    @cuda.jit
    def k(inp, out):
        if cuda.threadIdx.x != 0 or cuda.blockIdx.x != 0:
            return
        if inp[0] > 0.0:
            a = cuda.shared.array(0, dtype=float32)
            for i in range(inp.size):
                a[i] = inp[i]
            for i in range(out.size):
                out[i] = a[i]
        b = cuda.shared.array(0, dtype=float32)  # noqa: F841

    inp = np.array([5.0, 6.0], dtype=np.float32)
    out = cuda.to_device(np.zeros(2, dtype=np.float32))
    k[1, 1, 0, 8](cuda.to_device(inp), out)
    np.testing.assert_allclose(out.copy_to_host(), inp)


def test_runtime_shaped_shared_in_branch_then_later_request_raises():
    """A runtime-shaped allocation inside one branch arm leaves the next
    allocation's offset undefined; lowering reports a clear error
    instead of emitting IR the MLIR verifier rejects."""

    @cuda.jit
    def k(n_arr, out):
        n = n_arr[0]
        if n > 1:
            a = cuda.shared.array(n, dtype=float32)
            a[0] = 1.0
            out[0] = a[0]
        b = cuda.shared.array(n, dtype=float32)
        b[0] = 2.0
        out[1] = b[0]

    n_arr = np.array([2], dtype=np.int32)
    out = cuda.to_device(np.zeros(2, dtype=np.float32))
    with pytest.raises(errors.LoweringError, match="conditional control flow"):
        k[1, 1](cuda.to_device(n_arr), out)
