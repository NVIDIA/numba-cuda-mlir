# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba_cuda_mlir import cuda
import pytest


@pytest.fixture
def global_arrays():
    """Populate module-level globals for this test only.

    Allocating at import time would not survive a ``current_context().reset()``
    from a sibling test in the same xdist worker. Assigning into module
    globals (rather than returning the arrays) keeps the kernels' captures on
    the ``lower_global_assign`` path instead of the freevar/closure one.
    """
    global _GLOBAL_1D, _GLOBAL_0D
    _GLOBAL_1D = cuda.to_device(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    _GLOBAL_0D = cuda.to_device(np.array(42.0, dtype=np.float64))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64], ids=str)
def test_freevar_capture(dtype):
    """Captured local device arrays (free variables) across dtypes and ranks."""
    # 1-d
    src = cuda.to_device(np.arange(5, dtype=dtype))
    out = cuda.device_array(5, dtype=dtype)

    @cuda.jit
    def copy_1d(r):
        i = cuda.grid(1)
        if i < 5:
            r[i] = src[i]

    copy_1d[1, 5](out)
    np.testing.assert_array_equal(out.copy_to_host(), src.copy_to_host())

    # 0-d (scalar)
    scalar = cuda.to_device(np.array(7, dtype=dtype))
    out0 = cuda.device_array(1, dtype=dtype)

    @cuda.jit
    def copy_0d(r):
        r[0] = scalar[()]

    copy_0d[1, 1](out0)
    np.testing.assert_array_equal(out0.copy_to_host(), [7])


def test_freevar_multiple():
    """Multiple captured arrays used together."""
    a = cuda.to_device(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = cuda.to_device(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    out = cuda.device_array(3, dtype=np.float32)

    @cuda.jit
    def mul(r):
        i = cuda.grid(1)
        if i < 3:
            r[i] = a[i] * b[i]

    mul[1, 3](out)
    np.testing.assert_array_equal(out.copy_to_host(), [4.0, 10.0, 18.0])


def test_global_capture(global_arrays):
    """Module-level (global) device arrays, 1-d and 0-d."""
    out1 = cuda.device_array(3, dtype=np.float32)

    @cuda.jit
    def read_global_1d(r):
        i = cuda.grid(1)
        if i < 3:
            r[i] = _GLOBAL_1D[i]

    read_global_1d[1, 3](out1)
    np.testing.assert_array_equal(out1.copy_to_host(), [10.0, 20.0, 30.0])

    out0 = cuda.device_array(1, dtype=np.float64)

    @cuda.jit
    def read_global_0d(r):
        r[0] = _GLOBAL_0D[()]

    read_global_0d[1, 1](out0)
    np.testing.assert_array_equal(out0.copy_to_host(), [42.0])


def test_global_and_freevar_mixed(global_arrays):
    """Global and local captures in the same kernel."""
    local = cuda.to_device(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    out = cuda.device_array(3, dtype=np.float32)

    @cuda.jit
    def add(r):
        i = cuda.grid(1)
        if i < 3:
            r[i] = _GLOBAL_1D[i] + local[i]

    add[1, 3](out)
    np.testing.assert_array_equal(out.copy_to_host(), [11.0, 22.0, 33.0])


def test_captures_keep_pyvals_alive(global_arrays):
    """Captured device arrays (both free-var and global) must be retained by
    the compiled kernel's library so their GPU memory cannot be freed and
    recycled while the pointers baked into the PTX are still in use."""
    local = cuda.to_device(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    out = cuda.device_array(3, dtype=np.float32)

    @cuda.jit
    def kernel(r):
        i = cuda.grid(1)
        if i < 3:
            r[i] = _GLOBAL_1D[i] + local[i]

    kernel[1, 3](out)
    sig = next(iter(kernel.overloads))
    refs = kernel.overloads[sig].library.referenced_objects
    assert refs[id(local)] is local
    assert refs[id(_GLOBAL_1D)] is _GLOBAL_1D
