# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from numba_cuda_mlir import cuda

FIDX = np.array([2, 0, 1], dtype=np.int32)
FVALS = np.array([1.5, -2.5, 4.0, 8.0], dtype=np.float32)
FGRID = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)


def test_frozen_index_array_correctness():
    """A frozen integer array gathers the expected elements."""

    @cuda.jit
    def k(inp, out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = inp[FIDX[i % 3]]

    inp = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    out = cuda.to_device(np.zeros(6, dtype=np.float32))
    k[1, 32](cuda.to_device(inp), out)
    expected = inp[FIDX[np.arange(6) % 3]]
    np.testing.assert_allclose(out.copy_to_host(), expected)


def test_frozen_value_array_correctness():
    """A frozen float array contributes the expected values."""

    @cuda.jit
    def k(inp, out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = inp[i] + FVALS[i % 4]

    inp = np.arange(8, dtype=np.float32)
    out = cuda.to_device(np.zeros(8, dtype=np.float32))
    k[1, 32](cuda.to_device(inp), out)
    expected = inp + FVALS[np.arange(8) % 4]
    np.testing.assert_allclose(out.copy_to_host(), expected)


def test_frozen_2d_array_correctness():
    """A frozen 2-D float64 array indexes correctly on both axes."""

    @cuda.jit
    def k(out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = FGRID[i // 2, i % 2]

    out = cuda.to_device(np.zeros(4, dtype=np.float64))
    k[1, 32](out)
    np.testing.assert_allclose(out.copy_to_host(), FGRID.ravel())


def test_frozen_array_lowers_to_constant_global():
    """Array literals lower to constant globals, not heap allocation.

    The kernel name is kept free of the substring "alloc" because the
    PTX embeds the mangled test name.
    """

    @cuda.jit
    def k(inp, out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = inp[i] + FVALS[i % 4] + inp[FIDX[i % 3]]

    inp = np.ones(8, dtype=np.float32)
    out = cuda.to_device(np.zeros(8, dtype=np.float32))
    k[1, 32](cuda.to_device(inp), out)
    ptx = str(k.inspect_lto_ptx())
    assert "__numba_cuda_mlir_array_literal" in ptx
    assert "malloc_param" not in ptx


def test_frozen_array_large_grid():
    """Large grids survive kernels that capture array literals.

    The bufferized element-wise lowering allocated a per-thread device
    malloc with no free; the 8 MB default heap was exhausted around
    64k threads and the kernel faulted with an illegal address.
    """
    n = 1 << 18

    @cuda.jit
    def k(inp, out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = inp[i] + FVALS[i % 4]

    inp = np.ones(n, dtype=np.float32)
    out = cuda.to_device(np.zeros(n, dtype=np.float32))
    threads = 256
    k[(n + threads - 1) // threads, threads](cuda.to_device(inp), out)
    res = out.copy_to_host()
    expected = inp + FVALS[np.arange(n) % 4]
    np.testing.assert_allclose(res, expected)
