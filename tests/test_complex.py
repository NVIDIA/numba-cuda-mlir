# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pytest

from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir import cuda


class TestAtomicOnComplexComponents(unittest.TestCase):
    def test_atomic_on_real_1d(self):
        @cuda.jit
        def atomic_add_one(values):
            i = cuda.grid(1)
            cuda.atomic.add(values.real, i, 1)

        N = 32
        arr1 = np.arange(N) + np.arange(N) * 1j
        arr2 = arr1.copy()
        atomic_add_one[1, N](arr2)
        np.testing.assert_equal(arr1 + 1, arr2)

    def test_atomic_on_imag_1d(self):
        @cuda.jit
        def atomic_add_one_j(values):
            i = cuda.grid(1)
            cuda.atomic.add(values.imag, i, 1)

        N = 32
        arr1 = np.arange(N) + np.arange(N) * 1j
        arr2 = arr1.copy()
        atomic_add_one_j[1, N](arr2)
        np.testing.assert_equal(arr1 + 1j, arr2)

    def test_atomic_on_real_2d(self):
        @cuda.jit
        def atomic_add_one_2d(values):
            i, j = cuda.grid(2)
            if i < values.shape[0] and j < values.shape[1]:
                cuda.atomic.add(values.real, (i, j), 1)

        M, N = 4, 8
        arr1 = (np.arange(M * N) + np.arange(M * N) * 1j).reshape(M, N)
        arr2 = arr1.copy()
        atomic_add_one_2d[(1, 1), (M, N)](arr2)
        np.testing.assert_equal(arr1 + 1, arr2)

    def test_atomic_on_imag_2d(self):
        @cuda.jit
        def atomic_add_one_j_2d(values):
            i, j = cuda.grid(2)
            if i < values.shape[0] and j < values.shape[1]:
                cuda.atomic.add(values.imag, (i, j), 1)

        M, N = 4, 8
        arr1 = (np.arange(M * N) + np.arange(M * N) * 1j).reshape(M, N)
        arr2 = arr1.copy()
        atomic_add_one_j_2d[(1, 1), (M, N)](arr2)
        np.testing.assert_equal(arr1 + 1j, arr2)


N_SHARED = 32


@pytest.mark.parametrize(
    "complex_dtype,float_dtype",
    [
        (np.complex64, np.float32),
        (np.complex128, np.float64),
    ],
)
def test_shared_memory_complex_real_imag(complex_dtype, float_dtype):
    """Regression: .real/.imag on shared-memory complex arrays must preserve address space."""

    @cuda.jit
    def kernel(inp, out_real, out_imag):
        tid = cuda.threadIdx.x
        sm = cuda.shared.array(N_SHARED, dtype=complex_dtype)
        sm[tid] = inp[tid]
        cuda.syncthreads()
        out_real[tid] = sm.real[tid]
        out_imag[tid] = sm.imag[tid]

    rng = np.random.default_rng(42)
    inp = (rng.standard_normal(N_SHARED) + 1j * rng.standard_normal(N_SHARED)).astype(complex_dtype)
    out_real = np.zeros(N_SHARED, dtype=float_dtype)
    out_imag = np.zeros(N_SHARED, dtype=float_dtype)

    kernel[1, N_SHARED](inp, out_real, out_imag)

    np.testing.assert_allclose(out_real, inp.real)
    np.testing.assert_allclose(out_imag, inp.imag)


@pytest.mark.parametrize(
    "complex_type",
    [
        np.complex64,
        np.complex128,
        complex,
    ],
)
def test_complex_constructor_2args(complex_type):
    @cuda.jit
    def kernel(out):
        out[0] = complex_type(1.5, -2.5)

    dtype = np.complex128 if complex_type is complex else complex_type
    out = np.zeros(1, dtype=dtype)
    kernel[1, 1](out)

    np.testing.assert_equal(out[0], complex_type(1.5 - 2.5j))
