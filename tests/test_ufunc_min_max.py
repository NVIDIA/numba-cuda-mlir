# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""np.maximum, np.minimum, np.fmax and np.fmin on NaN, signed zeros and complex numbers."""

import numpy as np
import pytest
from numba_cuda_mlir import cuda

UFUNCS = [np.maximum, np.minimum, np.fmax, np.fmin]

X = [np.nan, 1.0, np.nan, -0.0, 0.0, 1.0, -np.inf]
Y = [1.0, np.nan, np.nan, 0.0, -0.0, 2.0, np.nan]

# (x, y) as (real, imaginary) parts, which keep the sign of zeros: a nan part in
# x, in y or in both, orderings decided by the real or the imaginary parts, ties
COMPLEX_PAIRS = [
    ((np.nan, 0.0), (1.0, 0.0)),
    ((1.0, 0.0), (np.nan, 0.0)),
    ((0.0, np.nan), (1.0, 0.0)),
    ((1.0, 0.0), (0.0, np.nan)),
    ((2.0, 0.0), (1.0, np.nan)),
    ((1.0, np.nan), (2.0, 0.0)),
    ((np.nan, 0.0), (0.0, np.nan)),
    ((np.nan, 1.0), (np.nan, 2.0)),
    ((1.0, 2.0), (1.0, 3.0)),
    ((1.0, 5.0), (2.0, 0.0)),
    ((-0.0, 0.0), (0.0, 0.0)),
    ((1.0, -0.0), (1.0, 0.0)),
]


def make_kernels(ufunc, size):
    @cuda.jit
    def array_kernel(x, y, out):
        ufunc(x, y, out)

    @cuda.jit
    def scalar_kernel(x, y, out):
        i = cuda.grid(1)
        if i < out.size:
            ufunc(x[i], y[i], out[i:])

    return array_kernel[1, 1], scalar_kernel[1, size]


@pytest.mark.parametrize("ufunc", UFUNCS, ids=lambda u: u.__name__)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_min_max_nan(ufunc, dtype):
    x, y = np.array(X, dtype=dtype), np.array(Y, dtype=dtype)
    expected = ufunc(x, y)
    for kernel in make_kernels(ufunc, x.size):
        out = np.zeros_like(expected)
        kernel(x, y, out)
        np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("ufunc", UFUNCS, ids=lambda u: u.__name__)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_min_max_signed_zero(ufunc, dtype):
    # -0.0 orders below 0.0, as in IEEE 754-2019, the builtin min and max, and
    # NumPy on aarch64. NumPy on x86-64 returns y for maximum and minimum, and x
    # or y for fmax and fmin.
    x = np.array([-0.0, 0.0, -0.0, 0.0], dtype=dtype)
    y = np.array([0.0, -0.0, -0.0, 0.0], dtype=dtype)
    if ufunc in (np.minimum, np.fmin):
        expected = np.array([-0.0, -0.0, -0.0, 0.0], dtype=dtype)
    else:
        expected = np.array([0.0, 0.0, -0.0, 0.0], dtype=dtype)
    for kernel in make_kernels(ufunc, x.size):
        out = np.full_like(x, np.nan)
        kernel(x, y, out)
        np.testing.assert_array_equal(out, expected)
        np.testing.assert_array_equal(np.signbit(out), np.signbit(expected))


def complex_array(parts, dtype):
    z = np.empty(len(parts), dtype=dtype)
    z.real, z.imag = np.array(parts).T
    return z


@pytest.mark.parametrize("ufunc", UFUNCS, ids=lambda u: u.__name__)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_min_max(ufunc, dtype):
    x = complex_array([pair[0] for pair in COMPLEX_PAIRS], dtype)
    y = complex_array([pair[1] for pair in COMPLEX_PAIRS], dtype)
    # NumPy's complex loops return x or y unchanged, the same on every platform
    expected = ufunc(x, y)
    bits = np.dtype(f"u{x.real.itemsize}")
    for kernel in make_kernels(ufunc, x.size):
        out = np.zeros_like(expected)
        kernel(x, y, out)
        # Bit for bit, which tells nans and signed zeros apart
        np.testing.assert_array_equal(out.view(bits), expected.view(bits))
