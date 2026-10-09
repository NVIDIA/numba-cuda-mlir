# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""np.maximum, np.minimum, np.fmax and np.fmin handle NaN as NumPy does."""

import numpy as np
import pytest
from numba_cuda_mlir import cuda

X = [np.nan, 1.0, np.nan, -0.0, 0.0, 1.0, -np.inf]
Y = [1.0, np.nan, np.nan, 0.0, -0.0, 2.0, np.nan]


def make_kernels(ufunc):
    @cuda.jit
    def array_kernel(x, y, out):
        ufunc(x, y, out)

    @cuda.jit
    def scalar_kernel(x, y, out):
        i = cuda.grid(1)
        if i < out.size:
            ufunc(x[i], y[i], out[i:])

    return array_kernel[1, 1], scalar_kernel[1, len(X)]


@pytest.mark.parametrize(
    "ufunc", [np.maximum, np.minimum, np.fmax, np.fmin], ids=lambda u: u.__name__
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_min_max_nan(ufunc, dtype):
    x, y = np.array(X, dtype=dtype), np.array(Y, dtype=dtype)
    expected = ufunc(x, y)
    for kernel in make_kernels(ufunc):
        out = np.zeros_like(expected)
        kernel(x, y, out)
        # The sign of a zero result for equal zeros is left out: NumPy's own
        # vectorized loops and its documented definition disagree on it
        np.testing.assert_array_equal(out, expected)
