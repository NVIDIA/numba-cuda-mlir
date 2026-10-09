# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unary minus and plus on arrays inside a kernel, e.g. y = -x or y = +x."""

import numpy as np
import pytest
from numba_cuda_mlir import cuda

VALUES = {
    "int8": np.array([0, 1, -1, 127, -128], dtype=np.int8),
    "int32": np.array([0, 1, -1, 2**31 - 1, -(2**31)], dtype=np.int32),
    "int64": np.array([0, 1, -1, 2**63 - 1, -(2**63)], dtype=np.int64),
    "uint8": np.array([0, 1, 128, 255], dtype=np.uint8),
    "uint32": np.array([0, 1, 2**31, 2**32 - 1], dtype=np.uint32),
    # Negating a float flips the sign of zeros too
    "float32": np.array([0.0, -0.0, 1.5, -np.inf, np.nan, -np.nan], dtype=np.float32),
    "float64": np.array([0.0, -0.0, 1.5, -np.inf, np.nan, -np.nan]),
    "complex64": np.array(
        [0j, complex(-0.0, 0.0), 1 - 2j, complex(np.nan, -0.0)], dtype=np.complex64
    ),
    "complex128": np.array([0j, complex(-0.0, 0.0), 1 - 2j, complex(np.nan, -0.0)]),
}


def assert_same_values(out, expected):
    np.testing.assert_array_equal(out, expected)
    if out.dtype.kind in "fc":
        # PTX leaves the sign of a negated NaN unspecified, so check the other signs
        for part in (np.real, np.imag):
            signed = ~np.isnan(part(expected))
            np.testing.assert_array_equal(
                np.signbit(part(out))[signed], np.signbit(part(expected))[signed]
            )


@pytest.mark.parametrize("x", VALUES.values(), ids=VALUES.keys())
def test_negate_array(x):
    @cuda.jit
    def kernel(x, out):
        y = -x
        for i in range(out.size):
            out[i] = y[i]

    out = np.zeros_like(x)
    kernel[1, 1](x, out)
    assert_same_values(out, -x)


@pytest.mark.parametrize("x", VALUES.values(), ids=VALUES.keys())
def test_negative_ufunc_on_array(x):
    @cuda.jit
    def kernel(x, out):
        y = np.negative(x)
        for i in range(out.size):
            out[i] = y[i]

    out = np.zeros_like(x)
    kernel[1, 1](x, out)
    assert_same_values(out, np.negative(x))


def test_negate_2d_array_expression():
    @cuda.jit
    def kernel(x, out):
        y = -x + 1.0
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = y[i, j]

    x = np.arange(12.0).reshape(3, 4)
    out = np.zeros_like(x)
    kernel[1, 1](x, out)
    np.testing.assert_array_equal(out, -x + 1.0)


@pytest.mark.parametrize("x", VALUES.values(), ids=VALUES.keys())
def test_unary_plus_array(x):
    @cuda.jit
    def kernel(x, out):
        y = +x
        for i in range(out.size):
            out[i] = y[i]

    out = np.zeros_like(x)
    kernel[1, 1](x, out)
    assert_same_values(out, +x)


def test_unary_plus_returns_a_copy():
    @cuda.jit
    def kernel(x, out):
        y = +x
        y[0] = 99.0
        out[0] = x[0]
        out[1] = y[0]

    x = np.array([1.0, 2.0])
    out = np.zeros(2)
    kernel[1, 1](x, out)
    np.testing.assert_array_equal(out, [1.0, 99.0])
