# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NumPy ufuncs called on scalars inside a kernel, e.g. out[i] = np.maximum(x[i], y[i])."""

import numpy as np
import pytest
from numba_cuda_mlir import cuda

BINARY = [
    np.maximum,
    np.minimum,
    np.fmax,
    np.fmin,
    np.greater,
    np.greater_equal,
    np.less,
    np.less_equal,
    np.equal,
    np.not_equal,
    np.logical_and,
    np.logical_or,
    np.logical_xor,
    np.arctan2,
    np.hypot,
]
INTEGER_BINARY = [np.bitwise_and, np.bitwise_or, np.bitwise_xor]
UNARY = [
    np.sin,
    np.cos,
    np.tan,
    np.arcsin,
    np.arccos,
    np.arctan,
    np.sinh,
    np.cosh,
    np.tanh,
    np.arcsinh,
    np.arccosh,
    np.arctanh,
    np.deg2rad,
    np.radians,
    np.rad2deg,
    np.degrees,
    np.log,
    np.log2,
    np.log10,
    np.exp,
    np.logical_not,
]
INTEGER_UNARY = [np.invert, np.bitwise_not]

FLOATS = np.array([-1.5, -0.0, 0.0, 0.25, 2.0])
OTHER_FLOATS = np.array([2.0, 0.25, -0.0, 0.0, -1.5])
INTS = np.array([-3, 0, 1, 2, 7])
OTHER_INTS = np.array([7, 2, 1, 0, -3])


def make_kernel(ufunc, nin):
    if nin == 1:

        @cuda.jit
        def kernel(out, x):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = ufunc(x[i])

    else:

        @cuda.jit
        def kernel(out, x, y):
            i = cuda.grid(1)
            if i < out.size:
                out[i] = ufunc(x[i], y[i])

    return kernel


def check(ufunc, *inputs):
    with np.errstate(all="ignore"):
        expected = ufunc(*inputs)
    out = np.zeros_like(expected)
    make_kernel(ufunc, len(inputs))[1, out.size](out, *inputs)
    if expected.dtype.kind == "f":
        np.testing.assert_allclose(out, expected, rtol=1e-12)
    else:
        np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("ufunc", BINARY, ids=lambda u: u.__name__)
@pytest.mark.parametrize(
    "x, y",
    [(FLOATS, OTHER_FLOATS), (INTS, OTHER_INTS), (INTS, FLOATS)],
    ids=["float", "int", "int-float"],
)
def test_binary_ufunc_on_scalars(ufunc, x, y):
    check(ufunc, x, y)


@pytest.mark.parametrize("ufunc", INTEGER_BINARY, ids=lambda u: u.__name__)
def test_integer_binary_ufunc_on_scalars(ufunc):
    check(ufunc, INTS, OTHER_INTS)


@pytest.mark.parametrize("ufunc", UNARY, ids=lambda u: u.__name__)
@pytest.mark.parametrize("x", [FLOATS, INTS], ids=["float", "int"])
def test_unary_ufunc_on_scalars(ufunc, x):
    check(ufunc, x)


# np.bitwise_not is an alias of np.invert in NumPy 2, so name the cases explicitly
@pytest.mark.parametrize("ufunc", INTEGER_UNARY, ids=["invert", "bitwise_not"])
def test_integer_unary_ufunc_on_scalars(ufunc):
    check(ufunc, INTS)
