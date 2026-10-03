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
INTS = np.array([-3, 0, 1, 2, 7], dtype=np.int64)
OTHER_INTS = np.array([7, 2, 1, 0, -3], dtype=np.int64)
# Unsigned values above the signed range must compare as unsigned
UINT32S = np.array([0, 1, 2**31, 3_000_000_000, 2**32 - 1], dtype=np.uint32)
OTHER_UINT32S = np.array([2**32 - 1, 2**31 - 1, 2**31, 1, 0], dtype=np.uint32)
INT32S = np.array([-3, 0, 2**31 - 1, -(2**31), 7], dtype=np.int32)
OTHER_INT32S = np.array([7, -(2**31), 1, 2**31 - 1, -3], dtype=np.int32)
FLOAT32S = np.array([-1.5, -0.0, 0.0, 0.25, 3e9], dtype=np.float32)
OTHER_FLOAT32S = np.array([2.0, 0.25, -0.0, 0.0, -1.5], dtype=np.float32)
# NumPy compares a signed integer and a uint64 exactly, not as float64 or
# as their bit patterns
UINT64S = np.array([0, 2**63, 2**64 - 1, 2**63 - 1, 2**63], dtype=np.uint64)
INT64S = np.array([-1, 2**63 - 1, -1, 2**63 - 1, -(2**63)], dtype=np.int64)


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
        rtol = 1e-6 if expected.dtype == np.float32 else 1e-12
        np.testing.assert_allclose(out, expected, rtol=rtol)
    else:
        np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("ufunc", BINARY, ids=lambda u: u.__name__)
@pytest.mark.parametrize(
    "x, y",
    [
        (FLOATS, OTHER_FLOATS),
        (INTS, OTHER_INTS),
        (INTS, FLOATS),
        (UINT32S, OTHER_UINT32S),
        (INT32S, OTHER_INT32S),
        (UINT32S, INT32S),
        (FLOAT32S, OTHER_FLOAT32S),
        (UINT64S, INT64S),
    ],
    ids=[
        "float",
        "int",
        "int-float",
        "uint32",
        "int32",
        "uint32-int32",
        "float32",
        "uint64-int64",
    ],
)
def test_binary_ufunc_on_scalars(ufunc, x, y):
    check(ufunc, x, y)


@pytest.mark.parametrize("ufunc", INTEGER_BINARY, ids=lambda u: u.__name__)
@pytest.mark.parametrize(
    "x, y",
    [(INTS, OTHER_INTS), (UINT32S, OTHER_UINT32S), (UINT32S, INT32S)],
    ids=["int", "uint32", "uint32-int32"],
)
def test_integer_binary_ufunc_on_scalars(ufunc, x, y):
    check(ufunc, x, y)


@pytest.mark.parametrize("ufunc", UNARY, ids=lambda u: u.__name__)
@pytest.mark.parametrize("x", [FLOATS, INTS, UINT32S], ids=["float", "int", "uint32"])
def test_unary_ufunc_on_scalars(ufunc, x):
    check(ufunc, x)


# np.bitwise_not is an alias of np.invert in NumPy 2, so name the cases explicitly
@pytest.mark.parametrize("ufunc", INTEGER_UNARY, ids=["invert", "bitwise_not"])
@pytest.mark.parametrize("x", [INTS, UINT32S], ids=["int", "uint32"])
def test_integer_unary_ufunc_on_scalars(ufunc, x):
    check(ufunc, x)
