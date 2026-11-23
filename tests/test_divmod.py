# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`divmod()` on scalars in a kernel.

`divmod` shares its quotient and remainder with `//` and `%`, so these also pin
the three against each other: CPython guarantees `divmod(a, b) == (a // b, a % b)`
and a kernel must not disagree with itself about that.
"""

import itertools

import numpy as np
import pytest
from numba_cuda_mlir import cuda

SIGNED = [np.int8, np.int16, np.int32, np.int64]
UNSIGNED = [np.uint8, np.uint16, np.uint32, np.uint64]
FLOATS = [np.float32, np.float64]

INT_MAGNITUDES = [1, 2, 3, 5, 7, 8, 13, 100, 127]
FLOAT_MAGNITUDES = [0.5, 1.0, 2.0, 3.0, 7.0, 0.1, 1e3, 1e-3]


def _signed(values):
    return [*values, *[-v for v in values]]


def _run(dtype, pairs):
    @cuda.jit
    def kernel(a, b, out):
        i = cuda.grid(1)
        if i < a.size:
            quotient, remainder = divmod(a[i], b[i])
            out[i, 0] = quotient
            out[i, 1] = remainder
            out[i, 2] = a[i] // b[i]
            out[i, 3] = a[i] % b[i]

    a = np.array([p[0] for p in pairs], dtype=dtype)
    b = np.array([p[1] for p in pairs], dtype=dtype)
    out = cuda.device_array((len(pairs), 4), dtype=dtype)
    kernel[(len(pairs) + 63) // 64, 64](cuda.to_device(a), cuda.to_device(b), out)
    return out.copy_to_host()


@pytest.mark.parametrize("dtype", SIGNED)
def test_divmod_signed_matches_python(dtype):
    pairs = list(itertools.product(_signed(INT_MAGNITUDES), repeat=2))
    got = _run(dtype, pairs)
    expected = [divmod(dtype(x), dtype(y)) for x, y in pairs]
    np.testing.assert_array_equal(got[:, 0], [q for q, _ in expected])
    np.testing.assert_array_equal(got[:, 1], [r for _, r in expected])


@pytest.mark.parametrize("dtype", UNSIGNED)
def test_divmod_unsigned_matches_python(dtype):
    pairs = list(itertools.product(INT_MAGNITUDES, repeat=2))
    got = _run(dtype, pairs)
    expected = [divmod(dtype(x), dtype(y)) for x, y in pairs]
    np.testing.assert_array_equal(got[:, 0], [q for q, _ in expected])
    np.testing.assert_array_equal(got[:, 1], [r for _, r in expected])


@pytest.mark.parametrize("dtype", FLOATS)
def test_divmod_float_matches_python(dtype):
    pairs = list(itertools.product(_signed(FLOAT_MAGNITUDES), repeat=2))
    got = _run(dtype, pairs)
    expected = [divmod(dtype(x), dtype(y)) for x, y in pairs]
    np.testing.assert_array_equal(got[:, 0], [q for q, _ in expected])
    np.testing.assert_array_equal(got[:, 1], [r for _, r in expected])


@pytest.mark.parametrize("dtype", SIGNED + UNSIGNED + FLOATS)
def test_divmod_agrees_with_floordiv_and_mod(dtype):
    """A kernel must not disagree with itself about `divmod` versus `//` and `%`."""
    magnitudes = FLOAT_MAGNITUDES if dtype in FLOATS else INT_MAGNITUDES
    values = magnitudes if dtype in UNSIGNED else _signed(magnitudes)
    got = _run(dtype, list(itertools.product(values, repeat=2)))
    np.testing.assert_array_equal(got[:, 0], got[:, 2])
    np.testing.assert_array_equal(got[:, 1], got[:, 3])


def test_divmod_of_literals():
    """Both operands constant: the result is still a usable tuple."""

    @cuda.jit
    def kernel(out):
        quotient, remainder = divmod(17, 5)
        out[0] = quotient
        out[1] = remainder

    out = cuda.device_array(2, dtype=np.int64)
    kernel[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), list(divmod(17, 5)))


def test_divmod_tuple_is_indexable():
    """The result is a real tuple, not only unpackable."""

    @cuda.jit
    def kernel(a, b, out):
        pair = divmod(a[0], b[0])
        out[0] = pair[0]
        out[1] = pair[1]

    a = cuda.to_device(np.array([-7], dtype=np.int64))
    b = cuda.to_device(np.array([3], dtype=np.int64))
    out = cuda.device_array(2, dtype=np.int64)
    kernel[1, 1](a, b, out)
    np.testing.assert_array_equal(out.copy_to_host(), list(divmod(-7, 3)))
