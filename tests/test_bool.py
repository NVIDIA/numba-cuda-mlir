# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from numba_cuda_mlir import cuda


def test_boolean():
    @cuda.jit("void(float64[:], bool_)")
    def k(A, vertial):
        if vertial:
            A[0] = 123
        else:
            A[0] = 321

    A = np.array([0], dtype="float64")
    A = cuda.to_device(A)
    k[1, 1](A, True)
    assert A.copy_to_host()[0] == 123
    k[1, 1](A, False)
    assert A.copy_to_host()[0] == 321


def test_boolean_equality_comparisons():
    @cuda.jit
    def k(inp, out):
        a = inp[0] > 0
        b = inp[1] > 0
        out[0] = 1 if a == b else 0
        out[1] = 1 if a != b else 0

    out = cuda.to_device(np.zeros(2, dtype=np.int32))
    inp = cuda.to_device(np.array([1, 0], dtype=np.int32))
    k[1, 1](inp, out)
    np.testing.assert_array_equal(out.copy_to_host(), [0, 1])

    inp = cuda.to_device(np.array([1, 1], dtype=np.int32))
    k[1, 1](inp, out)
    np.testing.assert_array_equal(out.copy_to_host(), [1, 0])


def test_boolean_ordering_comparisons():
    """Booleans must compare as unsigned i1: True > False."""

    @cuda.jit
    def k(inp, out):
        a = inp[0] > 0
        b = inp[1] > 0
        out[0] = 1 if a < b else 0
        out[1] = 1 if a <= b else 0
        out[2] = 1 if a > b else 0
        out[3] = 1 if a >= b else 0

    out = cuda.to_device(np.zeros(4, dtype=np.int32))
    for lhs, rhs in [(1, 0), (0, 1), (1, 1), (0, 0)]:
        inp = cuda.to_device(np.array([lhs, rhs], dtype=np.int32))
        k[1, 1](inp, out)
        a, b = bool(lhs), bool(rhs)
        expected = [int(a < b), int(a <= b), int(a > b), int(a >= b)]
        np.testing.assert_array_equal(out.copy_to_host(), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_to_boolean_conversion(dtype):
    """A float converts to bool by comparing against zero, not by truncating.

    Lowering the conversion to fptoui/fptosi with an i1 target keeps the low bit of
    the truncated integer, so 0.0 came out True and 2.0 came out False.
    """

    @cuda.jit
    def k(out, a):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = a[i]

    values = np.array([0.0, -0.0, 0.5, 1.0, 2.0, -1.0, 1e-45, np.inf, -np.inf, np.nan], dtype=dtype)
    a = cuda.to_device(values)
    out = cuda.to_device(np.zeros(values.size, dtype=np.bool_))
    k[1, values.size](out, a)
    np.testing.assert_array_equal(out.copy_to_host(), values.astype(np.bool_))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_to_wider_integer_still_truncates(dtype):
    """Only the i1 target compares against zero; wider integers keep truncation."""

    @cuda.jit
    def k(out, a):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = a[i]

    values = np.array([0.0, 2.5, -2.5, 7.9], dtype=dtype)
    a = cuda.to_device(values)
    out = cuda.to_device(np.zeros(values.size, dtype=np.int32))
    k[1, values.size](out, a)
    np.testing.assert_array_equal(out.copy_to_host(), values.astype(np.int32))
