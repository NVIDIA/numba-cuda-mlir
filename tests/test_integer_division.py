# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from numba_cuda_mlir import cuda
from numba_cuda_mlir.types import int32

WRAP_SEGMENTS = int32(5)


def test_signed_mod_negative_dividend():
    """% on runtime int32 values follows Python floored-modulo semantics."""

    @cuda.jit
    def k(a, b, out):
        i = cuda.grid(1)
        if i < a.size:
            out[i] = int32(a[i]) % int32(b[i])

    a = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 6, -6, 6, 7, -7], dtype=np.int32)
    b = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, -5, -5, -3, 3], dtype=np.int32)
    out = cuda.to_device(np.zeros_like(a))
    k[1, 32](cuda.to_device(a), cuda.to_device(b), out)
    expected = np.array([int(x) % int(y) for x, y in zip(a, b)], dtype=np.int32)
    np.testing.assert_array_equal(out.copy_to_host(), expected)


def test_signed_mod_frozen_constant_divisor():
    """% with a frozen-constant divisor and a runtime dividend.

    Literal-on-literal modulo folds in Python before lowering, so this
    is the smallest form that exercises the emitted MLIR.
    """

    @cuda.jit
    def k(a, out):
        i = cuda.grid(1)
        if i < a.size:
            out[i] = int32(a[i]) % WRAP_SEGMENTS

    a = np.array([-6, -5, -1, 0, 4, 6, -2147483648, 2147483647], dtype=np.int32)
    out = cuda.to_device(np.zeros_like(a))
    k[1, 32](cuda.to_device(a), out)
    expected = np.array([int(x) % 5 for x in a], dtype=np.int32)
    np.testing.assert_array_equal(out.copy_to_host(), expected)


def test_signed_floordiv_negative_operands():
    """// on runtime int32 values floors toward negative infinity."""

    @cuda.jit
    def k(a, b, out):
        i = cuda.grid(1)
        if i < a.size:
            out[i] = int32(a[i]) // int32(b[i])

    a = np.array([-6, -5, -1, 0, 6, -7, 7, -2147483648], dtype=np.int32)
    b = np.array([5, 5, 5, 5, 5, 3, -3, 7], dtype=np.int32)
    out = cuda.to_device(np.zeros_like(a))
    k[1, 32](cuda.to_device(a), cuda.to_device(b), out)
    expected = np.array([int(x) // int(y) for x, y in zip(a, b)], dtype=np.int32)
    np.testing.assert_array_equal(out.copy_to_host(), expected)


def test_signed_mod_floordiv_int64():
    """64-bit signed % and // keep floored semantics (no widening involved)."""

    @cuda.jit
    def k(a, b, mod_out, div_out):
        i = cuda.grid(1)
        if i < a.size:
            mod_out[i] = a[i] % b[i]
            div_out[i] = a[i] // b[i]

    a = np.array([-(2**40) - 3, 2**40 + 3, -6, 6], dtype=np.int64)
    b = np.array([7, -7, 5, -5], dtype=np.int64)
    mod_out = cuda.to_device(np.zeros_like(a))
    div_out = cuda.to_device(np.zeros_like(a))
    k[1, 32](cuda.to_device(a), cuda.to_device(b), mod_out, div_out)
    np.testing.assert_array_equal(
        mod_out.copy_to_host(),
        np.array([int(x) % int(y) for x, y in zip(a, b)], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        div_out.copy_to_host(),
        np.array([int(x) // int(y) for x, y in zip(a, b)], dtype=np.int64),
    )


def test_unsigned_mod_floordiv():
    """Unsigned % and // use unsigned ops on zero-extended operands."""

    @cuda.jit
    def k(a, b, mod_out, div_out):
        i = cuda.grid(1)
        if i < a.size:
            mod_out[i] = a[i] % b[i]
            div_out[i] = a[i] // b[i]

    a = np.array([4294967290, 7, 4294967295, 10], dtype=np.uint32)
    b = np.array([5, 5, 5, 3], dtype=np.uint32)
    mod_out = cuda.to_device(np.zeros_like(a))
    div_out = cuda.to_device(np.zeros_like(a))
    k[1, 32](cuda.to_device(a), cuda.to_device(b), mod_out, div_out)
    np.testing.assert_array_equal(mod_out.copy_to_host(), a % b)
    np.testing.assert_array_equal(div_out.copy_to_host(), a // b)


def test_float_mod_unchanged():
    """Float % keeps the floored a - floor(a/b) * b lowering."""

    @cuda.jit
    def k(a, b, out):
        i = cuda.grid(1)
        if i < a.size:
            out[i] = a[i] % b[i]

    a = np.array([-7.5, 7.5, -1.0, 5.25], dtype=np.float64)
    b = np.array([3.0, -3.0, 4.0, 2.0], dtype=np.float64)
    out = cuda.to_device(np.zeros_like(a))
    k[1, 32](cuda.to_device(a), cuda.to_device(b), out)
    np.testing.assert_allclose(out.copy_to_host(), a % b, rtol=1e-15)
