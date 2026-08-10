# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from numba_cuda_mlir import cuda


def test_boolean_bitwise_and_or_xor():
    @cuda.jit
    def k(inp, out):
        a = inp[0] > 0
        b = inp[1] > 0
        out[0] = 1 if a & b else 0
        out[1] = 1 if a | b else 0
        out[2] = 1 if a ^ b else 0

    inp = cuda.to_device(np.array([1, 0], dtype=np.int32))
    out = cuda.to_device(np.zeros(3, dtype=np.int32))
    k[1, 1](inp, out)
    np.testing.assert_array_equal(out.copy_to_host(), [0, 1, 1])

    inp = cuda.to_device(np.array([1, 1], dtype=np.int32))
    k[1, 1](inp, out)
    np.testing.assert_array_equal(out.copy_to_host(), [1, 1, 0])


def test_boolean_inplace_bitwise():
    @cuda.jit
    def k(inp, out):
        acc_and = True
        acc_or = False
        acc_xor = False
        for i in range(inp.shape[0]):
            flag = inp[i] > 0
            acc_and &= flag
            acc_or |= flag
            acc_xor ^= flag
        out[0] = 1 if acc_and else 0
        out[1] = 1 if acc_or else 0
        out[2] = 1 if acc_xor else 0

    inp = cuda.to_device(np.array([1, 0, 1], dtype=np.int32))
    out = cuda.to_device(np.zeros(3, dtype=np.int32))
    k[1, 1](inp, out)
    np.testing.assert_array_equal(out.copy_to_host(), [0, 1, 0])


def test_mixed_boolean_integer_bitwise():
    @cuda.jit
    def k(inp, out):
        flag = inp[0] > 0
        out[0] = inp[1] & flag
        out[1] = inp[1] | flag
        out[2] = flag ^ inp[1]

    inp = cuda.to_device(np.array([1, 6], dtype=np.int32))
    out = cuda.to_device(np.zeros(3, dtype=np.int32))
    k[1, 1](inp, out)
    np.testing.assert_array_equal(out.copy_to_host(), [6 & 1, 6 | 1, 1 ^ 6])


def test_boolean_invert():
    @cuda.jit
    def k(inp, out):
        a = inp[0] > 0
        b = ~a
        out[0] = 1 if b else 0
        out[1] = 1 if ~b else 0

    inp = cuda.to_device(np.array([0], dtype=np.int32))
    out = cuda.to_device(np.zeros(2, dtype=np.int32))
    k[1, 1](inp, out)
    np.testing.assert_array_equal(out.copy_to_host(), [1, 0])


def test_integer_invert():
    @cuda.jit
    def k(i32_io, u32_io, i8_io):
        i32_io[1] = ~i32_io[0]
        u32_io[1] = ~u32_io[0]
        i8_io[1] = ~i8_io[0]

    i32_io = cuda.to_device(np.array([5, 0], dtype=np.int32))
    u32_io = cuda.to_device(np.array([5, 0], dtype=np.uint32))
    i8_io = cuda.to_device(np.array([-1, 0], dtype=np.int8))
    k[1, 1](i32_io, u32_io, i8_io)
    assert i32_io.copy_to_host()[1] == ~np.int32(5)
    assert u32_io.copy_to_host()[1] == np.uint32(~np.uint32(5))
    assert i8_io.copy_to_host()[1] == ~np.int8(-1)
