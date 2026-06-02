# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import sys
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda import bf16

pytestmark = [
    pytest.mark.requires_cc_min((8, 0), "bfloat16"),
    pytest.mark.skipif(sys.platform == "win32", reason="bfloat16 tests are disabled on Windows"),
]

BF16_UNARY_OPS = [
    (bf16.htrunc, 2.7, 2.0),
    (bf16.hceil, 2.3, 3.0),
    (bf16.hfloor, 2.7, 2.0),
    (bf16.hsqrt, 4.0, 2.0),
    (bf16.hrsqrt, 4.0, 0.5),
    (bf16.hrcp, 2.0, 0.5),
    (bf16.hlog, np.e, 1.0),
    (bf16.hlog2, 8.0, 3.0),
    (bf16.hcos, 0.0, 1.0),
    (bf16.hsin, 0.0, 0.0),
    (bf16.hexp, 0.0, 1.0),
    (bf16.hexp2, 3.0, 8.0),
]

BF16_BINARY_OPS = [
    (bf16.hadd, 2.0, 3.0, 5.0),
    (bf16.hsub, 5.0, 3.0, 2.0),
    (bf16.hmul, 2.0, 3.0, 6.0),
    (bf16.hdiv, 6.0, 2.0, 3.0),
    (bf16.hmax, 2.0, 5.0, 5.0),
    (bf16.hmin, 2.0, 5.0, 2.0),
]

BF16_COMPARISON_OPS = [
    (bf16.heq, 2.0, 2.0, True),
    (bf16.heq, 2.0, 3.0, False),
    (bf16.hne, 2.0, 3.0, True),
    (bf16.hne, 2.0, 2.0, False),
    (bf16.hgt, 3.0, 2.0, True),
    (bf16.hgt, 2.0, 3.0, False),
    (bf16.hlt, 2.0, 3.0, True),
    (bf16.hlt, 3.0, 2.0, False),
    (bf16.hge, 3.0, 2.0, True),
    (bf16.hge, 2.0, 2.0, True),
    (bf16.hle, 2.0, 3.0, True),
    (bf16.hle, 2.0, 2.0, True),
]


@pytest.mark.parametrize("op,input_val,expected", BF16_UNARY_OPS)
def test_bf16_unary_intrinsics(op, input_val, expected):
    @cuda.jit
    def kernel(out, x):
        out[0] = op(bf16.bfloat16(x))

    out = np.zeros(1, dtype=np.float32)
    kernel[1, 1](out, input_val)
    np.testing.assert_allclose(out[0], expected, rtol=0.1)


@pytest.mark.parametrize("op,a,b,expected", BF16_BINARY_OPS)
def test_bf16_binary_intrinsics(op, a, b, expected):
    @cuda.jit
    def kernel(out, x, y):
        out[0] = op(bf16.bfloat16(x), bf16.bfloat16(y))

    out = np.zeros(1, dtype=np.float32)
    kernel[1, 1](out, a, b)
    np.testing.assert_allclose(out[0], expected, rtol=0.1)


@pytest.mark.parametrize("op,a,b,expected", BF16_COMPARISON_OPS)
def test_bf16_comparison_intrinsics(op, a, b, expected):
    @cuda.jit
    def kernel(out, x, y):
        out[0] = 1 if op(bf16.bfloat16(x), bf16.bfloat16(y)) else 0

    out = np.zeros(1, dtype=np.int32)
    kernel[1, 1](out, a, b)
    assert out[0] == (1 if expected else 0)


@pytest.mark.parametrize("value", [3.14, 2.5, 1.5, 5.0, 10.0, 42.0])
def test_bf16_constructor(value):
    @cuda.jit
    def kernel(out, x):
        out[0] = bf16.bfloat16(x)

    out = np.zeros(1, dtype=np.float32)
    kernel[1, 1](out, value)
    np.testing.assert_allclose(out[0], float(value), rtol=0.1)


def test_bf16_fma():
    @cuda.jit
    def kernel(out, a, b, c):
        out[0] = bf16.hfma(bf16.bfloat16(a), bf16.bfloat16(b), bf16.bfloat16(c))

    out = np.zeros(1, dtype=np.float32)
    kernel[1, 1](out, 2.0, 3.0, 4.0)
    np.testing.assert_allclose(out[0], 10.0, rtol=0.1)


def test_bf16_fma_relu():
    @cuda.jit
    def kernel(out, a, b, c):
        out[0] = bf16.hfma_relu(bf16.bfloat16(a), bf16.bfloat16(b), bf16.bfloat16(c))

    out = np.zeros(1, dtype=np.float32)
    kernel[1, 1](out, 2.0, 3.0, -10.0)
    np.testing.assert_allclose(out[0], 0.0, rtol=0.1)


def test_bf16_saturating_add():
    @cuda.jit
    def kernel(out, a, b):
        out[0] = bf16.hadd_sat(bf16.bfloat16(a), bf16.bfloat16(b))

    out = np.zeros(1, dtype=np.float32)
    kernel[1, 1](out, 0.8, 0.5)
    np.testing.assert_allclose(out[0], 1.0, rtol=0.1)


def test_bf16_special_value_checks():
    @cuda.jit
    def kernel(out_inf, out_nan):
        out_inf[0] = 1 if bf16.hisinf(bf16.bfloat16(float("inf"))) else 0
        out_nan[0] = 1 if bf16.hisnan(bf16.bfloat16(float("nan"))) else 0

    out_inf = np.zeros(1, dtype=np.int32)
    out_nan = np.zeros(1, dtype=np.int32)
    kernel[1, 1](out_inf, out_nan)
    assert out_inf[0] == 1
    assert out_nan[0] == 1


def test_bf16_operators():
    @cuda.jit
    def kernel(out, a, b):
        x = bf16.bfloat16(a)
        y = bf16.bfloat16(b)
        out[0] = x + y
        out[1] = x - y
        out[2] = x * y
        out[3] = x / y

    out = np.zeros(4, dtype=np.float32)
    kernel[1, 1](out, 6.0, 2.0)
    np.testing.assert_allclose(out, [8.0, 4.0, 12.0, 3.0], rtol=0.1)


def test_bf16_comparison_operators():
    @cuda.jit
    def kernel(out, a, b):
        x = bf16.bfloat16(a)
        y = bf16.bfloat16(b)
        out[0] = 1 if x == y else 0
        out[1] = 1 if x != y else 0
        out[2] = 1 if x < y else 0
        out[3] = 1 if x > y else 0

    out = np.zeros(4, dtype=np.int32)
    kernel[1, 1](out, 2.0, 3.0)
    np.testing.assert_array_equal(out, [0, 1, 1, 0])


def test_bf16_float2bfloat16():
    @cuda.jit
    def kernel(out, x):
        out[0] = bf16.float2bfloat16(x[0])

    out = np.zeros(1, dtype=np.float32)
    x = np.array([3.14], dtype=np.float32)
    kernel[1, 1](out, x)
    np.testing.assert_allclose(out[0], 3.14, rtol=0.1)


def test_bf16_double2bfloat16():
    @cuda.jit
    def kernel(out, x):
        out[0] = bf16.double2bfloat16(x)

    out = np.zeros(1, dtype=np.float32)
    kernel[1, 1](out, np.float64(2.718))
    np.testing.assert_allclose(out[0], 2.718, rtol=0.1)


def test_bf16_bfloat162float():
    @cuda.jit
    def kernel(out, x):
        out[0] = bf16.bfloat162float(bf16.bfloat16(x))

    out = np.zeros(1, dtype=np.float32)
    kernel[1, 1](out, 3.14)
    np.testing.assert_allclose(out[0], 3.14, rtol=0.1)
