# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
import numpy as np
import pytest
import operator

DTYPES = (np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)


# @pytest.mark.onerous
@pytest.mark.parametrize("dt1", DTYPES)
@pytest.mark.parametrize("dt2", DTYPES)
def test_broadcasted_floor_div(dt1, dt2):
    @cuda.jit
    def kernel(a, b, c):
        x = b // c
        for i in range(a.shape[0]):
            a[i] = x[i]

    b = np.ones([5], dtype=dt1)
    c = np.array(range(1, 6), dtype=dt2)
    expect = b // c
    a = np.zeros_like(expect)

    da = cuda.to_device(a)
    db = cuda.to_device(b)
    dc = cuda.to_device(c)
    kernel[1, 1](da, db, dc)
    a = da.copy_to_host()
    print(b, c)
    print(a, expect)
    assert np.allclose(expect, a), f"{expect=} != {a=}"


@pytest.mark.parametrize(
    "dtype, b, c",
    [
        (np.int32, [-7, 7, -7, 7, -1, 3], [2, -2, -2, 2, 3, 0]),
        (np.int64, [-7, 7, -7, 7, -1, 3], [2, -2, -2, 2, 3, 0]),
        (np.uint8, [200, 250, 7, 128, 255, 3], [3, 2, 2, 2, 255, 0]),
        (np.uint32, [3000000000, 7, 2**32 - 1, 3, 0, 3], [2, 2, 3, 2, 5, 0]),
        (np.float32, [-7.5, 7.5, 1.0, 0.75, -1.0, 3.0], [2.0, -2.0, 0.25, 0.25, 3.0, 0.0]),
        (np.float64, [-7.5, 7.5, 1.0, 1.0, -1.0, 3.0], [2.0, -2.0, 0.25, 0.1, 3.0, 0.0]),
    ],
)
def test_broadcasted_floor_div_values(dtype, b, c):
    """Rounds towards -inf, keeps fractional float operands, and x // 0 follows NumPy."""

    @cuda.jit
    def kernel(a, b, c):
        x = b // c
        for i in range(a.shape[0]):
            a[i] = x[i]

    b = np.array(b, dtype=dtype)
    c = np.array(c, dtype=dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        expect = b // c
    da = cuda.to_device(np.zeros_like(expect))
    kernel[1, 1](da, cuda.to_device(b), cuda.to_device(c))
    np.testing.assert_array_equal(da.copy_to_host(), expect)


# @pytest.mark.onerous
@pytest.mark.parametrize("dt1", DTYPES)
@pytest.mark.parametrize("dt2", DTYPES)
def test_broadcasted_div(dt1, dt2):
    @cuda.jit
    def kernel(a, b, c):
        x = b / c
        for i in range(a.shape[0]):
            a[i] = x[i]

    b = np.ones([5], dtype=dt1)
    c = np.array(range(1, 6), dtype=dt2)
    expect = b / c
    a = np.zeros_like(expect)

    da = cuda.to_device(a)
    db = cuda.to_device(b)
    dc = cuda.to_device(c)
    kernel[1, 1](da, db, dc)
    a = da.copy_to_host()
    print(b, c)
    print(a, expect)
    assert np.allclose(expect, a), f"{expect=} != {a=}"


# @pytest.mark.onerous
@pytest.mark.parametrize(
    "op",
    [
        operator.sub,
        operator.add,
        operator.mul,
    ],
)
@pytest.mark.parametrize("dt1", DTYPES)
@pytest.mark.parametrize("dt2", DTYPES)
def test_broadcasted_binary_ops(op, dt1, dt2):
    @cuda.jit
    def kernel(a, b, c):
        x = op(b, c)
        for i in range(a.shape[0]):
            a[i] = x[i]

    b = np.ones([5], dtype=dt1)
    c = np.array(range(1, 6), dtype=dt2)
    expect = op(b, c)
    a = np.zeros_like(expect)

    da = cuda.to_device(a)
    db = cuda.to_device(b)
    dc = cuda.to_device(c)
    kernel[1, 1](da, db, dc)
    a = da.copy_to_host()
    print(op, b, c)
    print(a, expect)
    assert np.allclose(expect, a), f"{expect=} != {a=}"


@pytest.mark.parametrize(
    "op,expected",
    [
        (operator.eq, True),
        (operator.ne, False),
        (operator.lt, False),
        (operator.le, True),
        (operator.gt, False),
        (operator.ge, True),
    ],
)
def test_int32_negative_compared_to_negative_int_literal(op, expected):
    @cuda.jit
    def kernel(neg_a, lit_a, out):
        # neg_a[0] is int32(-1); -1 below is a Python int literal that
        # promotes to int64. The coerce helper widens neg_a to int64 to
        # match; this must use signed extension.
        out[0] = op(neg_a[0], -1)

    neg = np.array([-1], dtype=np.int32)
    lit = np.array([-1], dtype=np.int64)
    out = np.zeros(1, dtype=np.bool_)
    dn = cuda.to_device(neg)
    dl = cuda.to_device(lit)
    do = cuda.to_device(out)
    kernel[1, 1](dn, dl, do)
    out = do.copy_to_host()
    assert bool(out[0]) is expected, (
        f"int32(-1) {op.__name__} -1 -> {bool(out[0])} (expected {expected})"
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    test_broadcasted_div(np.int8, np.float16)
    # test_broadcasted_binary_ops(operator.sub, np.int8, np.float16)
    # test_broadcasted_binary_ops(operator.floordiv, np.int8, np.float16)
