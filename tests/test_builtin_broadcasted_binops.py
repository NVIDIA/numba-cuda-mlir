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
