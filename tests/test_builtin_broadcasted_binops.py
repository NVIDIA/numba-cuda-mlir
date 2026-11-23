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


# Elements that lose their value when widened with the wrong signedness: negative numbers, and
# unsigned numbers with the top bit set.
VALUES = {
    np.bool_: [True, False, True, True],
    np.int8: [-1, 2, -128, 127],
    np.uint8: [200, 2, 128, 255],
    np.int16: [-1, 2, -32768, 32767],
    np.int32: [-1, 2, -(2**31), 2**31 - 1],
    np.uint32: [3000000000, 2, 2**31, 2**32 - 1],
    np.int64: [-100, 2, -3, 4],
    np.uint64: [2**63, 2, 2**63 + 2**62, 2**64 - 2**11],
    np.float32: [0.5, -2.5, 3.0, -4.0],
    np.float64: [0.5, -2.5, 3.0, -4.0],
}


def assert_matches_numpy(actual, expect):
    if expect.dtype.kind == "f":
        np.testing.assert_allclose(actual, expect, rtol=1e-6)
    else:
        np.testing.assert_array_equal(actual, expect)


@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, np.add, np.multiply])
@pytest.mark.parametrize(
    "dt1, dt2",
    [
        (np.int32, np.int64),
        (np.int8, np.int32),
        (np.int8, np.uint8),
        (np.int32, np.uint32),
        (np.uint8, np.uint32),
        (np.uint8, np.int16),
        (np.bool_, np.int32),
        (np.uint32, np.float64),
        (np.uint64, np.float64),
        (np.uint8, np.float32),
        (np.int32, np.float64),
    ],
)
def test_mixed_array_operands_keep_their_signedness(op, dt1, dt2):
    @cuda.jit
    def kernel(a, b, c):
        x = op(b, c)
        for i in range(a.shape[0]):
            a[i] = x[i]

    b = np.array(VALUES[dt1], dtype=dt1)
    c = np.array(VALUES[dt2], dtype=dt2)
    expect = op(b, c)
    da = cuda.to_device(np.zeros_like(expect))
    kernel[1, 1](da, cuda.to_device(b), cuda.to_device(c))
    assert_matches_numpy(da.copy_to_host(), expect)


@pytest.mark.parametrize(
    "dtype, op, scalar",
    [
        (np.uint8, operator.mul, 1.5),
        (np.uint8, operator.truediv, 2.0),
        (np.uint8, operator.add, 0.5),
        (np.uint32, operator.mul, 1.5),
        (np.uint32, operator.truediv, 2.0),
        (np.uint64, operator.add, 0.5),
        (np.bool_, operator.mul, 2),
        (np.bool_, operator.add, 1),
        (np.int32, operator.mul, 1.5),
    ],
)
def test_array_and_scalar_keep_their_signedness(dtype, op, scalar):
    @cuda.jit
    def kernel(a, b, s):
        x = op(b, s)
        y = op(s, b)
        for i in range(a.shape[1]):
            a[0, i] = x[i]
            a[1, i] = y[i]

    b = np.array(VALUES[dtype], dtype=dtype)
    expect = np.array([op(b, scalar), op(scalar, b)], dtype=np.float64)
    da = cuda.to_device(np.zeros_like(expect))
    kernel[1, 1](da, cuda.to_device(b), scalar)
    assert_matches_numpy(da.copy_to_host(), expect)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    test_broadcasted_div(np.int8, np.float16)
    # test_broadcasted_binary_ops(operator.sub, np.int8, np.float16)
    # test_broadcasted_binary_ops(operator.floordiv, np.int8, np.float16)
