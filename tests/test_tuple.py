# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import namedtuple

from numba_cuda_mlir import cuda
import numpy as np
import pytest


def test_compile_time_tuple_slice():
    @cuda.jit
    def kernel(out):
        rev = (2, 4)[::-1]
        out[0] = rev[0]

    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](out)
    np.testing.assert_array_equal(out, (4,))


def test_tuple_builtin_hetero_tuple():
    @cuda.jit
    def k(r_int, r_float, x):
        t = tuple(x)
        r_int[0] = t[0]
        r_int[1] = t[1]
        r_float[0] = t[2]

    x = (1, 2, 3.5)
    r_int = np.zeros(2, dtype=np.int64)
    r_float = np.zeros(1, dtype=np.float64)
    k[1, 1](r_int, r_float, x)
    assert r_int[0] == x[0]
    assert r_int[1] == x[1]
    assert r_float[0] == x[2]


@pytest.mark.parametrize(
    "x, y",
    [
        ((1, 2), (3, 4, 5)),
        ((1, 2.5), (3, 4.5)),
        ((1, 2, 3), (4, 5.5)),
        ((1, 2.5), (3, 4, 5)),
    ],
)
def test_tuple_concat(x, y):
    from numba_cuda_mlir.cuda.experimental import consteval

    expected = x + y
    n = len(expected)
    r = np.zeros(n, dtype=np.float64)

    @cuda.jit
    def k(r, x, y):
        t = x + y
        for i in consteval(range(n)):
            r[i] = t[i]

    k[1, 1](r, x, y)
    np.testing.assert_array_equal(r, expected)


def test_unituple_arg():
    @cuda.jit
    def f(r, x):
        r[0] = x[0]
        r[1] = x[1]
        r[2] = x[2]

    x = (1, 2, 3)
    r = np.zeros(len(x), dtype=np.int64)
    f[1, 1](r, x)
    np.testing.assert_array_equal(r, x)


def test_tuple_cast():
    @cuda.jit
    def k(r, x):
        t = tuple(x)
        r[0] = t[0]
        r[1] = t[1]
        r[2] = t[2]

    x = (10, 20, 30)
    r = np.zeros(3, dtype=np.int64)
    k[1, 1](r, x)
    np.testing.assert_array_equal(r, x)


def test_hetero_tuple_arg():
    @cuda.jit
    def f(r1, r2, x):
        r1[0] = x[0]
        r1[1] = x[1]
        r1[2] = x[2]
        r2[0] = x[3]
        r2[1] = x[4]
        r2[2] = x[5]

    x = (1, 2, 3, 4.5, 5.5, 6.5)
    r1 = np.zeros(3, dtype=np.int64)
    r2 = np.zeros(3, dtype=np.float64)
    f[1, 1](r1, r2, x)
    np.testing.assert_array_equal(r1, x[:3])
    np.testing.assert_array_equal(r2, x[3:])


def test_captured_namedtuple_getattr():
    Prec = namedtuple("Prec", ["a", "b"])
    prec = Prec(np.float32, np.float64)
    r = np.zeros(1, dtype=np.float32)

    @cuda.jit
    def f(r):
        tmp = cuda.local.array(1, prec.a)
        tmp[0] = 1
        r[0] = tmp[0]

    f[1, 1](r)
    assert r[0] == np.float32(1)


def test_captured_namedtuple_unpack():
    LD = namedtuple("LD", ["a", "b", "c"])
    dim = LD(2, 4, 8)
    out = np.zeros(1, dtype=np.int64)

    @cuda.jit
    def kernel(out):
        a, b, c = dim
        out[0] = a + b + c

    kernel[1, 1](out)
    np.testing.assert_array_equal(out, 14)


@pytest.mark.parametrize(
    "Point, vals, dtypes",
    [
        (namedtuple("Point", ("x", "y")), (1, 2), [np.int64]),
        (namedtuple("Point", ("x", "y", "r")), (1, 2, 2.236), [np.int64, np.float64]),
    ],
    ids=["uniform", "mixed"],
)
def test_namedtuple_arg(Point, vals, dtypes):
    p = Point(*vals)
    if len(dtypes) == 1:
        r = np.zeros(len(p), dtype=dtypes[0])

        @cuda.jit
        def f(r, x):
            r[0] = x.x
            r[1] = x.y

        f[1, 1](r, p)
        np.testing.assert_array_equal(r, vals)
    else:
        r1 = np.zeros(2, dtype=dtypes[0])
        r2 = np.zeros(1, dtype=dtypes[1])

        @cuda.jit
        def f(r1, r2, x):
            r1[0] = x.x
            r1[1] = x.y
            r2[0] = x.r

        f[1, 1](r1, r2, p)
        assert r1[0] == p.x
        assert r1[1] == p.y
        assert r2[0] == p.r


def test_empty_tuple():
    @cuda.jit
    def f(r, x):
        r[0] = len(x)

    x = tuple()
    r = np.ones(1, dtype=np.int64)
    f[1, 1](r, x)
    assert r[0] == 0


def test_tuple_of_empty_tuples():
    @cuda.jit
    def f(r, x):
        r[0] = len(x)
        r[1] = len(x[0])

    x = ((), (), ())
    r = np.ones(2, dtype=np.int64)
    f[1, 1](r, x)
    assert r[0] == 3
    assert r[1] == 0


def test_tuple_of_tuples():
    @cuda.jit
    def f(r, x):
        r[0] = len(x)
        r[1] = len(x[0])
        r[2] = len(x[1])
        r[3] = len(x[2])
        r[4] = x[1][0]
        r[5] = x[1][1]
        r[6] = x[2][0]
        r[7] = x[2][1]
        r[8] = x[2][2]

    x = ((), (5, 6), (8, 9, 10))
    r = np.ones(9, dtype=np.int64)
    f[1, 1](r, x)
    expected = [3, 0, 2, 3, 5, 6, 8, 9, 10]
    np.testing.assert_array_equal(r, expected)


def test_tuple_of_tuples_and_scalars():
    @cuda.jit
    def f(r, x):
        r[0] = len(x)
        r[1] = len(x[0])
        r[2] = x[0][0]
        r[3] = x[0][1]
        r[4] = x[0][2]
        r[5] = x[1]

    x = ((6, 5, 4), 7)
    r = np.ones(9, dtype=np.int64)
    f[1, 1](r, x)
    assert r[0] == 2
    assert r[1] == 3
    assert r[2] == 6
    assert r[3] == 5
    assert r[4] == 4
    assert r[5] == 7


def test_tuple_of_arrays():
    @cuda.jit
    def f(x):
        i = cuda.grid(1)
        if i < len(x[0]):
            x[0][i] = x[1][i] + x[2][i]

    N = 10
    x0 = np.zeros(N)
    x1 = np.ones_like(x0)
    x2 = x1 * 3
    f[1, N]((x0, x1, x2))
    np.testing.assert_equal(x0, x1 + x2)


def test_tuple_of_array_scalar_tuple():
    @cuda.jit
    def f(r, x):
        r[0] = x[0][0]
        r[1] = x[0][1]
        r[2] = x[1]
        r[3] = x[2][0]
        r[4] = x[2][1]

    z = np.arange(2, dtype=np.int64)
    x = (2 * z, 10, (4, 3))
    r = np.zeros(5, dtype=np.int64)
    f[1, 1](r, x)
    expected = [0, 2, 10, 4, 3]
    np.testing.assert_array_equal(r, expected)


def test_tuple_multi_assign_from_device_function():
    @cuda.jit(device=True)
    def two_tuple_returns(cond):
        if cond:
            return (1, 2, 3)
        return (4, 5, 6)

    @cuda.jit
    def kernel(out, n):
        t = two_tuple_returns(n > 0)
        out[0] = t[0]

    out = cuda.device_array((1,), dtype=np.int64)
    kernel[1, 1](out, 1)
    assert out.copy_to_host()[0] == 1

    kernel[1, 1](out, -1)
    assert out.copy_to_host()[0] == 4


def test_hetero_tuple_multi_assign_from_inlined_device_function():
    @cuda.jit(device=True, forceinline=True)
    def strides(flag, ld, m, k):
        shape = (m, k)
        if flag:
            return (ld * shape[1], 1, ld)
        return (ld * shape[0], ld, 1)

    @cuda.jit
    def kernel(out, flag):
        r = strides(flag, 2, 7, 3)
        out[0] = r[0]
        out[1] = r[1]
        out[2] = r[2]

    out = cuda.device_array((3,), dtype=np.int64)

    kernel[1, 1](out, True)
    np.testing.assert_array_equal(out.copy_to_host(), [6, 1, 2])

    kernel[1, 1](out, False)
    np.testing.assert_array_equal(out.copy_to_host(), [14, 2, 1])


def test_dynamic_index_tuple_of_tuples():
    """Runtime indexing into a tuple of tuples (e.g. coefficient rows)."""
    rows = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))

    @cuda.jit
    def kernel(out):
        for i in range(len(rows)):
            row = rows[i]
            out[i] = row[0] + row[1]

    out = np.zeros(3, dtype=np.float64)
    kernel[1, 1](out)
    np.testing.assert_allclose(out, [3.0, 7.0, 11.0])


def test_dynamic_index_tuple_of_hetero_tuples():
    rows = ((1, 2.5), (3, 4.5))

    @cuda.jit
    def kernel(out):
        for i in range(len(rows)):
            row = rows[i]
            out[i] = row[0] + row[1]

    out = np.zeros(2, dtype=np.float64)
    kernel[1, 1](out)
    np.testing.assert_allclose(out, [3.5, 7.5])


def test_dynamic_index_deeply_nested_tuple():
    nested = (((1.0, 2.0), (3.0, 4.0)), ((5.0, 6.0), (7.0, 8.0)))

    @cuda.jit
    def kernel(out):
        for i in range(len(nested)):
            block = nested[i]
            out[i] = block[0][0] + block[1][1]

    out = np.zeros(2, dtype=np.float64)
    kernel[1, 1](out)
    np.testing.assert_allclose(out, [5.0, 13.0])


def test_dynamic_index_scalar_tuple_still_works():
    values = (10.0, 20.0, 30.0)

    @cuda.jit
    def kernel(out):
        for i in range(len(values)):
            out[i] = values[i]

    out = np.zeros(3, dtype=np.float64)
    kernel[1, 1](out)
    np.testing.assert_allclose(out, [10.0, 20.0, 30.0])


def test_dynamic_index_tuple_of_empty_tuples_bounds_check():
    """Out-of-range indexing raises even when the selected elements
    are empty tuples, which emit no leaf switches."""
    empties = ((), ())

    @cuda.jit(debug=True, opt=False)
    def kernel(idx_arr, out):
        row = empties[idx_arr[0]]
        out[0] = len(row)

    out = cuda.to_device(np.zeros(1, dtype=np.int32))
    kernel[1, 1](cuda.to_device(np.array([1], dtype=np.int64)), out)
    assert out.copy_to_host()[0] == 0
    with pytest.raises(IndexError):
        kernel[1, 1](cuda.to_device(np.array([5], dtype=np.int64)), out)


if __name__ == "__main__":
    test_tuple_concat((1, 2), (3, 4, 5))
