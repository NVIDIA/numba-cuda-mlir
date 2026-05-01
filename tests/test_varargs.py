# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
import numpy as np
import pytest


def test_varargs_len():
    @cuda.jit
    def k(r, *args):
        r[0] = len(args)

    r = np.zeros(1, dtype=np.int64)
    k[1, 1](r, np.zeros(5, dtype=np.float32), np.zeros(5, dtype=np.float32))
    assert r[0] == 2

    r[:] = 0
    k[1, 1](
        r,
        np.zeros(5, dtype=np.float32),
        np.zeros(5, dtype=np.float32),
        np.zeros(5, dtype=np.float32),
    )
    assert r[0] == 3


def test_varargs_indexing():
    @cuda.jit
    def k(r, *args):
        r[0] = args[0][0]
        r[1] = args[1][0]
        r[2] = args[2][0]

    a = np.array([10.0], dtype=np.float32)
    b = np.array([20.0], dtype=np.float32)
    c = np.array([30.0], dtype=np.float32)
    r = np.zeros(3, dtype=np.float32)
    k[1, 1](r, a, b, c)
    np.testing.assert_array_equal(r, [10.0, 20.0, 30.0])


def test_varargs_only():
    @cuda.jit
    def k(*args):
        args[0][0] = len(args)

    r = np.zeros(1, dtype=np.float32)
    k[1, 1](r, np.zeros(1, dtype=np.float32))
    assert r[0] == 2


@pytest.mark.parametrize("n_extra", [1, 2, 4])
def test_varargs_len_parametrized(n_extra):
    @cuda.jit
    def k(r, *args):
        r[0] = len(args)

    extras = [np.zeros(1, dtype=np.float32) for _ in range(n_extra)]
    r = np.zeros(1, dtype=np.int64)
    k[1, 1](r, *extras)
    assert r[0] == n_extra


def test_varargs_device_function():
    @cuda.jit(device=True)
    def udf(a, b):
        return a + b

    @cuda.jit
    def kernel(answers, size, *input_cols):
        i = cuda.grid(1)
        if i < size:
            answers[i] = udf(input_cols[0][i], input_cols[1][i])

    n = 8
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 10
    ans = np.zeros(n, dtype=np.float32)
    kernel[1, n](ans, n, a, b)
    np.testing.assert_allclose(ans, a + b)


def test_varargs_forall():
    @cuda.jit(device=True)
    def udf(a, b):
        return a + b

    @cuda.jit
    def kernel(answers, size, *input_cols):
        i = cuda.grid(1)
        if i < size:
            answers[i] = udf(input_cols[0][i], input_cols[1][i])

    n = 8
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 10
    ans = np.zeros(n, dtype=np.float32)
    kernel.forall(n)(ans, n, a, b)
    np.testing.assert_allclose(ans, a + b)


@pytest.mark.xfail(
    reason="Star-unpacking in device function calls is not yet supported",
    raises=Exception,
)
def test_varargs_star_unpack_into_device_call():
    @cuda.jit(device=True)
    def udf(a, b):
        return a + b

    @cuda.jit
    def kernel(answers, a, b):
        i = cuda.grid(1)
        if i < len(answers):
            tup = (a[i], b[i])
            answers[i] = udf(*tup)

    n = 4
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 10
    ans = np.zeros(n, dtype=np.float32)
    kernel[1, n](ans, a, b)
    np.testing.assert_allclose(ans, a + b)


@pytest.mark.xfail(
    reason="Generator expressions (yield) are not supported in kernels",
    raises=Exception,
)
def test_varargs_generator_expression():
    @cuda.jit(device=True)
    def udf(a, b):
        return a + b

    @cuda.jit
    def kernel(answers, size, *input_cols):
        i = cuda.grid(1)
        if i < size:
            fargs = tuple(col[i] for col in input_cols)
            answers[i] = udf(*fargs)

    n = 4
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 10
    ans = np.zeros(n, dtype=np.float32)
    kernel[1, n](ans, n, a, b)
    np.testing.assert_allclose(ans, a + b)
