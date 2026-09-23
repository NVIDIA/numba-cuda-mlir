# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir.errors import TypingError
import inspect
import numpy as np
import pytest

import logging

logging.basicConfig(level=logging.DEBUG)


def test_device_functions():
    @cuda.jit(device=False)
    def bar():
        return cuda.grid(1)

    @cuda.jit(dump=True, print_after_all=False, dump_cubin=False)
    def foo(a: cuda.DeviceNDArray):
        a[0] = bar()

    a = cuda.to_device(np.array([1], dtype=np.int32))
    foo[1, 1, 0, 0](a)
    a = a.copy_to_host()
    print(a)
    assert a[0] == 0


def test_nonvoid_kernel_rejected_before_launch():
    """A kernel with a non-void return type is rejected at compile time."""

    @cuda.jit
    def returns_value(a, out):
        out[0] = a[0]
        return 1

    source, first_line = inspect.getsourcelines(returns_value.py_func)
    return_line = first_line + next(i for i, line in enumerate(source) if "return 1" in line)

    with pytest.raises(TypingError) as excinfo:
        returns_value[1, 1](np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64))
    msg = str(excinfo.value)
    assert "must have void return type" in msg
    assert 'test_device_functions.py", line %d:' % return_line in msg


def test_self_recursion():
    """Self-recursive device function"""

    @cuda.jit(device=True)
    def fib(n):
        if n <= 1:
            return n
        return fib(n - 1) + fib(n - 2)

    @cuda.jit
    def kernel(out):
        out[0] = fib(10)

    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](out)
    assert out[0] == 55


def test_none_equality_in_device_func():
    """Device function that compares a value against None."""

    @cuda.jit(device=True)
    def maybe_default(x):
        z = None
        if x == z:
            return -1
        return x

    @cuda.jit
    def kernel(out):
        out[0] = maybe_default(10)

    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](out)
    assert out[0] == 10


def test_factory_device_functions_keep_captures_separate():
    def make_leaf(offset):
        @cuda.jit(device=True, inline="never")
        def leaf(value):
            return value + offset

        return leaf

    first = make_leaf(3)
    second = make_leaf(5)

    @cuda.jit
    def kernel(out):
        out[0] = second(first(0))

    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](out)
    assert out[0] == 8


if __name__ == "__main__":
    test_device_functions()
    test_self_recursion()
    test_none_equality_in_device_func()
