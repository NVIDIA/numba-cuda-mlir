# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
import numpy as np

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


if __name__ == "__main__":
    test_device_functions()
    test_self_recursion()
    test_none_equality_in_device_func()
