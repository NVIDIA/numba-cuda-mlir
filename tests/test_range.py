# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)


def test_range1():
    @cuda.jit
    def k(x: cuda.DeviceNDArray):
        for i in range(5):
            x[0] += i

    x = cuda.to_device(np.zeros(1, dtype=np.int32))
    k[1, 1](x)
    x = x.copy_to_host()
    assert x[0] == sum(range(5))


def test_range2():
    @cuda.jit
    def k(x: cuda.DeviceNDArray):
        for i in range(3, 7):
            x[0] += i

    x = cuda.to_device(np.zeros(1, dtype=np.int32))
    k[1, 1](x)
    x = x.copy_to_host()
    assert x[0] == sum(range(3, 7))


def test_range3():
    @cuda.jit
    def k(x: cuda.DeviceNDArray):
        for i in range(3, 10, 2):
            x[0] += i

    x = cuda.to_device(np.zeros(1, dtype=np.int32))
    k[1, 1](x)
    x = x.copy_to_host()
    assert x[0] == sum(range(3, 10, 2))


def test_range_neg_stride():
    @cuda.jit
    def k(x: cuda.DeviceNDArray):
        for i in range(10, 3, -2):
            x[0] += i

    x = cuda.to_device(np.zeros(1, dtype=np.int32))
    k[1, 1](x)
    x = x.copy_to_host()
    assert x[0] == sum(range(10, 3, -2))


def test_range_float_bounds():
    @cuda.jit
    def k(bounds, x):
        for i in range(bounds[0], bounds[1]):
            x[0] += i

    x = cuda.to_device(np.zeros(1, dtype=np.int64))
    k[1, 1](cuda.to_device(np.array([2.0, 6.0])), x)
    x = x.copy_to_host()
    assert x[0] == sum(range(2, 6))


def test_range_uint32_bounds_above_int32_max():
    @cuda.jit
    def k(bounds, x):
        for i in range(np.int64(bounds[0]), bounds[1]):
            x[0] += 1
        for i in range(bounds[0], np.int64(bounds[1])):
            x[1] = i
            break

    n = 3_000_000_000
    x = cuda.to_device(np.zeros(2, dtype=np.int64))
    k[1, 1](cuda.to_device(np.array([n - 3, n], dtype=np.uint32)), x)
    x = x.copy_to_host()
    assert list(x) == [3, n - 3]


if __name__ == "__main__":
    test_range1()
    test_range2()
    test_range3()
    test_range_neg_stride()
    test_range_float_bounds()
    test_range_uint32_bounds_above_int32_max()
