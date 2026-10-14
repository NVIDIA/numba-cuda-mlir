# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
import numpy as np
import logging
import pytest

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


@pytest.mark.parametrize(
    "args,expected",
    [
        ((-(2**63), 2**63 - 1, 2**63 - 1), 3),
        ((2**63 - 1, -(2**63), -(2**63)), 2),
        ((np.int8(-128), np.int8(127), np.int8(1)), 255),
        ((np.uint64(0), np.uint64(2**64 - 1), np.uint64(2**63)), 2),
    ],
)
def test_range_trip_count_boundaries(args, expected):
    @cuda.jit
    def count(out, start, stop, step):
        total = 0
        for _ in range(start, stop, step):
            total += 1
        out[0] = total

    out = cuda.to_device(np.zeros(1, dtype=np.int64))
    count[1, 1](out, *args)
    assert out.copy_to_host()[0] == expected


def test_range_reuse():
    @cuda.jit
    def compute(out, n):
        values = range(n)
        total = 0
        for value in values:
            total += value
        for value in values:
            total += value
        for outer in values:
            for inner in values:
                total += outer * inner
        out[0] = total

    out = cuda.to_device(np.zeros(1, dtype=np.int64))
    compute[1, 1](out, 4)
    assert out.copy_to_host()[0] == 48


if __name__ == "__main__":
    test_range1()
    test_range2()
    test_range3()
    test_range_neg_stride()
