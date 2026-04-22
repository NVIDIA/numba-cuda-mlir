# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cuda.simt as cuda
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


if __name__ == "__main__":
    test_range1()
    test_range2()
    test_range3()
    test_range_neg_stride()
