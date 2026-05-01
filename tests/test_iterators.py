# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba_cuda_mlir import cuda
import numpy as np


def test_array_iteration():
    @cuda.jit
    def sum_array(x, result):
        total = 0
        for v in x:
            total += v
        result[0] = total

    x = cuda.to_device(np.asarray([1, 2, 3, 4, 5], dtype=np.int64))
    result = cuda.to_device(np.zeros(1, dtype=np.int64))

    sum_array[1, 1](x, result)
    assert result.copy_to_host()[0] == 15


def test_array_iteration_float():
    @cuda.jit
    def sum_array(x, result):
        total = 0.0
        for v in x:
            total += v
        result[0] = total

    x = cuda.to_device(np.asarray([1.5, 2.5, 3.0, 4.0, 5.0], dtype=np.float64))
    result = cuda.to_device(np.zeros(1, dtype=np.float64))

    sum_array[1, 1](x, result)
    assert result.copy_to_host()[0] == 16.0
