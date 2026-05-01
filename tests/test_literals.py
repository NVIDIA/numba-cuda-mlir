# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
import numpy as np
import pytest

literal_unroll = cuda.misc.special.literal_unroll
literally = cuda.misc.special.literally


def test_literal_unroll():
    @cuda.jit
    def k(x: cuda.DeviceNDArray):
        for i in literal_unroll(range(10)):
            x[i] = i

    x = cuda.to_device(np.zeros(10, dtype=np.int32))
    k[1, 1](x)
    x = x.copy_to_host()
    assert np.all(x == np.arange(10))


@pytest.mark.xfail
def test_literal_eval():
    arr = np.float32([1, 2, 3])

    @cuda.jit
    def k(x: cuda.DeviceNDArray):
        arr = literally(arr)
        for i in range(3):
            x[i] = arr[i]

    x = cuda.to_device(np.zeros(3, dtype=np.float32))
    k[1, 1](x)
    x = x.copy_to_host()
    assert np.all(x == np.float32([1, 2, 3]))
