# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda.extending import typeof_impl

from numba_cuda_mlir import types as mlir_types


class MyCustomDType:
    pass


my_dtype = MyCustomDType()


@typeof_impl.register(MyCustomDType)
def typeof_my_dtype(val, c):
    return mlir_types.NumberClass(mlir_types.float32)


def test_custom_dtype_local_array():
    """
    Test that a custom Python object mimicking a type specification
    can be properly lowered when passed as `dtype` to `cuda.local.array`.
    """

    @cuda.jit
    def kernel(out):
        arr = cuda.local.array((10,), dtype=my_dtype)
        arr[0] = 42.0
        out[0] = arr[0]

    out = np.zeros(1, dtype=np.float32)
    kernel[1, 1](out)
    assert out[0] == 42.0


def test_custom_dtype_shared_array():
    """
    Test that a custom Python object mimicking a type specification
    can be properly lowered when passed as `dtype` to `cuda.shared.array`.
    """

    @cuda.jit
    def kernel(out):
        arr = cuda.shared.array((10,), dtype=my_dtype)
        if cuda.threadIdx.x == 0:
            arr[0] = 42.0
        cuda.syncthreads()
        out[0] = arr[0]

    out = np.zeros(1, dtype=np.float32)
    kernel[1, 1](out)
    assert out[0] == 42.0
