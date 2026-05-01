# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.numba_cuda.typing import typeof
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.np import numpy_support
import numpy as np
import cupy


@typeof.typeof_impl.register(cupy.ndarray)
def typeof_cupy_ndarray(val, c):
    numba_dtype = numpy_support.from_dtype(np.dtype(val.dtype))
    return types.Array(dtype=numba_dtype, ndim=len(val.shape), layout="C")
