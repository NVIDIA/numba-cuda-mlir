# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.numba_cuda.typing import typeof
import cupy


@typeof.typeof_impl.register(cupy.ndarray)
def typeof_cupy_ndarray(val, c):
    return typeof._typeof_cuda_array_interface(val.__cuda_array_interface__, c, val.dtype)
