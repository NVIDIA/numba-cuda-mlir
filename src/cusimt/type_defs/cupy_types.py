# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt.numba_cuda.typing import typeof
from cusimt.numba_cuda import types
import numpy as np
import cupy


@typeof.typeof_impl.register(cupy.ndarray)
def typeof_cupy_ndarray(val, c):
    np_dtype = np.dtype(val.dtype)

    dtype_map = {
        np.dtype("bool"): types.boolean,
        np.dtype("int8"): types.int8,
        np.dtype("int16"): types.int16,
        np.dtype("int32"): types.int32,
        np.dtype("int64"): types.int64,
        np.dtype("uint8"): types.uint8,
        np.dtype("uint16"): types.uint16,
        np.dtype("uint32"): types.uint32,
        np.dtype("uint64"): types.uint64,
        np.dtype("float16"): types.float16,
        np.dtype("float32"): types.float32,
        np.dtype("float64"): types.float64,
        np.dtype("complex64"): types.complex64,
        np.dtype("complex128"): types.complex128,
    }

    if np_dtype not in dtype_map:
        raise TypeError(f"Unsupported cupy dtype: {val.dtype}")

    numba_dtype = dtype_map[np_dtype]
    ty = types.Array(dtype=numba_dtype, ndim=len(val.shape), layout="C")
    return ty
