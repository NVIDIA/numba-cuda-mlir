# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from numba_cuda_mlir import cuda


def test_captured_dtype_type_attribute():
    idx_dtype = np.dtype(np.int64)

    @cuda.jit
    def kernel(dst):
        i = cuda.grid(1)
        if i < dst.size:
            dst[i] = idx_dtype.type(0) if i == 0 else i

    dst = np.zeros(4, dtype=np.int64)
    kernel[1, 32](dst)
    np.testing.assert_array_equal(dst, [0, 1, 2, 3])


def test_captured_dtype_kind_attribute():
    int_dtype = np.dtype(np.int64)
    float_dtype = np.dtype(np.float64)

    @cuda.jit
    def kernel(dst):
        dst[0] = int_dtype.kind == "i"
        dst[1] = float_dtype.kind == "f"

    dst = np.zeros(2, dtype=np.bool_)
    kernel[1, 1](dst)
    np.testing.assert_array_equal(dst, [True, True])
