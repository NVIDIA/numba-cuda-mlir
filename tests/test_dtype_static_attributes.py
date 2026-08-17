# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from numba_cuda_mlir import cuda


def test_captured_dtype_scalar_metadata():
    int_dtype = np.dtype(np.int32)
    float_dtype = np.dtype(np.float64)

    @cuda.jit
    def kernel(out):
        out[0] = int_dtype.itemsize
        out[1] = int_dtype.num
        out[2] = int_dtype.alignment
        out[3] = int_dtype.isbuiltin
        out[4] = int_dtype.hasobject
        out[5] = int_dtype.isalignedstruct
        out[6] = int_dtype.isnative
        out[7] = int_dtype.char == "i"
        out[8] = int_dtype.name == "int32"
        out[9] = int_dtype.str == "<i4"
        out[10] = int_dtype.byteorder == "="
        out[11] = float_dtype.kind == "f"
        out[12] = int_dtype.base.itemsize
        out[13] = int_dtype.names is None
        out[14] = len(int_dtype.shape)

    out = np.zeros(15, dtype=np.int64)
    kernel[1, 1](out)
    np.testing.assert_array_equal(out, [4, 5, 4, 1, 0, 0, 1, 1, 1, 1, 1, 1, 4, 1, 0])


def test_captured_dtype_structured_metadata():
    record_dtype = np.dtype([("x", np.int32), ("y", np.float64)])
    subarray_dtype = np.dtype((np.int32, (2, 3)))

    @cuda.jit
    def kernel(out):
        out[0] = record_dtype.names[0] == "x"
        out[1] = record_dtype.names[1] == "y"
        out[2] = subarray_dtype.shape[0]
        out[3] = subarray_dtype.shape[1]
        out[4] = subarray_dtype.base.itemsize

    out = np.zeros(5, dtype=np.int64)
    kernel[1, 1](out)
    np.testing.assert_array_equal(out, [1, 1, 2, 3, 4])
