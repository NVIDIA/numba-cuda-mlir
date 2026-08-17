# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from numba_cuda_mlir import cuda


def test_captured_dtype_scalar_metadata():
    int_dtype = np.dtype(np.int32)
    float_dtype = np.dtype(np.float64)
    int_char = int_dtype.char
    int_name = int_dtype.name
    int_str = int_dtype.str
    int_byteorder = int_dtype.byteorder
    float_kind = float_dtype.kind

    @cuda.jit
    def kernel(out):
        out[0] = int_dtype.itemsize
        out[1] = int_dtype.num
        out[2] = int_dtype.alignment
        out[3] = int_dtype.isbuiltin
        out[4] = int_dtype.hasobject
        out[5] = int_dtype.isalignedstruct
        out[6] = int_dtype.isnative
        out[7] = int_dtype.char == int_char
        out[8] = int_dtype.name == int_name
        out[9] = int_dtype.str == int_str
        out[10] = int_dtype.byteorder == int_byteorder
        out[11] = float_dtype.kind == float_kind
        out[12] = int_dtype.base.itemsize
        out[13] = int_dtype.names is None
        out[14] = len(int_dtype.shape)

    out = np.zeros(15, dtype=np.int64)
    kernel[1, 1](out)
    np.testing.assert_array_equal(
        out,
        [
            int_dtype.itemsize,
            int_dtype.num,
            int_dtype.alignment,
            int_dtype.isbuiltin,
            int_dtype.hasobject,
            int_dtype.isalignedstruct,
            int_dtype.isnative,
            1,
            1,
            1,
            1,
            1,
            int_dtype.base.itemsize,
            1,
            0,
        ],
    )


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
