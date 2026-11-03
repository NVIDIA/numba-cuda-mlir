# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from numba_cuda_mlir import cuda


@cuda.jit
def integer_minmax(a, b, minimum, maximum):
    i = cuda.grid(1)
    if i < a.size:
        minimum[i] = min(a[i], b[i])
        maximum[i] = max(a[i], b[i])


INTEGER_TYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]


@pytest.mark.parametrize("dtype", INTEGER_TYPES)
def test_builtin_integer_minmax(dtype):
    info = np.iinfo(dtype)
    high_bit = 1 << (info.bits - 1) if info.min == 0 else info.min
    a = np.array([info.max, 1, high_bit, 0, info.min, info.max], dtype=dtype)
    b = np.array([1, info.max, 2, 0, info.max, info.min], dtype=dtype)
    minimum, maximum = np.empty_like(a), np.empty_like(a)
    integer_minmax[1, a.size](a, b, minimum, maximum)
    np.testing.assert_array_equal(minimum, np.minimum(a, b))
    np.testing.assert_array_equal(maximum, np.maximum(a, b))


@pytest.mark.parametrize(
    "dtype_left,dtype_right",
    [
        (np.uint8, np.uint16),
        (np.uint16, np.uint32),
        (np.uint32, np.uint64),
        (np.int8, np.int16),
        (np.int16, np.int32),
        (np.int32, np.int64),
        (np.uint8, np.int8),
        (np.uint32, np.int64),
        (np.uint64, np.int64),
        (np.uint32, np.float64),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_builtin_minmax_promotes_operands(dtype_left, dtype_right, reverse):
    left_info = np.iinfo(dtype_left)
    a = np.array([left_info.max, left_info.min, 1], dtype=dtype_left)
    right_min = np.iinfo(dtype_right).min if np.issubdtype(dtype_right, np.integer) else -1.5
    b = np.array([1, right_min, 2], dtype=dtype_right)
    if reverse:
        a, b = b, a
    result_dtype = np.result_type(a.dtype, b.dtype)
    minimum = np.empty(a.size, dtype=result_dtype)
    maximum = np.empty_like(minimum)
    integer_minmax[1, a.size](a, b, minimum, maximum)
    np.testing.assert_array_equal(minimum, np.minimum(a, b))
    np.testing.assert_array_equal(maximum, np.maximum(a, b))
