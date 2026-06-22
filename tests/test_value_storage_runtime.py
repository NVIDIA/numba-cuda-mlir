# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from numba_cuda_mlir import cuda, extending


def test_float16_array_constructors_use_storage_values():
    @cuda.jit
    def kernel(out):
        zeros = np.zeros(4, dtype=np.float16)
        ones = np.ones(4, dtype=np.float16)
        full = np.full(4, 2.5, dtype=np.float16)
        out[0] = zeros[0]
        out[1] = ones[1]
        out[2] = full[2]

    out = np.zeros(3, dtype=np.float16)
    kernel[1, 1](out)
    np.testing.assert_allclose(out, np.array([0.0, 1.0, 2.5], dtype=np.float16))


def test_bool_array_constructors_use_storage_values():
    @cuda.jit
    def kernel(out):
        zeros = np.zeros(4, dtype=np.bool_)
        ones = np.ones(4, dtype=np.bool_)
        out[0] = 1 if zeros[0] else 0
        out[1] = 1 if ones[1] else 0

    out = np.zeros(2, dtype=np.int32)
    kernel[1, 1](out)
    np.testing.assert_array_equal(out, np.array([0, 1], dtype=np.int32))


def test_float16_reductions_read_value_elements():
    @cuda.jit
    def kernel(sum_out, flags_out, values):
        sum_out[0] = np.sum(values)
        flags_out[0] = 1 if np.any(values) else 0
        flags_out[1] = 1 if np.all(values) else 0

    values = np.array([1.0, 2.0, 0.0, 3.0], dtype=np.float16)
    sum_out = np.zeros(1, dtype=np.float16)
    flags_out = np.zeros(2, dtype=np.int32)
    kernel[1, 1](sum_out, flags_out, values)
    np.testing.assert_allclose(sum_out, np.array([6.0], dtype=np.float16))
    np.testing.assert_array_equal(flags_out, np.array([1, 0], dtype=np.int32))


def test_bool_reductions_read_value_elements():
    @cuda.jit
    def kernel(flags_out, values):
        flags_out[0] = 1 if np.any(values) else 0
        flags_out[1] = 1 if np.all(values) else 0

    values = np.array([True, False, True], dtype=np.bool_)
    flags_out = np.zeros(2, dtype=np.int32)
    kernel[1, 1](flags_out, values)
    np.testing.assert_array_equal(flags_out, np.array([1, 0], dtype=np.int32))


def test_heterogeneous_tuple_reassignment_uses_storage_slots():
    @cuda.jit
    def kernel(out, values, flags):
        pair = (values[0], flags[0] > 0)
        if flags[1] > 0:
            pair = (values[1], flags[2] > 0)
        out[0] = pair[0]
        out[1] = 1 if pair[1] else 0

    values = np.array([1.5, 2.5], dtype=np.float16)
    flags = np.array([1, 1, 0], dtype=np.int32)
    out = np.zeros(2, dtype=np.float16)
    kernel[1, 1](out, values, flags)
    np.testing.assert_allclose(out, np.array([2.5, 0.0], dtype=np.float16))


def test_overload_call_converts_storage_return_to_value():
    def is_positive(x):
        raise NotImplementedError

    @extending.overload(is_positive, typing_registry=extending.typing_registry)
    def ol_is_positive(x):
        def impl(x):
            return x > np.float16(0)

        return impl

    @cuda.jit
    def kernel(out, values):
        out[0] = 1 if is_positive(values[0]) else 0
        out[1] = 1 if is_positive(values[1]) else 0

    values = np.array([1.0, -1.0], dtype=np.float16)
    out = np.zeros(2, dtype=np.int32)
    kernel[1, 1](out, values)
    np.testing.assert_array_equal(out, np.array([1, 0], dtype=np.int32))
