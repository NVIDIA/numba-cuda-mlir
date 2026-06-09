# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from numba_cuda_mlir import cuda


def test_vector_add_scalar():
    @cuda.jit
    def kernel(arr):
        v = cuda.float32x4(1.0, 2.0, 3.0, 4.0)
        v2 = v + 10.0
        arr[0] = v2.x
        arr[1] = v2.y
        arr[2] = v2.z
        arr[3] = v2.w

    arr = np.zeros(4, dtype=np.float32)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [11.0, 12.0, 13.0, 14.0])


def test_scalar_add_vector():
    @cuda.jit
    def kernel(arr):
        v = cuda.float32x4(1.0, 2.0, 3.0, 4.0)
        v2 = 10.0 + v
        arr[0] = v2.x
        arr[1] = v2.y
        arr[2] = v2.z
        arr[3] = v2.w

    arr = np.zeros(4, dtype=np.float32)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [11.0, 12.0, 13.0, 14.0])


def test_vector_add_vector():
    @cuda.jit
    def kernel(arr):
        v = cuda.float32x4(1.0, 2.0, 3.0, 4.0)
        v2 = cuda.float32x4(10.0, 20.0, 30.0, 40.0)
        v3 = v + v2
        arr[0] = v3.x
        arr[1] = v3.y
        arr[2] = v3.z
        arr[3] = v3.w

    arr = np.zeros(4, dtype=np.float32)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [11.0, 22.0, 33.0, 44.0])


def test_vector_sub_scalar():
    @cuda.jit
    def kernel(arr):
        v = cuda.float32x4(1.0, 2.0, 3.0, 4.0)
        v2 = v - 1.0
        arr[0] = v2.x
        arr[1] = v2.y
        arr[2] = v2.z
        arr[3] = v2.w

    arr = np.zeros(4, dtype=np.float32)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [0.0, 1.0, 2.0, 3.0])


def test_vector_mul_scalar():
    @cuda.jit
    def kernel(arr):
        v = cuda.float32x4(1.0, 2.0, 3.0, 4.0)
        v2 = v * 2.0
        arr[0] = v2.x
        arr[1] = v2.y
        arr[2] = v2.z
        arr[3] = v2.w

    arr = np.zeros(4, dtype=np.float32)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [2.0, 4.0, 6.0, 8.0])


def test_vector_div_scalar():
    @cuda.jit
    def kernel(arr):
        v = cuda.float32x4(2.0, 4.0, 6.0, 8.0)
        v2 = v / 2.0
        arr[0] = v2.x
        arr[1] = v2.y
        arr[2] = v2.z
        arr[3] = v2.w

    arr = np.zeros(4, dtype=np.float32)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0, 4.0])


def test_vector_floordiv_scalar():
    @cuda.jit
    def kernel(arr):
        v = cuda.int32x4(3, 5, 7, 9)
        v2 = v // 2
        arr[0] = v2.x
        arr[1] = v2.y
        arr[2] = v2.z
        arr[3] = v2.w

    arr = np.zeros(4, dtype=np.int32)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [1, 2, 3, 4])


def test_vector_mod_scalar():
    @cuda.jit
    def kernel(arr):
        v = cuda.int32x4(3, 5, 7, 9)
        v2 = v % 2
        arr[0] = v2.x
        arr[1] = v2.y
        arr[2] = v2.z
        arr[3] = v2.w

    arr = np.zeros(4, dtype=np.int32)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [1, 1, 1, 1])


def test_vector_neg():
    @cuda.jit
    def kernel(arr):
        v = cuda.float32x4(1.0, -2.0, 3.0, -4.0)
        v2 = -v
        arr[0] = v2.x
        arr[1] = v2.y
        arr[2] = v2.z
        arr[3] = v2.w

    arr = np.zeros(4, dtype=np.float32)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [-1.0, 2.0, -3.0, 4.0])


def test_vector_abs():
    @cuda.jit
    def kernel(arr):
        v = cuda.float32x4(1.0, -2.0, 3.0, -4.0)
        v2 = abs(v)
        arr[0] = v2.x
        arr[1] = v2.y
        arr[2] = v2.z
        arr[3] = v2.w

    arr = np.zeros(4, dtype=np.float32)
    kernel[1, 1](arr)
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0, 4.0])


def test_vector_mul_vector_fails():
    @cuda.jit
    def kernel(arr):
        v = cuda.float32x4(1.0, 2.0, 3.0, 4.0)
        v2 = cuda.float32x4(1.0, 2.0, 3.0, 4.0)
        v3 = v * v2
        arr[0] = v3.x

    arr = np.zeros(1, dtype=np.float32)
    with pytest.raises(Exception, match="No implementation of function Function"):
        kernel[1, 1](arr)


def test_vector_div_vector_fails():
    @cuda.jit
    def kernel(arr):
        v = cuda.float32x4(1.0, 2.0, 3.0, 4.0)
        v2 = cuda.float32x4(1.0, 2.0, 3.0, 4.0)
        v3 = v / v2
        arr[0] = v3.x

    arr = np.zeros(1, dtype=np.float32)
    with pytest.raises(Exception, match="No implementation of function Function"):
        kernel[1, 1](arr)
