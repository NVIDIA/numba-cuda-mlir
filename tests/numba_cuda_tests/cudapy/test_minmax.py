# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

import numba_cuda_mlir
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda.types import float64
from numba_cuda_mlir.testing import NumbaCUDATestCase
import pytest


def builtin_max(A, B, C):
    i = cuda.grid(1)

    if i >= len(C):
        return

    C[i] = float64(max(A[i], B[i]))


def builtin_min(A, B, C):
    i = cuda.grid(1)

    if i >= len(C):
        return

    C[i] = float64(min(A[i], B[i]))


class TestCudaMinMax(NumbaCUDATestCase):
    def _run(
        self,
        kernel,
        reference_function,
        ptx_instruction,
        dtype_left,
        dtype_right,
    ):
        kernel = numba_cuda_mlir.cuda.jit(kernel)

        a = np.array([1.0, np.nan, 1.0, np.nan, -0.0, 0.0], dtype=dtype_left)
        b = np.array([2.0, 2.0, np.nan, np.nan, 0.0, -0.0], dtype=dtype_right)
        c = np.empty(a.size, dtype=np.float64)
        expected = reference_function(a, b)

        kernel[1, c.shape](a, b, c)
        np.testing.assert_array_equal(c, expected)
        zero = expected == 0
        np.testing.assert_array_equal(np.signbit(c[zero]), np.signbit(expected[zero]))

        ptx = next(p for p in kernel.inspect_asm().values())
        self.assertIn(ptx_instruction, ptx)
        self.assertNotIn(ptx_instruction.replace(".", ".NaN.", 1), ptx)

    def test_max_f8f8(self):
        self._run(builtin_max, np.fmax, "max.f64", np.float64, np.float64)

    def test_max_f4f8(self):
        self._run(builtin_max, np.fmax, "max.f64", np.float32, np.float64)

    def test_max_f8f4(self):
        self._run(builtin_max, np.fmax, "max.f64", np.float64, np.float32)

    def test_max_f4f4(self):
        self._run(builtin_max, np.fmax, "max.f32", np.float32, np.float32)

    def test_min_f8f8(self):
        self._run(builtin_min, np.fmin, "min.f64", np.float64, np.float64)

    def test_min_f4f8(self):
        self._run(builtin_min, np.fmin, "min.f64", np.float32, np.float64)

    def test_min_f8f4(self):
        self._run(builtin_min, np.fmin, "min.f64", np.float64, np.float32)

    def test_min_f4f4(self):
        self._run(builtin_min, np.fmin, "min.f32", np.float32, np.float32)
