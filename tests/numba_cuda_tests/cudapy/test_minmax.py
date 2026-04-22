# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import re

import numpy as np

import cusimt
import cuda.simt as cuda
from numba import float64
from cusimt.testing import NumbaCUDATestCase
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
        numpy_equivalent,
        ptx_instruction,
        dtype_left,
        dtype_right,
        n=5,
    ):
        kernel = cusimt.jit(kernel)

        c = np.zeros(n, dtype=np.float64)
        a = np.arange(n, dtype=dtype_left) + 0.5
        b = np.full(n, fill_value=2, dtype=dtype_right)

        kernel[1, c.shape](a, b, c)
        np.testing.assert_allclose(c, numpy_equivalent(a, b))

        ptx = next(p for p in kernel.inspect_asm().values())
        # sm_100 may emit e.g. "max.NaN.f32" instead of "max.f32"
        pattern = re.escape(ptx_instruction).replace(r"\.", r"\.(?:NaN\.)?", 1)
        self.assertRegex(ptx, pattern)

    def test_max_f8f8(self):
        self._run(builtin_max, np.maximum, "max.f64", np.float64, np.float64)

    def test_max_f4f8(self):
        self._run(builtin_max, np.maximum, "max.f64", np.float32, np.float64)

    def test_max_f8f4(self):
        self._run(builtin_max, np.maximum, "max.f64", np.float64, np.float32)

    def test_max_f4f4(self):
        self._run(builtin_max, np.maximum, "max.f32", np.float32, np.float32)

    def test_min_f8f8(self):
        self._run(builtin_min, np.minimum, "min.f64", np.float64, np.float64)

    def test_min_f4f8(self):
        self._run(builtin_min, np.minimum, "min.f64", np.float32, np.float64)

    def test_min_f8f4(self):
        self._run(builtin_min, np.minimum, "min.f64", np.float64, np.float32)

    def test_min_f4f4(self):
        self._run(builtin_min, np.minimum, "min.f32", np.float32, np.float32)
