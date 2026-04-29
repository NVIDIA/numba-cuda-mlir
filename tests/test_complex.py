# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pytest

from cusimt.numba_cuda import types
import cuda.simt as cuda


class TestAtomicOnComplexComponents(unittest.TestCase):

    def test_atomic_on_real_1d(self):
        @cuda.jit
        def atomic_add_one(values):
            i = cuda.grid(1)
            cuda.atomic.add(values.real, i, 1)

        N = 32
        arr1 = np.arange(N) + np.arange(N) * 1j
        arr2 = arr1.copy()
        atomic_add_one[1, N](arr2)
        np.testing.assert_equal(arr1 + 1, arr2)

    def test_atomic_on_imag_1d(self):
        @cuda.jit
        def atomic_add_one_j(values):
            i = cuda.grid(1)
            cuda.atomic.add(values.imag, i, 1)

        N = 32
        arr1 = np.arange(N) + np.arange(N) * 1j
        arr2 = arr1.copy()
        atomic_add_one_j[1, N](arr2)
        np.testing.assert_equal(arr1 + 1j, arr2)

    def test_atomic_on_real_2d(self):
        @cuda.jit
        def atomic_add_one_2d(values):
            i, j = cuda.grid(2)
            if i < values.shape[0] and j < values.shape[1]:
                cuda.atomic.add(values.real, (i, j), 1)

        M, N = 4, 8
        arr1 = (np.arange(M * N) + np.arange(M * N) * 1j).reshape(M, N)
        arr2 = arr1.copy()
        atomic_add_one_2d[(1, 1), (M, N)](arr2)
        np.testing.assert_equal(arr1 + 1, arr2)

    def test_atomic_on_imag_2d(self):
        @cuda.jit
        def atomic_add_one_j_2d(values):
            i, j = cuda.grid(2)
            if i < values.shape[0] and j < values.shape[1]:
                cuda.atomic.add(values.imag, (i, j), 1)

        M, N = 4, 8
        arr1 = (np.arange(M * N) + np.arange(M * N) * 1j).reshape(M, N)
        arr2 = arr1.copy()
        atomic_add_one_j_2d[(1, 1), (M, N)](arr2)
        np.testing.assert_equal(arr1 + 1j, arr2)
