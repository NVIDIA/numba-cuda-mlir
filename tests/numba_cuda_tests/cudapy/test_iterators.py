# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numba_cuda_mlir
from numba_cuda_mlir.testing import NumbaCUDATestCase
import pytest
import numpy as np


class TestIterators(NumbaCUDATestCase):
    @pytest.mark.xfail(True, reason="Typing error")
    def test_enumerate(self):
        @numba_cuda_mlir.cuda.jit
        def enumerator(x, error):
            count = 0

            for i, v in enumerate(x):
                if count != i:
                    error[0] = 1
                if v != x[i]:
                    error[0] = 2

                count += 1

            if count != len(x):
                error[0] = 3

        x = np.asarray((10, 9, 8, 7, 6))
        error = np.zeros(1, dtype=np.int32)

        enumerator[1, 1](x, error)
        self.assertEqual(error[0], 0)

    def _test_twoarg_function(self, f):
        x = np.asarray((10, 9, 8, 7, 6))
        y = np.asarray((1, 2, 3, 4, 5))
        error = np.zeros(1, dtype=np.int32)

        f[1, 1](x, y, error)
        self.assertEqual(error[0], 0)

    @pytest.mark.xfail(True, reason="Typing error")
    def test_zip(self):
        @numba_cuda_mlir.cuda.jit
        def zipper(x, y, error):
            i = 0

            for xv, yv in zip(x, y):
                if xv != x[i]:
                    error[0] = 1
                if yv != y[i]:
                    error[0] = 2

                i += 1

            if i != len(x):
                error[0] = 3

        self._test_twoarg_function(zipper)

    @pytest.mark.xfail(True, reason="Typing error")
    def test_enumerate_zip(self):
        @numba_cuda_mlir.cuda.jit
        def enumerator_zipper(x, y, error):
            count = 0

            for i, (xv, yv) in enumerate(zip(x, y)):
                if i != count:
                    error[0] = 1
                if xv != x[i]:
                    error[0] = 2
                if yv != y[i]:
                    error[0] = 3

                count += 1

            if count != len(x):
                error[0] = 4

        self._test_twoarg_function(enumerator_zipper)

    @pytest.mark.xfail(True, reason="Typing error")
    def test_zip_enumerate(self):
        @numba_cuda_mlir.cuda.jit
        def zipper_enumerator(x, y, error):
            count = 0

            for (i, xv), yv in zip(enumerate(x), y):
                if i != count:
                    error[0] = 1
                if xv != x[i]:
                    error[0] = 2
                if yv != y[i]:
                    error[0] = 3

                count += 1

            if count != len(x):
                error[0] = 4

        self._test_twoarg_function(zipper_enumerator)
