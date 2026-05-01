# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from numba_cuda_mlir.testing import NumbaCUDATestCase
import numba_cuda_mlir
from numba_cuda_mlir import cuda


def boolean_func(A, vertial):
    if vertial:
        A[0] = 123
    else:
        A[0] = 321


class TestCudaBoolean(NumbaCUDATestCase):
    def test_boolean(self):
        func = numba_cuda_mlir.jit("void(float64[:], bool_)")(boolean_func)
        A = np.array([0], dtype="float64")
        func[1, 1](A, True)
        self.assertTrue(A[0] == 123)
        func[1, 1](A, False)
        self.assertTrue(A[0] == 321)
