# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import numba_cuda_mlir
from numba_cuda_mlir import cuda
from numba_cuda_mlir.testing import NumbaCUDATestCase


class TestCudaComplex(NumbaCUDATestCase):
    def test_cuda_complex_arg(self):
        @numba_cuda_mlir.jit("void(complex128[:], complex128)")
        def foo(a, b):
            i = cuda.grid(1)
            a[i] += b

        a = np.arange(5, dtype=np.complex128)
        a0 = a.copy()
        foo[1, a.shape](a, 2j)
        self.assertTrue(np.allclose(a, a0 + 2j))
