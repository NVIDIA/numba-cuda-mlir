# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from numba_cuda_mlir.cuda import vectorize
import numba_cuda_mlir
from numba_cuda_mlir.numba_cuda.types import float64
from numba_cuda_mlir.testing import NumbaCUDATestCase
import pytest

sig = [float64(float64, float64)]


@pytest.mark.xfail(True, reason="Vectorize not supported")
class TestCUDAVectorizeScalarArg(NumbaCUDATestCase):
    def test_vectorize_scalar_arg(self):
        @vectorize(sig, target="cuda")
        def vector_add(a, b):
            return a + b

        A = np.arange(10, dtype=np.float64)
        dA = cuda.to_device(A)
        v = vector_add(1.0, dA)

        np.testing.assert_array_almost_equal(v.copy_to_host(), np.arange(1, 11, dtype=np.float64))

    def test_vectorize_all_scalars(self):
        @vectorize(sig, target="cuda")
        def vector_add(a, b):
            return a + b

        v = vector_add(1.0, 1.0)

        np.testing.assert_almost_equal(2.0, v)
