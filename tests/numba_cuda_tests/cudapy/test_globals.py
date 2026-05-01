# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import numba_cuda_mlir
from numba_cuda_mlir import cuda
import pytest
from numba_cuda_mlir.numba_cuda.types import int32, float32
from numba_cuda_mlir.testing import NumbaCUDATestCase

N = 100


def simple_smem(ary):
    sm = cuda.shared.array(N, int32)
    i = cuda.grid(1)
    if i == 0:
        for j in range(N):
            sm[j] = j
    cuda.syncthreads()
    ary[i] = sm[i]


S0 = 10
S1 = 20


def coop_smem2d(ary):
    i, j = cuda.grid(2)
    sm = cuda.shared.array((S0, S1), float32)
    sm[i, j] = (i + 1) / (j + 1)
    cuda.syncthreads()
    ary[i, j] = sm[i, j]


class TestCudaTestGlobal(NumbaCUDATestCase):
    def test_global_int_const(self):
        """Test simple_smem"""
        compiled = numba_cuda_mlir.jit("void(int32[:])")(simple_smem)

        nelem = 100
        ary = np.empty(nelem, dtype=np.int32)
        compiled[1, nelem](ary)

        self.assertTrue(np.all(ary == np.arange(nelem, dtype=np.int32)))

    @pytest.mark.skip(reason="unknown")
    def test_global_tuple_const(self):
        """Test coop_smem2d"""
        compiled = numba_cuda_mlir.jit("void(float32[:,:])")(coop_smem2d)

        shape = 10, 20
        ary = np.empty(shape, dtype=np.float32)
        compiled[1, shape](ary)

        exp = np.empty_like(ary)
        for i in range(ary.shape[0]):
            for j in range(ary.shape[1]):
                exp[i, j] = float(i + 1) / (j + 1)
        self.assertTrue(np.allclose(ary, exp))
