# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

import numba_cuda_mlir
from numba_cuda_mlir import cuda
from numba_cuda_mlir.testing import NumbaCUDATestCase


class TestFreeVar(NumbaCUDATestCase):
    def test_freevar(self):
        """Make sure we can compile the following kernel with freevar reference
        in arguments to shared.array
        """
        from numba_cuda_mlir.numba_cuda.types import float32

        size = 1024
        nbtype = float32

        @numba_cuda_mlir.jit("(float32[::1], intp)")
        def foo(A, i):
            "Dummy function"
            sdata = cuda.shared.array(
                size,  # size is freevar
                dtype=nbtype,
            )  # nbtype is freevar
            A[i] = sdata[i]

        A = np.arange(2, dtype="float32")
        foo[1, 1](A, 0)
