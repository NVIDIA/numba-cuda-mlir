# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from numba_cuda_mlir import cuda


def test_boolean():
    @cuda.jit("void(float64[:], bool_)")
    def k(A, vertial):
        if vertial:
            A[0] = 123
        else:
            A[0] = 321

    A = np.array([0], dtype="float64")
    A = cuda.to_device(A)
    k[1, 1](A, True)
    assert A.copy_to_host()[0] == 123
    k[1, 1](A, False)
    assert A.copy_to_host()[0] == 321
