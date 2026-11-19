# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binary ufuncs on arrays with static shapes, such as local and shared arrays."""

import numpy as np
from numba_cuda_mlir import cuda


def test_binary_ufuncs_on_fixed_shape_arrays():
    @cuda.jit
    def kernel(out):
        local = cuda.local.array((2, 3), np.float64)
        shared = cuda.shared.array(4, np.float64)
        for i in range(2):
            for j in range(3):
                local[i, j] = i * 3 + j + 1
        for i in range(4):
            shared[i] = i + 1
        a = np.maximum(local, local * 2.0)
        b = np.arctan2(shared, shared)
        for i in range(2):
            for j in range(3):
                out[i * 3 + j] = a[i, j]
        for i in range(4):
            out[6 + i] = b[i]

    out = np.zeros(10)
    kernel[1, 1](out)
    values = np.concatenate([np.arange(1.0, 7.0), np.arange(1.0, 5.0)])
    expected = np.concatenate([values[:6] * 2.0, np.arctan2(values[6:], values[6:])])
    np.testing.assert_allclose(out, expected)
