# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from numba_cuda_mlir import cuda

NP_TRUE = np.bool_(True)
NP_FALSE = np.bool_(False)
NP_F32 = np.float32(3.5)
NP_I8 = np.int8(7)


def test_np_bool_global_in_branch():
    @cuda.jit
    def k(out):
        if NP_TRUE:
            out[0] = 1
        if NP_FALSE:
            out[1] = 1

    out = cuda.to_device(np.zeros(2, dtype=np.int32))
    k[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), [1, 0])


def test_np_bool_global_assignment():
    @cuda.jit
    def k(out):
        flag = NP_TRUE
        out[0] = 1 if flag else 0
        other = NP_FALSE
        out[1] = 1 if other else 0

    out = cuda.to_device(np.zeros(2, dtype=np.int32))
    k[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), [1, 0])


def test_np_bool_closure_freevar():
    def make_kernel(flag):
        @cuda.jit
        def k(out):
            captured = flag
            out[0] = 1 if captured else 0

        return k

    out = cuda.to_device(np.zeros(1, dtype=np.int32))
    make_kernel(np.bool_(True))[1, 1](out)
    assert out.copy_to_host()[0] == 1
    make_kernel(np.bool_(False))[1, 1](out)
    assert out.copy_to_host()[0] == 0


def test_np_scalar_globals_arithmetic():
    @cuda.jit
    def k(out):
        out[0] = NP_F32 * 2.0
        out[1] = NP_I8 + 1

    out = cuda.to_device(np.zeros(2, dtype=np.float64))
    k[1, 1](out)
    np.testing.assert_allclose(out.copy_to_host(), [7.0, 8.0])
