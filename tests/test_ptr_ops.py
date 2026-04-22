# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cuda.simt as cuda
from cuda.simt import compiler, types
import ctypes
import numpy as np


def test_ptr_arith():
    @cuda.jit(types.void(types.int32[:], types.int32), dump=True)
    def ptr_arith(a, b):
        a_as_ptr = ctypes.cast(a, ctypes.POINTER(ctypes.c_int32))
        a_as_ptr[0] = 5
        a_as_ptr += b
        a_as_ptr[0] = 6

    a = cuda.device_array(2, dtype=np.int32)
    b = 1
    ptr_arith[1, 1](a, b)
    ah = a.copy_to_host()
    assert ah[0] == 5
    assert ah[1] == 6

    @cuda.jit(types.void(types.int32[:], types.int32), dump=True)
    def ptr_arith(a, b):
        a_as_ptr = ctypes.cast(a, ctypes.POINTER(ctypes.c_int32))
        a_as_ptr += 1
        a_as_ptr[0] = 2
        a_as_ptr -= b
        a_as_ptr[0] = 1

    ptr_arith[1, 1](a, b)
    ah = a.copy_to_host()
    assert ah[0] == 1
    assert ah[1] == 2


def test_cpointer_getitem():
    """Test CPointer getitem with explicit signature (like numba-cuda's test_dispatcher_cpointer_arguments)"""
    ptr = types.CPointer(types.int32)
    sig = types.void(ptr, types.int32, ptr, ptr, types.uint32)

    @cuda.jit(sig)
    def axpy(r, a, x, y, n):
        i = cuda.grid(1)
        if i < n:
            r[i] = a * x[i] + y[i]

    N = 16
    a = 5
    hx = np.arange(N, dtype=np.int32)
    hy = np.arange(N, dtype=np.int32) * 2
    dx = cuda.to_device(hx)
    dy = cuda.to_device(hy)
    dr = cuda.device_array(N, dtype=np.int32)

    r_ptr = dr.__cuda_array_interface__["data"][0]
    x_ptr = dx.__cuda_array_interface__["data"][0]
    y_ptr = dy.__cuda_array_interface__["data"][0]

    axpy[1, N](r_ptr, a, x_ptr, y_ptr, N)

    hr = dr.copy_to_host()
    expected = a * hx + hy
    np.testing.assert_array_equal(hr, expected)


if __name__ == "__main__":
    test_ptr_arith()
    test_cpointer_getitem()
