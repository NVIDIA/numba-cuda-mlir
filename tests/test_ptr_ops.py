# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir import compiler, types
from numba_cuda_mlir.cuda.experimental import intrin
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir._mlir.extras import types as T
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


def test_array_slice_ptr_preserves_offset():
    """ctypes.cast(array_slice, POINTER(...)) points at the slice start."""

    @cuda.jit
    def copy_from_slice(src, dst, start):
        i = cuda.threadIdx.x
        ptr = ctypes.cast(src[start:], ctypes.POINTER(ctypes.c_int32))
        dst[i] = ptr[i]

    @cuda.jit
    def copy_bool_from_slice(src, dst, start):
        i = cuda.threadIdx.x
        ptr = ctypes.cast(src[start:], ctypes.POINTER(ctypes.c_bool))
        dst[i] = ptr[i]

    @intrin.define
    def load_i32(ptr: llvm.PointerType.get, idx: types.int64) -> types.int32:
        llvm_kDynamic = -2147483648
        offset_ptr = llvm.getelementptr(
            llvm.PointerType.get(), ptr, [idx], [llvm_kDynamic], T.i32(), None
        )
        return llvm.load(T.i32(), offset_ptr)

    @cuda.jit
    def copy_from_types_ptr_slice(src, dst, start):
        dst[0] = load_i32(types.ptr(src[start:]), 0)

    h_src = np.arange(32, dtype=np.int32)
    h_dst = np.zeros(8, dtype=np.int32)

    copy_from_slice[1, h_dst.size](h_src, h_dst, 11)

    np.testing.assert_array_equal(h_dst, h_src[11 : 11 + h_dst.size])

    h_bool_src = np.array(
        [False, True, False, True, True, False, True, False],
        dtype=np.bool_,
    )
    h_bool_dst = np.zeros(4, dtype=np.bool_)

    copy_bool_from_slice[1, h_bool_dst.size](h_bool_src, h_bool_dst, 3)

    np.testing.assert_array_equal(h_bool_dst, h_bool_src[3 : 3 + h_bool_dst.size])

    h_types_ptr_dst = np.zeros(1, dtype=np.int32)

    copy_from_types_ptr_slice[1, 1](h_src, h_types_ptr_dst, 11)

    assert h_types_ptr_dst[0] == h_src[11]


if __name__ == "__main__":
    test_ptr_arith()
    test_cpointer_getitem()
    test_array_slice_ptr_preserves_offset()
