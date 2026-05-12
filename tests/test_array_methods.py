# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from numba_cuda_mlir import cuda


def test_slicing():
    @cuda.jit(dump=False)
    def slicing(byte_arr: cuda.DeviceNDArray, start: int, stop: int, output: cuda.DeviceNDArray):
        val = byte_arr[start:stop, start:stop]
        output[0] = val[0, 0]

    byte_arr_host = np.array(range(9), dtype=np.uint8).reshape(3, 3)
    byte_arr = cuda.to_device(byte_arr_host)
    output = cuda.to_device(np.zeros(1, dtype=np.int32))
    slicing[1, 1, 0, 0](byte_arr, 1, 2, output)
    output = output.copy_to_host()
    expect = byte_arr_host[1, 1]
    assert output[0] == expect, f"output: {output} != {expect}"


def test_array_view():
    @cuda.jit(dump=False)
    def reinterpret_array_type(
        byte_arr: cuda.DeviceNDArray, start: int, stop: int, output: cuda.DeviceNDArray
    ):
        # Tested with just one thread
        val = byte_arr[start:stop].view(np.int32)[0]
        output[0] = val

    h = np.array(range(10), dtype=np.uint64)
    d = cuda.to_device(h)
    output = cuda.to_device(np.zeros(1, dtype=np.int32))
    reinterpret_array_type[1, 1](d, 1, 3, output)
    output = output.copy_to_host()
    expect = h[1:2].view(np.int32)[0]
    assert output[0] == expect, f"output: {output} != {expect}"


def test_array_view_same_dtype():
    """view() with the same element type is a no-op and must not fail."""

    @cuda.jit
    def kernel(arr):
        v = arr.view(np.int64)
        v[0] = 42

    d = cuda.to_device(np.zeros(4, dtype=np.int64))
    kernel[1, 1](d)
    out = d.copy_to_host()
    assert out[0] == 42, f"expected 42, got {out[0]}"


def test_array_view_reinterpret_dtype():
    """view() with a different same-sized dtype reinterprets the bits."""

    @cuda.jit
    def kernel(arr, out):
        v = arr.view(np.int32)
        out[0] = v[0]

    d = cuda.to_device(np.array([1.5], dtype=np.float32))
    out_d = cuda.to_device(np.zeros(1, dtype=np.int32))
    kernel[1, 1](d, out_d)

    expected = np.array([1.5], dtype=np.float32).view(np.int32)[0]
    result = out_d.copy_to_host()[0]
    assert result == expected, f"expected {expected}, got {result}"


def test_array_view_custom_dtype():
    """view() dispatches through to_mlir_type for externally-registered types."""
    from numba_cuda_mlir.lowering_utilities import to_mlir_type
    from numba_cuda_mlir._mlir.extras import types as T
    from numba_cuda_mlir.numba_cuda.extending import typeof_impl
    from numba_cuda_mlir.numba_cuda import types as nb_types

    class _MyBoxedDtype:
        pass

    my_dtype = _MyBoxedDtype()

    @typeof_impl.register(_MyBoxedDtype)
    def _typeof_my(val, c):
        return nb_types.NumberClass(nb_types.int64)

    @to_mlir_type.register(_MyBoxedDtype)
    def _my_to_mlir(val):
        return T.i64()

    @cuda.jit
    def kernel(arr):
        v = arr.view(my_dtype)
        v[0] = 42

    d = cuda.to_device(np.zeros(4, dtype=np.int64))
    kernel[1, 1](d)
    out = d.copy_to_host()
    assert out[0] == 42, f"expected 42, got {out[0]}"


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    # test_slicing()
    test_array_view()
