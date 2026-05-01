# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.numba_cuda.cudadrv.devicearray import DeviceNDArray
from numba_cuda_mlir import cuda
from numba_cuda_mlir import types, compiler, testing
import numpy as np
import pytest
import tempfile
import logging
from numba_cuda_mlir import linker
from numba_cuda_mlir.numba_cuda.cudadrv.driver import CudaAPIError, LinkerError

logging.basicConfig(level=logging.DEBUG)

ADD_CU_SOURCE = """
extern "C" __device__ int add(float *ret, float a, float b) {
    *ret = a + b;
    return 0;
}
"""

CALL_LIBDEVICE_CU_SOURCE = """
extern "C" __device__ float __nv_floorf(float a);
extern "C" __device__ int call_floor(float *ret, float a) {
    *ret = __nv_floorf(a);
    return 0;
}
"""


def test_link_libraries():
    with tempfile.NamedTemporaryFile(suffix=".cu", delete=False) as temp_file:
        temp_file.write(ADD_CU_SOURCE.encode())
        temp_file.flush()
        add = cuda.declare_device(
            "add", "float32(float32, float32)", link=temp_file.name
        )

    @cuda.jit(dump=False, print_after_all=False, opt_level=3, lineinfo=True)
    def call_external_function(x: DeviceNDArray):
        x[0] = add(x[0], x[1])

    cres = call_external_function.compile("(float32[:],)")
    print(cres.metadata["mlir_module_str"])
    assert cres

    x = np.array([1.0, 2.0], dtype=np.float32)
    x = cuda.to_device(x)
    call_external_function[1, 1](x)
    x = x.copy_to_host()
    print(x)
    assert x[0] == 3.0


def test_link_libdevice():
    with tempfile.NamedTemporaryFile(suffix=".cu", delete=False) as temp_file:
        temp_file.write(CALL_LIBDEVICE_CU_SOURCE.encode())
        temp_file.flush()
        floor = cuda.declare_device(
            "call_floor",
            "float32(float32)",
            link=(temp_file.name,),
        )

    @cuda.jit(dump=False, print_after_all=False, opt_level=3, lineinfo=True)
    def call_external_function(x: DeviceNDArray):
        x[0] = floor(x[0])

    cres = call_external_function.compile("(float32[:],)")
    print(cres.metadata["mlir_module_str"])
    assert cres

    x = np.array([1.5], dtype=np.float32)
    x = cuda.to_device(x)
    call_external_function[1, 1](x)
    x = x.copy_to_host()
    print(x)
    assert x[0] == 1.0


DOUBLE_CU_SOURCE = """
extern "C" __device__ int double_elements(int *ret, int *arr) {
    arr[0] = arr[0] * 2;
    arr[1] = arr[1] * 2;
    *ret = 0;
    return 0;
}
"""


def test_ffi_from_buffer():
    """ffi.from_buffer extracts a CPointer from an array."""
    cffi = pytest.importorskip("cffi")
    ffi = cffi.FFI()

    with tempfile.NamedTemporaryFile(suffix=".cu", delete=False) as temp_file:
        temp_file.write(DOUBLE_CU_SOURCE.encode())
        temp_file.flush()
        double_elements = cuda.declare_device(
            "double_elements",
            types.int32(types.CPointer(types.int32)),
            link=temp_file.name,
        )

    @cuda.jit
    def kernel(x: DeviceNDArray):
        ptr = ffi.from_buffer(x)
        double_elements(ptr)

    x = np.array([3, 7], dtype=np.int32)
    x = cuda.to_device(x)
    kernel[1, 1](x)
    x = x.copy_to_host()
    assert x[0] == 6
    assert x[1] == 14


VOID_MUTATE_CU_SOURCE = """
extern "C" __device__ int mutate_array(int *ret, int *arr) {
    arr[0] = arr[0] * 10;
    *ret = 0;
    return 0;
}
"""


def test_link_void_return():
    """declare_device with void return type passes a dummy return pointer per the ABI."""
    cffi = pytest.importorskip("cffi")
    ffi = cffi.FFI()

    with tempfile.NamedTemporaryFile(suffix=".cu", delete=False) as temp_file:
        temp_file.write(VOID_MUTATE_CU_SOURCE.encode())
        temp_file.flush()
        mutate = cuda.declare_device(
            "mutate_array",
            types.void(types.CPointer(types.int32)),
            link=temp_file.name,
        )

    @cuda.jit
    def kernel(x: DeviceNDArray):
        ptr = ffi.from_buffer(x)
        mutate(ptr)

    x = np.array([5], dtype=np.int32)
    x = cuda.to_device(x)
    kernel[1, 1](x)
    x = x.copy_to_host()
    assert x[0] == 50


def test_linker():
    my_linker = linker.Linker(cc=(7, 5))
    my_linker.add_cu(ADD_CU_SOURCE, "add")
    code = my_linker.complete()
    assert code.code


if __name__ == "__main__":
    test_link_libdevice()
    test_link_libraries()
    test_ffi_from_buffer()
    test_link_void_return()
    test_linker()
