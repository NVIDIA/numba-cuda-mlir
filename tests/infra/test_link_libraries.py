# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cuda import pathfinder
from cuda.core import Device, Program, ProgramOptions
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
        add = cuda.declare_device("add", "float32(float32, float32)", link=temp_file.name)

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


def _compile_ltoir(source):
    dev = Device(0)
    dev.set_current()
    cc = dev.compute_capability
    program = Program(
        source,
        "c++",
        ProgramOptions(
            arch=f"sm_{cc.major}{cc.minor}",
            link_time_optimization=True,
            relocatable_device_code=True,
            include_path=pathfinder.find_nvidia_header_directory("cudart"),
        ),
    )
    return bytes(program.compile("ltoir").code)


SHARED_FROM_BUFFER_CABI_LTOIR = """
extern "C" __device__ void write_smem(void* p) {
    *(unsigned int*)p = 0x12345678u;
}
"""


HALF_NOOP_CABI_LTOIR = """
#include <cuda_fp16.h>
extern "C" __device__ void noop_half(__half* p) {
    (void)p;
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


def test_ffi_from_buffer_shared_memory_ltoir_cabi():
    """ffi.from_buffer preserves shared-memory pointers passed to C ABI LTOIR."""
    cffi = pytest.importorskip("cffi")
    ffi = cffi.FFI()
    write_smem = cuda.declare_device(
        "write_smem",
        types.void(types.CPointer(types.int32)),
        link=cuda.LTOIR(_compile_ltoir(SHARED_FROM_BUFFER_CABI_LTOIR)),
        abi="c",
    )

    @cuda.jit(opt_level=3)
    def kernel(out):
        smem = cuda.shared.array(shape=(16,), dtype=np.int32, alignment=16)
        tid = cuda.threadIdx.x
        smem[tid] = -1
        cuda.syncthreads()

        if tid == 0:
            write_smem(ffi.from_buffer(smem))

        cuda.syncthreads()
        out[tid] = smem[tid]

    out = cuda.to_device(np.zeros(16, dtype=np.int32))
    kernel[1, 16](out)
    cuda.synchronize()

    got = out.copy_to_host()
    assert got[0] == np.int32(0x12345678)
    np.testing.assert_equal(got[1:], np.full(15, -1, dtype=np.int32))


def test_ltoir_cabi_preserves_float16_shared_init_stores():
    """C ABI LTOIR calls do not make LTO drop preceding float16 shared stores."""
    cffi = pytest.importorskip("cffi")
    ffi = cffi.FFI()
    noop_half = cuda.declare_device(
        "noop_half",
        types.void(types.CPointer(types.float16)),
        link=cuda.LTOIR(_compile_ltoir(HALF_NOOP_CABI_LTOIR)),
        abi="c",
    )

    @cuda.jit(opt_level=3)
    def kernel(out):
        smem = cuda.shared.array(shape=(16,), dtype=np.float16, alignment=16)
        tid = cuda.threadIdx.x
        val = np.float16(3.0)

        smem[tid * 4 + 0] = val
        smem[tid * 4 + 1] = val
        smem[tid * 4 + 2] = val
        smem[tid * 4 + 3] = val
        cuda.syncthreads()

        if tid == 0:
            noop_half(ffi.from_buffer(smem))

        cuda.syncthreads()
        out[tid * 4 + 0] = smem[tid * 4 + 0]
        out[tid * 4 + 1] = smem[tid * 4 + 1]
        out[tid * 4 + 2] = smem[tid * 4 + 2]
        out[tid * 4 + 3] = smem[tid * 4 + 3]

    out = cuda.to_device(np.zeros(16, dtype=np.float16))
    kernel[1, 4](out)
    cuda.synchronize()

    got = out.copy_to_host()
    expected = np.full(16, 3.0, dtype=np.float16)
    np.testing.assert_equal(got, expected)


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
