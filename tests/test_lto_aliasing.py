# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import cffi
import numpy as np
from cuda import pathfinder
from cuda.core import Device, Program, ProgramOptions
from numba_cuda_mlir import cuda, types as t
from numba_cuda_mlir.compiler import declare_device
from numba_cuda_mlir.numba_cuda.cudadrv.linkable_code import LTOIR
from numba_cuda_mlir.extending import overload, typing_registry
from numba_cuda_mlir.cuda import vector_types

# We need to compile the LTO module once
dev = Device(0)
dev.set_current()
cc = dev.compute_capability


def build_lto(src):
    return bytes(
        Program(
            src,
            "c++",
            ProgramOptions(
                arch=f"sm_{cc.major}{cc.minor}",
                link_time_optimization=True,
                relocatable_device_code=True,
                include_path=pathfinder.find_nvidia_header_directory("cudart"),
            ),
        )
        .compile("ltoir")
        .code
    )


# External LTO function
lto_noop = build_lto(
    """
    #include <cuda_fp16.h>
    extern "C" __device__ void s_noop(void* p) {
        (void)p;
    }
    """
)

ffi = cffi.FFI()

_call_noops = {}


def get_call_noop(dtype):
    if dtype not in _call_noops:
        stub = declare_device("s_noop", t.void(t.CPointer(dtype)), link=LTOIR(lto_noop), abi="c")

        def call_noop(smem):
            pass

        @overload(call_noop, strict=False, typing_registry=typing_registry)
        def ol_call_noop(smem):
            def impl(smem):
                stub(ffi.from_buffer(smem))

            return impl

        _call_noops[dtype] = call_noop
    return _call_noops[dtype]


types_map = {
    "float16": (t.float16, vector_types.float16x2, vector_types.float16x4, np.float16, 2),
    "float32": (t.float32, vector_types.float32x2, vector_types.float32x4, np.float32, 4),
    "float64": (t.float64, vector_types.float64x2, vector_types.float64x4, np.float64, 8),
}


@pytest.mark.parametrize("dtype_name", ["float16", "float32", "float64"])
@pytest.mark.parametrize("base_len", [1, 2, 4])
@pytest.mark.parametrize("input_len", [1, 2, 4])
def test_lto_aliasing(dtype_name, base_len, input_len):
    scalar_t, vec2_t, vec4_t, np_dtype, itemsize = types_map[dtype_name]

    type_dict = {1: scalar_t, 2: vec2_t, 4: vec4_t}
    base_type = type_dict[base_len]
    input_type = type_dict[input_len]

    call_noop = get_call_noop(base_type)

    num_write_threads = 16 // input_len
    num_read_threads = 16 // base_len
    num_threads = max(num_write_threads, num_read_threads)

    @cuda.jit
    def k_vec(inp, out):
        smem = cuda.shared.array(shape=(0,), dtype=base_type, alignment=16)
        smem_input = smem.view(input_type)
        smem_output = smem.view(base_type)

        tid = cuda.threadIdx.x

        if tid < num_write_threads:
            if input_len == 1:
                smem_input[tid] = inp[tid]
            elif input_len == 2:
                smem_input[tid] = input_type(inp[tid * 2 + 0], inp[tid * 2 + 1])
            elif input_len == 4:
                smem_input[tid] = input_type(
                    inp[tid * 4 + 0], inp[tid * 4 + 1], inp[tid * 4 + 2], inp[tid * 4 + 3]
                )

        cuda.syncthreads()

        if tid == 0:
            call_noop(smem)

        cuda.syncthreads()

        if tid < num_read_threads:
            if base_len == 1:
                out[tid] = smem_output[tid]
            elif base_len == 2:
                out[tid * 2 + 0] = smem_output[tid].x
                out[tid * 2 + 1] = smem_output[tid].y
            elif base_len == 4:
                out[tid * 4 + 0] = smem_output[tid].x
                out[tid * 4 + 1] = smem_output[tid].y
                out[tid * 4 + 2] = smem_output[tid].z
                out[tid * 4 + 3] = smem_output[tid].w

    input_h = np.full((16,), 3.0, dtype=np_dtype)
    output_h = np.zeros((16,), dtype=np_dtype)

    input_d = cuda.to_device(input_h)
    output_d = cuda.to_device(output_h)

    # 16 elements * itemsize bytes
    k_vec[1, num_threads, 0, 16 * itemsize](input_d, output_d)
    cuda.synchronize()

    got = output_d.copy_to_host()
    expected = np.full((16,), 3.0, dtype=np_dtype)
    np.testing.assert_array_equal(got, expected)
