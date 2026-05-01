# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
import numpy as np
import pytest
from cuda.bindings.driver import (
    CUresult,
    cuGetErrorName,
    cuLibraryGetGlobal,
    cuMemcpyHtoD,
)
from numba_cuda_mlir.numba_cuda.cudadrv.linkable_code import CUSource

MODULE_WITH_GLOBAL = """
__device__ int value = 0;
extern "C" __device__ int get_value(int &retval) {
    retval = value;
    return 0;
}
"""


def _counter_callbacks():
    state = {"n": 0}
    return (
        state,
        lambda obj: state.__setitem__("n", state["n"] + 1),
        lambda obj: state.__setitem__("n", state["n"] - 1),
    )


def _check_cuda_result(result):
    err = result[0]
    if err != CUresult.CUDA_SUCCESS:
        _, name = cuGetErrorName(err)
        raise RuntimeError(f"CUDA driver call failed: {name}")
    return result[1:]


def test_setup_and_teardown():
    state, setup, teardown = _counter_callbacks()
    lib = CUSource("", setup_callback=setup, teardown_callback=teardown)

    @cuda.jit(link=[lib])
    def kernel():
        pass

    assert state["n"] == 0
    kernel[1, 1]()
    assert state["n"] == 1
    kernel[1, 1]()  # cached
    assert state["n"] == 1
    cuda.current_context().reset()
    assert state["n"] == 0


def test_per_argtype_callbacks():
    state, setup, teardown = _counter_callbacks()
    lib = CUSource("", setup_callback=setup, teardown_callback=teardown)

    @cuda.jit(link=[lib])
    def kernel(x):
        pass

    kernel[1, 1](np.int32(1))
    assert state["n"] == 1
    kernel[1, 1](np.float64(1.0))
    assert state["n"] == 2
    cuda.current_context().reset()
    assert state["n"] == 0


@pytest.mark.parametrize("use_jit_link", [True, False], ids=["jit_link", "declare_device_link"])
def test_callback_sets_device_global(use_jit_link):
    def set_value(obj):
        dptr, size = _check_cuda_result(cuLibraryGetGlobal(obj.handle, b"value"))
        arr = np.array([42], np.int32)
        _check_cuda_result(cuMemcpyHtoD(dptr, arr.ctypes.data, size))

    lib = CUSource(MODULE_WITH_GLOBAL, setup_callback=set_value)

    if use_jit_link:
        get_value = cuda.declare_device("get_value", "int32()")

        @cuda.jit(link=[lib])
        def kernel(out):
            out[0] = get_value()

    else:
        get_value = cuda.declare_device("get_value", "int32()", link=[lib])

        @cuda.jit
        def kernel(out):
            out[0] = get_value()

    out = np.zeros(1, np.int32)
    kernel[1, 1](out)
    assert out[0] == 42
