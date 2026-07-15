# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for kernel exception handling via error code mechanism."""

import numpy as np
import pytest
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda import require_context, types
from numba_cuda_mlir.numba_cuda.testing import skip_if_nvjitlink_missing
from numba_cuda_mlir.tools import get_gpu_compute_capability


@cuda.jit(debug=True, opt=False)
def tuple_bounds_check_kernel(out, idx):
    """Kernel that accesses a tuple with bounds checking."""
    t = (10, 20, 30)
    out[0] = t[idx]


def test_tuple_index_out_of_bounds_raises():
    """Test that out-of-bounds tuple access raises IndexError."""
    out = cuda.device_array(1, dtype=np.int32)

    # Valid access should work
    tuple_bounds_check_kernel[1, 1](out, 0)
    result = out.copy_to_host()
    assert result[0] == 10

    tuple_bounds_check_kernel[1, 1](out, 2)
    result = out.copy_to_host()
    assert result[0] == 30

    # Out-of-bounds access should raise IndexError
    with pytest.raises(IndexError, match="out of bounds"):
        tuple_bounds_check_kernel[1, 1](out, 5)


def test_multiple_errors_first_wins():
    """Test that when multiple errors occur, the first one wins."""
    out = cuda.device_array(1, dtype=np.int32)

    # First error should be captured
    with pytest.raises(IndexError):
        tuple_bounds_check_kernel[1, 1](out, 100)

    # Error should be reset, so another error can be raised
    with pytest.raises(IndexError):
        tuple_bounds_check_kernel[1, 1](out, 200)


def test_error_global_in_ptx():
    """Test that the error global is present in compiled PTX."""
    from numba_cuda_mlir.compiler import compile_ptx
    from numba_cuda_mlir.numba_cuda import types

    sig = types.void(types.int32[:], types.int64)
    ptx, _ = compile_ptx(tuple_bounds_check_kernel, sig)

    assert "__numba_cuda_mlir_error_code" in ptx, "Error global not found in PTX"
    assert ".common .global" in ptx, "Error global should be common"


@skip_if_nvjitlink_missing("nvJitLink missing")
@require_context
def test_ltoir_device_functions_share_error_global():
    """Independently compiled device functions can be linked together."""

    def op_a(x):
        return x + 1

    def op_b(x):
        return x + 2

    signature = types.int32(types.int32)
    lto_a, _ = cuda.compile(
        op_a,
        signature,
        device=True,
        abi="c",
        abi_info={"abi_name": "op_a"},
        output="ltoir",
    )
    lto_b, _ = cuda.compile(
        op_b,
        signature,
        device=True,
        abi="c",
        abi_info={"abi_name": "op_b"},
        output="ltoir",
    )
    major, minor = get_gpu_compute_capability(tuple)

    from cuda.bindings import nvjitlink

    handle = nvjitlink.create(2, ["-lto", f"-arch=sm_{major}{minor}"])
    try:
        nvjitlink.add_data(handle, nvjitlink.InputType.LTOIR, lto_a, len(lto_a), "op_a")
        nvjitlink.add_data(handle, nvjitlink.InputType.LTOIR, lto_b, len(lto_b), "op_b")
        nvjitlink.complete(handle)
    finally:
        nvjitlink.destroy(handle)


@pytest.mark.parametrize(
    "debug, opt",
    [
        (False, False),
        (False, True),
        (True, False),
    ],
)
def test_raise_only_kernel(debug, opt):
    """Raise-only kernels compile; the raise surfaces only under debug."""

    @cuda.jit(debug=debug, opt=opt)
    def k():
        raise RuntimeError("Error")

    if debug:
        with pytest.raises(RuntimeError, match="Runtime error in kernel"):
            k[1, 1]()
    else:
        k[1, 1]()
        cuda.synchronize()
