# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for kernel exception handling via error code mechanism.

Kernel exceptions (``raise``/``assert``/bounds checks) surface to the host only
when the kernel is compiled with ``debug=True``. This matches numba-cuda and the
documented behavior in ``docs/source/user/cudapysupported.rst``: with
``debug=False`` the launch stays asynchronous and no device-side error readback
(``cuCtxSynchronize`` + status ``cuMemcpyDtoH``) is performed. See issue #232.
"""

import numpy as np
import pytest
from numba_cuda_mlir import cuda
from numba_cuda_mlir.compiler import compile_ptx
from numba_cuda_mlir.cuda import literal_unroll
from numba_cuda_mlir.numba_cuda import require_context, types
from numba_cuda_mlir.numba_cuda.testing import skip_if_nvjitlink_missing
from numba_cuda_mlir.tools import get_gpu_compute_capability


@cuda.jit
def tuple_bounds_check_kernel(out, idx):
    """Kernel that accesses a tuple with bounds checking (debug=False)."""
    t = (10, 20, 30)
    out[0] = t[idx]


@cuda.jit(debug=True, opt=False)
def tuple_bounds_check_kernel_debug(out, idx):
    """Same kernel compiled with debug=True, so kernel exceptions surface."""
    t = (10, 20, 30)
    out[0] = t[idx]


def test_tuple_index_in_bounds():
    """In-bounds tuple access works regardless of debug."""
    out = cuda.device_array(1, dtype=np.int32)

    tuple_bounds_check_kernel[1, 1](out, 0)
    cuda.synchronize()
    assert out.copy_to_host()[0] == 10

    tuple_bounds_check_kernel[1, 1](out, 2)
    cuda.synchronize()
    assert out.copy_to_host()[0] == 30


def test_tuple_index_out_of_bounds_raises_with_debug():
    """Out-of-bounds tuple access raises IndexError when debug=True."""
    out = cuda.device_array(1, dtype=np.int32)

    # Valid access should still work.
    tuple_bounds_check_kernel_debug[1, 1](out, 0)
    cuda.synchronize()
    assert out.copy_to_host()[0] == 10

    # Out-of-bounds access should raise IndexError. With debug=True the check
    # runs inside the (now synchronous) launch, so the launch call itself raises.
    with pytest.raises(IndexError, match="out of bounds"):
        tuple_bounds_check_kernel_debug[1, 1](out, 5)
        cuda.synchronize()


def test_tuple_index_out_of_bounds_no_raise_without_debug():
    """With debug=False, kernel exceptions do not surface: the launch stays
    asynchronous and no device-side error readback is performed."""
    out = cuda.device_array(1, dtype=np.int32)

    # Out-of-bounds index, but debug=False -> must not raise.
    tuple_bounds_check_kernel[1, 1](out, 5)
    cuda.synchronize()


def test_error_resets_between_launches():
    """The device error global is reset after each debug=True readback, so a
    later launch surfaces its own error instead of being masked (or falsely
    triggered) by a prior launch's code.

    With debug=True the readback runs inside the launch call, so each launch
    raises directly.
    """
    out = cuda.device_array(1, dtype=np.int32)

    # First launch's out-of-bounds access is read back and raises.
    with pytest.raises(IndexError):
        tuple_bounds_check_kernel_debug[1, 1](out, 100)

    # The global was reset to 0, so the second launch surfaces its own error.
    with pytest.raises(IndexError):
        tuple_bounds_check_kernel_debug[1, 1](out, 200)


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
    """A raise-only kernel compiles; the RuntimeError surfaces only with debug=True."""

    @cuda.jit(debug=debug, opt=opt)
    def k():
        raise RuntimeError("Error")

    if debug:
        with pytest.raises(RuntimeError, match="Runtime error in kernel"):
            k[1, 1]()
            cuda.synchronize()
    else:
        # debug=False: exceptions are not checked; the launch is asynchronous
        # and must not raise.
        k[1, 1]()
        cuda.synchronize()


def _launch(kernel, debug, *values):
    """Launch ``kernel`` on (out, x); return out and whether the launch raised ValueError."""
    out = cuda.to_device(np.zeros(2, dtype=np.int64))
    x = cuda.to_device(np.array(values, dtype=np.int64))
    raised = False
    try:
        kernel[1, 1](out, x)
        cuda.synchronize()
    except ValueError:
        raised = True
    assert raised == (debug and values[0] < 0)
    return out.copy_to_host().tolist()


@pytest.mark.parametrize("debug", [False, True])
def test_raise_skips_code_after_raise(debug):
    """A taken raise returns from the kernel; the store after it does not run."""

    @cuda.jit(debug=debug, opt=not debug)
    def k(out, x):
        if x[0] < 0:
            raise ValueError("negative")
        out[0] = 5

    assert _launch(k, debug, 1, 0) == [5, 0]
    assert _launch(k, debug, -1, 0) == [0, 0]


@pytest.mark.parametrize("debug", [False, True])
def test_raise_before_branching_tail(debug):
    """A value defined after the raise and used past a later branch compiles.

    The raise used to branch to the last Return block, which here holds
    ``out[0] = v``; ``v`` does not dominate the raise, so the module failed to
    verify ("operand #0 does not dominate this use").
    """

    @cuda.jit(debug=debug, opt=not debug)
    def k(out, x):
        if x[0] < 0:
            raise ValueError("negative")
        v = x[0] * 2
        if x[1] > 0:
            out[1] = 1
        out[0] = v

    assert _launch(k, debug, 3, 0) == [6, 0]
    assert _launch(k, debug, 3, 1) == [6, 1]
    assert _launch(k, debug, -1, 1) == [0, 0]


def test_literal_unroll_before_branching_tail():
    """literal_unroll's unmatched-case raise does not break a branching tail."""

    @cuda.jit
    def k(out, x):
        value = 0
        for item in literal_unroll((1, 2)):
            value += item
        v = value * x[0]
        if x[1] > 0:
            out[1] = 1
        out[0] = v

    assert _launch(k, False, 2, 1) == [6, 1]


@cuda.jit(device=True, inline="never")
def _checked_scale(v, flag):
    if v < 0:
        raise ValueError("negative")
    w = v * 2
    if flag:
        v += 1
    return w


@cuda.jit(device=True, inline="never")
def _checked_pair(v):
    if v < 0:
        raise ValueError("negative")
    w = (v, v * 2.0)
    if v > 3:
        v += 1
    return w


@pytest.mark.parametrize("debug", [False, True])
def test_raise_in_value_returning_device_function(debug):
    """A device function that raises returns from the raise site, for scalar and tuple results."""

    @cuda.jit(debug=debug, opt=not debug)
    def k(out, x):
        out[0] = _checked_scale(x[0], x[1] > 0)
        p = _checked_pair(x[0])
        out[1] = p[0] + int(p[1])

    assert _launch(k, debug, 3, 1) == [6, 9]
    _launch(k, debug, -1, 1)


@pytest.mark.parametrize("ndim", [1, 2])
def test_raise_in_array_returning_device_function(ndim):
    """A device function returning an array view compiles when it can raise."""

    def view_1d(a, n):
        if n < 0:
            raise ValueError("negative")
        b = a[n:]
        if n > 3:
            a[0] = 1
        return b

    def view_2d(a, n):
        if n < 0:
            raise ValueError("negative")
        b = a[n:, 1:]
        if n > 3:
            a[0, 0] = 1
        return b

    pyfunc = view_1d if ndim == 1 else view_2d
    array_type = types.Array(types.float32, ndim, "C")
    ptx, _ = compile_ptx(pyfunc, (array_type, types.int64), device=True)
    assert "ret;" in ptx
