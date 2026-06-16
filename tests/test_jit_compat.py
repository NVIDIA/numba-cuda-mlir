# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for JIT decorator compatibility with numba-cuda options."""

import numpy as np
import pytest
import subprocess
import sys
import textwrap
from numba_cuda_mlir import cuda
from numba_cuda_mlir import types


def _run_in_subprocess(code: str):
    """Run Python code in a subprocess and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


# --- JIT Kwargs Compatibility ---


@pytest.mark.parametrize(
    "kwargs",
    [
        {"launch_bounds": 128},
        {"launch_bounds": (256, 2)},
        {"launch_bounds": (256, 2, 1)},
        {"inline": True},
        {"inline": False},
        {"inline": "never"},  # Use string instead of lambda for serialization
        {"opt": 0},
        {"opt": 1},
    ],
)
def test_jit_kwargs_accepted(kwargs):
    """Test that cuda.jit accepts various numba-cuda compatible kwargs."""

    @cuda.jit(**kwargs)
    def kernel(arr):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = i

    arr = cuda.device_array(10, dtype=np.int32)
    kernel[1, 10](arr)
    np.testing.assert_array_equal(arr.copy_to_host(), np.arange(10, dtype=np.int32))


def test_jit_kwargs_dbg_optnone():
    """Test _dbg_optnone kwarg (run in subprocess for isolation)."""
    code = """
        import numpy as np
        from numba_cuda_mlir import cuda

        @cuda.jit(_dbg_optnone=True)
        def kernel(arr):
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] = i

        arr = cuda.device_array(10, dtype=np.int32)
        kernel[1, 10](arr)
        result = arr.copy_to_host()
        assert all(result == range(10)), f"Expected [0..9], got {result}"
        print("OK")
    """
    rc, stdout, stderr = _run_in_subprocess(code)
    assert rc == 0, f"subprocess failed: {stderr}"
    assert "OK" in stdout


def test_jit_kwargs_inline_callable():
    """Test inline=callable kwarg (run in subprocess for isolation)."""
    code = """
        import numpy as np
        from numba_cuda_mlir import cuda

        @cuda.jit(inline=lambda expr: True)
        def kernel(arr):
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] = i

        arr = cuda.device_array(10, dtype=np.int32)
        kernel[1, 10](arr)
        result = arr.copy_to_host()
        assert all(result == range(10)), f"Expected [0..9], got {result}"
        print("OK")
    """
    rc, stdout, stderr = _run_in_subprocess(code)
    assert rc == 0, f"subprocess failed: {stderr}"
    assert "OK" in stdout


def test_cuda_cudadrv_reexports_legacy_namespace():
    from numba_cuda_mlir import cuda as compat_cuda
    from numba_cuda_mlir.numba_cuda.cudadrv import devicearray, driver
    import numba_cuda_mlir.cuda.cudadrv.devicearray as compat_devicearray

    assert compat_cuda.cudadrv.devicearray is devicearray
    assert compat_cuda.cudadrv.driver is driver
    assert compat_devicearray is devicearray


# --- MLIRDispatcher Resource Methods ---


@pytest.mark.parametrize(
    "method,min_val",
    [
        ("get_regs_per_thread", 0),
        ("get_max_threads_per_block", 1),
        ("get_shared_mem_per_block", 0),
        ("get_const_mem_size", 0),
        ("get_local_mem_per_thread", 0),
    ],
)
def test_dispatcher_resource_methods(method, min_val):
    """Test MLIRDispatcher resource query methods (run in subprocess for isolation)."""
    code = f"""
        import numpy as np
        from numba_cuda_mlir import cuda

        @cuda.jit
        def kernel(arr):
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] = i

        arr = cuda.device_array(10, dtype=np.int32)
        kernel[1, 10](arr)
        sig = tuple(kernel.overloads.keys())[0]
        result = kernel.{method}(sig)
        assert isinstance(result, int), f"Expected int, got {{type(result)}}"
        assert result >= {min_val}, f"Expected >= {min_val}, got {{result}}"
        print("OK")
    """
    rc, stdout, stderr = _run_in_subprocess(code)
    assert rc == 0, f"subprocess failed: {stderr}"
    assert "OK" in stdout


def test_compile_device():
    """Test compile_device and internal callee compilation work."""

    @cuda.jit(device=True)
    def device_func(x):
        return x * 2

    result = device_func.compile_device((types.int32,))
    assert result is not None

    @cuda.jit(device=True)
    def callee_func(x):
        return x * 2

    callee_result = callee_func._compile_as_device_callee((types.int32,))
    assert callee_result.metadata.get("cubin") is None


# --- Operator is/is_not Lowering ---


@pytest.mark.parametrize(
    "expr,input_val,expected",
    [
        ("val is None", 42, 0),
        ("val is not None", 42, 1),
    ],
)
def test_is_none_with_int(expr, input_val, expected):
    """Test 'x is None' and 'x is not None' for non-None values."""
    # We use exec to create kernels with different expressions
    kernel_code = f"""
@cuda.jit
def kernel(arr, val):
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] = 1 if {expr} else 0
"""
    local_ns = {"cuda": cuda}
    exec(kernel_code, local_ns)
    kernel = local_ns["kernel"]

    arr = cuda.device_array(1, dtype=np.int32)
    kernel[1, 1](arr, input_val)
    assert arr.copy_to_host()[0] == expected


def test_none_is_none():
    """Test 'None is None' returns True."""

    @cuda.jit
    def kernel(arr):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = 1 if None is None else 0

    arr = cuda.device_array(1, dtype=np.int32)
    kernel[1, 1](arr)
    assert arr.copy_to_host()[0] == 1


def test_none_is_not_none():
    """Test 'None is not None' returns False."""

    @cuda.jit
    def kernel(arr):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = 1 if None is not None else 0

    arr = cuda.device_array(1, dtype=np.int32)
    kernel[1, 1](arr)
    assert arr.copy_to_host()[0] == 0


def test_optional_return_from_device_function():
    @cuda.jit(device=True)
    def maybe_value(cond):
        if cond:
            return 42
        return None

    @cuda.jit
    def kernel(out, n):
        v = maybe_value(n > 0)
        if v is not None:
            out[0] = v

    out = cuda.device_array((1,), dtype=np.int64)
    out.copy_to_device(np.array([0], dtype=np.int64))
    kernel[1, 1](out, 1)
    assert out.copy_to_host()[0] == 42


@pytest.mark.parametrize(
    "expr,input_val,expected",
    [
        ("val is True", True, 1),
        ("val is True", False, 0),
        ("val is False", False, 1),
        ("val is False", True, 0),
        ("val is not True", True, 0),
        ("val is not True", False, 1),
    ],
)
def test_bool_identity(expr, input_val, expected):
    """Test bool identity comparisons (is True, is False, is not True)."""
    kernel_code = f"""
@cuda.jit
def kernel(arr, val):
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] = 1 if {expr} else 0
"""
    local_ns = {"cuda": cuda}
    exec(kernel_code, local_ns)
    kernel = local_ns["kernel"]

    arr = cuda.device_array(1, dtype=np.int32)
    kernel[1, 1](arr, input_val)
    assert arr.copy_to_host()[0] == expected


# --- Intrinsics ---


def test_aligned_dynamic_shared_memory_ptx_llvm70():
    def kernel():
        smem = cuda.shared.array(shape=(0,), dtype=np.byte, alignment=16)
        smem[0] = 0

    ptx, _ = cuda.compile_ptx(kernel, (), cc=(8, 0))
    assert ptx


def test_nanosleep_ptx():
    """Test nanosleep emits correct PTX."""

    def use_nanosleep(x):
        cuda.nanosleep(32)
        cuda.nanosleep(x)

    ptx, _ = cuda.compile_ptx(use_nanosleep, (types.uint32,))
    assert ptx.count("nanosleep.u32") == 2


def test_nanosleep_kernel():
    """Test nanosleep works in kernel execution."""

    @cuda.jit
    def kernel(arr):
        i = cuda.grid(1)
        if i < arr.size:
            cuda.nanosleep(32)
            arr[i] = i

    arr = cuda.device_array(10, dtype=np.int32)
    kernel[1, 10](arr)
    np.testing.assert_array_equal(arr.copy_to_host(), np.arange(10, dtype=np.int32))


@pytest.mark.parametrize(
    "intrinsic,input_val,expected",
    [
        # Use 64-bit values since numba widens scalar args to int64
        ("brev", np.uint64(0x80000000), np.uint64(0x0000000100000000)),
        ("clz", np.int64(0x00100000), 43),  # 64-bit clz
        ("popc", np.int64(0b11010111), 6),
        ("ffs", np.int64(0b11010100), 3),
        ("ffs", np.int64(0), 0),
    ],
)
def test_bit_intrinsics(intrinsic, input_val, expected):
    """Test bit manipulation intrinsics (brev, clz, popc, ffs)."""
    kernel_code = f"""
@cuda.jit
def kernel(arr, x):
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] = cuda.{intrinsic}(x)
"""
    local_ns = {"cuda": cuda}
    exec(kernel_code, local_ns)
    kernel = local_ns["kernel"]

    # Use int64 for result storage since intrinsics may return 64-bit
    arr = cuda.device_array(1, dtype=np.int64)
    kernel[1, 1](arr, input_val)
    assert arr.copy_to_host()[0] == expected


def test_chip_forward_compat():
    """Test targeting a lower chip than the current device."""
    from numba_cuda_mlir.numba_cuda.cudadrv import nvrtc

    current_cc = cuda.get_current_device().compute_capability
    supported_ccs = set(nvrtc.get_supported_ccs())
    target_cc = next(
        (cc for cc in [(7, 5), (7, 0), (6, 0), (5, 0)] if cc < current_cc and cc in supported_ccs),
        None,
    )
    if target_cc is None:
        supported = ", ".join(f"sm_{cc[0]}{cc[1]}" for cc in sorted(supported_ccs))
        pytest.skip(
            f"no supported lower chip target available for sm_{current_cc[0]}{current_cc[1]}; "
            f"supported targets: {supported}"
        )

    @cuda.jit(chip=f"sm_{target_cc[0]}{target_cc[1]}")
    def kernel(arr):
        arr[0] = 42

    arr = np.zeros(1, dtype=np.int32)
    kernel[1, 1](arr)
    assert arr[0] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
