# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re

import numpy as np
import pytest
from numba_cuda_mlir import compiler, cuda, tools, types
from numba_cuda_mlir.types import float32


def test_dynamic_shared_after_control_flow():
    """shared.array(0) lowers after a branch without a compiler crash."""

    @cuda.jit
    def k(inp, out):
        if cuda.threadIdx.x != 0 or cuda.blockIdx.x != 0:
            return
        shared = cuda.shared.array(0, dtype=float32)
        for i in range(inp.size):
            shared[i] = inp[i]
        for i in range(out.size):
            out[i] = shared[i]

    inp = np.array([7.0, 8.0], dtype=np.float32)
    out = cuda.to_device(np.zeros(2, dtype=np.float32))
    k[1, 1, 0, 8](cuda.to_device(inp), out)
    np.testing.assert_allclose(out.copy_to_host(), inp)


def test_runtime_shaped_shared_after_control_flow():
    """Runtime-shaped shared arrays lower after a branch without a
    compiler crash."""

    @cuda.jit
    def k(n_arr, inp, out):
        if cuda.threadIdx.x != 0 or cuda.blockIdx.x != 0:
            return
        n = n_arr[0]
        shared = cuda.shared.array(n, dtype=float32)
        for i in range(inp.size):
            shared[i] = inp[i]
        for i in range(out.size):
            out[i] = shared[i]

    n_arr = np.array([2], dtype=np.int32)
    inp = np.array([9.0, 10.0], dtype=np.float32)
    out = cuda.to_device(np.zeros(2, dtype=np.float32))
    k[1, 1, 0, 8](cuda.to_device(n_arr), cuda.to_device(inp), out)
    np.testing.assert_allclose(out.copy_to_host(), inp)


def test_dynamic_shared_after_conditional_runtime_shaped_shared():
    """Multiple shared arrays across control flow keep offset values valid."""

    @cuda.jit
    def k(flag, n_arr, out):
        if flag[0] != 0:
            n = n_arr[0]
            scratch = cuda.shared.array(n, dtype=float32)
            scratch[0] = 13.0

        shared = cuda.shared.array(0, dtype=float32)
        shared[0] = 21.0

        if flag[0] != 0:
            out[0] = scratch[0]
        out[1] = shared[0]

    flag = np.array([1], dtype=np.int32)
    n_arr = np.array([2], dtype=np.int32)
    out = cuda.to_device(np.zeros(2, dtype=np.float32))
    k[1, 1, 0, 16](cuda.to_device(flag), cuda.to_device(n_arr), out)
    np.testing.assert_allclose(out.copy_to_host(), np.array([13.0, 21.0], dtype=np.float32))


def _mixed_shared_kernel(dtype, static_first):
    @cuda.jit
    def kernel(out, extent):
        if static_first:
            fixed = cuda.shared.array(16, dtype=dtype)
            dynamic = cuda.shared.array(0, dtype=dtype)
        else:
            dynamic = cuda.shared.array(0, dtype=dtype)
            fixed = cuda.shared.array(16, dtype=dtype)
        t = cuda.threadIdx.x
        fixed[t] = 10 + t
        dynamic[t] = 100 + t
        cuda.syncthreads()
        out[t] = fixed[(t + 1) % 16] + dynamic[t]
        if t == 0:
            extent[0] = dynamic.size

    return kernel


@pytest.mark.parametrize("static_first", [True, False])
@pytest.mark.parametrize("dtype", [np.uint8, np.int32, np.float64])
def test_static_and_dynamic_shared_are_disjoint(dtype, static_first):
    kernel = _mixed_shared_kernel(dtype, static_first)
    out = cuda.device_array(16, dtype=np.int64)
    extent = cuda.device_array(1, dtype=np.int64)
    kernel[1, 16, 0, 32 * np.dtype(dtype).itemsize](out, extent)
    t = np.arange(16)
    np.testing.assert_array_equal(out.copy_to_host(), 110 + (t + 1) % 16 + t)
    np.testing.assert_array_equal(extent.copy_to_host(), [32])


@pytest.mark.parametrize("cc", [(9, 0), (12, 0)])
@pytest.mark.parametrize("static_first", [True, False])
def test_dynamic_shared_ptx_uses_external_region_and_launch_extent(monkeypatch, cc, static_first):
    # A target snapshot suffices for compilation; no CUDA context is needed.
    monkeypatch.setattr(
        tools,
        "get_gpu_compute_capability",
        lambda as_type=str: cc if as_type is tuple else f"sm_{cc[0]}{cc[1]}",
    )
    kernel = _mixed_shared_kernel(np.int32, static_first)
    ptx, _ = compiler.compile_ptx(kernel, types.void(types.int64[::1], types.int64[::1]), cc=cc)
    assert re.search(
        r"\.extern \.shared \.align 8 \.b8 __numba_cuda_mlir_dynamic_shared_\w+\[\];", ptx
    )
    assert re.search(r"\.shared \.align 8 \.b8 static_shared_memory_\d+\[64\];", ptx)
    assert "%dynamic_smem_size" in ptx
    assert "st.shared" in ptx
    assert "ld.shared" in ptx


@pytest.mark.parametrize("cc", [(9, 0), (12, 0)])
def test_explicit_alignment_reaches_ptx_and_optimized_ir(monkeypatch, cc):
    monkeypatch.setattr(
        tools,
        "get_gpu_compute_capability",
        lambda as_type=str: cc if as_type is tuple else f"sm_{cc[0]}{cc[1]}",
    )

    @cuda.jit
    def kernel(n, out):
        prefix = cuda.shared.array((n[0], n[1]), dtype=np.int32, alignment=16)
        tail = cuda.shared.array(0, dtype=np.float64)
        prefix[0, 0] = 7
        tail[0] = 21.0
        out[0] = prefix[0, 0]
        out[1] = tail[0]

    sig = types.void(types.int64[::1], types.int64[::1])
    # The strongest requested alignment is applied to the external symbol.
    ptx, _ = compiler.compile_ptx(kernel, sig, cc=cc)
    assert re.search(
        r"\.extern \.shared \.align 16 \.b8 __numba_cuda_mlir_dynamic_shared_\w+\[\];", ptx
    )
    # The runtime-sized array's alignment assumption must survive optimization
    # instead of being erased as a dead op with an unused result.
    optimized = compiler.compile_mlir(kernel, sig, optimized=True)
    assert re.search(r'llvm\.intr\.assume .*\["align"\(', optimized)


def test_runtime_allocations_align_each_window():
    @cuda.jit
    def kernel(n, out):
        prefix = cuda.shared.array(n[0], dtype=np.uint8)
        middle = cuda.shared.array((n[0], n[1]), dtype=np.int32, alignment=16)
        tail = cuda.shared.array(0, dtype=np.float64, alignment=32)
        prefix[0] = 7
        middle[0, 0] = 13
        tail[0] = 21
        out[0] = prefix[0]
        out[1] = middle[0, 0]
        out[2] = tail[0]
        out[3] = tail.size

    out = cuda.device_array(4, dtype=np.int64)
    # [0,3), padding, [16,40), padding, [64,96): four float64s.
    kernel[1, 1, 0, 96](cuda.to_device(np.array([3, 2], dtype=np.int64)), out)
    np.testing.assert_array_equal(out.copy_to_host(), [7, 13, 21, 4])


@pytest.mark.parametrize("prefix_size,expected", [(3, 3), (24, 1), (31, 0), (32, 0), (40, 0)])
def test_remaining_extent_clamps_after_runtime_allocation(prefix_size, expected):
    @cuda.jit
    def kernel(n, out):
        prefix = cuda.shared.array(n[0], dtype=np.uint8)
        tail = cuda.shared.array(0, dtype=np.int64)
        again = cuda.shared.array(0, dtype=np.int64)
        # Only inspect shapes: an oversized prefix must not be dereferenced.
        out[0] = prefix.size
        out[1] = tail.size
        out[2] = again.size

    out = cuda.device_array(3, dtype=np.int64)
    kernel[1, 1, 0, 32](cuda.to_device(np.array([prefix_size], dtype=np.int64)), out)
    np.testing.assert_array_equal(out.copy_to_host(), [prefix_size, expected, 0])


@pytest.mark.parametrize("take_branch", [0, 1])
def test_remaining_extent_after_conditional_allocation(take_branch):
    @cuda.jit
    def kernel(flag, n, out):
        if flag[0]:
            prefix = cuda.shared.array(n[0], dtype=np.uint8)
            prefix[0] = 13
        tail = cuda.shared.array(0, dtype=np.int32)
        tail[0] = 21
        out[0] = tail.size
        out[1] = tail[0]
        if flag[0]:
            out[2] = prefix[0]
        else:
            out[2] = 0

    out = cuda.device_array(3, dtype=np.int64)
    flag = cuda.to_device(np.array([take_branch], dtype=np.int64))
    n = cuda.to_device(np.array([3], dtype=np.int64))
    kernel[1, 1, 0, 32](flag, n, out)
    np.testing.assert_array_equal(
        out.copy_to_host(), [6 if take_branch else 8, 21, 13 * take_branch]
    )
