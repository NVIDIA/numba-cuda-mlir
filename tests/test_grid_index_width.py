# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that ``cuda.grid`` and ``cuda.gridsize`` are 64-bit.

A one-dimensional launch may legally contain up to ``(2**31 - 1) * 1024``
threads, so a 32-bit global thread index wraps to a negative value for threads
beyond ``2**31``.
"""

import numpy as np
import pytest

from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.compiler import compile_mlir, compile_ptx

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")

# A launch that is 32768 threads wider than 2**31.
BIG_LAUNCH_BLOCKS = 67109888
BIG_LAUNCH_THREADS = 32
BIG_LAUNCH_TOTAL = BIG_LAUNCH_BLOCKS * BIG_LAUNCH_THREADS


def _big_launch_skip_reason():
    """Return why this device cannot run the >2**31 thread launch, or None."""
    if not cuda.is_available():
        return "CUDA GPU required"
    device = cuda.get_current_device()
    max_blocks = device.MAX_GRID_DIM_X
    max_threads = device.MAX_THREADS_PER_BLOCK
    if max_blocks < BIG_LAUNCH_BLOCKS or max_threads < BIG_LAUNCH_THREADS:
        return (
            f"device supports at most {max_blocks} blocks of {max_threads} threads, "
            f"which cannot express a {BIG_LAUNCH_TOTAL} thread launch"
        )
    return None


_BIG_LAUNCH_SKIP_REASON = _big_launch_skip_reason()

requires_big_launch = pytest.mark.skipif(
    _BIG_LAUNCH_SKIP_REASON is not None,
    reason=_BIG_LAUNCH_SKIP_REASON or "",
)


def grid_1d():
    return cuda.grid(1)


def grid_2d():
    return cuda.grid(2)


def grid_3d():
    return cuda.grid(3)


def gridsize_1d():
    return cuda.gridsize(1)


def gridsize_2d():
    return cuda.gridsize(2)


def gridsize_3d():
    return cuda.gridsize(3)


@pytest.mark.parametrize(
    "device_func, expected",
    [
        (grid_1d, types.int64),
        (grid_2d, types.UniTuple(types.int64, 2)),
        (grid_3d, types.UniTuple(types.int64, 3)),
        (gridsize_1d, types.int64),
        (gridsize_2d, types.UniTuple(types.int64, 2)),
        (gridsize_3d, types.UniTuple(types.int64, 3)),
    ],
    ids=["grid1", "grid2", "grid3", "gridsize1", "gridsize2", "gridsize3"],
)
def test_declared_return_type_is_64_bit(device_func, expected):
    _, return_type = compile_ptx(device_func, (), device=True)
    assert return_type == expected


def test_grid_is_not_truncated_in_mlir():
    """The global thread index must never be narrowed to 32 bits."""

    def kernel(out):
        out[0] = cuda.grid(1)

    mlir = str(compile_mlir(kernel, "void(int64[::1])"))
    assert "index to i32" not in mlir


def test_gridsize_is_not_truncated_in_mlir():
    def kernel(out):
        out[0] = cuda.gridsize(1)

    mlir = str(compile_mlir(kernel, "void(int64[::1])"))
    assert "index to i32" not in mlir


@requires_big_launch
def test_large_launch_grid_is_never_negative():
    @cuda.jit("void(int64[::1])")
    def count_negative_indices(counter):
        if cuda.grid(1) < 0:
            cuda.atomic.add(counter, 0, 1)

    counter = cuda.to_device(np.zeros(1, dtype=np.int64))
    count_negative_indices[BIG_LAUNCH_BLOCKS, BIG_LAUNCH_THREADS](counter)
    cuda.synchronize()

    assert counter.copy_to_host()[0] == 0


@requires_big_launch
def test_large_launch_gridsize_is_the_true_thread_count():
    @cuda.jit("void(int64[::1])")
    def report_gridsize(out):
        if cuda.grid(1) == 0:
            out[0] = cuda.gridsize(1)

    out = cuda.to_device(np.zeros(1, dtype=np.int64))
    report_gridsize[BIG_LAUNCH_BLOCKS, BIG_LAUNCH_THREADS](out)
    cuda.synchronize()

    assert out.copy_to_host()[0] == BIG_LAUNCH_TOTAL


def test_small_launch_grid_1d():
    @cuda.jit("void(int64[::1])")
    def write_grid(out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = i

    blocks, threads = 7, 3
    out = np.full(blocks * threads, -1, dtype=np.int64)
    write_grid[blocks, threads](out)

    np.testing.assert_array_equal(out, np.arange(blocks * threads))


def test_small_launch_grid_2d():
    @cuda.jit("void(int64[:, ::1], int64[:, ::1])")
    def write_grid(out_x, out_y):
        i, j = cuda.grid(2)
        out_x[i, j] = i
        out_y[i, j] = j

    blocks, threads = (5, 6), (4, 3)
    shape = (blocks[0] * threads[0], blocks[1] * threads[1])
    out_x = np.full(shape, -1, dtype=np.int64)
    out_y = np.full(shape, -1, dtype=np.int64)
    write_grid[blocks, threads](out_x, out_y)

    expected_x, expected_y = np.indices(shape)
    np.testing.assert_array_equal(out_x, expected_x)
    np.testing.assert_array_equal(out_y, expected_y)


def test_small_launch_grid_3d():
    @cuda.jit("void(int64[:, :, ::1])")
    def write_grid(out):
        i, j, k = cuda.grid(3)
        out[i, j, k] = i * 100 + j * 10 + k

    blocks, threads = (2, 3, 4), (3, 2, 1)
    shape = tuple(b * t for b, t in zip(blocks, threads))
    out = np.full(shape, -1, dtype=np.int64)
    write_grid[blocks, threads](out)

    i, j, k = np.indices(shape)
    np.testing.assert_array_equal(out, i * 100 + j * 10 + k)


def test_small_launch_gridsize():
    @cuda.jit("void(int64[::1])")
    def write_gridsize(out):
        out[0] = cuda.gridsize(1)
        out[1], out[2] = cuda.gridsize(2)
        out[3], out[4], out[5] = cuda.gridsize(3)

    blocks, threads = (5, 6, 7), (4, 3, 2)
    out = np.zeros(6, dtype=np.int64)
    write_gridsize[blocks, threads](out)

    expected = [b * t for b, t in zip(blocks, threads)]
    np.testing.assert_array_equal(out, [expected[0]] + expected[:2] + expected)
