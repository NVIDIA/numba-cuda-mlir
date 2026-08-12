# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from numba_cuda_mlir import cuda


def test_host_array_capture_uses_constant_memory():
    table = np.arange(64, dtype=np.int64)

    @cuda.jit(cache=False)
    def kernel(out):
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = table[i % table.shape[0]]

    out = cuda.device_array(128, dtype=np.int64)
    for _ in range(150):
        kernel[1, 128](out)
    cuda.synchronize()

    np.testing.assert_array_equal(out.copy_to_host()[:64], table)
    ptx = "\n".join(kernel.inspect_asm().values())
    assert ".const" in ptx
    assert "malloc" not in ptx


def test_host_array_capture_materializes_noncontiguous_input():
    table = np.arange(12, dtype=np.int32)[::2]

    @cuda.jit
    def kernel(out):
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = table[i]

    out = cuda.device_array(table.shape, dtype=table.dtype)
    kernel[1, table.size](out)

    np.testing.assert_array_equal(out.copy_to_host(), table)


def test_host_array_capture_column_slice():
    table = np.arange(12, dtype=np.int64).reshape(3, 4)

    @cuda.jit
    def kernel(out):
        col = table[:, 2]
        for i in range(col.shape[0]):
            out[i] = col[i]

    out = cuda.device_array(3, dtype=table.dtype)
    kernel[1, 1](out)

    np.testing.assert_array_equal(out.copy_to_host(), table[:, 2])


def test_host_array_tuple_capture_column_slice():
    tables = (
        np.arange(12, dtype=np.int64).reshape(3, 4),
        np.arange(12, 24, dtype=np.int64).reshape(3, 4),
    )

    @cuda.jit
    def kernel(out, table_index):
        col = tables[table_index][:, 2]
        for i in range(col.shape[0]):
            out[i] = col[i]

    out = cuda.device_array(3, dtype=tables[1].dtype)
    kernel[1, 1](out, 1)

    np.testing.assert_array_equal(out.copy_to_host(), tables[1][:, 2])


def test_host_array_capture_f_layout_column_slice_and_reduction():
    table = np.asfortranarray(np.arange(12, dtype=np.int64).reshape(3, 4))

    @cuda.jit
    def kernel(out, any_out):
        col = table[:, 2]
        for i in range(col.shape[0]):
            out[i] = col[i]
        any_out[0] = np.any(table)

    out = cuda.device_array(table.shape[0], dtype=table.dtype)
    any_out = cuda.device_array(1, dtype=np.bool_)
    kernel[1, 1](out, any_out)

    np.testing.assert_array_equal(out.copy_to_host(), table[:, 2])
    np.testing.assert_array_equal(any_out.copy_to_host(), [np.any(table)])


def test_host_array_capture_ravel():
    table = np.arange(12, dtype=np.int64).reshape(3, 4)

    @cuda.jit
    def kernel(out):
        flat = table.ravel()
        for i in range(flat.shape[0]):
            out[i] = flat[i]

    out = cuda.device_array(table.size, dtype=table.dtype)
    kernel[1, 1](out)

    np.testing.assert_array_equal(out.copy_to_host(), table.ravel())


def test_host_array_capture_transpose():
    table = np.arange(12, dtype=np.int64).reshape(3, 4).T

    @cuda.jit
    def kernel(out):
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                out[i, j] = table[i, j]

    out = cuda.device_array(table.shape, dtype=table.dtype)
    kernel[1, 1](out)

    np.testing.assert_array_equal(out.copy_to_host(), table)


def test_host_array_capture_multidimensional_negative_stride():
    table = np.arange(12, dtype=np.int64).reshape(3, 4)[::-1, ::-1]

    @cuda.jit
    def kernel(out):
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                out[i, j] = table[i, j]

    out = cuda.device_array(table.shape, dtype=table.dtype)
    kernel[1, 1](out)

    np.testing.assert_array_equal(out.copy_to_host(), table)
