# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir import compiler, testing
import numpy as np
import pytest


def test_numba_issue_9324():
    # https://github.com/numba/numba/issues/9324
    @cuda.jit(dump=True)
    def f_gpu(array):
        i = cuda.grid(1)

        array1 = array[:, 1]
        array0 = array[:, 0]

        if i < array.shape[0]:
            array1[i] += 1.0
            array1[i] = array0[i]

    array_cpu = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    f_gpu.forall(array_cpu.shape[0])(array_cpu)
    print(array_cpu)
    assert np.allclose(array_cpu, np.array([[1, 1, 3], [4, 4, 6]]))


INCOMPLETE_SLICE_CASES = (
    (
        [5, 2, 4],
        np.array(
            [
                [[1, 1, 1, 1], [1, 1, 1, 1]],
                [[1, 1, 1, 1], [1, 1, 1, 1]],
                [[1, 1, 1, 1], [1, 1, 1, 1]],
                [[5, 5, 5, 5], [5, 5, 5, 5]],
                [[1, 1, 1, 1], [1, 1, 1, 1]],
            ],
            dtype=np.int32,
        ),
    ),
    ([4, 1, 2], np.array([[[1, 1]], [[1, 1]], [[1, 1]], [[5, 5]]], dtype=np.int32)),
)


@pytest.mark.parametrize("shape,answer", INCOMPLETE_SLICE_CASES)
def test_incomplete_slice(shape, answer):
    shape = tuple(shape)

    @cuda.jit(dump=True, print_after_all=False)
    def k(array: cuda.DeviceNDArray):
        array[3] = 5

    h = np.ones(shape, dtype=np.int32)
    d = cuda.to_device(h)
    k[1, 1](d)
    assert np.allclose(d.copy_to_host(), answer), f"Expected {answer}, got {d.copy_to_host()}"

    # CHECK-LABEL: gpu.func
    # CHECK-SAME: (%[[ARG:.+]]: memref
    # CHECK-SAME: kernel
    # CHECK: scf.forall
    # CHECK: memref.store %{{.+}}, %[[ARG]]

    cres = compiler.compile_for(k, d)
    mlir = cres.mlir_module_str
    testing.filecheck_with_comments(mlir)


@pytest.mark.parametrize("start", [0, 1, 3])
def test_slice_axis0_offset_2d(start):
    rows, cols, window = 6, 4, 2

    @cuda.jit
    def k(src, dst, s):
        view = src[s:]
        for r in range(window):
            for c in range(cols):
                dst[r, c] = view[r, c]

    src = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    dst = cuda.to_device(np.zeros((window, cols), dtype=np.float64))
    k[1, 1](cuda.to_device(src), dst, start)
    np.testing.assert_array_equal(dst.copy_to_host(), src[start : start + window])


def test_slice_axis0_offset_3d_per_block():
    num_blocks, chunk, rows, cols = 4, 2, 3, 5

    @cuda.jit
    def k(src, dst):
        base = cuda.blockIdx.x * chunk
        view = src[base:]
        for m in range(chunk):
            for r in range(rows):
                for c in range(cols):
                    dst[base + m, r, c] = view[m, r, c]

    n = num_blocks * chunk
    src = np.arange(n * rows * cols, dtype=np.float64).reshape(n, rows, cols)
    dst = cuda.to_device(np.zeros_like(src))
    k[num_blocks, 1](cuda.to_device(src), dst)
    np.testing.assert_array_equal(dst.copy_to_host(), src)


if __name__ == "__main__":
    test_incomplete_slice(*INCOMPLETE_SLICE_CASES[0])
    test_slice_axis0_offset_2d(3)
    test_slice_axis0_offset_3d_per_block()
