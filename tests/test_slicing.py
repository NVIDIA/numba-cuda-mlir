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


FROZEN_SLICE_CASES = (
    ("empty_tail", slice(4, 4)),
    ("empty_head", slice(0, 0)),
    ("head", slice(0, 2)),
    ("mid", slice(1, 3)),
    ("full", slice(0, 4)),
    ("no_stop", slice(2, None)),
    ("empty_no_stop", slice(4, None)),
    ("no_bounds", slice(None)),
    ("step", slice(0, 4, 2)),
    ("empty_step", slice(4, 4, 2)),
    ("overlong_step", slice(0, 4, 3)),
    ("negative_start", slice(-3, 3)),
    ("negative_stop", slice(1, -1)),
    ("clamped", slice(0, 9)),
    ("clamped_empty", slice(9, 9)),
    ("reversed", slice(None, None, -1)),
    ("reversed_bounded", slice(3, 1, -1)),
    ("reversed_step2", slice(None, None, -2)),
)


@pytest.mark.parametrize(
    "frozen", [c[1] for c in FROZEN_SLICE_CASES], ids=[c[0] for c in FROZEN_SLICE_CASES]
)
def test_frozen_slice_of_statically_shaped_array(frozen):
    n = 4

    @cuda.jit
    def k(out):
        scratch = cuda.local.array(n, dtype=np.float32)
        for i in range(n):
            scratch[i] = np.float32(i)
        view = scratch[frozen]
        out[0] = np.float32(view.shape[0])
        total = np.float32(0.0)
        for i in range(view.shape[0]):
            total += view[i]
        out[1] = total

    out = np.zeros(2, dtype=np.float32)
    k[1, 1](out)
    expected = np.arange(n, dtype=np.float32)[frozen]
    assert out[0] == expected.shape[0]
    assert out[1] == expected.sum()


def test_inline_constant_slice_of_statically_shaped_array():
    @cuda.jit
    def k(out):
        scratch = cuda.local.array(4, dtype=np.float32)
        for i in range(4):
            scratch[i] = np.float32(i)
        empty = scratch[4:4]
        out[0] = np.float32(empty.shape[0])
        stepped = scratch[0:4:2]
        total = np.float32(0.0)
        for i in range(stepped.shape[0]):
            total += stepped[i]
        out[1] = total

    out = np.zeros(2, dtype=np.float32)
    k[1, 1](out)
    assert out[0] == 0
    assert out[1] == np.arange(4, dtype=np.float32)[0:4:2].sum()


def test_frozen_slice_in_tuple_index():
    frozen = slice(1, 3)

    @cuda.jit
    def k(out):
        scratch = cuda.local.array((4, 2), dtype=np.float32)
        for i in range(4):
            for j in range(2):
                scratch[i, j] = np.float32(i * 2 + j)
        view = scratch[frozen, 1]
        out[0] = np.float32(view.shape[0])
        total = np.float32(0.0)
        for i in range(view.shape[0]):
            total += view[i]
        out[1] = total

    out = np.zeros(2, dtype=np.float32)
    k[1, 1](out)
    expected = (np.arange(8, dtype=np.float32).reshape(4, 2))[frozen, 1]
    assert out[0] == expected.shape[0]
    assert out[1] == expected.sum()


def test_frozen_slice_through_device_function():
    frozen = slice(4, 4)

    @cuda.jit(device=True)
    def view_len(arr):
        v = arr[frozen]
        return np.float32(v.shape[0])

    @cuda.jit
    def k(out):
        scratch = cuda.local.array(4, dtype=np.float32)
        for i in range(4):
            scratch[i] = np.float32(i)
        out[0] = view_len(scratch)

    out = np.zeros(1, dtype=np.float32)
    k[1, 1](out)
    assert out[0] == 0


def test_frozen_slice_setitem():
    frozen = slice(1, 3)

    @cuda.jit
    def k(out):
        scratch = cuda.local.array(4, dtype=np.float32)
        for i in range(4):
            scratch[i] = np.float32(0)
        scratch[frozen] = np.float32(5)
        for i in range(4):
            out[i] = scratch[i]

    out = np.zeros(4, dtype=np.float32)
    k[1, 1](out)
    expected = np.zeros(4, dtype=np.float32)
    expected[frozen] = 5
    assert np.array_equal(out, expected)


def test_inline_reversed_slices():
    @cuda.jit
    def k(out):
        scratch = cuda.local.array(4, dtype=np.float32)
        for i in range(4):
            scratch[i] = np.float32(i)
        rev = scratch[::-1]
        out[0] = np.float32(rev.shape[0])
        out[1] = rev[0]
        bounded = scratch[3:1:-1]
        out[2] = np.float32(bounded.shape[0])
        out[3] = bounded[1]
        stepped = scratch[::-2]
        out[4] = np.float32(stepped.shape[0])
        out[5] = stepped[1]

    out = np.zeros(6, dtype=np.float32)
    k[1, 1](out)
    ref = np.arange(4, dtype=np.float32)
    assert out[0] == 4 and out[1] == ref[::-1][0]
    assert out[2] == 2 and out[3] == ref[3:1:-1][1]
    assert out[4] == 2 and out[5] == ref[::-2][1]


def test_empty_slice_preserves_stride():
    @cuda.jit
    def k(out):
        scratch = cuda.local.array(4, dtype=np.float32)
        out[0] = scratch[4:4:2].strides[0]
        out[1] = scratch[1:1:-2].strides[0]

    out = np.zeros(2, dtype=np.int64)
    k[1, 1](out)
    assert np.array_equal(out, np.array([8, -8], dtype=np.int64))


RUNTIME_BOUNDS_CASES = (
    ("basic", 1, 3),
    ("negative_start", -3, 3),
    ("negative_stop", 1, -1),
    ("clamped", 0, 9),
    ("empty", 3, 1),
    ("empty_oob", 9, 9),
    ("negative_empty", -1, -3),
)


@pytest.mark.parametrize(
    "start,stop",
    [c[1:] for c in RUNTIME_BOUNDS_CASES],
    ids=[c[0] for c in RUNTIME_BOUNDS_CASES],
)
def test_runtime_slice_bounds(start, stop):
    @cuda.jit
    def k(arr, out, i, j):
        view = arr[i:j]
        out[0] = np.float32(view.shape[0])
        total = np.float32(0.0)
        for t in range(view.shape[0]):
            total += view[t]
        out[1] = total

    arr = np.arange(4, dtype=np.float32)
    out = np.zeros(2, dtype=np.float32)
    k[1, 1](arr, out, start, stop)
    expected = arr[start:stop]
    assert out[0] == expected.shape[0]
    assert out[1] == expected.sum()


@pytest.mark.parametrize(
    "step",
    [1, 2, 3, -1, -2, np.iinfo(np.intp).max, np.iinfo(np.intp).min],
    ids=lambda s: f"step{s}",
)
def test_runtime_slice_step(step):
    @cuda.jit
    def k(arr, out, s):
        view = arr[::s]
        out[0] = np.float32(view.shape[0])
        total = np.float32(0.0)
        for t in range(view.shape[0]):
            total += view[t]
        out[1] = total

    arr = np.arange(5, dtype=np.float32)
    out = np.zeros(2, dtype=np.float32)
    k[1, 1](arr, out, step)
    expected = arr[::step]
    assert out[0] == expected.shape[0]
    assert out[1] == expected.sum()


def test_constant_intp_min_step_on_dynamic_source():
    step = np.iinfo(np.intp).min

    @cuda.jit
    def k(arr, out):
        view = arr[::step]
        out[0] = view.shape[0]
        out[1] = view[0]

    arr = np.arange(5, dtype=np.int64)
    out = np.zeros(2, dtype=np.int64)
    k[1, 1](arr, out)
    assert np.array_equal(out, np.array([1, 4], dtype=np.int64))


def test_runtime_stepped_slice_inexact_length():
    @cuda.jit
    def k(arr, out, i, j):
        view = arr[i:j:2]
        out[0] = np.float32(view.shape[0])

    arr = np.arange(4, dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    k[1, 1](arr, out, 0, 3)
    assert out[0] == 2


def test_constant_zero_step_is_a_compile_error():
    @cuda.jit
    def k(arr, out):
        view = arr[::0]
        out[0] = np.float32(view.shape[0])

    arr = np.arange(4, dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        k[1, 1](arr, out)


def test_runtime_zero_step_raises_value_error():
    @cuda.jit(debug=True, opt=False)
    def k(arr, out, s):
        view = arr[::s]
        out[0] = np.float32(view.shape[0])

    arr = np.arange(4, dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    with pytest.raises(ValueError):
        k[1, 1](arr, out, 0)


def test_thread_index_derived_slice():
    @cuda.jit
    def k(arr, out):
        i = cuda.threadIdx.x
        view = arr[i:]
        out[i] = np.float32(view.shape[0])

    arr = np.arange(8, dtype=np.float32)
    out = np.zeros(8, dtype=np.float32)
    k[1, 8](arr, out)
    assert np.array_equal(out, np.arange(8, 0, -1, dtype=np.float32))


def test_overflowed_nonnegative_expression_bound():
    # Overflowed bounds wrap like any negative index.
    @cuda.jit
    def k(arr, out):
        start = arr.shape[0] * np.int64(1 << 62)
        view = arr[start:]
        out[0] = start
        out[1] = view.shape[0]

    arr = np.arange(3, dtype=np.int64)
    out = np.zeros(2, dtype=np.int64)
    k[1, 1](arr, out)
    assert out[0] == np.iinfo(np.int64).min // 2
    assert out[1] == 3


def test_dynamic_source_negative_step_unchanged():
    @cuda.jit
    def k(arr, out):
        view = arr[3:1:-1]
        out[0] = np.float32(view.shape[0])
        out[1] = view[0]
        out[2] = view[1]

    arr = np.arange(4, dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    k[1, 1](arr, out)
    assert out[0] == 2
    assert out[1] == 3
    assert out[2] == 2


def test_static_partial_index():
    @cuda.jit
    def k(out):
        scratch = cuda.local.array((4, 2), dtype=np.float32)
        for i in range(4):
            for j in range(2):
                scratch[i, j] = np.float32(i * 2 + j)
        row = scratch[2]
        out[0] = row[0]
        out[1] = row[1]

    out = np.zeros(2, dtype=np.float32)
    k[1, 1](out)
    assert out[0] == 4
    assert out[1] == 5


def test_slice_lowers_to_strided_view():
    @cuda.jit
    def k(arr, out, s):
        view = arr[s:]
        out[0] = view[0]

    arr = np.arange(4, dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    k[1, 1](arr, out, 1)
    assert out[0] == 1

    # CHECK-LABEL: gpu.func
    # CHECK: memref.extract_strided_metadata
    # CHECK: memref.reinterpret_cast
    cres = compiler.compile_for(k, arr, out, 1)
    mlir = cres.mlir_module_str
    testing.filecheck_with_comments(mlir)


if __name__ == "__main__":
    test_incomplete_slice(*INCOMPLETE_SLICE_CASES[0])
    test_slice_axis0_offset_2d(3)
    test_slice_axis0_offset_3d_per_block()
