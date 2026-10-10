# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLIR owners for NumPy operations previously backed by vendored intrinsics."""

import numpy as np
import pytest

from numba_cuda_mlir import compiler, cuda, tools, types


def round_kernel(x, out):
    out[0] = np.round(x[0], 1)


def expand_dims_kernel(x, out):
    out[0] = np.expand_dims(x, -1)[2, 0]


def concatenate_kernel(x, out):
    out[0] = np.concatenate((x, x), axis=0)[x.size]


def stack_kernel(x, out):
    out[0] = np.stack((x, x), axis=-1)[2, 1]


def column_stack_kernel(x, out):
    out[0] = np.column_stack((x, x))[2, 1]


def hstack_kernel(x, out):
    out[0] = np.hstack((x, x))[x.size]


def vstack_kernel(x, out):
    out[0] = np.vstack((x, x))[1, 2]


def dstack_kernel(x, out):
    out[0] = np.dstack((x, x))[0, 2, 1]


def ascontiguousarray_kernel(x, out):
    out[0] = np.ascontiguousarray(x)[2]


def asfortranarray_kernel(x, out):
    out[0] = np.asfortranarray(x)[2]


def flip_kernel(x, out):
    out[0] = np.flip(x)[0]


def empty_like_kernel(x, out):
    result = np.empty_like(x)
    result[1] = 7.0
    out[0] = result[1]


CASES = [
    (round_kernel, 2.2),
    (expand_dims_kernel, 3.0),
    (concatenate_kernel, 2.25),
    (stack_kernel, 3.0),
    (column_stack_kernel, 3.0),
    (hstack_kernel, 2.25),
    (vstack_kernel, 3.0),
    (dstack_kernel, 3.0),
    (ascontiguousarray_kernel, 3.0),
    (asfortranarray_kernel, 3.0),
    (flip_kernel, 3.0),
    (empty_like_kernel, 7.0),
]


@pytest.mark.parametrize("fn, expected", CASES, ids=lambda x: getattr(x, "__name__", None))
def test_intrinsic_has_mlir_lowering_without_gpu(monkeypatch, fn, expected):
    monkeypatch.setattr(tools, "_cached_cc", (8, 0))
    array_type = types.Array(types.float32, 1, "C")
    compiler.compile(
        fn,
        types.void(array_type, array_type),
        device=True,
        abi="c",
        abi_info={"abi_name": f"numpy_intrinsic_{fn.__name__}"},
        output="ltoir",
        cc=(8, 0),
    )


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("fn, expected", CASES, ids=lambda x: getattr(x, "__name__", None))
def test_intrinsic_result_on_gpu(fn, expected):
    kernel = cuda.jit(fn)
    x = cuda.to_device(np.array([2.25, 2.0, 3.0], dtype=np.float32))
    out = cuda.to_device(np.zeros(1, dtype=np.float32))
    kernel[1, 1](x, out)
    np.testing.assert_allclose(out.copy_to_host()[0], expected, rtol=1e-6)


def f_layout_kernel(x, out):
    converted = np.asfortranarray(x)
    out[0] = converted[1, 0]


def f_stack_kernel(x, out):
    stacked = np.stack((x, x), axis=1)
    out[0] = stacked[1, 0, 0]


def f_empty_like_kernel(x, out):
    result = np.empty_like(x)
    result[1, 0] = 9.0
    out[0] = result[1, 0]


@pytest.mark.parametrize("fn", [f_layout_kernel, f_stack_kernel, f_empty_like_kernel])
def test_fortran_layout_intrinsic_compiles_without_gpu(monkeypatch, fn):
    monkeypatch.setattr(tools, "_cached_cc", (8, 0))
    input_type = types.Array(types.float32, 2, "F")
    output_type = types.Array(types.float32, 1, "C")
    compiler.compile(
        fn,
        types.void(input_type, output_type),
        device=True,
        abi="c",
        abi_info={"abi_name": f"numpy_intrinsic_{fn.__name__}"},
        output="ltoir",
        cc=(8, 0),
    )


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize(
    "fn, expected",
    [
        (f_layout_kernel, 3.0),
        (f_stack_kernel, 3.0),
        (f_empty_like_kernel, 9.0),
    ],
)
def test_fortran_layout_intrinsic_result_on_gpu(fn, expected):
    kernel = cuda.jit(fn)
    x = cuda.to_device(np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3)))
    out = cuda.to_device(np.zeros(1, dtype=np.float32))
    kernel[1, 1](x, out)
    np.testing.assert_allclose(out.copy_to_host()[0], expected)


def concatenate_2d_kernel(x, out):
    out[0] = np.concatenate((x, x), axis=1)[1, 3]


def stack_2d_kernel(x, out):
    out[0] = np.stack((x, x), axis=1)[1, 1, 2]


def expand_dims_2d_kernel(x, out):
    out[0] = np.expand_dims(x, 1)[1, 0, 2]


def flip_2d_kernel(x, out):
    out[0] = np.flip(x)[0, 0]


def column_stack_2d_kernel(x, out):
    out[0] = np.column_stack((x[:, 0], x))[1, 2]


MULTIDIM_CASES = [
    (concatenate_2d_kernel, 3.0),
    (stack_2d_kernel, 5.0),
    (expand_dims_2d_kernel, 5.0),
    (flip_2d_kernel, 5.0),
    (column_stack_2d_kernel, 4.0),
]


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("fn, expected", MULTIDIM_CASES)
def test_multidimensional_intrinsic_result_on_gpu(fn, expected):
    kernel = cuda.jit(fn)
    x = cuda.to_device(np.arange(6, dtype=np.float32).reshape(2, 3))
    out = cuda.to_device(np.zeros(1, dtype=np.float32))
    kernel[1, 1](x, out)
    np.testing.assert_allclose(out.copy_to_host()[0], expected)


def dynamic_concatenate_kernel(x, axis, out):
    result = np.concatenate((x, x), axis=axis)
    if axis == 1:
        out[0] = result[1, 3]
    else:
        out[0] = result[2, 0]


def dynamic_stack_kernel(x, axis, out):
    result = np.stack((x, x), axis=axis)
    if axis == 1:
        out[0] = result[1, 1, 2]
    else:
        out[0] = result[1, 2, 1]


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("axis, expected", [(1, 3.0), (0, 0.0)])
def test_dynamic_concatenate_axis_on_gpu(axis, expected):
    kernel = cuda.jit(dynamic_concatenate_kernel)
    x = cuda.to_device(np.arange(6, dtype=np.float32).reshape(2, 3))
    out = cuda.to_device(np.zeros(1, dtype=np.float32))
    kernel[1, 1](x, axis, out)
    np.testing.assert_allclose(out.copy_to_host()[0], expected)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("axis", [1, -1])
def test_dynamic_stack_axis_on_gpu(axis):
    kernel = cuda.jit(dynamic_stack_kernel)
    x = cuda.to_device(np.arange(6, dtype=np.float32).reshape(2, 3))
    out = cuda.to_device(np.zeros(1, dtype=np.float32))
    kernel[1, 1](x, axis, out)
    np.testing.assert_allclose(out.copy_to_host()[0], 5.0)


def concatenate_promoted_dtype_kernel(x, y, out):
    out[0] = np.concatenate((x, y))[len(x)]


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_concatenate_promotes_input_dtype_on_gpu():
    kernel = cuda.jit(concatenate_promoted_dtype_kernel)
    x = cuda.to_device(np.array([1, 2], dtype=np.float32))
    y = cuda.to_device(np.array([3, 4], dtype=np.float64))
    out = cuda.to_device(np.zeros(1, dtype=np.float64))
    kernel[1, 1](x, y, out)
    np.testing.assert_allclose(out.copy_to_host()[0], 3.0)


def concatenate_int_promoted_dtype_kernel(x, y, out):
    out[0] = np.concatenate((x, y))[len(x)]


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_concatenate_promotes_integer_dtype_on_gpu():
    kernel = cuda.jit(concatenate_int_promoted_dtype_kernel)
    x = cuda.to_device(np.array([1, 2], dtype=np.int32))
    y = cuda.to_device(np.array([3, 4], dtype=np.int64))
    out = cuda.to_device(np.zeros(1, dtype=np.int64))
    kernel[1, 1](x, y, out)
    np.testing.assert_equal(out.copy_to_host()[0], 3)


def stack_int_promoted_dtype_kernel(x, y, out):
    out[0] = np.stack((x, y), axis=1)[0, 1]


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_stack_promotes_integer_dtype_on_gpu():
    kernel = cuda.jit(stack_int_promoted_dtype_kernel)
    x = cuda.to_device(np.array([1, 2], dtype=np.int32))
    y = cuda.to_device(np.array([3, 4], dtype=np.int64))
    out = cuda.to_device(np.zeros(1, dtype=np.int64))
    kernel[1, 1](x, y, out)
    np.testing.assert_equal(out.copy_to_host()[0], 3)


def flip_2d_c_kernel(x, out):
    out[0] = np.flip(x)[0, 0]


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_flip_multidimensional_c_layout_on_gpu():
    kernel = cuda.jit(flip_2d_c_kernel)
    x = cuda.to_device(np.arange(6, dtype=np.float32).reshape(2, 3))
    out = cuda.to_device(np.zeros(1, dtype=np.float32))
    kernel[1, 1](x, out)
    np.testing.assert_allclose(out.copy_to_host()[0], 5.0)
