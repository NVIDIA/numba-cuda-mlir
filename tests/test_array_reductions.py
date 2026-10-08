# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from numba_cuda_mlir import cuda
import logging

CASES = (
    np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5]),
    np.float64([-0.0, -1.5]),
    np.float64([-1.5, 2.5, float("inf")]),
    np.float64([-1.5, 2.5, -float("inf")]),
    np.float64([-1.5, 2.5, float("inf"), -float("inf")]),
    np.float64([np.nan, -1.5, 2.5, np.nan, 3.0, -0.0]),
    np.float64([np.nan, -1.5, 2.5, np.nan, float("inf"), -float("inf"), 3.0, 0.0]),
    np.float64([5.0, np.nan, -1.5, np.nan]),
    np.float64([np.nan, np.nan]),
)


def close_or_both_nan(a, b):
    return np.allclose(a, b) or (np.isnan(a) and np.isnan(b))


@pytest.mark.parametrize("func", [np.all, np.any, np.sum, np.mean, np.var])
@pytest.mark.parametrize("case", CASES)
def test_basic(func, case):
    expected = func(case)

    @cuda.jit(dump=True)
    def kernel(out, case):
        out[0] = func(case)

    out = cuda.to_device(np.zeros(1, dtype=case.dtype))
    case = cuda.to_device(case)
    kernel[1, 1](out, case)
    out = out.copy_to_host()
    assert close_or_both_nan(out, expected), f"{func.__name__}({case}) = {out} != {expected}"


@pytest.mark.parametrize(
    "func",
    [
        np.min,
        np.max,
        np.nanmin,
        np.nanmax,
        np.nanmean,
        np.nansum,
        np.nanprod,
    ],
)
@pytest.mark.parametrize("case", CASES)
def test_nan_reductions(func, case):
    # Skip all-NaN case for nan* functions - edge case with different behavior
    if np.all(np.isnan(case)) and func.__name__.startswith("nan"):
        pytest.skip("All-NaN case not supported for nan* functions")

    expected = func(case)

    @cuda.jit(opt_level=3, fastmath=True)
    def kernel(out, case):
        out[0] = func(case)

    out = cuda.to_device(np.zeros(1, dtype=case.dtype))
    case = cuda.to_device(case)
    kernel[1, 1](out, case)
    out = out.copy_to_host()
    assert close_or_both_nan(out, expected), f"{func.__name__}({case}) = {out} != {expected}"


@pytest.mark.parametrize("func", [np.argmin, np.argmax])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float32, np.float64, np.bool_])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "slice"])
@pytest.mark.parametrize("method", [False, True])
def test_arg_reductions(func, dtype, layout, method):
    case = np.array([[5, 3, 1], [4, 1, 5]], dtype=dtype, order="F" if layout == "F" else "C")
    if layout == "strided":
        case = case[:, ::-1]
    expected = func(case[1:, 1:] if layout == "slice" else case)
    is_argmin = func == np.argmin

    @cuda.jit
    def kernel(out, arr):
        if layout == "slice":
            arr = arr[1:, 1:]
        if method:
            if is_argmin:
                out[0] = arr.argmin()
            else:
                out[0] = arr.argmax()
        else:
            out[0] = func(arr)

    out = cuda.device_array(1, dtype=np.int64)
    # Device transfers require a contiguous host array; reverse on device below.
    if layout == "strided":
        arr = cuda.to_device(case[:, ::-1].copy())[:, ::-1]
    else:
        arr = cuda.to_device(case)
    kernel[1, 1](out, arr)
    assert out.copy_to_host()[0] == expected


@pytest.mark.parametrize("func", [np.argmin, np.argmax])
@pytest.mark.parametrize("case", [*CASES, np.array([5, 3, 1, 4, 2]), np.array([7])])
def test_arg_reductions_values(func, case):
    @cuda.jit
    def kernel(out, arr):
        out[0] = func(arr)

    out = cuda.device_array(1, dtype=np.int64)
    kernel[1, 1](out, cuda.to_device(case))
    assert out.copy_to_host()[0] == func(case)


@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (2, 3)])
def test_flat_iteration(shape):
    case = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)

    @cuda.jit
    def kernel(out, arr):
        total = 0
        count = 0
        for value in arr.flat:
            total += value
            count += 1
        out[0] = total
        out[1] = count

    out = cuda.device_array(2, dtype=np.int64)
    kernel[1, 1](out, cuda.to_device(case))
    np.testing.assert_array_equal(out.copy_to_host(), [case.sum(), case.size])


@pytest.mark.parametrize("func", [np.argmin, np.argmax])
def test_arg_reductions_empty(func):
    @cuda.jit(debug=True, opt=False)
    def kernel(out, arr):
        out[0] = func(arr)

    out = cuda.device_array(1, dtype=np.int64)
    with pytest.raises(ValueError, match="Invalid value in kernel"):
        kernel[1, 1](out, cuda.to_device(np.empty(0, dtype=np.int64)))
        cuda.synchronize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_basic(np.mean, CASES[0])
