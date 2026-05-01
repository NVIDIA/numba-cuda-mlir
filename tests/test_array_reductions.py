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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_basic(np.mean, CASES[0])
