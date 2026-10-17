# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""np.left_shift and np.right_shift on scalars and on arrays with an output array.

As in NumPy, shifting by the bit width or more, or by a negative count, gives 0, or -1
for a right shift of a negative value.
"""

import numpy as np
import pytest
from numba_cuda_mlir import cuda

SHIFTS = [np.left_shift, np.right_shift]


def shift_operands(dtype):
    """A few values of dtype, each shifted by counts below, at and above its bit width.

    The counts out of range include some whose low half is in range, like 2**32 + 1
    and -2**63 for 64-bit types.
    """
    info = np.iinfo(dtype)
    half = 1 << (info.bits // 2)
    values = [0, 1, 5, info.max, info.max // 3]
    counts = [0, 1, info.bits - 1, info.bits, info.bits + 1, half, half + 1, info.max]
    if info.min < 0:
        values += [-1, -8, info.min]
        counts += [-1, -info.bits, -half, info.min]
    else:
        counts += [info.max // 2 + 1]
    x = np.repeat(np.array(values, dtype=dtype), len(counts))
    n = np.tile(np.array(counts, dtype=dtype), len(values))
    return x, n


OPERANDS = {
    dtype.__name__: shift_operands(dtype)
    for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64)
}
# Mixed types shift in their promoted type: int64, int16, int32 and int64 here
OPERANDS["uint8-int64"] = (
    np.array([1, 255, 128, 255, 1, 7, 1, 255], dtype=np.uint8),
    np.array([3, 8, 56, 63, 64, -1, 2**32 + 1, -(2**63)], dtype=np.int64),
)
OPERANDS["int8-uint8"] = (
    np.array([-1, -128, 127, 1, -1, 3], dtype=np.int8),
    np.array([3, 8, 9, 15, 16, 255], dtype=np.uint8),
)
OPERANDS["int32-int8"] = (
    np.array([-7, 2**31 - 1, -(2**31), 1, -1, 12345], dtype=np.int32),
    np.array([1, 31, 31, 32, -1, -128], dtype=np.int8),
)
OPERANDS["uint32-int32"] = (
    np.array([1, 2**32 - 1, 2**32 - 1, 5, 7, 3], dtype=np.uint32),
    np.array([31, 32, 63, -1, 64, -(2**31)], dtype=np.int32),
)


def run_scalars(ufunc, x, n, out):
    @cuda.jit
    def kernel(x, n, out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = ufunc(x[i], n[i])

    kernel[1, out.size](x, n, out)


def run_scalars_to_array(ufunc, x, n, out):
    @cuda.jit
    def kernel(x, n, out):
        i = cuda.grid(1)
        if i < out.size:
            ufunc(x[i], n[i], out[i:])

    kernel[1, out.size](x, n, out)


def run_arrays(ufunc, x, n, out):
    @cuda.jit
    def kernel(x, n, out):
        ufunc(x, n, out)

    kernel[1, 1](x, n, out)


@pytest.mark.parametrize(
    "run",
    [run_scalars, run_scalars_to_array, run_arrays],
    ids=["scalars", "scalars-to-array", "arrays"],
)
@pytest.mark.parametrize("ufunc", SHIFTS, ids=lambda u: u.__name__)
@pytest.mark.parametrize("x, n", OPERANDS.values(), ids=OPERANDS.keys())
def test_shift(run, ufunc, x, n):
    expected = ufunc(x, n)
    out = np.zeros_like(expected)
    run(ufunc, x, n, out)
    np.testing.assert_array_equal(out, expected)
