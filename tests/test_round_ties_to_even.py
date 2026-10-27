# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-argument ``round()`` takes halfway cases to even.

MLIR's ``math.round`` takes them away from zero, so every ``.5`` input came
back off by one with no diagnostic.  Python, NumPy and numba-cuda all round to
even, which is ``math.roundeven``.
"""

import numpy as np
import pytest

from numba_cuda_mlir import cuda


HALFWAY = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
TO_EVEN = [-4.0, -2.0, -2.0, 0.0, 0.0, 2.0, 2.0, 4.0]


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f4", "f8"])
def test_halfway_values_round_to_even(dtype):
    """Every ``.5`` input lands on the nearest even integer, not away from zero."""

    @cuda.jit
    def k(a, out):
        i = cuda.grid(1)
        if i < a.size:
            out[i] = round(a[i])

    a = cuda.to_device(np.array(HALFWAY, dtype=dtype))
    out = cuda.to_device(np.zeros(len(HALFWAY), dtype=dtype))
    k[1, 32](a, out)
    cuda.synchronize()
    np.testing.assert_array_equal(out.copy_to_host(), np.array(TO_EVEN, dtype=dtype))


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f4", "f8"])
def test_non_halfway_values_round_to_nearest(dtype):
    """Rounding away from even is only for ties, so pin the ordinary cases too."""
    values = [-2.75, -2.25, -0.25, 0.25, 2.25, 2.75]
    expected = [-3.0, -2.0, -0.0, 0.0, 2.0, 3.0]

    @cuda.jit
    def k(a, out):
        i = cuda.grid(1)
        if i < a.size:
            out[i] = round(a[i])

    a = cuda.to_device(np.array(values, dtype=dtype))
    out = cuda.to_device(np.zeros(len(values), dtype=dtype))
    k[1, 32](a, out)
    cuda.synchronize()
    np.testing.assert_array_equal(out.copy_to_host(), np.array(expected, dtype=dtype))


def test_halfway_matches_python_round():
    """The kernel agrees with CPython's own ``round`` on the tie cases."""

    @cuda.jit
    def k(a, out):
        i = cuda.grid(1)
        if i < a.size:
            out[i] = round(a[i])

    a = cuda.to_device(np.array(HALFWAY, dtype=np.float64))
    out = cuda.to_device(np.zeros(len(HALFWAY), dtype=np.float64))
    k[1, 32](a, out)
    cuda.synchronize()
    np.testing.assert_array_equal(
        out.copy_to_host(), np.array([round(v) for v in HALFWAY], dtype=np.float64)
    )
