# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from numba_cuda_mlir import cuda


@cuda.jit
def _increment(arr):
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] += 1


@pytest.mark.parametrize(
    "size,tpb", [(1000, 0), (256, 64)], ids=["default_tpb", "explicit_tpb"]
)
def test_forall_basic(size, tpb):
    arr = np.zeros(size, dtype=np.float32)
    dev_arr = cuda.to_device(arr)
    _increment.forall(dev_arr.size, tpb=tpb)(dev_arr)
    result = dev_arr.copy_to_host()
    np.testing.assert_array_equal(result, np.ones(size, dtype=np.float32))


def test_forall_no_work():
    arr = np.arange(11, dtype=np.float32)
    dev_arr = cuda.to_device(arr)
    _increment.forall(0)(dev_arr)
    result = dev_arr.copy_to_host()
    np.testing.assert_array_equal(result, np.arange(11, dtype=np.float32))


def test_forall_negative_raises():
    with pytest.raises(
        ValueError, match="Can't create ForAll with negative task count"
    ):
        _increment.forall(-1)


@pytest.mark.parametrize("n", [1, 7, 128, 1000, 100_000])
def test_forall_various_sizes(n):
    arr = np.zeros(n, dtype=np.int32)
    dev_arr = cuda.to_device(arr)
    _increment.forall(n)(dev_arr)
    result = dev_arr.copy_to_host()
    np.testing.assert_array_equal(result, np.ones(n, dtype=np.int32))
