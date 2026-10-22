# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from numba_cuda_mlir import cuda


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
)
def test_round_integer(dtype):
    @cuda.jit
    def kernel(values, out):
        i = cuda.grid(1)
        if i < values.size:
            out[i] = round(values[i])

    limits = np.iinfo(dtype)
    values = [limits.min, 0, 1, 7, limits.max]
    if limits.min < 0:
        values.extend([-7, -1])
    if limits.bits == 64:
        values.append(2**53 + 1)
    values = np.array(values, dtype=dtype)
    out = np.zeros(values.size, dtype=np.uint64 if dtype == np.uint64 else np.int64)
    kernel[1, 32](values, out)
    np.testing.assert_array_equal(out, values)
