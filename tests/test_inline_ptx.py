# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from numba_cuda_mlir import cuda, tools
from numba_cuda_mlir.cuda import experimental

nanosleep = cuda.intrin.nanosleep
_cc = tools.get_gpu_compute_capability(tuple)
CHIP = _cc[0] * 10 + _cc[1]


@pytest.mark.skipif(
    condition=CHIP < 70,
    reason="nanosleep intrinsic requires sm_70 or higher compute capability",
)
def test_inline_ptx_outputs():
    """Single and multiple inline PTX results reach GPU output arrays."""

    @cuda.jit(opt_level=3)
    def kernel(a, b, out):
        i = cuda.threadIdx.x
        nanosleep(50)
        res = experimental.inline_ptx(
            """.reg .u32 t1;
                mul.lo.u32 t1, %1, %1;
                mul.lo.u32 %0, t1, %1;
                """,
            "=r",
            np.int32,
            "r",
            a[i],
        )

        # Multiple outputs share the same runtime inputs.
        res1, res2 = experimental.inline_ptx(
            """add.u32 %0, %2, %3;
                sub.u32 %1, %2, %3;
                """,
            "=r",
            np.int32,  # output 1: a + b
            "=r",
            np.int32,  # output 2: a - b
            "r",
            a[i],
            "r",
            b[i],
        )

        out[i, 0] = res
        out[i, 1] = res1
        out[i, 2] = res2

    a = np.array([-7, -1, 0, 2, 5], dtype=np.int32)
    b = np.array([3, -4, 6, 5, 3], dtype=np.int32)
    out = np.zeros((a.size, 3), dtype=np.int32)
    kernel[1, a.size](a, b, out)

    expected = np.column_stack((a * a * a, a + b, a - b))
    np.testing.assert_array_equal(out, expected)
