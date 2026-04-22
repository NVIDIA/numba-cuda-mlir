# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import cuda.simt as cuda


@pytest.mark.parametrize(
    "x, y, expected_z",
    [
        (5, 2, 2),  # break when i==y, z=y
        (5, 7, 8),  # no break, else clause: z = i*2 where i=4
    ],
)
def test_for_else_break(x, y, expected_z):
    """Variable assigned in both if-break and for-else branches.

    This creates a phi node where one incoming value may be
    ir.UNDEFINED (when the loop range is empty).
    """

    @cuda.jit
    def kernel(x, y, out):
        for i in range(x):
            if i == y:
                z = y
                break
        else:
            z = i * 2
        out[0] = z

    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](x, y, out)
    cuda.synchronize()
    if expected_z is not None:
        assert out[0] == expected_z


def test_conditional_variable_init():
    """Variable only assigned inside conditional branches."""

    @cuda.jit
    def kernel(flag, out):
        if flag:
            z = 10
        else:
            z = 20
        out[0] = z

    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](1, out)
    cuda.synchronize()
    assert out[0] == 10

    kernel[1, 1](0, out)
    cuda.synchronize()
    assert out[0] == 20
