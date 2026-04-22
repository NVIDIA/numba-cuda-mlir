# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cuda.simt as cs
import numpy as np
import pytest

literal_unroll = cs.misc.special.literal_unroll
literally = cs.misc.special.literally


def test_literal_unroll():
    @cs.jit
    def k(x: cs.DeviceNDArray):
        for i in literal_unroll(range(10)):
            x[i] = i

    x = cs.to_device(np.zeros(10, dtype=np.int32))
    k[1, 1](x)
    x = x.copy_to_host()
    assert np.all(x == np.arange(10))


@pytest.mark.xfail
def test_literal_eval():
    arr = np.float32([1, 2, 3])

    @cs.jit
    def k(x: cs.DeviceNDArray):
        arr = literally(arr)
        for i in range(3):
            x[i] = arr[i]

    x = cs.to_device(np.zeros(3, dtype=np.float32))
    k[1, 1](x)
    x = x.copy_to_host()
    assert np.all(x == np.float32([1, 2, 3]))
