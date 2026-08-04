# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from numba_cuda_mlir import cuda


def test_host_array_capture_uses_constant_memory():
    table = np.arange(64, dtype=np.int64)

    @cuda.jit(cache=False)
    def kernel(out):
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = table[i % table.shape[0]]

    out = cuda.device_array(128, dtype=np.int64)
    for _ in range(150):
        kernel[1, 128](out)
    cuda.synchronize()

    np.testing.assert_array_equal(out.copy_to_host()[:64], table)
    ptx = "\n".join(kernel.inspect_asm().values())
    assert ".const" in ptx
    assert "malloc" not in ptx


def test_host_array_capture_materializes_noncontiguous_input():
    table = np.arange(12, dtype=np.int32)[::2]

    @cuda.jit
    def kernel(out):
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = table[i]

    out = cuda.device_array(table.shape, dtype=table.dtype)
    kernel[1, table.size](out)

    np.testing.assert_array_equal(out.copy_to_host(), table)
