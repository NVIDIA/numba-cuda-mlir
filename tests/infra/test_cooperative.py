# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cooperative kernel launch support."""

import cuda.simt as cuda
from cusimt.numba_cuda.cudadrv.devicearray import DeviceNDArray
import numpy as np
import tempfile


COOP_GRID_SYNC_SOURCE = """
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __device__ int coop_identity(int *ret, int val) {
    cg::grid_group grid = cg::this_grid();
    grid.sync();
    *ret = val;
    return 0;
}
"""


def test_cooperative_grid_sync_extern():
    """Cooperative launch via declare_device linking external CUDA C++."""
    with tempfile.NamedTemporaryFile(suffix=".cu", delete=False) as f:
        f.write(COOP_GRID_SYNC_SOURCE.encode())
        f.flush()
        coop_identity = cuda.declare_device(
            "coop_identity",
            "int32(int32)",
            link=f.name,
            use_cooperative=True,
        )

    @cuda.jit
    def kernel(out: DeviceNDArray):
        tid = cuda.grid(1)
        if tid < out.shape[0]:
            out[tid] = coop_identity(tid * 10)

    n = 32
    out = cuda.to_device(np.zeros(n, dtype=np.int32))
    kernel[1, n](out)
    result = out.copy_to_host()
    expected = np.arange(n, dtype=np.int32) * 10
    np.testing.assert_array_equal(result, expected)


def test_cooperative_grid_sync_python():
    """Cooperative launch using Python cuda.cg API."""

    @cuda.jit
    def kernel(out):
        g = cuda.cg.this_grid()
        g.sync()
        tid = cuda.grid(1)
        if tid < out.shape[0]:
            out[tid] = tid * 10

    n = 32
    from cuda.simt import types

    sig = (types.int32[::1],)
    out = cuda.to_device(np.zeros(n, dtype=np.int32))
    kernel[1, n](out)
    result = out.copy_to_host()
    expected = np.arange(n, dtype=np.int32) * 10
    np.testing.assert_array_equal(result, expected)

    overload = kernel.overloads[sig]
    assert overload.cooperative
    mb = overload.max_cooperative_grid_blocks(n)
    assert mb > 0


if __name__ == "__main__":
    test_cooperative_grid_sync_extern()
    test_cooperative_grid_sync_python()
