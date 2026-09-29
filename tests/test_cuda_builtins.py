# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.errors import TypingError
import numpy as np
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)


def test_cuda_builtins():
    ctx = cuda.current_context()

    @cuda.jit(opt_level=3, dump=True)
    def kernel(x_dev):
        print(cuda.threadIdx.x)
        print(cuda.threadIdx.y)
        print(cuda.threadIdx.z)
        print(cuda.blockIdx.x)
        print(cuda.blockIdx.y)
        print(cuda.blockIdx.z)
        print(cuda.blockDim.x)
        print(cuda.gridsize(1))
        x, y = cuda.gridsize(2)
        print(x)
        print(y)
        x, y, z = cuda.gridsize(3)
        print(x)
        print(y)
        print(z)
        print(cuda.grid(1))
        x, y = cuda.grid(2)
        print(x)
        print(y)
        x, y, z = cuda.grid(3)
        print(x)
        print(y)
        print(z)

        s = x_dev.shape
        s0 = s[0]
        cuda.syncthreads()

    stream = int(cuda.default_stream())
    x_dev = cuda.to_device(np.zeros(2, dtype=np.int32))
    kernel[1, 1, stream, 0](x_dev)
    x_host = x_dev.copy_to_host()
    assert x_host[0] == 0


def _grid_runtime_dim(d, out):
    out[0] = cuda.grid(d[0])


def _gridsize_runtime_dim(d, out):
    out[0] = cuda.gridsize(d[0])


@pytest.mark.parametrize(
    "body, intrinsic",
    [(_grid_runtime_dim, "grid"), (_gridsize_runtime_dim, "gridsize")],
    ids=["grid", "gridsize"],
)
def test_grid_runtime_dimensions_error_handling(body, intrinsic):
    """A dimension that is not a compile-time constant is rejected with a TypingError."""

    with pytest.raises(TypingError) as excinfo:
        cuda.jit(body).compile(types.void(types.int64[:], types.int64[:]))
    msg = str(excinfo.value)
    assert "Cannot request literal type" in msg
    assert "During: resolving callee type: Function(<intrinsic %s>)" % intrinsic in msg
    assert "test_cuda_builtins.py" in msg


if __name__ == "__main__":
    test_cuda_builtins()
