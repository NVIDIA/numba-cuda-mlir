# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
import numpy as np
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


if __name__ == "__main__":
    test_cuda_builtins()
