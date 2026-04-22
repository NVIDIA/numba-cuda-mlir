# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cuda.simt as cs
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)


def test_cuda_builtins():
    ctx = cs.current_context()

    @cs.jit(opt_level=3, dump=True)
    def kernel(x_dev):
        print(cs.threadIdx.x)
        print(cs.threadIdx.y)
        print(cs.threadIdx.z)
        print(cs.blockIdx.x)
        print(cs.blockIdx.y)
        print(cs.blockIdx.z)
        print(cs.blockDim.x)
        print(cs.gridsize(1))
        x, y = cs.gridsize(2)
        print(x)
        print(y)
        x, y, z = cs.gridsize(3)
        print(x)
        print(y)
        print(z)
        print(cs.grid(1))
        x, y = cs.grid(2)
        print(x)
        print(y)
        x, y, z = cs.grid(3)
        print(x)
        print(y)
        print(z)

        s = x_dev.shape
        s0 = s[0]
        cs.syncthreads()

    stream = int(cs.default_stream())
    x_dev = cs.to_device(np.zeros(2, dtype=np.int32))
    kernel[1, 1, stream, 0](x_dev)
    x_host = x_dev.copy_to_host()
    assert x_host[0] == 0


if __name__ == "__main__":
    test_cuda_builtins()
