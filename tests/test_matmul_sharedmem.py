# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from numba_cuda_mlir import cuda
import math

TPB = 16


@cuda.jit(opt_level=3, fastmath=True, dump=True)
def kernel(A: cuda.DeviceNDArray, B: cuda.DeviceNDArray, C: cuda.DeviceNDArray):
    sA: cuda.DeviceNDArray = cuda.shared_array(shape=(TPB, TPB), dtype=np.float32)
    sB: cuda.DeviceNDArray = cuda.shared_array(shape=(TPB, TPB), dtype=np.float32)

    x, y = cuda.grid(2)
    tx: int = cuda.threadIdx.x
    ty: int = cuda.threadIdx.y
    bpg: int = math.ceil(A.shape[1] / TPB)

    if x >= C.shape[0] or y >= C.shape[1]:
        return

    tmp = 0.0
    for i in range(bpg):
        ax = x
        ay = i * TPB + ty
        bx = i * TPB + tx
        by = y
        if ax < A.shape[0] and ay < A.shape[1]:
            sA[tx, ty] = A[ax, ay]
        else:
            sA[tx, ty] = 0.0

        if bx < B.shape[0] and by < B.shape[1]:
            sB[tx, ty] = B[bx, by]
        else:
            sB[tx, ty] = 0.0

        cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]
        cuda.syncthreads()

    C[x, y] = tmp


def test_cuda() -> None:
    shape = (128, 128)
    A = np.random.randn(*shape).astype(np.float32)
    B = np.random.randn(*shape).astype(np.float32)
    C = np.zeros_like(A)
    A_dev = cuda.to_device(A)
    B_dev = cuda.to_device(B)
    C_dev = cuda.to_device(C)
    stream = int(cuda.default_stream())
    sharedmem = int(TPB * TPB * 2 * np.dtype(np.float32).itemsize)
    threads = (TPB, TPB, 1)
    blocks = (math.ceil(C.shape[0] / TPB), math.ceil(C.shape[1] / TPB), 1)

    kernel[blocks, threads, stream, sharedmem](A_dev, B_dev, C_dev)

    cuda.synchronize()

    C_dev.copy_to_host(C)

    expected = A @ B
    np.testing.assert_allclose(C, expected, rtol=1e-3, atol=1e-3)
    print("Tests passed")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    test_cuda()
