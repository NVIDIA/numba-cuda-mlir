# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import cuda.simt as cuda
import logging

logging.basicConfig(level=logging.DEBUG)


def test_cuda():
    @cuda.jit(dump=False)
    def kernel(A: cuda.DeviceNDArray, B: cuda.DeviceNDArray, C: cuda.DeviceNDArray):
        i = cuda.blockIdx.x
        j = cuda.blockIdx.y
        if i < C.shape[0] and j < C.shape[1]:
            tmp = 0.0
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp

    shape = (64, 64)
    A = np.random.randn(*shape).astype(np.float32)
    B = np.random.randn(*shape).astype(np.float32)
    C = np.zeros_like(A)
    A_dev = cuda.to_device(A)
    B_dev = cuda.to_device(B)
    C_dev = cuda.to_device(C)

    kernel[(C.shape[0], C.shape[1], 1), 1](A_dev, B_dev, C_dev)

    C_dev.copy_to_host(C)

    expected = A @ B
    np.testing.assert_allclose(C, expected, rtol=1e-5, atol=1e-5)
    print("Test passed")


if __name__ == "__main__":
    test_cuda()
