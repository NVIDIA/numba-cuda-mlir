# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda.experimental import consteval
import numpy as np


@cuda.jit
def gemm(A, B, C, i):
    print("gemm iteration ", i)
    C[i, i] = A[i, i] + B[i, i]
    return C


class Epilogue:
    @staticmethod
    def run(matrix):
        print("identity epilogue")
        return matrix


class ReLU(Epilogue):
    @staticmethod
    def run(matrix):
        print("ReLU epilogue")
        return matrix


def kernel_factory(Epilogue):
    @cuda.jit
    def kernel(A, B, C):
        matrix = C
        for i in range(C.shape[1]):
            matrix = gemm(A, B, C, i)
        consteval(cuda.jit(Epilogue.run))(matrix)

    return kernel


def test_example():
    A = np.random.randn(10, 10).astype(np.float32)
    B = np.random.randn(10, 10).astype(np.float32)
    C = np.zeros((10, 10), dtype=np.float32)

    kernel_factory(Epilogue)[1, 1](A, B, C)
    kernel_factory(ReLU)[1, 1](A, B, C)

    assert np.allclose(np.diag(C), np.diag(A + B))


if __name__ == "__main__":
    test_example()
