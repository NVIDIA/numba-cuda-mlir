# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir import types, compiler, testing
import numpy as np
import pytest


def test_export_mlir():
    @cuda.jit(opt_level=3)
    def k(x: cuda.DeviceNDArray):
        print(x)

    mlir = compiler.compile_mlir(k, types.void(types.float32[:]))
    assert mlir is not None
    testing.filecheck_with_comments(mlir)
    # CHECK-LABEL:     gpu.func @_ZN16test_export_mlir12_3clocals_3e1kE5ArrayIfLi1E1A7mutable7alignedE(
    # CHECK-SAME:      %{{.*}}: memref<?xf32, strided<[?], offset: ?>>
    # CHECK:                  gpu.printf "memref<shape=["
    # CHECK:                  gpu.printf "\0A"
    # CHECK-NEXT:             gpu.return
    # CHECK-NEXT:           }


def test_export_mlir_optimized():
    @cuda.jit(opt_level=3)
    def k(x: cuda.DeviceNDArray):
        print(x)

    mlir = compiler.compile_mlir(k, types.void(types.float32[:]), optimized=True)
    assert mlir is not None
    testing.filecheck_with_comments(mlir)
    # CHECK: llvm.func @_ZN26test_export_mlir_optimized12_3clocals_3e1kE5ArrayIfLi1E1A7mutable7alignedE
    # CHECK: llvm.call @vprintf


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    test_export_mlir()
