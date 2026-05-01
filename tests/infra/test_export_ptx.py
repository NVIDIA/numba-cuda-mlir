# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir import types, compiler, testing
import numpy as np
import pytest


def test_export_ptx():
    @cuda.jit(opt_level=3)
    def k(x: cuda.DeviceNDArray):
        print(x)

    ptx = compiler.compile_ptx(k, types.void(types.float32[:]))
    assert ptx is not None
    # CHECK: .visible .entry _ZN15test_export_ptx12_3clocals_3e1kE5ArrayIfLi1E1A7mutable7alignedE
    testing.filecheck_with_comments(ptx)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    test_export_ptx()
