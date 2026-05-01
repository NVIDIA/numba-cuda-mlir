# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir import types, compiler, testing
import numpy as np
import pytest


def test_export_cubin():
    @cuda.jit(opt_level=3, device=True)
    def k(x: cuda.DeviceNDArray):
        print(x)

    cubin = compiler.compile_cubin(k, types.void(types.float32[:]))
    assert cubin is not None
    assert cubin[:4] == b"\x7fELF"


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    test_export_cubin()
