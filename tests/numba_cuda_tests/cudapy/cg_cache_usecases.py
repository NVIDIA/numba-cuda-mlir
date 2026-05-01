# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numba_cuda_mlir
from numba_cuda_mlir import cuda
from numba_cuda_mlir.testing import NumbaCUDATestCase
import sys

from .cache_usecases import CUDAUseCase


# Usecase with cooperative groups


@numba_cuda_mlir.jit(cache=True)
def cg_usecase_kernel(r, x):
    grid = cuda.cg.this_grid()
    grid.sync()


cg_usecase = CUDAUseCase(cg_usecase_kernel)


class _TestModule(NumbaCUDATestCase):
    """
    Tests for functionality of this module's functions.
    Note this does not define any "test_*" method, instead check_module()
    should be called by hand.
    """

    def check_module(self, mod):
        mod.cg_usecase(0)


def self_test():
    mod = sys.modules[__name__]
    _TestModule().check_module(mod)
