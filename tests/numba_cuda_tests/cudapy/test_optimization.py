# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

import numba_cuda_mlir
from numba_cuda_mlir import cuda
from numba_cuda_mlir.testing import NumbaCUDATestCase
from numba_cuda_mlir.numba_cuda.types import float64
import pytest


def kernel_func(x):
    x[0] = 1


def device_func(x, y, z):
    return x * y + z


# Fragments of code that are removed from kernel_func's PTX when optimization
# is on. Previously this list was longer when kernel wrappers were used - if
# the test function were more complex it may be possible to isolate additional
# fragments of PTX we could check for the absence / presence of, but removal of
# the use of local memory is a good indicator that optimization was applied.
removed_by_opt = ("__local_depot0",)


class TestOptimization(NumbaCUDATestCase):
    def test_eager_opt(self):
        # Optimization should occur by default
        sig = (float64[::1],)
        kernel = numba_cuda_mlir.jit(sig)(kernel_func)
        ptx = kernel.inspect_asm()

        for fragment in removed_by_opt:
            self.assertNotIn(fragment, ptx[sig])

    @pytest.mark.xfail(True, reason="Regex doesn't match")
    def test_eager_noopt(self):
        # Optimization disabled
        sig = (float64[::1],)
        kernel = numba_cuda_mlir.jit(sig, opt=False)(kernel_func)
        ptx = kernel.inspect_asm()

        for fragment in removed_by_opt:
            self.assertIn(fragment, ptx[sig])

    def test_lazy_opt(self):
        # Optimization should occur by default
        kernel = numba_cuda_mlir.jit(kernel_func)
        x = np.zeros(1, dtype=np.float64)
        kernel[1, 1](x)

        # Grab the PTX for the one definition that has just been jitted
        ptx = next(iter(kernel.inspect_asm().items()))[1]

        for fragment in removed_by_opt:
            self.assertNotIn(fragment, ptx)

    @pytest.mark.xfail(True, reason="Regex doesn't match")
    def test_lazy_noopt(self):
        # Optimization disabled
        kernel = numba_cuda_mlir.jit(opt=False)(kernel_func)
        x = np.zeros(1, dtype=np.float64)
        kernel[1, 1](x)

        # Grab the PTX for the one definition that has just been jitted
        ptx = next(iter(kernel.inspect_asm().items()))[1]

        for fragment in removed_by_opt:
            self.assertIn(fragment, ptx)

    def test_device_opt(self):
        # Optimization should occur by default
        sig = (float64, float64, float64)
        device = numba_cuda_mlir.jit(sig, device=True)(device_func)
        ptx = device.inspect_asm(sig)
        self.assertIn("fma.rn.f64", ptx)

    def test_device_noopt(self):
        # Optimization disabled
        sig = (float64, float64, float64)
        device = numba_cuda_mlir.jit(sig, device=True, opt=False)(device_func)
        ptx = device.inspect_asm(sig)
        # Fused-multiply adds should be disabled when not optimizing
        self.assertNotIn("fma.rn.f64", ptx)
