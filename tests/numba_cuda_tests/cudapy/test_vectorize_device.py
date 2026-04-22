# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from cuda.simt import vectorize
import cusimt
from numba.types import float32
import numpy as np
from cusimt.testing import NumbaCUDATestCase
import pytest


@pytest.mark.xfail(True, reason="Vectorize not supported")
class TestCudaVectorizeDeviceCall(NumbaCUDATestCase):
    def test_cuda_vectorize_device_call(self):
        @cusimt.jit(float32(float32, float32, float32), device=True)
        def cu_device_fn(x, y, z):
            return x**y / z

        def cu_ufunc(x, y, z):
            return cu_device_fn(x, y, z)

        ufunc = vectorize([float32(float32, float32, float32)], target="cuda")(cu_ufunc)

        N = 100

        X = np.array(np.random.sample(N), dtype=np.float32)
        Y = np.array(np.random.sample(N), dtype=np.float32)
        Z = np.array(np.random.sample(N), dtype=np.float32) + 0.1

        out = ufunc(X, Y, Z)

        gold = (X**Y) / Z

        self.assertTrue(np.allclose(out, gold))
