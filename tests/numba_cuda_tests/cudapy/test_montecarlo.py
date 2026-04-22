# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import math
import cusimt
import cuda.simt as cuda
from cusimt.testing import NumbaCUDATestCase


class TestCudaMonteCarlo(NumbaCUDATestCase):
    def test_montecarlo(self):
        """Just make sure we can compile this"""

        @cusimt.jit("void(double[:], double[:], double, double, double, double[:])")
        def step(last, paths, dt, c0, c1, normdist):
            i = cuda.grid(1)
            if i >= paths.shape[0]:
                return
            noise = normdist[i]
            paths[i] = last[i] * math.exp(c0 * dt + c1 * noise)
