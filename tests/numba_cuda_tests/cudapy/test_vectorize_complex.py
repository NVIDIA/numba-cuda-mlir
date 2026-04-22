# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from cuda.simt import vectorize
from cusimt.testing import NumbaCUDATestCase
import pytest


@pytest.mark.xfail(True, reason="Vectorize not supported")
class TestVectorizeComplex(NumbaCUDATestCase):
    def test_vectorize_complex(self):
        @vectorize(["complex128(complex128)"], target="cuda")
        def vcomp(a):
            return a * a + 1.0

        A = np.arange(5, dtype=np.complex128)
        B = vcomp(A)
        self.assertTrue(np.allclose(A * A + 1.0, B))
