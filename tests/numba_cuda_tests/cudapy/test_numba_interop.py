# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

import cusimt
import cuda.simt as cuda
from cusimt.testing import NumbaCUDATestCase
from cusimt.extending import overload


class TestNumbaInterop(NumbaCUDATestCase):
    def test_overload_inline_always(self):
        # From Issue #624
        def get_42():
            raise NotImplementedError()

        @overload(get_42, target="cuda", inline="always")
        def ol_blas_get_accumulator():
            def impl():
                return 42

            return impl

        @cusimt.jit
        def kernel(a):
            a[0] = get_42()

        a = np.empty(1, dtype=np.float32)
        kernel[1, 1](a)
        np.testing.assert_equal(a[0], 42)
