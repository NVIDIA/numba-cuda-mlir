# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

import numba_cuda_mlir
from numba_cuda_mlir import extending
from numba_cuda_mlir import cuda
from numba_cuda_mlir.testing import NumbaCUDATestCase
from numba_cuda_mlir.extending import overload, typing_registry


class TestNumbaInterop(NumbaCUDATestCase):
    def test_overload_inline_always(self):
        # From Issue #624
        def get_42():
            raise NotImplementedError()

        @overload(
            get_42,
            target="cuda",
            inline="always",
            typing_registry=typing_registry,
        )
        def ol_blas_get_accumulator():
            def impl():
                return 42

            return impl

        extending.refresh_registries()

        @numba_cuda_mlir.cuda.jit
        def kernel(a):
            a[0] = get_42()

        a = np.empty(1, dtype=np.float32)
        kernel[1, 1](a)
        np.testing.assert_equal(a[0], 42)
