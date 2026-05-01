# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import numba_cuda_mlir
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda.np.numpy_support import from_dtype
from numba_cuda_mlir.testing import NumbaCUDATestCase
import pytest


class TestAlignment(NumbaCUDATestCase):
    def test_record_alignment(self):
        rec_dtype = np.dtype([("a", "int32"), ("b", "float64")], align=True)
        rec = from_dtype(rec_dtype)

        @numba_cuda_mlir.jit((rec[:],))
        def foo(a):
            i = cuda.grid(1)
            a[i].a = a[i].b

        a_recarray = np.recarray(3, dtype=rec_dtype)
        for i in range(a_recarray.size):
            a_rec = a_recarray[i]
            a_rec.a = 0
            a_rec.b = (i + 1) * 123

        foo[1, 3](a_recarray)

        self.assertTrue(np.all(a_recarray.a == a_recarray.b))

    @pytest.mark.xfail(True, reason="Assertion not raised")
    def test_record_alignment_error(self):
        rec_dtype = np.dtype([("a", "int32"), ("b", "float64")])
        rec = from_dtype(rec_dtype)

        msg = "type float64 is not aligned"
        with self.assertRaisesRegex(Exception, msg) as raises:

            @numba_cuda_mlir.jit((rec[:],))
            def foo(a):
                i = cuda.grid(1)
                a[i].a = a[i].b

        self.assertTrue("type float64 is not aligned" in str(raises.exception))
