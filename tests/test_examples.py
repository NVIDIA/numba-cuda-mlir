# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
It is important that we ensure the code examples we circulate are
not just "slide-ware" but demonstrate something about how numba-cuda-mlir
works. These tests contain examples that were at one point
used in a presentation.
"""

import numpy as np
from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda import literal_unroll
from numba_cuda_mlir.testing import filecheck


def test_metaprogramming_examples():
    """
    For metaprogramming slides delivered Jan 9th, 2026.
    """

    def factory(loop_body, N):
        def kernel_body(a):
            for i in literal_unroll(range(N)):
                loop_body(a, i)

        return cuda.jit(kernel_body)

    @cuda.jit
    def baz(x, i):
        """
        Assign value to index in array.
        Use easily identifiable numbers so it is easy to map
        the Python code to the generated PTX and MLIR.
        """
        x[i] = 777

    @cuda.jit
    def fiz(x, i):
        """Same as the above device function"""
        x[i] = 999

    array = np.zeros((5,), dtype=np.int32)
    kernel_baz_2 = factory(baz, 2)
    # factory(baz, 2)[1, 1](x)
    kernel_baz_2[1, 1](array)
    assert all(array == [777, 777, 0, 0, 0])

    (ptx,) = kernel_baz_2.inspect_ptx().values()
    filecheck(
        """
        CHECK: .visible .entry
        CHECK-SAME: kernel_body
        CHECK-COUNT-2: st.global.{{[bu]}}32 [%r{{.+}}], {{.+}};
        CHECK: ret;
        """,
        ptx,
    )

    array = np.zeros((5,), dtype=np.int32)
    # factory(fizz, 3)[1, 1](array)
    kernel_fiz_3 = factory(fiz, 3)
    kernel_fiz_3[1, 1](array)
    assert all(array == [999, 999, 999, 0, 0])

    (ptx,) = kernel_fiz_3.inspect_ptx().values()
    filecheck(
        """
        CHECK: .visible .entry
        CHECK-SAME: kernel_body
        CHECK-COUNT-3: st.global.{{[bu]}}32 [%r{{.+}}], {{.+}};
        CHECK: ret;
        """,
        ptx,
    )
