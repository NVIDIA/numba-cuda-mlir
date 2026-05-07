# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda import experimental
from numba_cuda_mlir import compiler, testing
import numpy as np
from numba_cuda_mlir.numba_cuda.types import float32
from numba_cuda_mlir import tools

brk = cuda.intrin.breakpoint
nanosleep = cuda.intrin.nanosleep
_cc = tools.get_gpu_compute_capability(tuple)
CHIP = _cc[0] * 10 + _cc[1]


@pytest.mark.skipif(
    condition=CHIP < 70,
    reason="nanosleep intrinsic requires sm_70 or higher compute capability",
)
def test_inline_ptx_manual():
    ctx = cuda.current_context()
    sig = (float32, float32)

    @cuda.jit(opt_level=3, signature=sig)
    def kernel(a, b):
        brk()
        nanosleep(50)
        experimental.inline_ptx("brkpt;")
        res = experimental.inline_ptx(
            """.reg .u32 t1;
                mul.lo.u32 t1, %1, %1;
                mul.lo.u32 %0, t1, %1;
                """,
            "=r",
            np.int32,
            "r",
            2,
        )

        # Test multiple return values from inline_ptx
        res1, res2 = experimental.inline_ptx(
            """add.u32 %0, %2, %3;
                sub.u32 %1, %2, %3;
                """,
            "=r",
            np.int32,  # output 1: a + b
            "=r",
            np.int32,  # output 2: a - b
            "r",
            5,  # input a
            "r",
            3,  # input b
        )

        print("res: ", res)
        print("res1 (5+3): ", res1)
        print("res2 (5-3): ", res2)

    cres = compiler.compile_result(kernel, sig)
    testing.filecheck(
        """
        CHECK: nvvm.inline_ptx "brkpt;"
        CHECK: nvvm.breakpoint
        CHECK: nvvm.nanosleep
        """,
        cres.mlir_module,
    )
    testing.filecheck(
        """
        CHECK:       .reg .u32 t1;
        CHECK-NEXT:     mul.lo.u32 t1, %r{{[0-9]+}}, %r{{[0-9]+}};
        CHECK-NEXT:     mul.lo.u32 %r{{[0-9]+}}, t1, %r{{[0-9]+}};
        CHECK:       add.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}};
        CHECK-NEXT:     sub.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}};
        """,
        cres.ptx,
    )


if __name__ == "__main__":
    test_inline_ptx_manual()
