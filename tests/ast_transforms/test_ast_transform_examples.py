# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from textwrap import dedent

from numba_cuda_mlir.testing import filecheck_with_comments, run_in_subprocess


def test_consteval_prints():
    """Test that consteval prints happen at compile time, not runtime."""

    stdout, stderr = run_in_subprocess(
        dedent(
            """
        import warnings
        warnings.filterwarnings("ignore")

        from numba_cuda_mlir import cuda
        from numba_cuda_mlir.cuda.experimental import consteval
        import numpy as np

        def should_assign_one(loop_iter: int):
            print(f"COMPTIME: should_assign_one({loop_iter})")
            return loop_iter < 3

        @cuda.jit
        def k(x):
            consteval(print("COMPTIME: start of kernel"))
            print("RUNTIME: start of kernel")

            for i in consteval(range(5)):
                consteval(print(f"COMPTIME: unrolled iteration {i}"))
                print("RUNTIME: unrolled iteration", i)
                if consteval(should_assign_one(i)):
                    x[i] = 1
                else:
                    x[i] = 2

            print("RUNTIME: kernel finished")
            consteval(print("COMPTIME: kernel finished"))

        x = np.zeros(5, dtype=np.float32)
        k[1, 1](x)

        cuda.synchronize()
        np.testing.assert_array_equal(x, [1, 1, 1, 2, 2])
        """
        )
    )
    output = stderr + stdout
    print(output)
    # CHECK: COMPTIME: start of kernel
    # CHECK-NEXT: COMPTIME: unrolled iteration 0
    # CHECK-NEXT: COMPTIME: should_assign_one(0)
    # CHECK-NEXT: COMPTIME: unrolled iteration 1
    # CHECK-NEXT: COMPTIME: should_assign_one(1)
    # CHECK-NEXT: COMPTIME: unrolled iteration 2
    # CHECK-NEXT: COMPTIME: should_assign_one(2)
    # CHECK-NEXT: COMPTIME: unrolled iteration 3
    # CHECK-NEXT: COMPTIME: should_assign_one(3)
    # CHECK-NEXT: COMPTIME: unrolled iteration 4
    # CHECK-NEXT: COMPTIME: should_assign_one(4)
    # CHECK-NEXT: COMPTIME: kernel finished
    # CHECK-NEXT: RUNTIME: start of kernel
    # CHECK-NEXT: RUNTIME: unrolled iteration 0
    # CHECK-NEXT: RUNTIME: unrolled iteration 1
    # CHECK-NEXT: RUNTIME: unrolled iteration 2
    # CHECK-NEXT: RUNTIME: unrolled iteration 3
    # CHECK-NEXT: RUNTIME: unrolled iteration 4
    # CHECK-NEXT: RUNTIME: kernel finished

    filecheck_with_comments(output)
