# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cuda.simt as cs
import numpy as np
from cusimt.testing import filecheck_with_comments


def test_consteval_prints(capfd):
    """Test that consteval prints happen at compile time, not runtime."""

    def should_assign_one(loop_iter: int):
        print(f"COMPTIME: should_assign_one({loop_iter})")
        return loop_iter < 3

    @cs.jit(experimental_ast_transforms=True)
    def k(x):
        cs.consteval(print("COMPTIME: start of kernel"))
        print("RUNTIME: start of kernel")

        for i in cs.consteval(range(5)):
            cs.consteval(print(f"COMPTIME: unrolled iteration {i}"))
            print("RUNTIME: unrolled iteration", i)
            if cs.consteval(should_assign_one(i)):
                x[i] = 1
            else:
                x[i] = 2

        print("RUNTIME: kernel finished")
        cs.consteval(print("COMPTIME: kernel finished"))

    x = np.zeros(5, dtype=np.float32)
    k[1, 1](x)
    import cuda.simt

    cuda.simt.synchronize()
    output = capfd.readouterr().out
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

    np.testing.assert_array_equal(x, [1, 1, 1, 2, 2])
