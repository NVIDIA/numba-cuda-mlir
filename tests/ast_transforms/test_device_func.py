# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda.experimental import consteval
import numpy as np
from numba_cuda_mlir import tools
from numba_cuda_mlir.testing import filecheck_with_comments

sm_arch = tools.get_gpu_compute_capability()


def epi_sm100(x):
    print("RUNTIME: running epilogue specialized for sm100")
    x[0] += 1


def epi_sm101(x):
    print("RUNTIME: running epilogue specialized for sm101")
    x[0] += 2


def epi_default(x):
    print("RUNTIME: running epilogue specialized for default arch")
    x[0] += 1


def epilogue_for(arch: str):
    print(f"COMPTIME: using arch={arch}")
    if arch == "sm_100":
        return epi_sm100
    elif arch == "sm_101":
        return epi_sm101
    else:
        return epi_default


def kernel_factory(M):  # M is CT-constant
    config = dict(assign_one=True, N=5, arch=sm_arch)  # CT-constant

    @cuda.jit
    def kernel(x):
        print("RUNTIME: Start of kernel")
        for i in consteval(range(config["N"] + 1)):
            consteval(print(f"COMPTIME: unrolled loop iteration {i}"))
            if consteval(config["assign_one"]):
                print("RUNTIME: assigning ", M, " to index: ", i)
                x[i] = M
            else:
                x[i] = 2
        epi = consteval(cuda.jit(epilogue_for(config["arch"])))
        epi(x)

    return kernel


def test_device_func(capfd):
    x = np.zeros(5, dtype=np.float32)
    x_d = cuda.to_device(x)
    kernel_factory(777)[1, 1](x_d)
    x = x_d.copy_to_host()
    output = capfd.readouterr().out
    print(output)

    # CHECK: COMPTIME: unrolled loop iteration 0
    # CHECK-NEXT: COMPTIME: unrolled loop iteration 1
    # CHECK-NEXT: COMPTIME: unrolled loop iteration 2
    # CHECK-NEXT: COMPTIME: unrolled loop iteration 3
    # CHECK-NEXT: COMPTIME: unrolled loop iteration 4
    # CHECK-NEXT: COMPTIME: unrolled loop iteration 5
    # CHECK-NEXT: COMPTIME: using arch={{.*}}
    # CHECK-NEXT: RUNTIME: Start of kernel
    # CHECK-NEXT: RUNTIME: assigning 777 to index: 0
    # CHECK: RUNTIME: running epilogue specialized for {{.*}}

    filecheck_with_comments(output)
    np.testing.assert_array_equal(x, [778.0, 777.0, 777.0, 777.0, 777.0])


if __name__ == "__main__":
    test_device_func()
