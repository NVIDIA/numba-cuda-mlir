# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.types import literal
import numpy as np
import pytest

literal_unroll = cuda.misc.special.literal_unroll
literally = cuda.misc.special.literally


def test_literal_unroll():
    @cuda.jit
    def k(x: cuda.DeviceNDArray):
        for i in literal_unroll(range(10)):
            x[i] = i

    x = cuda.to_device(np.zeros(10, dtype=np.int32))
    k[1, 1](x)
    x = x.copy_to_host()
    assert np.all(x == np.arange(10))


@pytest.mark.parametrize(
    "value,constant",
    [(8, "i64 8"), (-3, "i64 -3"), (2**64 - 1, "i64 -1"), (True, "i8 1"), (False, "i8 0")],
)
def test_scalar_literal_argument_is_constant(value, constant):
    @cuda.jit(device=True, chip="sm_100")
    def identity(value):
        return value

    llvm_ir = identity.inspect_llvm((literal(value),))
    assert f"ret {constant}" in llvm_ir


def test_literal_loop_bound_promotes_local_array():
    @cuda.jit(chip="sm_100")
    def kernel(inp, out, count):
        values = cuda.local.array(8, dtype=types.float32)
        for i in range(count):
            values[i] = inp[i]
        total = 0.0
        for i in range(count):
            total += values[i]
        out[0] = total

    ptx = kernel.compile((types.float32[::1], types.float32[::1], literal(8)), output="ptx").ptx
    assert ".local" not in ptx


def test_literal_argument_dispatch_keeps_runtime_arguments():
    @cuda.jit
    def kernel(out, value, scale):
        value = literally(value)
        out[0] = value * scale + 1

    out = cuda.to_device(np.zeros(1, dtype=np.int64))
    for value, scale in [(1, 3), (True, 5), (0, 7), (False, 9), (-3, 2), (8, 4), (1, 11)]:
        kernel[1, 1](out, value, scale)
        assert out.copy_to_host()[0] == value * scale + 1
    assert {args[1] for args in kernel.overloads} == {
        literal(value) for value in (1, True, 0, False, -3, 8)
    }


@pytest.mark.parametrize("first,second", [(3, 8), (False, True)])
def test_literal_device_specializations_have_distinct_symbols(first, second):
    @cuda.jit(device=True, inline="never")
    def twice(value):
        value = literally(value)
        return 2 * value

    @cuda.jit
    def kernel(out):
        out[0] = twice(first)
        out[1] = twice(second)

    out = cuda.to_device(np.zeros(2, dtype=np.int64))
    kernel[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), [2 * first, 2 * second])


@pytest.mark.xfail
def test_literal_eval():
    arr = np.float32([1, 2, 3])

    @cuda.jit
    def k(x: cuda.DeviceNDArray):
        arr = literally(arr)
        for i in range(3):
            x[i] = arr[i]

    x = cuda.to_device(np.zeros(3, dtype=np.float32))
    k[1, 1](x)
    x = x.copy_to_host()
    assert np.all(x == np.float32([1, 2, 3]))
